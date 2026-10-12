package provider

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"

	"hearim/internal/hearim/config"
)

// EmbedSide distinguishes the two roles of an embedded text: the state-side
// query and the option-side document. Asymmetric models prompt them
// differently.
type EmbedSide int

const (
	EmbedQuery EmbedSide = iota
	EmbedDocument
)

// embedCacheMax bounds the per-embedder vector cache; it is dropped whole
// when full (option texts and anchors refill it within a request or two).
const embedCacheMax = 4096

// Embedder calls one provider's embeddings API for one model. It speaks the
// OpenAI, TEI, Ollama, Cohere and Gemini wire formats and returns
// L2-normalized vectors, so a dot product is the cosine similarity.
type Embedder struct {
	providerID string
	model      string
	cfg        config.EmbeddingConfig
	extras     map[string]any
	http       *httpClient

	mu    sync.Mutex
	cache map[string][]float32
}

// NewEmbedder builds the embedder for an embedding-scored model; it returns
// nil when the model is not one (config.ProviderConfig.EmbeddingFor).
func NewEmbedder(cfg config.ProviderConfig, model string) *Embedder {
	ecfg := cfg.EmbeddingFor(model)
	if ecfg == nil {
		return nil
	}
	if ecfg.API == config.EmbedAPIGemini && cfg.APIKey != "" {
		// Gemini authenticates with x-goog-api-key, not a bearer token.
		heads := map[string]string{"x-goog-api-key": cfg.APIKey}
		for k, v := range cfg.Headers {
			heads[k] = v
		}
		cfg.Headers, cfg.APIKey = heads, ""
	}
	return &Embedder{
		providerID: cfg.ID,
		model:      model,
		cfg:        *ecfg,
		extras:     ModelExtras(cfg, model),
		http:       newHTTPClient(cfg),
		cache:      map[string][]float32{},
	}
}

// Config returns the resolved embedding settings.
func (e *Embedder) Config() config.EmbeddingConfig { return e.cfg }

// API returns the wire format in use.
func (e *Embedder) API() string { return e.cfg.API }

// Embed returns one normalized vector per text plus the upstream-reported
// input token count (0 when the API reports none). Results are cached by
// side and text.
func (e *Embedder) Embed(ctx context.Context, side EmbedSide, texts []string) ([][]float32, int, error) {
	prefix, params := e.cfg.QueryPrefix, e.cfg.QueryParams
	if side == EmbedDocument {
		prefix, params = e.cfg.DocumentPrefix, e.cfg.DocumentParams
	}
	out := make([][]float32, len(texts))
	var missing []string
	seen := map[string]bool{}
	e.mu.Lock()
	for i, t := range texts {
		if v, ok := e.cache[cacheKey(side, t)]; ok {
			out[i] = v
		} else if !seen[t] {
			seen[t] = true
			missing = append(missing, t)
		}
	}
	e.mu.Unlock()

	tokens := 0
	fresh := map[string][]float32{}
	batch := e.batchSize()
	for start := 0; start < len(missing); start += batch {
		end := min(start+batch, len(missing))
		chunk := missing[start:end]
		inputs := make([]string, len(chunk))
		for i, t := range chunk {
			inputs[i] = prefix + t
		}
		vecs, n, err := e.call(ctx, side, inputs, params)
		if err != nil {
			return nil, 0, err
		}
		if len(vecs) != len(chunk) {
			return nil, 0, fmt.Errorf("provider: embeddings: got %d vectors for %d inputs", len(vecs), len(chunk))
		}
		tokens += n
		for i, v := range vecs {
			if err := normalize(v); err != nil {
				return nil, 0, err
			}
			fresh[chunk[i]] = v
		}
	}

	if len(fresh) > 0 {
		e.mu.Lock()
		if len(e.cache)+len(fresh) > embedCacheMax {
			e.cache = map[string][]float32{}
		}
		for t, v := range fresh {
			e.cache[cacheKey(side, t)] = v
		}
		e.mu.Unlock()
		for i, t := range texts {
			if out[i] == nil {
				out[i] = fresh[t]
			}
		}
	}
	return out, tokens, nil
}

func cacheKey(side EmbedSide, text string) string {
	if side == EmbedDocument {
		return "d\x00" + text
	}
	return "q\x00" + text
}

// batchSize is the per-request input cap each API accepts by default.
func (e *Embedder) batchSize() int {
	switch e.cfg.API {
	case config.EmbedAPITEI:
		return 32
	case config.EmbedAPICohere:
		return 96
	case config.EmbedAPIGemini:
		return 100
	default:
		return 256
	}
}

func (e *Embedder) path(def string) string {
	if e.cfg.Path != "" {
		return e.cfg.Path
	}
	return def
}

// call performs one upstream request in the configured wire format.
// Precedence of body fields: hearim's shape < extra_params < per-side params.
func (e *Embedder) call(ctx context.Context, side EmbedSide, inputs []string, params map[string]any) ([][]float32, int, error) {
	merge := func(body map[string]any) map[string]any {
		for _, src := range []map[string]any{e.extras, params} {
			for k, v := range src {
				if k != "model" && k != "input" && k != "inputs" && k != "texts" && k != "requests" {
					body[k] = v
				}
			}
		}
		return body
	}
	switch e.cfg.API {
	case config.EmbedAPIOpenAI:
		var resp struct {
			Data []struct {
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
			Usage struct {
				PromptTokens int `json:"prompt_tokens"`
				TotalTokens  int `json:"total_tokens"`
			} `json:"usage"`
		}
		body := merge(map[string]any{"model": e.model, "input": inputs, "encoding_format": "float"})
		if err := e.http.do(ctx, "POST", e.path("/v1/embeddings"), body, &resp); err != nil {
			return nil, 0, err
		}
		sort.SliceStable(resp.Data, func(i, j int) bool { return resp.Data[i].Index < resp.Data[j].Index })
		vecs := make([][]float32, len(resp.Data))
		for i, d := range resp.Data {
			vecs[i] = d.Embedding
		}
		return vecs, max(resp.Usage.PromptTokens, resp.Usage.TotalTokens), nil

	case config.EmbedAPITEI:
		var vecs [][]float32
		body := merge(map[string]any{"inputs": inputs, "normalize": true, "truncate": true})
		if err := e.http.do(ctx, "POST", e.path("/embed"), body, &vecs); err != nil {
			return nil, 0, err
		}
		return vecs, 0, nil

	case config.EmbedAPIOllama:
		var resp struct {
			Embeddings      [][]float32 `json:"embeddings"`
			PromptEvalCount int         `json:"prompt_eval_count"`
		}
		body := merge(map[string]any{"model": e.model, "input": inputs})
		if err := e.http.do(ctx, "POST", e.path("/api/embed"), body, &resp); err != nil {
			return nil, 0, err
		}
		return resp.Embeddings, resp.PromptEvalCount, nil

	case config.EmbedAPICohere:
		var resp struct {
			Embeddings struct {
				Float [][]float32 `json:"float"`
			} `json:"embeddings"`
			Meta struct {
				BilledUnits struct {
					InputTokens int `json:"input_tokens"`
				} `json:"billed_units"`
			} `json:"meta"`
		}
		inputType := "search_query"
		if side == EmbedDocument {
			inputType = "search_document"
		}
		body := merge(map[string]any{
			"model": e.model, "texts": inputs, "input_type": inputType, "embedding_types": []string{"float"},
		})
		if err := e.http.do(ctx, "POST", e.path("/v2/embed"), body, &resp); err != nil {
			return nil, 0, err
		}
		return resp.Embeddings.Float, resp.Meta.BilledUnits.InputTokens, nil

	case config.EmbedAPIGemini:
		var resp struct {
			Embeddings []struct {
				Values []float32 `json:"values"`
			} `json:"embeddings"`
		}
		reqs := make([]map[string]any, len(inputs))
		for i, t := range inputs {
			reqs[i] = merge(map[string]any{
				"model":    "models/" + e.model,
				"content":  map[string]any{"parts": []map[string]any{{"text": t}}},
				"taskType": "SEMANTIC_SIMILARITY",
			})
			reqs[i]["model"] = "models/" + e.model
		}
		path := e.path("/v1beta/models/" + e.model + ":batchEmbedContents")
		if err := e.http.do(ctx, "POST", path, map[string]any{"requests": reqs}, &resp); err != nil {
			return nil, 0, err
		}
		vecs := make([][]float32, len(resp.Embeddings))
		for i, em := range resp.Embeddings {
			vecs[i] = em.Values
		}
		return vecs, 0, nil
	}
	return nil, 0, fmt.Errorf("provider: unknown embedding api %q", e.cfg.API)
}

func normalize(v []float32) error {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if len(v) == 0 || sum == 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		return fmt.Errorf("provider: embeddings: empty or degenerate vector")
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
	return nil
}
