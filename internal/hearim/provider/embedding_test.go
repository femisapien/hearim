package provider

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"hearim/internal/hearim/config"
)

// embedTexts pulls the input texts out of a request body in any of the five
// wire formats.
func embedTexts(body map[string]any) []string {
	var out []string
	add := func(v any) {
		for _, t := range v.([]any) {
			out = append(out, t.(string))
		}
	}
	switch {
	case body["input"] != nil:
		add(body["input"])
	case body["inputs"] != nil:
		add(body["inputs"])
	case body["texts"] != nil:
		add(body["texts"])
	case body["requests"] != nil:
		for _, r := range body["requests"].([]any) {
			parts := r.(map[string]any)["content"].(map[string]any)["parts"].([]any)
			out = append(out, parts[0].(map[string]any)["text"].(string))
		}
	}
	return out
}

func TestEmbedderWireFormats(t *testing.T) {
	// Each text embeds to (len, 1) unnormalized, so results are checkable.
	vec := func(s string) []float64 { return []float64{float64(len(s)), 1} }
	cases := []struct {
		api, engine, path, model string
		check                    func(t *testing.T, r *http.Request, body map[string]any)
		respond                  func(texts []string) any
		tokens                   int
	}{
		{
			api: "openai", engine: "vllm", path: "/v1/embeddings", model: "m",
			check: func(t *testing.T, r *http.Request, body map[string]any) {
				if body["model"] != "m" || r.Header.Get("Authorization") != "Bearer k" {
					t.Errorf("openai: model=%v auth=%q", body["model"], r.Header.Get("Authorization"))
				}
			},
			respond: func(texts []string) any {
				data := []any{} // reversed: the client must order by index
				for i := len(texts) - 1; i >= 0; i-- {
					data = append(data, map[string]any{"index": i, "embedding": vec(texts[i])})
				}
				return map[string]any{"data": data, "usage": map[string]any{"prompt_tokens": 7}}
			},
			tokens: 7,
		},
		{
			api: "tei", engine: "tei", path: "/embed", model: "m",
			check: func(t *testing.T, r *http.Request, body map[string]any) {
				if body["normalize"] != true || body["model"] != nil {
					t.Errorf("tei: body=%v", body)
				}
			},
			respond: func(texts []string) any {
				out := [][]float64{}
				for _, s := range texts {
					out = append(out, vec(s))
				}
				return out
			},
		},
		{
			api: "ollama", engine: "ollama", path: "/api/embed", model: "m",
			check: func(t *testing.T, r *http.Request, body map[string]any) {},
			respond: func(texts []string) any {
				out := [][]float64{}
				for _, s := range texts {
					out = append(out, vec(s))
				}
				return map[string]any{"embeddings": out, "prompt_eval_count": 5}
			},
			tokens: 5,
		},
		{
			api: "cohere", engine: "cohere", path: "/v2/embed", model: "m",
			check: func(t *testing.T, r *http.Request, body map[string]any) {
				if body["input_type"] != "search_document" {
					t.Errorf("cohere: input_type=%v", body["input_type"])
				}
			},
			respond: func(texts []string) any {
				out := [][]float64{}
				for _, s := range texts {
					out = append(out, vec(s))
				}
				return map[string]any{
					"embeddings": map[string]any{"float": out},
					"meta":       map[string]any{"billed_units": map[string]any{"input_tokens": 3}},
				}
			},
			tokens: 3,
		},
		{
			api: "gemini", engine: "gemini", path: "/v1beta/models/m:batchEmbedContents", model: "m",
			check: func(t *testing.T, r *http.Request, body map[string]any) {
				if r.Header.Get("x-goog-api-key") != "k" || r.Header.Get("Authorization") != "" {
					t.Errorf("gemini: auth headers = %q / %q", r.Header.Get("x-goog-api-key"), r.Header.Get("Authorization"))
				}
				req := body["requests"].([]any)[0].(map[string]any)
				if req["model"] != "models/m" || req["taskType"] != "CLASSIFICATION" {
					t.Errorf("gemini: request=%v", req)
				}
			},
			respond: func(texts []string) any {
				out := []any{}
				for _, s := range texts {
					out = append(out, map[string]any{"values": vec(s)})
				}
				return map[string]any{"embeddings": out}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.api, func(t *testing.T) {
			var calls int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				atomic.AddInt32(&calls, 1)
				if r.URL.Path != tc.path {
					t.Errorf("path = %s, want %s", r.URL.Path, tc.path)
				}
				var body map[string]any
				json.NewDecoder(r.Body).Decode(&body)
				tc.check(t, r, body)
				json.NewEncoder(w).Encode(tc.respond(embedTexts(body)))
			}))
			defer srv.Close()

			cfg := config.ProviderConfig{
				ID: "p", Engine: config.EngineKind(tc.engine), BaseURL: srv.URL, APIKey: "k",
				Models: config.ModelConfigs{{Name: tc.model, Type: config.ModelTypeEmbedding}},
				Embedding: &config.EmbeddingConfig{
					API:            tc.api, // explicit: the ollama engine defaults to its OpenAI surface
					DocumentPrefix: "d: ",
					DocumentParams: map[string]any{"taskType": "CLASSIFICATION"},
				},
			}
			emb := NewEmbedder(cfg, tc.model)
			if emb == nil || emb.API() != tc.api {
				t.Fatalf("embedder api = %v, want %s", emb, tc.api)
			}
			// "ab" and "abcd" get the 3-byte prefix: vectors (5,1) and (7,1).
			vecs, tokens, err := emb.Embed(context.Background(), EmbedDocument, []string{"ab", "abcd", "ab"})
			if err != nil {
				t.Fatal(err)
			}
			if tokens != tc.tokens {
				t.Errorf("tokens = %d, want %d", tokens, tc.tokens)
			}
			want := [][2]float64{{5, 1}, {7, 1}, {5, 1}}
			for i, w := range want {
				norm := math.Hypot(w[0], w[1])
				if math.Abs(float64(vecs[i][0])-w[0]/norm) > 1e-6 || math.Abs(float64(vecs[i][1])-w[1]/norm) > 1e-6 {
					t.Errorf("vec[%d] = %v, want normalized %v", i, vecs[i], w)
				}
			}
			// Second call is served from cache.
			if _, _, err := emb.Embed(context.Background(), EmbedDocument, []string{"abcd"}); err != nil {
				t.Fatal(err)
			}
			if calls != 1 {
				t.Errorf("upstream calls = %d, want 1 (cache)", calls)
			}
		})
	}
}

func TestEmbedderOnlyForEmbeddingModels(t *testing.T) {
	cfg := config.ProviderConfig{
		ID: "p", Engine: config.EngineVLLM, BaseURL: "http://x",
		Models: config.ModelConfigs{{Name: "llm"}, {Name: "emb", Type: config.ModelTypeEmbedding}},
	}
	if NewEmbedder(cfg, "llm") != nil {
		t.Error("plain model must not get an embedder")
	}
	if e := NewEmbedder(cfg, "emb"); e == nil || e.API() != "openai" || e.Config().Tau != 0.05 {
		t.Errorf("embedding model defaults = %+v", e)
	}
	cfg.Engine = config.EngineTEI
	if e := NewEmbedder(cfg, "anything"); e == nil || e.API() != "tei" {
		t.Error("every model on an embedding-only engine is embedding-scored")
	}
}
