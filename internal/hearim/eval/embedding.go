package eval

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"hearim/internal/hearim/compile"
	"hearim/internal/hearim/jev"
	"hearim/internal/hearim/provider"
	"hearim/internal/hearim/scoring"
)

// Embedding scoring methods reported in x-jev-scoring-method.
const (
	MethodEmbeddingCosine  = "embedding-cosine"
	MethodEmbeddingAnchors = "embedding-cosine+anchors"
)

// evaluateEmbedding answers one question from embedding similarity:
//
//	z_i = cos(embed(instructions + state), embed(option_i))
//	p   = softmax(z / tau)
//
// With labeled exemplars for the question (AnchorStore), each option also
// gets a soft-nearest-neighbour term over cos(state, exemplar state),
// centered across the options that have exemplars so options without any
// stay neutral. No text is generated and no logprob is read.
func (e *Evaluator) evaluateEmbedding(ctx context.Context, plan *compile.EvaluationPlan, qi int, route Route) (*QuestionResult, error) {
	q := plan.Questions[qi]
	if len(plan.Images) > 0 {
		return nil, jev.NewError(jev.CodeValidationFailed,
			"state declares %d image(s); embedding model %s scores text only", len(plan.Images), route.BackendModel)
	}
	emb := route.Embedder
	cfg := emb.Config()
	fail := func(err error) (*QuestionResult, error) {
		return nil, jev.NewQuestionError(jev.CodeBackendProbabilityUnavailable, q.ID, "embedding request failed: %v", err)
	}

	state := RenderState(plan.CanonicalState)
	queries := []string{strings.TrimSpace(q.Instructions + "\n" + state)}
	anchors := route.Anchors.For(q)
	if len(anchors) > 0 {
		queries = append(queries, state)
	}
	qv, tokens, err := emb.Embed(ctx, provider.EmbedQuery, queries)
	if err != nil {
		return fail(err)
	}
	options := make([]string, len(q.Candidates))
	for i, c := range q.Candidates {
		options[i] = optionText(q, c)
	}
	ov, n, err := emb.Embed(ctx, provider.EmbedDocument, options)
	if err != nil {
		return fail(err)
	}
	tokens += n

	z := make([]float64, len(options))
	for i := range options {
		z[i] = dot(qv[0], ov[i])
	}
	method := MethodEmbeddingCosine
	if len(anchors) > 0 {
		a := make([]float64, len(options))
		has := make([]bool, len(options))
		var sum float64
		var present int
		for i, c := range q.Candidates {
			texts := anchors[c.PublicKey]
			if len(texts) == 0 {
				continue
			}
			av, n, err := emb.Embed(ctx, provider.EmbedQuery, texts)
			if err != nil {
				return fail(err)
			}
			tokens += n
			sims := make([]float64, len(av))
			for j, v := range av {
				sims[j] = dot(qv[1], v) / cfg.AnchorTau
			}
			a[i] = cfg.AnchorTau * (logSumExp(sims) - math.Log(float64(len(sims))))
			has[i] = true
			sum += a[i]
			present++
		}
		mean := sum / float64(present)
		for i := range z {
			if has[i] {
				z[i] += *cfg.AnchorWeight * (a[i] - mean)
			}
		}
		method = MethodEmbeddingAnchors
	}

	for i := range z {
		z[i] /= cfg.Tau
	}
	probs := scoring.Softmax(z)
	answer, err := reduce(q, probs, cfg.Tau)
	if err != nil {
		return nil, err
	}
	prof := scoring.CalibrationProfile{
		Model: route.BackendModel, Engine: string(route.Engine),
		TemplateVersion: "embedding-v1", ScoringMethod: method, Tau: cfg.Tau,
	}
	return &QuestionResult{
		ID:            q.ID,
		Answer:        answer,
		InputTokens:   int64(tokens),
		CandidateMass: 1, // the distribution is defined over the candidates only
		ScoringMethod: method,
		Attempts:      []string{method},
		ProfileID:     prof.ID(),
	}, nil
}

// optionText is the text embedded for one candidate: "name: description"
// for choice, the level description for score, and for noul the true/false
// description or a plain yes/no sentence when none was given.
func optionText(q *compile.CompiledQuestion, c compile.Candidate) string {
	if q.Type != jev.TypeNoul {
		return c.SequenceText
	}
	if _, desc, ok := strings.Cut(c.SequenceText, ": "); ok {
		return desc
	}
	if c.PublicKey == "true" {
		return "Yes, this is true."
	}
	return "No, this is false."
}

// RenderState flattens canonical state JSON into readable "a · b: value"
// lines; embedding models read those far better than raw JSON. String
// states pass through unchanged.
func RenderState(canonical string) string {
	var v any
	if err := json.Unmarshal([]byte(canonical), &v); err != nil {
		return canonical
	}
	if s, ok := v.(string); ok {
		return s
	}
	var lines []string
	flattenState(v, "", &lines)
	return strings.Join(lines, "\n")
}

func flattenState(v any, path string, lines *[]string) {
	join := func(k string) string {
		k = strings.ReplaceAll(k, "_", " ")
		if path == "" {
			return k
		}
		return path + " · " + k
	}
	switch x := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			flattenState(x[k], join(k), lines)
		}
	case []any:
		nested := false
		for _, el := range x {
			switch el.(type) {
			case map[string]any, []any:
				nested = true
			}
		}
		if nested {
			for i, el := range x {
				flattenState(el, join(fmt.Sprint(i+1)), lines)
			}
			return
		}
		parts := make([]string, len(x))
		for i, el := range x {
			parts[i] = scalarText(el)
		}
		*lines = append(*lines, linePrefix(path)+strings.Join(parts, "; "))
	default:
		*lines = append(*lines, linePrefix(path)+scalarText(x))
	}
}

func linePrefix(path string) string {
	if path == "" {
		return ""
	}
	return path + ": "
}

func scalarText(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case nil:
		return "null"
	default:
		b, _ := json.Marshal(x)
		return string(b)
	}
}

func dot(a, b []float32) float64 {
	n := min(len(a), len(b))
	var s float64
	for i := 0; i < n; i++ {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func logSumExp(x []float64) float64 {
	m := math.Inf(-1)
	for _, v := range x {
		m = math.Max(m, v)
	}
	var s float64
	for _, v := range x {
		s += math.Exp(v - m)
	}
	return m + math.Log(s)
}

// AnchorStore holds labeled exemplars for embedding routes, grouped by
// question and answer. Adding an exemplar is the whole "training" step:
// a state close to it is pulled toward its answer.
type AnchorStore struct {
	// question signature -> answer public key -> rendered exemplar states
	byQuestion map[string]map[string][]string
	count      int
}

// LoadAnchors reads a JSONL file in the `hearim bench` corpus format
// ({"state", "question", "expected"}). Lines whose expected answer is not
// one of the question's options are rejected.
func LoadAnchors(path string, c *compile.Compiler) (*AnchorStore, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("eval: anchors: %w", err)
	}
	defer f.Close()
	store := &AnchorStore{byQuestion: map[string]map[string][]string{}}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	for line := 1; sc.Scan(); line++ {
		text := strings.TrimSpace(sc.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		var rec struct {
			State    json.RawMessage `json:"state"`
			Question *jev.Question   `json:"question"`
			Expected json.RawMessage `json:"expected"`
		}
		if err := json.Unmarshal([]byte(text), &rec); err != nil {
			return nil, fmt.Errorf("eval: anchors %s:%d: %w", path, line, err)
		}
		pr, err := jev.ValidateRequest(&jev.Request{
			Model: "anchor", State: rec.State, Questions: map[string]*jev.Question{"anchor": rec.Question},
		})
		if err != nil {
			return nil, fmt.Errorf("eval: anchors %s:%d: %w", path, line, err)
		}
		plan, err := c.Unlabeled().Compile(pr, "anchor")
		if err != nil {
			return nil, fmt.Errorf("eval: anchors %s:%d: %w", path, line, err)
		}
		q := plan.Questions[0]
		key := expectedKey(rec.Expected)
		known := false
		for _, cand := range q.Candidates {
			known = known || cand.PublicKey == key
		}
		if !known {
			return nil, fmt.Errorf("eval: anchors %s:%d: expected %s is not an option of the question", path, line, rec.Expected)
		}
		sig := questionSignature(q)
		if store.byQuestion[sig] == nil {
			store.byQuestion[sig] = map[string][]string{}
		}
		store.byQuestion[sig][key] = append(store.byQuestion[sig][key], RenderState(plan.CanonicalState))
		store.count++
	}
	return store, sc.Err()
}

// Len reports the number of loaded exemplars.
func (s *AnchorStore) Len() int {
	if s == nil {
		return 0
	}
	return s.count
}

// For returns the exemplar states per answer key for a question, matched on
// its type, instructions and option texts. Nil when there are none.
func (s *AnchorStore) For(q *compile.CompiledQuestion) map[string][]string {
	if s == nil {
		return nil
	}
	return s.byQuestion[questionSignature(q)]
}

func questionSignature(q *compile.CompiledQuestion) string {
	parts := []string{string(q.Type), q.Instructions}
	for _, c := range q.Candidates {
		parts = append(parts, c.SequenceText)
	}
	return strings.Join(parts, "\x00")
}

// expectedKey maps a corpus "expected" value (choice name, level index,
// or bool) to the candidate public key.
func expectedKey(raw json.RawMessage) string {
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return ""
	}
	switch x := v.(type) {
	case string:
		return x
	case bool:
		return fmt.Sprint(x)
	case float64:
		return fmt.Sprint(int(x))
	}
	return ""
}
