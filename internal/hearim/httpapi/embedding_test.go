package httpapi

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeEmbeddings is an OpenAI-shaped embeddings backend with a bag-of-words
// model: one dimension per vocabulary word, so cosine similarity is word
// overlap and the expected answers are checkable by eye.
func fakeEmbeddings(t *testing.T, vocab []string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		var body struct {
			Input []string `json:"input"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		data := []any{}
		for i, text := range body.Input {
			vec := make([]float64, len(vocab)+1)
			vec[len(vocab)] = 0.01 // keeps texts without vocabulary words non-zero
			for j, word := range vocab {
				vec[j] = float64(strings.Count(strings.ToLower(text), word))
			}
			data = append(data, map[string]any{"index": i, "embedding": vec})
		}
		json.NewEncoder(w).Encode(map[string]any{"data": data, "usage": map[string]any{"prompt_tokens": 11}})
	}))
}

func newEmbeddingServer(t *testing.T, upstreamURL, embeddingYAML string) http.Handler {
	t.Helper()
	cfg, err := configParseForTest(fmt.Sprintf(`
gateway:
  default_model: jev-embed
auth:
  api_keys: []
providers:
  - id: emb
    engine: generic-openai
    base_url: %s
    models:
      - name: bow
        type: embedding
%s
model_aliases:
  jev-embed: emb:bow
`, upstreamURL, embeddingYAML))
	if err != nil {
		t.Fatal(err)
	}
	srv, err := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), "")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { srv.Shutdown(defaultCtx()) })
	if ok, detail := srv.Ready(); !ok {
		t.Fatalf("embedding route not ready: %s", detail)
	}
	return srv.Handler()
}

func postSystemOne(t *testing.T, h http.Handler, body string) (*httptest.ResponseRecorder, map[string]map[string]any) {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(body)))
	var resp struct {
		Answers map[string]map[string]any `json:"answers"`
	}
	json.Unmarshal(rec.Body.Bytes(), &resp)
	return rec, resp.Answers
}

func TestSystemOneEmbeddingRoute(t *testing.T) {
	up := fakeEmbeddings(t, []string{"refund", "login", "delivery", "angry", "calm"})
	defer up.Close()
	h := newEmbeddingServer(t, up.URL, "")

	rec, answers := postSystemOne(t, h, `{
	  "model": "jev-embed",
	  "state": {"ticket": "I cannot login to my account"},
	  "questions": {
	    "queue": {"type": "choice", "instructions": "Pick the queue.",
	      "criteria": {"billing": "refund or invoice", "account": "login problems", "shipping": "delivery issues"}},
	    "upset": {"type": "noul", "instructions": "The customer is upset.",
	      "criteria": {"true": "angry tone", "false": "calm tone"}},
	    "level": {"type": "score", "criteria": ["calm", "login trouble", "angry"]}
	  }}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if got := answers["queue"]["choice"]; got != "account" {
		t.Errorf("queue = %v, want account (%v)", got, answers["queue"])
	}
	if p := answers["upset"]["noul"].(float64); p < 0.49 || p > 0.51 {
		t.Errorf("noul = %v, want ~0.5: the state shares no word with either side", p)
	}
	if s := answers["level"]["score"].(float64); s < 0.9 || s > 1.1 {
		t.Errorf("score = %v, want ~1 (the login level)", s)
	}
	if got := rec.Header().Get("x-jev-scoring-method"); got != "embedding-cosine" {
		t.Errorf("scoring method header = %q", got)
	}
	if got := rec.Header().Get("x-jev-implementation"); got != "openai-embedding-v1" {
		t.Errorf("implementation header = %q", got)
	}
}

// Option counts beyond the single-token label alphabets are the case the
// logprob routes reject; the embedding route has no such limit.
func TestSystemOneEmbeddingManyOptions(t *testing.T) {
	vocab := make([]string, 80)
	criteria := []string{}
	for i := range vocab {
		vocab[i] = fmt.Sprintf("topic%02dx", i)
		criteria = append(criteria, fmt.Sprintf(`"opt%02d": "about %s"`, i, vocab[i]))
	}
	up := fakeEmbeddings(t, vocab)
	defer up.Close()
	h := newEmbeddingServer(t, up.URL, "")

	rec, answers := postSystemOne(t, h, fmt.Sprintf(
		`{"model":"jev-embed","state":"a note on topic57x","questions":{"q":{"type":"choice","criteria":{%s}}}}`,
		strings.Join(criteria, ",")))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if got := answers["q"]["choice"]; got != "opt57" {
		t.Errorf("choice = %v, want opt57", got)
	}
	if n := len(answers["q"]["probabilities"].(map[string]any)); n != 80 {
		t.Errorf("probabilities has %d options, want 80", n)
	}
}

func TestSystemOneEmbeddingAnchors(t *testing.T) {
	up := fakeEmbeddings(t, []string{"refund", "login", "zebra"})
	defer up.Close()
	question := `{"type":"choice","instructions":"Pick the queue.","criteria":{"billing":"refund or invoice","account":"login problems"}}`
	request := `{"model":"jev-embed","state":"the zebra thing happened again","questions":{"q":` + question + `}}`

	// Without anchors the state matches neither option.
	_, answers := postSystemOne(t, newEmbeddingServer(t, up.URL, ""), request)
	probs := answers["q"]["probabilities"].(map[string]any)
	if d := probs["billing"].(float64) - probs["account"].(float64); d < -0.01 || d > 0.01 {
		t.Fatalf("zero-shot should be a tie, got %v", probs)
	}

	// One labeled exemplar teaches that "zebra" tickets go to billing.
	path := filepath.Join(t.TempDir(), "anchors.jsonl")
	lines := `{"state":"zebra charge on my card","question":` + question + `,"expected":"billing"}` + "\n" +
		`{"state":"cannot login","question":` + question + `,"expected":"account"}` + "\n"
	if err := os.WriteFile(path, []byte(lines), 0o600); err != nil {
		t.Fatal(err)
	}
	h := newEmbeddingServer(t, up.URL, "        embedding:\n          anchors: "+path)
	rec, answers := postSystemOne(t, h, request)
	if got := answers["q"]["choice"]; got != "billing" {
		t.Errorf("choice = %v, want billing (%s)", got, rec.Body)
	}
	if got := rec.Header().Get("x-jev-scoring-method"); got != "embedding-cosine+anchors" {
		t.Errorf("scoring method header = %q", got)
	}
}
