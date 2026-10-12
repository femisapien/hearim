# Embedding-scored models

hearim can answer `POST /v1/systemone` from an **embedding model** instead of
next-token logprobs. Each option's probability comes from the cosine
similarity between the embedded state and the embedded option text. No text
is generated and no logprob is read, so the route works with any embeddings
API and has no single-token label limit.

한국어: [EMBEDDING.ko.md](EMBEDDING.ko.md)

## When to use it

Measured on one machine (M2 Ultra), 150 records per corpus (±7pp), accuracy.
The logprob column is hearim's method on a local Qwen3.8-27B; the embedding
column is `embeddinggemma-300m` through this route, zero-shot.

| Corpus | Question | LLM logprob | Embedding | Embedding + anchors |
|---|---|---|---|---|
| AG News | choice, 4 topics | 0.847 | 0.647 | 0.793 |
| Banking77 | choice, 77 intents | cannot run (77 > label alphabet) | 0.773 | 0.887 |
| YNAT (Korean) | choice, 7 topics | 0.707 | 0.440 | 0.593 |
| SST-2 | noul, sentiment | 0.960 | 0.633 | 0.787 |
| KLUE NLI (Korean) | choice, entailment | 0.933 | 0.293 | 0.340 |

Latency was 34–46 ms per request for the embedding route against 2.4–6.9 s
for the LLM under an 8-way parallel load on a shared server (indicative only).

- **Good fit:** "what is this text about" questions — topic and intent
  routing, especially with more options than the label alphabets hold, and
  paths that must answer in tens of milliseconds.
- **Poor fit:** judgments that need reasoning — entailment, sentiment
  polarity, decisions that combine several state fields. Texts that mean
  opposite things sit close together in embedding space; use a logprob route.
- Results swing with the model and its task prompt. Run `hearim bench` on
  your own labeled records before relying on a route.

The embedding route is never chosen automatically. A request uses it only
when its `model` resolves to an embedding-scored model.

## Enable it

Mark a model `type: embedding` on any provider and point an alias at it:

```yaml
providers:
  - id: embed-local
    engine: generic-openai
    base_url: http://embeddings:8080
    models:
      - name: embeddinggemma-300m
        type: embedding
        embedding:
          query_prefix: "task: sentence similarity | query: "
          document_prefix: "task: sentence similarity | query: "

model_aliases:
  jev-embed: embed-local:embeddinggemma-300m
```

Providers whose engine is `tei`, `cohere` or `gemini` are embedding-only:
every model on them is embedding-scored and no `models` list is needed.

```yaml
providers:
  - id: tei
    engine: tei
    base_url: http://tei:8080
  - id: cohere
    engine: cohere
    api_key: ${COHERE_API_KEY}
  - id: gemini
    engine: gemini
    api_key: ${GEMINI_API_KEY}

model_aliases:
  jev-tei: tei:embeddinggemma-300m
  jev-cohere: cohere:embed-v4.0
  jev-gemini: gemini:gemini-embedding-001
```

`hearim probe` makes one embeddings call per embedding model and reports the
wire format and vector size. `hearim bench -model <alias>` works unchanged.

## Wire formats

`embedding.api` selects the format. The default follows the engine.

| `api` | Request | Default for engine | Auth | Reported tokens |
|---|---|---|---|---|
| `openai` | `POST /v1/embeddings` `{model, input[]}` | `generic-openai`, `vllm`, `llama.cpp`, `ollama`, `sglang` | `Authorization: Bearer` | `usage.prompt_tokens` |
| `tei` | `POST /embed` `{inputs[], normalize, truncate}` | `tei` | `Authorization: Bearer` | none |
| `ollama` | `POST /api/embed` `{model, input[]}` | (set explicitly) | `Authorization: Bearer` | `prompt_eval_count` |
| `cohere` | `POST /v2/embed` `{model, texts[], input_type, embedding_types}` | `cohere` | `Authorization: Bearer` | `meta.billed_units.input_tokens` |
| `gemini` | `POST /v1beta/models/{model}:batchEmbedContents` | `gemini` | `x-goog-api-key` | none |

The OpenAI format also covers LM Studio, Jina, Voyage and TEI's
OpenAI-compatible path. Anthropic has no embeddings API.

Inputs are batched per request (32 for TEI, 96 for Cohere, 100 for Gemini,
256 otherwise). Vectors are L2-normalized by hearim and cached per route, so
a repeated option set costs one upstream call for the state.

**Verification status.** The `openai`, `tei` and `ollama` formats were run
against a live sentence-transformers server (`extensions/embedding/serve.py`)
and gave identical benchmark results. `cohere` and `gemini` are tested only
against servers that imitate the documented request and response shapes; they
have not been run against the real services.

## Settings

All fields are optional and may be set on the provider (defaults for its
models) or on a model (overrides field by field).

| Field | Default | Meaning |
|---|---|---|
| `api` | by engine | `openai` \| `tei` \| `ollama` \| `cohere` \| `gemini` |
| `path` | by `api` | embeddings URL path override |
| `query_prefix` | `""` | text prepended to the state-side input |
| `document_prefix` | `""` | text prepended to each option |
| `query_params` | none | JSON fields merged into state-side requests |
| `document_params` | none | JSON fields merged into option-side requests |
| `tau` | `0.05` | softmax temperature over cosine similarities |
| `anchors` | none | JSONL file of labeled exemplars |
| `anchor_weight` | `1` | scale of the anchor term |
| `anchor_tau` | `0.05` | soft-nearest-neighbour temperature of the anchor term |

Prefixes carry the task prompt an embedding model was trained with
(EmbeddingGemma: `task: sentence similarity | query: `). Per-side params
carry the same idea for APIs that take it as a field:

```yaml
embedding:                       # Jina
  query_params: {task: text-matching}
  document_params: {task: text-matching}
embedding:                       # Cohere (defaults: search_query / search_document)
  query_params: {input_type: classification}
  document_params: {input_type: classification}
embedding:                       # Gemini (default: SEMANTIC_SIMILARITY)
  query_params: {taskType: CLASSIFICATION}
  document_params: {taskType: CLASSIFICATION}
```

Provider and model `extra_params` are merged into every embeddings request
as well (for example `dimensions`).

## How a question is scored

```text
query    = instructions + "\n" + state        (state flattened to "a · b: value" lines)
option_i = "name: description"                 choice
           level description                   score
           true / false description            noul ("Yes, this is true." / "No, this is false." when absent)

z_i = cos(embed(query), embed(option_i))
p   = softmax(z / tau)
```

`choice`, `score` and `noul` are then reduced exactly as on logprob routes
(argmax, probability-weighted level, P(true)); `confidence` is the same
normalized entropy. Diagnostics:

- `x-jev-scoring-method: embedding-cosine` (or `embedding-cosine+anchors`)
- `x-jev-implementation: <api>-embedding-v1`
- `usage.input_tokens` is what the embeddings API reported, `output_tokens` is 0

The probabilities are a temperature-scaled similarity ranking, not a model's
belief about the answer. `tau` sets how sharp they are; fit it on your own
data if you consume the probabilities and not only the top answer.

Limits: image-bearing states are rejected (422), and an embedding-scored
default model has no `/v1/completions` proxy surface.

## Anchors: teaching by example

An anchor is a labeled record: a state whose answer to a question is known.
With anchors loaded, a state close to an exemplar is pulled toward that
exemplar's answer:

```text
a_i   = anchor_tau * log mean_j exp(cos(embed(state), embed(exemplar_ij)) / anchor_tau)
z_i  += anchor_weight * (a_i - mean of a over options that have exemplars)
```

Options without exemplars are left as they are. No weights are trained;
adding a line to the file and restarting is the whole update.

The file uses the `hearim bench` corpus format, one record per line:

```jsonl
{"state": "the card I ordered still has not arrived", "question": {"type": "choice", "instructions": "Which intent?", "criteria": {"card_arrival": null, "lost_card": null}}, "expected": "card_arrival"}
```

Anchors apply to a request only when its question has the same type,
instructions and option texts as the anchor's question. `GET /v1/routes`
shows the number of loaded exemplars.

Measured gain with `embeddinggemma-300m` (accuracy at 0 / 1 / 4 / 16
exemplars per option): Banking77 0.773 / 0.820 / 0.880 / 0.887, SST-2
0.633 / 0.767 / 0.780 / 0.787, YNAT 0.440 / 0.447 / 0.613 / 0.593. Emotion
and Yelp did not move beyond sampling error, and NLI stayed at chance:
anchors shift boundaries the model can already see, nothing more. Most of
the gain arrives by four exemplars per option.

## Training an embedding model (extension)

`extensions/embedding/` holds an optional Python recipe; hearim itself needs
none of it.

- `train.py` fine-tunes a sentence-transformers encoder on labeled records
  with the objective this route serves: cross-entropy over
  `softmax(cos(query, option) / tau)` within each question, the other options
  acting as hard negatives. Only the encoder is trained, so the result is an
  ordinary embedding model that any embeddings server can host.
- `serve.py` serves a model directory over the `openai`, `tei` and `ollama`
  formats for local use.
- `jevtext.py` renders states and options exactly as the gateway does.

```bash
pip install -r extensions/embedding/requirements.txt
python extensions/embedding/train.py \
  --model google/embeddinggemma-300m \
  --train train.jsonl --eval eval.jsonl --out ./jev-embed \
  --query-prefix "task: sentence similarity | query: " \
  --document-prefix "task: sentence similarity | query: "
python extensions/embedding/serve.py --model ./jev-embed --port 8090
```

Use the same prefixes and `tau` in training and in the hearim config.

### What fine-tuning does (measured)

`embeddinggemma-300m`, accuracy before and after, one seed, 150 records per
corpus (400 for typed-decisions). "Specialist" trained on 3,000 typed-decisions
records for 3 epochs; "mixed" trained on 2,392 records from typed-decisions,
AG News, Emotion and Banking77 for 2 epochs. About 18 minutes each on an
M2 Ultra that was also serving other models.

| Corpus | Base | Specialist | Mixed |
|---|---|---|---|
| typed-decisions | 0.463 | **0.693** (trained) | 0.568 (trained) |
| AG News | 0.647 | 0.647 | **0.820** (trained) |
| Emotion | 0.533 | 0.387 | **0.633** (trained) |
| Banking77 | 0.773 | 0.467 | **0.853** (trained) |
| SST-2 | 0.633 | 0.733 | 0.707 |
| Yelp | 0.553 | 0.407 | 0.453 |
| YNAT (Korean) | 0.440 | 0.540 | 0.693 |
| KLUE NLI (Korean) | 0.327 | 0.327 | 0.287 |

- Fine-tuning lifts the schemas it was trained on by 10 to 23 points.
- **A model trained on one schema family is a specialist.** The
  typed-decisions model lost 31 points on Banking77 and 15 on Emotion and
  Yelp. Give such a model its own alias and keep the base model for
  everything else.
- Mixing tasks avoided that collapse and carried over to two of the four
  corpora it never saw (SST-2, YNAT), while Yelp fell by 10 points. One run
  is not enough to call that a rule; evaluate on held-out records of your own.
- Nothing here fixes reasoning-type questions: NLI stayed at chance.
- `score` questions are trained with hard labels; neighbouring levels are not
  treated as closer than distant ones.
- The model served through hearim reproduced the training script's accuracy
  (Banking77 0.853, AG News 0.820, YNAT 0.693), which confirms that the
  script and the gateway render the same text.

**Why not a trained head on top of the embeddings?** A head (as in
`mrm8488/embeddinggemma-2-ft-typed-decision`) scores higher on the schemas it
was trained on (0.763 on typed-decisions), but it cannot be served through an
embeddings API, and on seven corpora with unseen schemas that model's head
was worse than the plain cosine of its own encoder every time (AG News 0.507
against 0.853).
