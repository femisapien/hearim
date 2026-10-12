# 임베딩 채점 모델

hearim은 `POST /v1/systemone`을 next-token logprob 대신 **임베딩 모델**로
답할 수 있다. 각 선택지의 확률은 임베딩한 state와 임베딩한 선택지 텍스트
사이의 코사인 유사도에서 나온다. 텍스트를 생성하지 않고 logprob도 읽지
않으므로, 어떤 임베딩 API와도 동작하고 한 토큰 라벨 제한이 없다.

English: [EMBEDDING.md](EMBEDDING.md)

## 언제 쓰는가

장비 한 대(M2 Ultra)에서 코퍼스당 150건(±7%p)으로 잰 정확도다. logprob
열은 로컬 Qwen3.8-27B에 hearim 방식을 적용한 값이고, 임베딩 열은
`embeddinggemma-300m`을 이 경로로 zero-shot 채점한 값이다.

| 코퍼스 | 질문 | LLM logprob | 임베딩 | 임베딩 + 앵커 |
|---|---|---|---|---|
| AG News | choice, 주제 4개 | 0.847 | 0.647 | 0.793 |
| Banking77 | choice, 의도 77개 | 실행 불가 (77 > 라벨 알파벳) | 0.773 | 0.887 |
| YNAT (한국어) | choice, 주제 7개 | 0.707 | 0.440 | 0.593 |
| SST-2 | noul, 감성 | 0.960 | 0.633 | 0.787 |
| KLUE NLI (한국어) | choice, 함의 | 0.933 | 0.293 | 0.340 |

지연시간은 임베딩 경로가 요청당 34~46ms, LLM은 공유 서버의 8개 병렬
부하에서 2.4~6.9초였다(규모만 참고).

- **잘 맞는 경우:** "이 글이 무엇에 관한 것인가"를 묻는 질문. 주제·의도
  라우팅, 특히 선택지가 라벨 알파벳보다 많을 때와 수십 ms 안에 답해야 하는
  경로.
- **맞지 않는 경우:** 추론이 필요한 판정. 함의, 감성 극성, state의 여러
  필드를 종합하는 판단. 뜻이 반대인 글도 임베딩 공간에서는 가깝다. 이런
  질문에는 logprob 경로를 쓴다.
- 결과는 모델과 task prompt에 따라 크게 달라진다. 경로를 신뢰하기 전에
  자신의 라벨 데이터로 `hearim bench`를 돌려 본다.

임베딩 경로는 자동으로 선택되지 않는다. 요청의 `model`이 임베딩 채점
모델로 해석될 때만 쓰인다.

## 켜는 법

어떤 provider든 모델에 `type: embedding`을 붙이고 alias를 연결한다.

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

엔진이 `tei`, `cohere`, `gemini`인 provider는 임베딩 전용이다. 그 위의 모든
모델이 임베딩으로 채점되고 `models` 목록이 필요 없다.

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

`hearim probe`는 임베딩 모델마다 임베딩 호출을 한 번 하고 API 형식과 벡터
크기를 보고한다. `hearim bench -model <alias>`는 그대로 동작한다.

## API 형식

`embedding.api`로 형식을 고른다. 기본값은 엔진을 따른다.

| `api` | 요청 | 기본 적용 엔진 | 인증 | 보고되는 토큰 |
|---|---|---|---|---|
| `openai` | `POST /v1/embeddings` `{model, input[]}` | `generic-openai`, `vllm`, `llama.cpp`, `ollama`, `sglang` | `Authorization: Bearer` | `usage.prompt_tokens` |
| `tei` | `POST /embed` `{inputs[], normalize, truncate}` | `tei` | `Authorization: Bearer` | 없음 |
| `ollama` | `POST /api/embed` `{model, input[]}` | (명시적으로 지정) | `Authorization: Bearer` | `prompt_eval_count` |
| `cohere` | `POST /v2/embed` `{model, texts[], input_type, embedding_types}` | `cohere` | `Authorization: Bearer` | `meta.billed_units.input_tokens` |
| `gemini` | `POST /v1beta/models/{model}:batchEmbedContents` | `gemini` | `x-goog-api-key` | 없음 |

OpenAI 형식은 LM Studio, Jina, Voyage, TEI의 OpenAI 호환 경로도 포함한다.
Anthropic에는 임베딩 API가 없다.

입력은 요청당 묶음으로 보낸다(TEI 32, Cohere 96, Gemini 100, 그 외 256).
벡터는 hearim이 L2 정규화하고 route별로 캐시하므로, 같은 선택지 집합이
반복되면 state에 대한 upstream 호출 한 번만 든다.

**검증 상태.** `openai`, `tei`, `ollama` 형식은 실제 sentence-transformers
서버(`extensions/embedding/serve.py`)로 돌렸고 벤치마크 결과가 서로 같았다.
`cohere`와 `gemini`는 문서화된 요청·응답 형태를 흉내 낸 서버로만
테스트했고, 실제 서비스에는 돌려 보지 않았다.

## 설정

모든 필드는 선택이며 provider(그 모델들의 기본값)나 모델(필드 단위
덮어쓰기)에 둘 수 있다.

| 필드 | 기본값 | 의미 |
|---|---|---|
| `api` | 엔진에 따름 | `openai` \| `tei` \| `ollama` \| `cohere` \| `gemini` |
| `path` | `api`에 따름 | 임베딩 URL 경로 덮어쓰기 |
| `query_prefix` | `""` | state 쪽 입력 앞에 붙는 텍스트 |
| `document_prefix` | `""` | 각 선택지 앞에 붙는 텍스트 |
| `query_params` | 없음 | state 쪽 요청에 병합되는 JSON 필드 |
| `document_params` | 없음 | 선택지 쪽 요청에 병합되는 JSON 필드 |
| `tau` | `0.05` | 코사인 유사도에 대한 softmax 온도 |
| `anchors` | 없음 | 라벨 붙은 예시 JSONL 파일 |
| `anchor_weight` | `1` | 앵커 항의 크기 |
| `anchor_tau` | `0.05` | 앵커 항의 soft 최근접 이웃 온도 |

prefix에는 임베딩 모델이 학습 때 쓴 task prompt를 넣는다(EmbeddingGemma:
`task: sentence similarity | query: `). 같은 것을 필드로 받는 API는 쪽별
params로 넘긴다.

```yaml
embedding:                       # Jina
  query_params: {task: text-matching}
  document_params: {task: text-matching}
embedding:                       # Cohere (기본: search_query / search_document)
  query_params: {input_type: classification}
  document_params: {input_type: classification}
embedding:                       # Gemini (기본: SEMANTIC_SIMILARITY)
  query_params: {taskType: CLASSIFICATION}
  document_params: {taskType: CLASSIFICATION}
```

provider와 모델의 `extra_params`도 모든 임베딩 요청에 병합된다(예:
`dimensions`).

## 질문 하나가 채점되는 방식

```text
query    = instructions + "\n" + state        (state는 "a · b: value" 줄로 펼친다)
option_i = "이름: 설명"                        choice
           단계 설명                           score
           true / false 설명                   noul (없으면 "Yes, this is true." / "No, this is false.")

z_i = cos(embed(query), embed(option_i))
p   = softmax(z / tau)
```

그 뒤 `choice`, `score`, `noul`은 logprob 경로와 똑같이 환원된다(argmax,
확률 가중 단계, P(true)). `confidence`도 같은 정규화 entropy다. 진단 정보:

- `x-jev-scoring-method: embedding-cosine` (또는 `embedding-cosine+anchors`)
- `x-jev-implementation: <api>-embedding-v1`
- `usage.input_tokens`는 임베딩 API가 보고한 값, `output_tokens`는 0

이 확률은 온도로 조정한 유사도 순위이지, 답에 대한 모델의 믿음이 아니다.
`tau`가 분포의 날카로움을 정한다. 최상위 답만이 아니라 확률값을 쓴다면
자신의 데이터로 맞춘다.

제한: 이미지가 들어 있는 state는 거부한다(422). 기본 모델이 임베딩 채점
모델이면 `/v1/completions` 프록시 표면이 없다.

## 앵커: 예시로 가르치기

앵커는 라벨 붙은 레코드, 즉 어떤 질문에 대한 답이 알려진 state다. 앵커를
로드하면 예시에 가까운 state가 그 예시의 답 쪽으로 끌린다.

```text
a_i   = anchor_tau * log mean_j exp(cos(embed(state), embed(exemplar_ij)) / anchor_tau)
z_i  += anchor_weight * (a_i - 예시가 있는 선택지들의 a 평균)
```

예시가 없는 선택지는 그대로 둔다. 가중치 학습은 없다. 파일에 한 줄을
추가하고 재시작하는 것이 갱신의 전부다.

파일은 `hearim bench` 코퍼스 형식이고 한 줄에 레코드 하나다.

```jsonl
{"state": "주문한 카드가 아직 안 왔어요", "question": {"type": "choice", "instructions": "어떤 의도인가?", "criteria": {"card_arrival": null, "lost_card": null}}, "expected": "card_arrival"}
```

앵커는 요청의 질문이 앵커의 질문과 유형, instructions, 선택지 텍스트가
같을 때만 적용된다. `GET /v1/routes`가 로드된 예시 수를 보여 준다.

`embeddinggemma-300m`으로 잰 효과(선택지당 예시 0 / 1 / 4 / 16개일 때
정확도): Banking77 0.773 / 0.820 / 0.880 / 0.887, SST-2 0.633 / 0.767 /
0.780 / 0.787, YNAT 0.440 / 0.447 / 0.613 / 0.593. Emotion과 Yelp는 표본
오차 범위를 벗어나지 않았고 NLI는 무작위 수준에 머물렀다. 앵커는 모델이
이미 구분할 수 있는 경계를 옮길 뿐이다. 효과의 대부분은 선택지당 4개에서
나온다.

## 임베딩 모델 학습 (확장)

`extensions/embedding/`에 선택적 Python 레시피가 있다. hearim 자체는 이것을
필요로 하지 않는다.

- `train.py`는 라벨 붙은 레코드로 sentence-transformers 인코더를 이 경로가
  서빙하는 목적함수 그대로 미세조정한다. 질문 안에서
  `softmax(cos(query, option) / tau)`에 대한 cross-entropy이고, 같은 질문의
  다른 선택지가 hard negative가 된다. 인코더만 학습하므로 결과물은 어떤
  임베딩 서버에도 올릴 수 있는 일반 임베딩 모델이다.
- `serve.py`는 모델 디렉터리를 `openai`, `tei`, `ollama` 형식으로 서빙한다.
- `jevtext.py`는 state와 선택지를 게이트웨이와 똑같이 렌더링한다.

```bash
pip install -r extensions/embedding/requirements.txt
python extensions/embedding/train.py \
  --model google/embeddinggemma-300m \
  --train train.jsonl --eval eval.jsonl --out ./jev-embed \
  --query-prefix "task: sentence similarity | query: " \
  --document-prefix "task: sentence similarity | query: "
python extensions/embedding/serve.py --model ./jev-embed --port 8090
```

학습과 hearim 설정에서 같은 prefix와 `tau`를 쓴다.

### 미세조정의 효과 (실측)

`embeddinggemma-300m`, 학습 전후 정확도, 시드 하나, 코퍼스당 150건
(typed-decisions는 400건). "전용"은 typed-decisions 3,000건으로 3 epoch,
"혼합"은 typed-decisions, AG News, Emotion, Banking77에서 모은 2,392건으로
2 epoch 학습했다. 다른 모델도 서빙 중인 M2 Ultra에서 각각 약 18분.

| 코퍼스 | Base | 전용 | 혼합 |
|---|---|---|---|
| typed-decisions | 0.463 | **0.693** (학습) | 0.568 (학습) |
| AG News | 0.647 | 0.647 | **0.820** (학습) |
| Emotion | 0.533 | 0.387 | **0.633** (학습) |
| Banking77 | 0.773 | 0.467 | **0.853** (학습) |
| SST-2 | 0.633 | 0.733 | 0.707 |
| Yelp | 0.553 | 0.407 | 0.453 |
| YNAT (한국어) | 0.440 | 0.540 | 0.693 |
| KLUE NLI (한국어) | 0.327 | 0.327 | 0.287 |

- 미세조정은 학습한 스키마를 10~23%p 올린다.
- **한 스키마 계열로만 학습한 모델은 전용 모델이다.** typed-decisions
  모델은 Banking77에서 31%p, Emotion과 Yelp에서 15%p를 잃었다. 그런 모델은
  별도 alias로 두고 나머지 질문에는 base 모델을 쓴다.
- 과제를 섞으면 그 붕괴가 없었고, 학습에 없던 네 코퍼스 중 둘(SST-2,
  YNAT)로 효과가 옮겨갔다. Yelp는 10%p 떨어졌다. 한 번의 실행으로 규칙이라
  말할 수는 없으니 자신의 held-out 레코드로 평가한다.
- 추론형 질문은 고쳐지지 않는다. NLI는 무작위 수준에 머물렀다.
- `score` 질문은 hard label로 학습한다. 인접한 단계를 먼 단계보다 가깝게
  취급하지 않는다.
- hearim으로 서빙한 모델이 학습 스크립트의 정확도를 재현했다(Banking77
  0.853, AG News 0.820, YNAT 0.693). 스크립트와 게이트웨이가 같은 텍스트를
  렌더링한다는 확인이다.

**임베딩 위에 학습된 헤드를 얹지 않는 이유.** 헤드
(`mrm8488/embeddinggemma-2-ft-typed-decision` 방식)는 학습한 스키마에서 더
높지만(typed-decisions 0.763), 임베딩 API로 서빙할 수 없다. 그리고 처음 보는
스키마의 코퍼스 7개 모두에서 그 모델의 헤드는 자기 인코더의 순수 코사인보다
나빴다(AG News 0.507 대 0.853).
