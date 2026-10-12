# Embedding extension

Optional Python tooling for hearim's embedding-scored models
([docs/EMBEDDING.md](../../docs/EMBEDDING.md)). The gateway does not depend
on any of it.

| File | Purpose |
|---|---|
| `train.py` | fine-tune a sentence-transformers encoder on labeled `hearim bench` records |
| `serve.py` | serve a model directory over the OpenAI, TEI and Ollama embeddings formats |
| `jevtext.py` | state and option rendering, kept identical to the gateway's |

```bash
pip install -r requirements.txt
python train.py --model google/embeddinggemma-300m --train train.jsonl --eval eval.jsonl --out ./jev-embed
python serve.py --model ./jev-embed --port 8090
```

Read the training notes in the document above before using a fine-tuned
model for questions it was not trained on.
