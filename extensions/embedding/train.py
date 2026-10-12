"""Fine-tune an embedding model for Jev (System One) questions.

Only the encoder is trained, with the objective hearim serves:

    p(option_i) = softmax_i( cos(embed(instructions + state), embed(option_i)) / tau )

The result is an ordinary sentence-transformers model: serve it behind any
embeddings API (TEI, vLLM, llama.cpp, serve.py) and point a hearim
`type: embedding` model at it. No head, no custom inference code.

    python train.py --model ~/models/google_embeddinggemma-300m \
        --train train.jsonl --eval eval.jsonl --out ./jev-embed \
        --query-prefix "task: sentence similarity | query: " \
        --document-prefix "task: sentence similarity | query: "

Corpora are `hearim bench` JSONL: {"state", "question", "expected"}.
Use the same prefixes in hearim's embedding config as in training.
"""
import argparse
import random

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import jevtext


def encode(model, texts, grad):
    feats = (getattr(model, "preprocess", None) or model.tokenize)(texts)
    feats = {k: v.to(model.device) for k, v in feats.items() if torch.is_tensor(v)}
    with torch.set_grad_enabled(grad):
        return F.normalize(model(feats)["sentence_embedding"].float(), dim=-1)


def logits_for(model, rows, a, grad):
    """rows: [(query, state, options, gold)] -> list of per-question cosine vectors."""
    queries = [a.query_prefix + r[0] for r in rows]
    bank = {}
    for r in rows:
        for o in r[2]:
            bank.setdefault(a.document_prefix + o, len(bank))
    q = encode(model, queries, grad)
    d = encode(model, list(bank), grad)
    return [d[[bank[a.document_prefix + o] for o in r[2]]] @ q[i] for i, r in enumerate(rows)]


def accuracy(model, rows, a):
    model.eval()
    hit = 0
    for i in range(0, len(rows), a.batch_size):
        chunk = rows[i:i + a.batch_size]
        for z, r in zip(logits_for(model, chunk, a, grad=False), chunk):
            hit += int(z.argmax()) == r[3]
    return hit / max(len(rows), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="sentence-transformers model directory or hub id")
    ap.add_argument("--train", required=True, nargs="+")
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--document-prefix", default="")
    ap.add_argument("--tau", type=float, default=0.05, help="softmax temperature; match hearim's embedding.tau")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    model = SentenceTransformer(a.model, device=a.device)
    model.max_seq_length = a.max_seq_length
    train = [r for p in a.train for r in jevtext.load(p)]
    evals = {p: jevtext.load(p) for p in a.eval}
    print(f"train: {len(train)} decisions; device {model.device}")
    before = {p: accuracy(model, rows, a) for p, rows in evals.items()}

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(train) + a.batch_size - 1) // a.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1, anneal_strategy="linear")
    step = 0
    for epoch in range(a.epochs):
        random.shuffle(train)
        model.train()
        total = 0.0
        for i in range(0, len(train), a.batch_size):
            chunk = train[i:i + a.batch_size]
            loss = sum(F.cross_entropy(z.unsqueeze(0) / a.tau, torch.tensor([r[3]], device=z.device))
                       for z, r in zip(logits_for(model, chunk, a, grad=True), chunk)) / len(chunk)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(chunk)
            step += 1
        print(f"epoch {epoch + 1}/{a.epochs} loss {total / len(train):.4f}", flush=True)

    model.save(a.out)
    for p, rows in evals.items():
        print(f"{p}: accuracy {before[p]:.3f} -> {accuracy(model, rows, a):.3f}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
