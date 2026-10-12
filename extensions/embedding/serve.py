"""Serve a sentence-transformers model over the embedding APIs hearim speaks.

    python serve.py --model ~/models/google_embeddinggemma-300m --port 8090

Exposes POST /v1/embeddings (OpenAI), /embed (TEI) and /api/embed (Ollama).
Texts are embedded as sent: task prompts are hearim's job
(embedding.query_prefix / document_prefix), so the same model directory
behaves identically here, in TEI, and in training.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    model = SentenceTransformer(args.model, device=args.device)
    lock = threading.Lock()

    def embed(texts):
        with lock:
            return model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=False).tolist()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {"status": "ok", "model": args.model})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            texts = body.get("input", body.get("inputs"))
            if isinstance(texts, str):
                texts = [texts]
            if not texts:
                return self.reply(400, {"error": "no input texts"})
            vecs = embed(texts)
            if self.path == "/v1/embeddings":
                return self.reply(200, {"object": "list", "model": body.get("model", ""),
                                        "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]})
            if self.path == "/embed":
                return self.reply(200, vecs)
            if self.path == "/api/embed":
                return self.reply(200, {"model": body.get("model", ""), "embeddings": vecs})
            self.reply(404, {"error": f"unknown path {self.path}"})

    print(f"serving {args.model} on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
