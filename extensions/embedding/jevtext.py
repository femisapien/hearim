"""Text rendering shared by the training recipe and the gateway.

These functions must produce the same strings as hearim's embedding route
(internal/hearim/eval/embedding.go: RenderState, optionText): the model is
trained on exactly the text it will be served.
"""
import json


def _scalar(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _flatten(v, path, lines):
    def join(k):
        k = str(k).replace("_", " ")
        return f"{path} · {k}" if path else k

    prefix = f"{path}: " if path else ""
    if isinstance(v, dict):
        for k in sorted(v):
            _flatten(v[k], join(k), lines)
    elif isinstance(v, list):
        if any(isinstance(x, (dict, list)) for x in v):
            for i, x in enumerate(v):
                _flatten(x, join(i + 1), lines)
        else:
            lines.append(prefix + "; ".join(_scalar(x) for x in v))
    else:
        lines.append(prefix + _scalar(v))


def render_state(state):
    if isinstance(state, str):
        return state
    lines = []
    _flatten(state, "", lines)
    return "\n".join(lines)


def query_text(state, question):
    instructions = question.get("instructions") or ""
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{instructions}\n{render_state(state)}".strip()


def options(question):
    """-> (answer keys, option texts) in the order hearim scores them."""
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        keys = list(criteria)
        return keys, [f"{k}: {criteria[k]}" if criteria[k] else k for k in keys]
    if kind == "score":
        return [str(i) for i in range(len(criteria))], list(criteria)
    criteria = criteria or {}
    return ["true", "false"], [criteria.get("true") or "Yes, this is true.", criteria.get("false") or "No, this is false."]


def expected_key(expected):
    if isinstance(expected, bool):
        return "true" if expected else "false"
    return str(expected)


def load(path):
    """Read a `hearim bench` JSONL corpus -> [(query, state, option texts, gold index)]."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            r = json.loads(line)
            keys, opts = options(r["question"])
            rows.append((query_text(r["state"], r["question"]), render_state(r["state"]), opts, keys.index(expected_key(r["expected"]))))
    return rows
