#!/usr/bin/env python3
"""
Rerank helper: read JSON {query, contents, model} from stdin, output JSON {scores} to stdout.
Used when the add-on runs Cross-Encoder in an external Python (e.g. Python 3.11).
Install sentence-transformers in that Python: pip install sentence-transformers
"""
import sys
import json

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
_MODEL_CACHE = {}


def _normalize_model(model):
    return (model or "").strip() or DEFAULT_RERANK_MODEL


def _truncate_for_rerank(text):
    return (text or "")[:512]


def _get_model(model_name):
    model_name = _normalize_model(model_name)
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder
        _MODEL_CACHE[model_name] = CrossEncoder(model_name)
    return _MODEL_CACHE[model_name]


def main():
    model_name = DEFAULT_RERANK_MODEL
    try:
        data = json.load(sys.stdin)
        query = data.get("query", "")
        contents = data.get("contents", [])
        model_name = _normalize_model(data.get("model"))
        if not contents:
            json.dump({"scores": [], "model": model_name}, sys.stdout)
            sys.stdout.flush()
            return
        model = _get_model(model_name)
        pairs = [(query, _truncate_for_rerank(c)) for c in contents]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
        json.dump({"scores": [float(s) for s in scores], "model": model_name}, sys.stdout)
        sys.stdout.flush()
    except Exception as e:
        json.dump({"error": str(e), "model": model_name}, sys.stdout)
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
