#!/usr/bin/env python3
"""
Rerank helper: read JSON {query, contents} from stdin, output JSON {scores} to stdout.
Used when the add-on runs Cross-Encoder in an external Python (e.g. Python 3.11).
Install sentence-transformers in that Python: pip install sentence-transformers
"""
import sys
import json


def main():
    try:
        data = json.load(sys.stdin)
        query = data.get("query", "")
        contents = data.get("contents", [])
        if not contents:
            json.dump({"scores": []}, sys.stdout)
            sys.stdout.flush()
            return
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, (c[:512] if c else "")) for c in contents]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
        json.dump({"scores": [float(s) for s in scores]}, sys.stdout)
        sys.stdout.flush()
    except Exception as e:
        json.dump({"error": str(e)}, sys.stdout)
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
