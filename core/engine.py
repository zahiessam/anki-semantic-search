"""Embedding and API logic (no Qt). Used by search, indexing, and UI."""

import json
import os
import time
import urllib.request
import urllib.error

from ..utils.log import log_debug
from ..utils.config import load_config

OLLAMA_EMBED_CHUNK_SIZE = 64


def get_embedding_via_voyage(text, is_query=False, api_key=None, model=None):
    """Fetch a single embedding vector from the Voyage AI embeddings API."""
    effective_key = (api_key or "").strip() if api_key is not None else os.environ.get("VOYAGE_API_KEY", "").strip()
    if not effective_key:
        raise RuntimeError("Voyage API key is not set. Enter it in Settings → Search → Embeddings, or set VOYAGE_API_KEY.")
    model = (model or "voyage-3.5-lite").strip()
    url = "https://api.voyageai.com/v1/embeddings"
    payload = {"input": [text], "model": model, "input_type": "query" if is_query else "document"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data, {"Content-Type": "application/json", "Authorization": f"Bearer {effective_key}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            if isinstance(resp_data, dict) and resp_data.get("data"):
                first = resp_data["data"][0]
                if isinstance(first, dict) and "embedding" in first:
                    return first["embedding"]
            raise RuntimeError(f"Unexpected embeddings response structure: {type(resp_data)}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < 2:
                time.sleep(1.0 * (2 ** attempt))
                continue
            raise
    return None


def get_embeddings_via_voyage_batch(texts, input_type="document", api_key=None, model=None):
    """Fetch multiple embedding vectors from the Voyage AI embeddings API in a single call."""
    if not texts:
        return []
    effective_key = (api_key or "").strip() if api_key is not None else os.environ.get("VOYAGE_API_KEY", "").strip()
    if not effective_key:
        raise RuntimeError("Voyage API key is not set. Enter it in Settings → Search → Embeddings, or set VOYAGE_API_KEY.")
    model = (model or "voyage-3.5-lite").strip()
    url = "https://api.voyageai.com/v1/embeddings"
    payload = {"input": texts, "model": model, "input_type": input_type}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data, {"Content-Type": "application/json", "Authorization": f"Bearer {effective_key}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            if isinstance(resp_data, dict) and resp_data.get("data"):
                embeddings = [item["embedding"] for item in resp_data["data"] if isinstance(item, dict) and "embedding" in item]
                if len(embeddings) == len(texts):
                    return embeddings
                raise RuntimeError(f"Expected {len(texts)} embeddings but got {len(embeddings)}")
            raise RuntimeError(f"Unexpected embeddings response structure: {type(resp_data)}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < 2:
                time.sleep(1.0 * (2 ** attempt))
                continue
            raise
    return []


def get_embedding_via_openai(text, api_key=None, model=None):
    """Fetch a single embedding from OpenAI embeddings API."""
    effective_key = (api_key or "").strip() if api_key is not None else os.environ.get("OPENAI_API_KEY", "").strip()
    if not effective_key:
        raise RuntimeError("OpenAI API key is not set. Enter it in Settings → Search → Embeddings, or set OPENAI_API_KEY.")
    model = (model or "text-embedding-3-small").strip()
    url = "https://api.openai.com/v1/embeddings"
    payload = {"input": text, "model": model}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data, {"Content-Type": "application/json", "Authorization": f"Bearer {effective_key}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            if isinstance(resp_data, dict) and resp_data.get("data"):
                first = resp_data["data"][0]
                if isinstance(first, dict) and "embedding" in first:
                    return first["embedding"]
            raise RuntimeError(f"Unexpected OpenAI embeddings response: {type(resp_data)}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            if attempt < 2:
                time.sleep(1.0 * (2 ** attempt))
                continue
            raise
    return None


def get_embeddings_via_openai_batch(texts, api_key=None, model=None):
    """Fetch multiple embeddings from OpenAI in one call."""
    if not texts:
        return []
    effective_key = (api_key or "").strip() if api_key is not None else os.environ.get("OPENAI_API_KEY", "").strip()
    if not effective_key:
        raise RuntimeError("OpenAI API key is not set. Enter it in Settings → Search → Embeddings, or set OPENAI_API_KEY.")
    model = (model or "text-embedding-3-small").strip()
    url = "https://api.openai.com/v1/embeddings"
    payload = {"input": texts, "model": model}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data, {"Content-Type": "application/json", "Authorization": f"Bearer {effective_key}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            if isinstance(resp_data, dict) and resp_data.get("data"):
                embeddings = [item["embedding"] for item in resp_data["data"] if isinstance(item, dict) and "embedding" in item]
                if len(embeddings) == len(texts):
                    return embeddings
            raise RuntimeError(f"Unexpected OpenAI embeddings response: {type(resp_data)}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            if attempt < 2:
                time.sleep(1.0 * (2 ** attempt))
                continue
            raise
    return []


def get_embedding_via_cohere(text, is_query=False, api_key=None, model=None):
    """Fetch a single embedding from Cohere embed API."""
    effective_key = (api_key or "").strip() if api_key is not None else os.environ.get("COHERE_API_KEY", "").strip()
    if not effective_key:
        raise RuntimeError("Cohere API key is not set. Enter it in Settings → Search → Embeddings, or set COHERE_API_KEY.")
    model = (model or "embed-english-v3.0").strip()
    url = "https://api.cohere.com/v1/embed"
    input_type = "search_query" if is_query else "search_document"
    payload = {"texts": [text], "model": model, "input_type": input_type}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data, {"Content-Type": "application/json", "Authorization": f"Bearer {effective_key}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            if isinstance(resp_data, dict) and "embeddings" in resp_data:
                emb = resp_data["embeddings"]
                if isinstance(emb, list) and len(emb) == 1:
                    return emb[0]
            raise RuntimeError(f"Unexpected Cohere response: {type(resp_data)}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            if attempt < 2:
                time.sleep(1.0 * (2 ** attempt))
                continue
            raise
    return None


def get_embeddings_via_cohere_batch(texts, input_type="document", api_key=None, model=None):
    """Fetch multiple embeddings from Cohere in one call."""
    if not texts:
        return []
    effective_key = (api_key or "").strip() if api_key is not None else os.environ.get("COHERE_API_KEY", "").strip()
    if not effective_key:
        raise RuntimeError("Cohere API key is not set. Enter it in Settings → Search → Embeddings, or set COHERE_API_KEY.")
    model = (model or "embed-english-v3.0").strip()
    url = "https://api.cohere.com/v1/embed"
    itype = "search_query" if input_type == "query" else "search_document"
    payload = {"texts": texts, "model": model, "input_type": itype}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data, {"Content-Type": "application/json", "Authorization": f"Bearer {effective_key}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            if isinstance(resp_data, dict) and "embeddings" in resp_data:
                embeddings = resp_data["embeddings"]
                if isinstance(embeddings, list) and len(embeddings) == len(texts):
                    return embeddings
            raise RuntimeError(f"Unexpected Cohere response: {type(resp_data)}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            if attempt < 2:
                time.sleep(1.0 * (2 ** attempt))
                continue
            raise
    return []


def get_ollama_models(base_url="http://localhost:11434"):
    """Fetch list of available model names from Ollama (GET /api/tags)."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("models") or []
        names = []
        for m in models:
            if isinstance(m, dict) and m.get("name"):
                names.append(m["name"].strip())
            elif isinstance(m, str):
                names.append(m.strip())
        return sorted(names)
    except Exception as e:
        log_debug(f"Ollama list models failed: {str(e)[:100]}")
        return []


def get_embedding_via_ollama(text, base_url="http://localhost:11434", model="nomic-embed-text"):
    """Fetch a single embedding vector from a local Ollama server."""
    url = base_url.rstrip("/") + "/api/embed"
    payload = {"model": model, "input": text}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        if isinstance(resp_data, dict) and "embeddings" in resp_data:
            emb = resp_data["embeddings"]
            if isinstance(emb, list) and len(emb) >= 1:
                return emb[0]
        raise RuntimeError(f"Unexpected Ollama response: {type(resp_data)}")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        log_debug(f"Ollama embedding request failed: {str(e)[:100]}")
        raise


def get_embeddings_via_ollama_batch(texts, base_url="http://localhost:11434", model="nomic-embed-text"):
    """Fetch multiple embeddings from Ollama. Chunks to avoid timeouts."""
    if not texts:
        return []
    url = base_url.rstrip("/") + "/api/embed"
    out = []
    for start in range(0, len(texts), OLLAMA_EMBED_CHUNK_SIZE):
        chunk = texts[start : start + OLLAMA_EMBED_CHUNK_SIZE]
        payload = {"model": model, "input": chunk}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            if isinstance(resp_data, dict) and "embeddings" in resp_data:
                embeddings = resp_data["embeddings"]
                if isinstance(embeddings, list) and len(embeddings) == len(chunk):
                    out.extend(embeddings)
                    continue
        except Exception as e:
            log_debug(f"Ollama batch chunk failed: {str(e)[:80]} - falling back to per-text for chunk")
        for t in chunk:
            emb = get_embedding_via_ollama(t, base_url=base_url, model=model)
            out.append(emb)
    return out


def get_embedding_for_query(text, config=None):
    """Get embedding for a query using the configured engine."""
    if config is None:
        config = load_config()
    sc = config.get("search_config") or {}
    engine = sc.get("embedding_engine") or "voyage"
    if engine == "ollama":
        base_url = (sc.get("ollama_base_url") or "http://localhost:11434").strip()
        model = (sc.get("ollama_embed_model") or "nomic-embed-text").strip()
        return get_embedding_via_ollama(text, base_url=base_url, model=model)
    if engine == "openai":
        key = (sc.get("openai_embedding_api_key") or "").strip() or None
        model = (sc.get("openai_embedding_model") or "text-embedding-3-small").strip() or None
        return get_embedding_via_openai(text, api_key=key, model=model)
    if engine == "cohere":
        key = (sc.get("cohere_api_key") or "").strip() or None
        model = (sc.get("cohere_embedding_model") or "embed-english-v3.0").strip() or None
        return get_embedding_via_cohere(text, is_query=True, api_key=key, model=model)
    voyage_key = (sc.get("voyage_api_key") or "").strip() or None
    voyage_model = (sc.get("voyage_embedding_model") or "voyage-3.5-lite").strip() or None
    return get_embedding_via_voyage(text, is_query=True, api_key=voyage_key, model=voyage_model)


def get_embeddings_batch(texts, input_type="document", config=None):
    """Get embeddings for multiple texts using the configured engine."""
    if not texts:
        return []
    if config is None:
        config = load_config()
    sc = config.get("search_config") or {}
    engine = sc.get("embedding_engine") or "voyage"
    if engine == "ollama":
        base_url = (sc.get("ollama_base_url") or "http://localhost:11434").strip()
        model = (sc.get("ollama_embed_model") or "nomic-embed-text").strip()
        return get_embeddings_via_ollama_batch(texts, base_url=base_url, model=model)
    if engine == "openai":
        key = (sc.get("openai_embedding_api_key") or "").strip() or None
        model = (sc.get("openai_embedding_model") or "text-embedding-3-small").strip() or None
        return get_embeddings_via_openai_batch(texts, api_key=key, model=model)
    if engine == "cohere":
        key = (sc.get("cohere_api_key") or "").strip() or None
        model = (sc.get("cohere_embedding_model") or "embed-english-v3.0").strip() or None
        return get_embeddings_via_cohere_batch(texts, input_type=input_type, api_key=key, model=model)
    voyage_key = (sc.get("voyage_api_key") or "").strip() or None
    voyage_model = (sc.get("voyage_embedding_model") or "voyage-3.5-lite").strip() or None
    return get_embeddings_via_voyage_batch(texts, input_type=input_type, api_key=voyage_key, model=voyage_model)


def estimate_tokens(text):
    """Rough estimate: ~4 characters per token for English text."""
    if not text:
        return 0
    return len(text) // 4
