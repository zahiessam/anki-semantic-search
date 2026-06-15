"""Embedding, collection helpers, and persistence for the addon UI/workers."""

# ============================================================================
# Imports
# ============================================================================

import datetime
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict

from aqt import mw
from aqt.qt import QApplication

from ..utils import (
    get_checkpoint_path,
    get_embeddings_db_path,
    get_embeddings_storage_path_for_read,
    load_config,
    log_debug,
)
from .errors import _is_embedding_dimension_mismatch
from .keyword_scoring import (
    _simple_stem,
    aggregate_scored_notes_by_note_id,
    compute_bm25_scores,
    extract_keywords_improved,
    get_extended_stop_words,
)
from ..utils.text import EMBEDDING_CONTENT_VERSION

# ============================================================================
# Module Constants And Caches
# ============================================================================

HTML_TAG_RE = re.compile(r"<.*?>", re.DOTALL)
OLLAMA_EMBED_CHUNK_SIZE = 64
OLLAMA_FREE_CLOUD_MODEL_CANDIDATES = (
    "gpt-oss:20b-cloud",
    "gpt-oss:120b-cloud",
    "gemma4:31b-cloud",
    "gemma3:27b-cloud",
    "gemma3:12b-cloud",
    "gemma3:4b-cloud",
    "ministral-3:3b-cloud",
    "ministral-3:8b-cloud",
    "ministral-3:14b-cloud",
    "rnj-1:8b-cloud",
    "devstral-small-2:24b-cloud",
    "nemotron-3-nano:30b-cloud",
)
OLLAMA_CLOUD_MODEL_RE = re.compile(
    r"(?<![\w./-])([A-Za-z0-9][A-Za-z0-9_.\-/]*:(?:[A-Za-z0-9_.\-]*cloud|cloud)|[A-Za-z0-9][A-Za-z0-9_.\-/]*-cloud)(?![\w./-])"
)

_embedding_batch_cache = {}
_embeddings_file_cache = None
_embeddings_file_cache_path = None
_embeddings_file_cache_time = 0
_corrupted_files = set()
MAX_EMBEDDING_CACHE_SIZE = 100
MAX_QUERY_EMBEDDING_CACHE_SIZE = 128
_query_embedding_cache = OrderedDict()
_query_embedding_cache_lock = threading.Lock()


# ============================================================================
# Embedding Provider Clients
# ============================================================================

# --- Shared Request Helpers ---

def estimate_tokens(text):
    return len(text or "") // 4


def _request_json(url, payload=None, headers=None, timeout=30, method=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, body, headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8")
        return json.loads(text) if text else {}


# --- Voyage AI Embeddings ---

def get_embedding_via_voyage(text, is_query=False, api_key=None, model=None):
    effective_key = (api_key or "").strip() or os.environ.get("VOYAGE_API_KEY", "").strip()
    if not effective_key:
        raise RuntimeError("Voyage API key is not set.")
    payload = {
        "input": [text],
        "model": (model or "voyage-3.5-lite").strip(),
        "input_type": "query" if is_query else "document",
    }
    data = _request_json(
        "https://api.voyageai.com/v1/embeddings",
        payload=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {effective_key}",
        },
        timeout=30,
    )
    return data["data"][0]["embedding"]


def get_embeddings_via_voyage_batch(texts, input_type="document", api_key=None, model=None):
    if not texts:
        return []
    effective_key = (api_key or "").strip() or os.environ.get("VOYAGE_API_KEY", "").strip()
    if not effective_key:
        raise RuntimeError("Voyage API key is not set.")
    payload = {
        "input": texts,
        "model": (model or "voyage-3.5-lite").strip(),
        "input_type": input_type,
    }
    data = _request_json(
        "https://api.voyageai.com/v1/embeddings",
        payload=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {effective_key}",
        },
        timeout=60,
    )
    return [item["embedding"] for item in data.get("data", [])]


# --- OpenAI Embeddings ---

def get_embedding_via_openai(text, api_key=None, model=None):
    effective_key = (api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not effective_key:
        raise RuntimeError("OpenAI API key is not set.")
    data = _request_json(
        "https://api.openai.com/v1/embeddings",
        payload={"input": text, "model": (model or "text-embedding-3-small").strip()},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {effective_key}",
        },
        timeout=30,
    )
    return data["data"][0]["embedding"]


def get_embeddings_via_openai_batch(texts, api_key=None, model=None):
    if not texts:
        return []
    effective_key = (api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not effective_key:
        raise RuntimeError("OpenAI API key is not set.")
    data = _request_json(
        "https://api.openai.com/v1/embeddings",
        payload={"input": texts, "model": (model or "text-embedding-3-small").strip()},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {effective_key}",
        },
        timeout=60,
    )
    return [item["embedding"] for item in data.get("data", [])]


# --- Cohere Embeddings ---

def get_embedding_via_cohere(text, is_query=False, api_key=None, model=None):
    effective_key = (api_key or "").strip() or os.environ.get("COHERE_API_KEY", "").strip()
    if not effective_key:
        raise RuntimeError("Cohere API key is not set.")
    data = _request_json(
        "https://api.cohere.com/v1/embed",
        payload={
            "texts": [text],
            "model": (model or "embed-english-v3.0").strip(),
            "input_type": "search_query" if is_query else "search_document",
        },
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {effective_key}",
        },
        timeout=30,
    )
    embeddings = data.get("embeddings") or []
    return embeddings[0] if embeddings else None


def get_embeddings_via_cohere_batch(texts, input_type="document", api_key=None, model=None):
    if not texts:
        return []
    effective_key = (api_key or "").strip() or os.environ.get("COHERE_API_KEY", "").strip()
    if not effective_key:
        raise RuntimeError("Cohere API key is not set.")
    data = _request_json(
        "https://api.cohere.com/v1/embed",
        payload={
            "texts": texts,
            "model": (model or "embed-english-v3.0").strip(),
            "input_type": "search_query" if input_type == "query" else "search_document",
        },
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {effective_key}",
        },
        timeout=60,
    )
    return data.get("embeddings") or []


# --- Ollama Embeddings ---

def _normalize_ollama_base_url(base_url="http://localhost:11434"):
    url = (base_url or "http://localhost:11434").strip()
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")
    for suffix in ("/v1", "/api", "/api/tags", "/api/embed", "/api/chat", "/api/generate"):
        if url.lower().endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url.rstrip("/")


def get_ollama_models(base_url="http://localhost:11434"):
    try:
        base_url = _normalize_ollama_base_url(base_url)
        names, _stale_cloud_models = get_ollama_models_with_stale_cloud(base_url)
        return names
    except Exception as exc:
        log_debug(f"Ollama list models failed: {exc}")
        return []


def get_ollama_models_with_stale_cloud(base_url="http://localhost:11434"):
    """Return visible Ollama models plus locally listed cloud models outside the free allowlist."""
    base_url = _normalize_ollama_base_url(base_url)
    data = _request_json(f"{base_url}/api/tags", timeout=10)
    models = data.get("models") or []
    local_names = []
    for item in models:
        if isinstance(item, dict) and item.get("name"):
            local_names.append(item["name"].strip())
        elif isinstance(item, str):
            local_names.append(item.strip())

    names = _include_available_ollama_cloud_models(base_url, local_names)
    available = set(names)
    stale_cloud_models = sorted(
        name
        for name in local_names
        if is_ollama_cloud_model(name) and name not in available
    )
    return sorted(names), stale_cloud_models


def is_ollama_cloud_model(model):
    return bool(OLLAMA_CLOUD_MODEL_RE.search((model or "").strip()))


def delete_ollama_model(base_url, model):
    base_url = _normalize_ollama_base_url(base_url)
    _request_json(
        f"{base_url}/api/delete",
        payload={"name": model},
        headers={"Content-Type": "application/json"},
        timeout=30,
        method="DELETE",
    )


def get_ollama_model_capabilities(base_url, models, timeout=3, max_models=200):
    base_url = _normalize_ollama_base_url(base_url)
    capabilities = {}
    for model in list(models or [])[:max_models]:
        model = (model or "").strip()
        if not model:
            continue
        try:
            data = _request_json(f"{base_url}/api/show", payload={"name": model}, timeout=timeout)
            values = data.get("capabilities") or []
            capabilities[model] = [str(value).strip().lower() for value in values if str(value).strip()]
        except Exception as exc:
            log_debug(f"Ollama capability probe skipped {model}: {exc}")
            capabilities[model] = []
    return capabilities


def _include_available_ollama_cloud_models(base_url, names):
    """Add known free-tier cloud models that Ollama can show but omits from /api/tags."""
    out = {
        name
        for name in names
        if name and (not is_ollama_cloud_model(name) or name in OLLAMA_FREE_CLOUD_MODEL_CANDIDATES)
    }
    candidates = set(OLLAMA_FREE_CLOUD_MODEL_CANDIDATES)
    try:
        sc = (load_config().get("search_config") or {})
        for key in ("ollama_chat_model", "answer_local_model", "local_llm_model"):
            model = (sc.get(key) or "").strip()
            if model in OLLAMA_FREE_CLOUD_MODEL_CANDIDATES:
                candidates.add(model)
    except Exception:
        pass

    for model in sorted(candidates):
        if not model or model in out:
            continue
        try:
            data = _request_json(f"{base_url}/api/show", payload={"name": model}, timeout=3)
            if data.get("details") or data.get("model_info") or data.get("capabilities"):
                out.add(model)
        except Exception as exc:
            log_debug(f"Ollama cloud model probe skipped {model}: {exc}")
    return list(out)


def get_embedding_via_ollama(text, base_url="http://localhost:11434", model="nomic-embed-text"):
    base_url = _normalize_ollama_base_url(base_url)
    data = _request_json(
        f"{base_url}/api/embed",
        payload={"model": model, "input": text},
        headers={"Content-Type": "application/json"},
        timeout=90,
    )
    embeddings = data.get("embeddings") or []
    return embeddings[0] if embeddings else None


def get_embeddings_via_ollama_batch(texts, base_url="http://localhost:11434", model="nomic-embed-text"):
    if not texts:
        return []
    base_url = _normalize_ollama_base_url(base_url)
    out = []
    for start in range(0, len(texts), OLLAMA_EMBED_CHUNK_SIZE):
        chunk = texts[start : start + OLLAMA_EMBED_CHUNK_SIZE]
        try:
            data = _request_json(
                f"{base_url.rstrip('/')}/api/embed",
                payload={"model": model, "input": chunk},
                headers={"Content-Type": "application/json"},
                timeout=90,
            )
            embeddings = data.get("embeddings") or []
            if len(embeddings) == len(chunk):
                out.extend(embeddings)
                continue
        except Exception as exc:
            log_debug(f"Ollama batch chunk failed: {exc}")

        for text in chunk:
            out.append(get_embedding_via_ollama(text, base_url=base_url, model=model))
    return out


# --- Local OpenAI-Compatible Embeddings ---

def get_embedding_via_local_openai(text, base_url="http://localhost:1234/v1", model=None):
    data = _request_json(
        f"{base_url.rstrip('/')}/embeddings",
        payload={"input": text, "model": model or "text-embedding-3-small"},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    return data["data"][0]["embedding"]


def get_embeddings_via_local_openai_batch(texts, base_url="http://localhost:1234/v1", model=None):
    if not texts:
        return []
    data = _request_json(
        f"{base_url.rstrip('/')}/embeddings",
        payload={"input": texts, "model": model or "text-embedding-3-small"},
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    return [item["embedding"] for item in data.get("data", [])]


# --- Provider Routing And Engine Identity ---

def _resolve_embedding_engine_identity(config=None):
    """Return the canonical embedding engine id plus compatible historical aliases."""
    # explanation: resolve the new embedding UX fields before reading legacy engine keys.
    from ..utils.config import get_effective_embedding_config
    config = get_effective_embedding_config(config or load_config())
    sc = config.get("search_config") or {}
    engine = (sc.get("embedding_engine") or "voyage").strip().lower()

    aliases = []
    if engine == "cloud":
        # "cloud" is a UI-level label; current execution falls through to Voyage.
        model = (sc.get("voyage_embedding_model") or "voyage-3.5-lite").strip()
        canonical_engine = "voyage"
        aliases.append("cloud:default")
    elif engine == "ollama":
        model = (sc.get("ollama_embed_model") or "nomic-embed-text").strip()
        canonical_engine = "ollama"
    elif engine == "voyage":
        model = (sc.get("voyage_embedding_model") or "voyage-3.5-lite").strip()
        canonical_engine = "voyage"
        aliases.append("cloud:default")
    elif engine == "openai":
        model = (sc.get("openai_embedding_model") or "text-embedding-3-small").strip()
        canonical_engine = "openai"
    elif engine == "cohere":
        model = (sc.get("cohere_embedding_model") or "embed-english-v3.0").strip()
        canonical_engine = "cohere"
    elif engine == "local_openai":
        model = (sc.get("local_llm_model") or "text-embedding-3-small").strip()
        canonical_engine = "local_openai"
    else:
        model = "default"
        canonical_engine = engine or "default"

    canonical_id = f"{canonical_engine}:{model}"
    aliases = [alias for alias in aliases if alias and alias != canonical_id]
    return canonical_id, aliases


def get_embedding_engine_id(config=None):
    canonical_id, _aliases = _resolve_embedding_engine_identity(config=config)
    return canonical_id


def get_embedding_engine_candidates(config=None, engine_id=None):
    """Candidate engine ids to try when loading embeddings, in priority order."""
    if engine_id is not None:
        return [engine_id]

    canonical_id, aliases = _resolve_embedding_engine_identity(config=config)
    candidates = [canonical_id]
    for alias in aliases:
        if alias not in candidates:
            candidates.append(alias)
    if "legacy" not in candidates:
        candidates.append("legacy")
    return candidates


def _query_embedding_cache_key_for_config(text, config=None):
    # explanation: resolve same-provider and independent embedding choices into backend fields.
    from ..utils.config import get_effective_embedding_config
    config = get_effective_embedding_config(config or load_config())
    engine_id = get_embedding_engine_id(config)
    cache_key = (engine_id, " ".join((text or "").split()))
    return cache_key, config


def get_cached_query_embedding(text, config=None):
    cache_key, _config = _query_embedding_cache_key_for_config(text, config)

    with _query_embedding_cache_lock:
        cached = _query_embedding_cache.get(cache_key)
        if cached is not None:
            _query_embedding_cache.move_to_end(cache_key)
            return {
                "embedding": list(cached),
                "cache_key": cache_key,
                "cache_hit": True,
            }

    return {
        "embedding": None,
        "cache_key": cache_key,
        "cache_hit": False,
    }


def get_embedding_for_query(text, config=None):
    cache_key, config = _query_embedding_cache_key_for_config(text, config)
    sc = config.get("search_config") or {}
    engine = (sc.get("embedding_engine") or "voyage").strip().lower()

    with _query_embedding_cache_lock:
        cached = _query_embedding_cache.get(cache_key)
        if cached is not None:
            _query_embedding_cache.move_to_end(cache_key)
            return list(cached)

    if engine == "local_openai":
        embedding = get_embedding_via_local_openai(
            text,
            base_url=(sc.get("local_llm_url") or "http://localhost:1234/v1").strip(),
            model=(sc.get("local_llm_model") or "text-embedding-3-small").strip(),
        )
    elif engine == "ollama":
        embedding = get_embedding_via_ollama(
            text,
            base_url=(sc.get("ollama_base_url") or "http://localhost:11434").strip(),
            model=(sc.get("ollama_embed_model") or "nomic-embed-text").strip(),
        )
    elif engine == "openai":
        embedding = get_embedding_via_openai(
            text,
            api_key=(sc.get("openai_embedding_api_key") or "").strip(),
            model=(sc.get("openai_embedding_model") or "text-embedding-3-small").strip(),
        )
    elif engine == "cohere":
        embedding = get_embedding_via_cohere(
            text,
            is_query=True,
            api_key=(sc.get("cohere_api_key") or "").strip(),
            model=(sc.get("cohere_embedding_model") or "embed-english-v3.0").strip(),
        )
    else:
        embedding = get_embedding_via_voyage(
            text,
            is_query=True,
            api_key=(sc.get("voyage_api_key") or "").strip(),
            model=(sc.get("voyage_embedding_model") or "voyage-3.5-lite").strip(),
        )

    if embedding:
        with _query_embedding_cache_lock:
            _query_embedding_cache[cache_key] = tuple(float(x) for x in embedding)
            _query_embedding_cache.move_to_end(cache_key)
            while len(_query_embedding_cache) > MAX_QUERY_EMBEDDING_CACHE_SIZE:
                _query_embedding_cache.popitem(last=False)
    return embedding


def get_embeddings_batch(texts, input_type="document", config=None):
    if not texts:
        return []
    # explanation: resolve same-provider and independent embedding choices into backend fields.
    from ..utils.config import get_effective_embedding_config
    config = get_effective_embedding_config(config or load_config())
    sc = config.get("search_config") or {}
    engine = (sc.get("embedding_engine") or "voyage").strip().lower()

    if engine == "local_openai":
        return get_embeddings_via_local_openai_batch(
            texts,
            base_url=(sc.get("local_llm_url") or "http://localhost:1234/v1").strip(),
            model=(sc.get("local_llm_model") or "text-embedding-3-small").strip(),
        )
    if engine == "ollama":
        return get_embeddings_via_ollama_batch(
            texts,
            base_url=(sc.get("ollama_base_url") or "http://localhost:11434").strip(),
            model=(sc.get("ollama_embed_model") or "nomic-embed-text").strip(),
        )
    if engine == "openai":
        return get_embeddings_via_openai_batch(
            texts,
            api_key=(sc.get("openai_embedding_api_key") or "").strip(),
            model=(sc.get("openai_embedding_model") or "text-embedding-3-small").strip(),
        )
    if engine == "cohere":
        return get_embeddings_via_cohere_batch(
            texts,
            input_type=input_type,
            api_key=(sc.get("cohere_api_key") or "").strip(),
            model=(sc.get("cohere_embedding_model") or "embed-english-v3.0").strip(),
        )
    return get_embeddings_via_voyage_batch(
        texts,
        input_type=input_type,
        api_key=(sc.get("voyage_api_key") or "").strip(),
        model=(sc.get("voyage_embedding_model") or "voyage-3.5-lite").strip(),
    )


# ============================================================================
# Anki Collection Queries
# ============================================================================

def get_notes_count_per_model():
    try:
        if not mw or not mw.col:
            return {}
        rows = mw.col.db.execute("SELECT mid, COUNT(*) FROM notes GROUP BY mid")
        mid_to_count = {row[0]: row[1] for row in rows}
        out = {}
        for model in mw.col.models.all():
            out[model["name"]] = mid_to_count.get(model["id"], 0)
        return out
    except Exception as exc:
        log_debug(f"get_notes_count_per_model error: {exc}")
        return {}


def get_models_with_fields():
    try:
        if not mw or not mw.col:
            return []
        counts = get_notes_count_per_model()
        return [
            (model["name"], counts.get(model["name"], 0), [field["name"] for field in model["flds"]])
            for model in mw.col.models.all()
        ]
    except Exception as exc:
        log_debug(f"get_models_with_fields error: {exc}")
        return []


def get_models_with_fields_for_deck(deck_name):
    """Return note types used in a deck scope, including its subdecks."""
    try:
        if not mw or not mw.col or not deck_name:
            return []

        note_ids = mw.col.find_notes(_build_deck_query([deck_name]))
        if not note_ids:
            return []

        models_by_id = {int(model["id"]): model for model in mw.col.models.all()}
        counts_by_mid = {}
        for chunk_start in range(0, len(note_ids), 900):
            chunk = note_ids[chunk_start:chunk_start + 900]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT mid, COUNT(*) FROM notes WHERE id IN ({placeholders}) GROUP BY mid"
            for mid, count in mw.col.db.execute(query, *chunk):
                counts_by_mid[int(mid)] = counts_by_mid.get(int(mid), 0) + int(count or 0)

        rows = []
        for mid, count in counts_by_mid.items():
            model = models_by_id.get(mid)
            if model:
                rows.append((
                    model["name"],
                    count,
                    [field["name"] for field in model["flds"]],
                ))
        return sorted(rows, key=lambda row: row[1], reverse=True)
    except Exception as exc:
        log_debug(f"get_models_with_fields_for_deck error: {exc}")
        return []


def get_deck_names():
    try:
        if not mw or not mw.col:
            return []
        names = []
        for deck in mw.col.decks.all():
            name = deck.get("name", "")
            if name and not deck.get("dyn", False):
                names.append(name)
        return sorted(names)
    except Exception as exc:
        log_debug(f"get_deck_names error: {exc}")
        return []


def get_notes_count_per_deck():
    """Get note counts per deck using optimized SQL query with fallback to find_notes."""
    try:
        if not mw or not mw.col:
            return {}
        
        # Try optimized SQL approach first
        try:
            return get_notes_count_per_deck_optimized()
        except Exception as sql_exc:
            log_debug(f"Optimized deck count query failed, falling back to find_notes: {sql_exc}")
            # Fallback to original approach
            out = {}
            for index, deck_name in enumerate(get_deck_names(), start=1):
                try:
                    out[deck_name] = len(mw.col.find_notes(f'deck:"{deck_name}"'))
                    if index % 5 == 0:
                        QApplication.processEvents()
                except Exception:
                    out[deck_name] = 0
            return out
    except Exception as exc:
        log_debug(f"get_notes_count_per_deck error: {exc}")
        return {}


def get_notes_count_per_deck_optimized():
    """Count distinct notes per deck in one pass.

    Top-level deck rows include notes from their subdecks. Filtered deck cards
    are counted under their original deck via cards.odid.
    """
    try:
        if not mw or not mw.col:
            return {}

        deck_rows = []
        did_to_name = {}
        for deck in mw.col.decks.all():
            name = deck.get("name", "")
            if not name or deck.get("dyn", False):
                continue
            did = int(deck.get("id"))
            did_to_name[did] = name
            deck_rows.append((did, name))

        out = {name: 0 for _did, name in deck_rows}
        if not deck_rows:
            return out

        rows = mw.col.db.execute(
            """
            SELECT CASE WHEN c.odid IS NOT NULL AND c.odid != 0 THEN c.odid ELSE c.did END AS real_did,
                   c.nid
            FROM cards c
            GROUP BY real_did, c.nid
            """
        )
        notes_by_deck = {}
        for did, nid in rows:
            if did is None:
                continue
            did = int(did)
            if did in did_to_name:
                notes_by_deck.setdefault(did_to_name[did], set()).add(int(nid))

        main_decks = [name for _did, name in deck_rows if "::" not in name]
        main_note_ids = {name: set() for name in main_decks}
        for _did, deck_name in deck_rows:
            note_ids = notes_by_deck.get(deck_name, set())
            if note_ids:
                out[deck_name] = len(note_ids)
            for main_name in main_decks:
                if deck_name == main_name or deck_name.startswith(f"{main_name}::"):
                    main_note_ids[main_name].update(note_ids)
                    break
        for main_name, note_ids in main_note_ids.items():
            out[main_name] = len(note_ids)

        return out
    except Exception as exc:
        log_debug(f"get_notes_count_per_deck_optimized error: {exc}")
        raise


def get_notes_count_per_tag():
    """Get note counts per tag using optimized SQL query with fallback to find_notes."""
    try:
        if not mw or not mw.col:
            return {}
        
        # Try optimized SQL approach first
        try:
            return get_notes_count_per_tag_optimized()
        except Exception as sql_exc:
            log_debug(f"Optimized tag count query failed, falling back to find_notes: {sql_exc}")
            # Fallback to original approach
            tags = sorted(mw.col.tags.all())
            out = {}
            for index, tag in enumerate(tags, start=1):
                try:
                    out[tag] = len(mw.col.find_notes(f"tag:{tag}"))
                    if index % 5 == 0:
                        QApplication.processEvents()
                except Exception:
                    out[tag] = 0
            return out
    except Exception as exc:
        log_debug(f"get_notes_count_per_tag error: {exc}")
        return {}


def get_notes_count_per_tag_optimized():
    """Optimized version using direct SQL query for much better performance."""
    try:
        if not mw or not mw.col:
            return {}
        
        # Use direct SQL to get tag counts in a single query
        # Tags in Anki are stored in the notes.tags field as space-separated values
        # We need to parse them and count distinct notes per tag
        query = """
        SELECT 
            TRIM(tag) as tag_name,
            COUNT(DISTINCT id) as note_count
        FROM (
            SELECT id, TRIM(tag) as tag
            FROM notes, (
                SELECT value AS tag
                FROM json_each('["' || REPLACE(tags, ' ', '", "') || '"]')
            )
            WHERE tag != '' AND tag != ' '
        ) grouped_tags
        WHERE tag_name != '' AND tag_name != ' '
        GROUP BY tag_name
        ORDER BY tag_name
        """
        
        out = {}
        try:
            for row in mw.col.db.execute(query):
                tag_name = row[0]
                note_count = row[1] if row[1] else 0
                out[tag_name] = note_count
        except Exception as json_exc:
            # JSON approach might not work in older SQLite versions
            # Fall back to a simpler approach using string parsing
            log_debug(f"JSON tag parsing failed, using alternative approach: {json_exc}")
            return get_notes_count_per_tag_optimized_fallback()
        
        # Ensure all tags from mw.col.tags.all() are included
        # (in case some tags have no notes)
        for tag in sorted(mw.col.tags.all()):
            if tag not in out:
                out[tag] = 0
        
        return out
    except Exception as exc:
        log_debug(f"get_notes_count_per_tag_optimized error: {exc}")
        raise


def get_notes_count_per_tag_optimized_fallback():
    """Fallback optimized tag counting using string parsing instead of JSON."""
    try:
        if not mw or not mw.col:
            return {}
        
        # Alternative approach: get all notes with tags and parse them in Python
        # This is still much faster than calling find_notes for each tag
        query = "SELECT id, tags FROM notes WHERE tags IS NOT NULL AND tags != ''"
        
        tag_counts = {}
        for row in mw.col.db.execute(query):
            note_id = row[0]
            tags_str = row[1]
            # Tags are space-separated
            tags = tags_str.split()
            for tag in tags:
                tag = tag.strip()
                if tag:
                    if tag not in tag_counts:
                        tag_counts[tag] = set()
                    tag_counts[tag].add(note_id)
        
        # Convert sets to counts
        out = {tag: len(note_ids) for tag, note_ids in tag_counts.items()}
        
        # Ensure all tags from mw.col.tags.all() are included
        for tag in sorted(mw.col.tags.all()):
            if tag not in out:
                out[tag] = 0
        
        return out
    except Exception as exc:
        log_debug(f"get_notes_count_per_tag_optimized_fallback error: {exc}")
        raise


def _build_deck_query(enabled_decks):
    if not enabled_decks:
        return ""
    parts = []
    for deck_name in enabled_decks:
        if " " in deck_name or ":" in deck_name or "\\" in deck_name:
            parts.append(f'deck:"{deck_name}"')
        else:
            parts.append(f"deck:{deck_name}")
    return " or ".join(parts)


def resolve_scoped_note_ids_and_fields(ntf):
    """Resolve deck+note-type scoped note ids and per-note-type fields.

    When scope_fields is present, selected note types are applied inside each
    selected deck instead of globally across every selected deck.
    """
    scope_mode = (ntf or {}).get("scope_mode", "deck")
    scope_fields = (ntf or {}).get("scope_fields") or {}
    ntf_fields = (ntf or {}).get("note_type_fields") or {}

    if not mw or not mw.col:
        return [], dict(ntf_fields)

    if scope_mode != "deck" or not scope_fields:
        deck_q = _build_deck_query((ntf or {}).get("enabled_decks"))
        note_ids = mw.col.find_notes(deck_q) if deck_q else mw.col.find_notes("")
        return list(note_ids), dict(ntf_fields)

    models_by_id = {int(model["id"]): model["name"] for model in mw.col.models.all()}
    eligible_note_ids = set()
    scoped_fields = {}

    for deck_name, note_type_map in scope_fields.items():
        if not note_type_map:
            continue
        allowed_note_types = set(note_type_map.keys())
        deck_note_ids = mw.col.find_notes(_build_deck_query([deck_name]))
        for chunk_start in range(0, len(deck_note_ids), 900):
            chunk = deck_note_ids[chunk_start:chunk_start + 900]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            for nid, mid in mw.col.db.execute(
                f"SELECT id, mid FROM notes WHERE id IN ({placeholders})",
                *chunk,
            ):
                model_name = models_by_id.get(int(mid))
                if model_name not in allowed_note_types:
                    continue
                eligible_note_ids.add(int(nid))
                scoped_fields.setdefault(model_name, [])
                for field_name in note_type_map.get(model_name) or []:
                    if field_name not in scoped_fields[model_name]:
                        scoped_fields[model_name].append(field_name)

    return sorted(eligible_note_ids), scoped_fields


def analyze_note_eligibility(ntf):
    try:
        if not mw or not mw.col:
            return {
                "total_notes": 0,
                "eligible_count": 0,
                "eligible_note_ids": [],
                "filtered_out_note_type_count": 0,
                "no_selected_fields_count": 0,
                "empty_selected_fields_count": 0,
                "ineligible_notes": [],
            }
        # Support both legacy and new scope-based config
        scope_mode = ntf.get("scope_mode", "deck")
        scope_fields = ntf.get("scope_fields") or {}
        
        # Build deck query based on scope mode
        if scope_mode == "deck":
            note_ids, scoped_note_type_fields = resolve_scoped_note_ids_and_fields(ntf)
        elif scope_mode == "tag":
            scoped_note_type_fields = ntf.get("note_type_fields") or {}
            enabled_tags = ntf.get("enabled_tags")
            if enabled_tags:
                tag_q = " or ".join(f"tag:{tag}" for tag in enabled_tags)
                note_ids = mw.col.find_notes(tag_q)
            else:
                note_ids = mw.col.find_notes("")
        else:
            scoped_note_type_fields = ntf.get("note_type_fields") or {}
            # Fallback to legacy behavior
            deck_q = _build_deck_query(ntf.get("enabled_decks"))
            note_ids = mw.col.find_notes(deck_q) if deck_q else mw.col.find_notes("")
        if not note_ids:
            return {
                "total_notes": 0,
                "eligible_count": 0,
                "eligible_note_ids": [],
                "filtered_out_note_type_count": 0,
                "no_selected_fields_count": 0,
                "empty_selected_fields_count": 0,
                "ineligible_notes": [],
            }

        enabled = ntf.get("enabled_note_types")
        enabled_set = None if enabled is None else set(enabled or [])
        search_all = bool(ntf.get("search_all_fields", False))
        ntf_fields = scoped_note_type_fields or ntf.get("note_type_fields") or {}
        use_first = bool(ntf.get("use_first_field_fallback", True))

        model_map = {}
        for model in mw.col.models.all():
            model_name = model["name"]
            if enabled_set and model_name not in enabled_set:
                continue
            fields = model["flds"]
            if search_all:
                indices = list(range(len(fields)))
            else:
                # Try to get fields from scope_fields first (new format)
                wanted = None
                if scope_mode == "deck" and scope_fields:
                    # Merge fields from all enabled decks for this note type
                    for deck_fields in scope_fields.values():
                        if model_name in deck_fields:
                            deck_wanted = set(f.lower() for f in deck_fields[model_name])
                            if wanted is None:
                                wanted = deck_wanted
                            else:
                                wanted = wanted.union(deck_wanted)
                elif scope_mode == "tag" and scope_fields:
                    # Merge fields from all enabled tags for this note type
                    for tag_fields in scope_fields.values():
                        if model_name in tag_fields:
                            tag_wanted = set(f.lower() for f in tag_fields[model_name])
                            if wanted is None:
                                wanted = tag_wanted
                            else:
                                wanted = wanted.union(tag_wanted)
                
                # Fallback to legacy note_type_fields
                if wanted is None:
                    wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))
                
                if not wanted and use_first and fields:
                    wanted = {field["name"].lower() for field in fields[:2]}
                indices = [i for i, field in enumerate(fields) if field["name"].lower() in wanted]
            model_map[model["id"]] = {
                "indices": indices,
                "model_name": model_name,
                "field_names": [fields[i]["name"] for i in indices if i < len(fields)],
            }

        eligible_count = 0
        eligible_note_ids = []
        filtered_out_note_type_count = 0
        no_selected_fields_count = 0
        empty_selected_fields_count = 0
        ineligible_notes = []
        id_list = ",".join(map(str, note_ids))
        for nid, mid, flds_str in mw.col.db.execute(
            f"select id, mid, flds from notes where id in ({id_list})"
        ):
            model_info = model_map.get(mid)
            if model_info is None:
                filtered_out_note_type_count += 1
                ineligible_notes.append({
                    "id": nid,
                    "reason": "note type not selected",
                    "model_name": None,
                    "field_names": [],
                })
                continue
            indices = model_info["indices"]
            if not indices:
                no_selected_fields_count += 1
                ineligible_notes.append({
                    "id": nid,
                    "reason": "no embedding fields selected for this note type",
                    "model_name": model_info["model_name"],
                    "field_names": [],
                })
                continue
            fields = flds_str.split("\x1f")
            if any(i < len(fields) and fields[i].strip() for i in indices):
                eligible_count += 1
                eligible_note_ids.append(nid)
                continue
            empty_selected_fields_count += 1
            ineligible_notes.append({
                "id": nid,
                "reason": "selected embedding fields are empty",
                "model_name": model_info["model_name"],
                "field_names": list(model_info["field_names"]),
            })
        return {
            "total_notes": len(note_ids),
            "eligible_count": eligible_count,
            "eligible_note_ids": eligible_note_ids,
            "filtered_out_note_type_count": filtered_out_note_type_count,
            "no_selected_fields_count": no_selected_fields_count,
            "empty_selected_fields_count": empty_selected_fields_count,
            "ineligible_notes": ineligible_notes,
        }
    except Exception as exc:
        log_debug(f"analyze_note_eligibility error: {exc}")
        return {
            "total_notes": 0,
            "eligible_count": 0,
            "eligible_note_ids": [],
            "filtered_out_note_type_count": 0,
            "no_selected_fields_count": 0,
            "empty_selected_fields_count": 0,
            "ineligible_notes": [],
        }


def count_notes_matching_config(ntf):
    return analyze_note_eligibility(ntf).get("eligible_count", 0)


# ============================================================================
# Embedding Persistence
# ============================================================================

# --- SQLite Storage Helpers ---

def _embeddings_db_ensure_table(conn):
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
    if cur.fetchone():
        cur = conn.execute("PRAGMA table_info(embeddings)")
        columns = [row[1] for row in cur.fetchall()]
        if "engine_id" not in columns:
            old_chunk_expr = "chunk_index" if "chunk_index" in columns else "NULL"
            conn.execute(
                """
                CREATE TABLE embeddings_new (
                    engine_id TEXT NOT NULL,
                    note_id INTEGER NOT NULL,
                    chunk_index INTEGER,
                    content_hash TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    timestamp TEXT,
                    PRIMARY KEY (engine_id, note_id, content_hash)
                )
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO embeddings_new (engine_id, note_id, chunk_index, content_hash, embedding_blob, timestamp) "
                f"SELECT ?, note_id, {old_chunk_expr}, content_hash, embedding_blob, timestamp FROM embeddings",
                ("legacy",),
            )
            conn.execute("DROP TABLE embeddings")
            conn.execute("ALTER TABLE embeddings_new RENAME TO embeddings")
            log_debug("Migrated embeddings table to multi-engine schema (existing rows as engine_id='legacy')")
        elif "chunk_index" not in columns:
            conn.execute("ALTER TABLE embeddings ADD COLUMN chunk_index INTEGER")
            log_debug("Migrated embeddings table to add chunk_index column")
    else:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                engine_id TEXT NOT NULL,
                note_id INTEGER NOT NULL,
                chunk_index INTEGER,
                content_hash TEXT NOT NULL,
                embedding_blob BLOB NOT NULL,
                timestamp TEXT,
                PRIMARY KEY (engine_id, note_id, content_hash)
            )
            """
        )
    conn.commit()


def _embedding_to_blob(embedding):
    try:
        import numpy as np

        return np.array(embedding, dtype=np.float32).tobytes()
    except ImportError:
        import array

        return array.array("f", (float(value) for value in embedding)).tobytes()


def _blob_to_embedding(blob):
    try:
        import numpy as np

        return np.frombuffer(blob, dtype=np.float32)
    except ImportError:
        import array

        values = array.array("f")
        values.frombytes(blob)
        return values


def save_embedding(note_id, content_hash, embedding, batch_mode=True, storage_path=None, engine_id=None, chunk_index=None):
    try:
        import sqlite3

        if engine_id is None:
            engine_id = get_embedding_engine_id()
        db_path = storage_path if storage_path is not None else get_embeddings_db_path()
        key = f"{engine_id}_{note_id}_{content_hash}"
        emb_values = [float(value) for value in embedding]
        embedding_data = {
            "engine_id": engine_id,
            "note_id": note_id,
            "content_hash": content_hash,
            "chunk_index": chunk_index,
            "embedding": emb_values,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        if batch_mode:
            global _embedding_batch_cache
            if len(_embedding_batch_cache) >= MAX_EMBEDDING_CACHE_SIZE:
                flush_embedding_batch(storage_path=storage_path)
            _embedding_batch_cache[key] = embedding_data
            return True

        conn = sqlite3.connect(db_path, timeout=30)
        try:
            _embeddings_db_ensure_table(conn)
            conn.execute(
                "DELETE FROM embeddings WHERE engine_id = ? AND note_id = ?",
                (engine_id, note_id),
            )
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (engine_id, note_id, chunk_index, content_hash, embedding_blob, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    engine_id,
                    note_id,
                    chunk_index,
                    content_hash,
                    _embedding_to_blob(emb_values),
                    embedding_data["timestamp"],
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception as exc:
        log_debug(f"Error saving embedding: {exc}")
        return False


def flush_embedding_batch(storage_path=None):
    global _embedding_batch_cache, _embeddings_file_cache, _embeddings_file_cache_path, _embeddings_file_cache_time
    if not _embedding_batch_cache:
        return True

    try:
        import sqlite3

        db_path = storage_path if storage_path is not None else get_embeddings_db_path()
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            _embeddings_db_ensure_table(conn)
            rows = []
            for data in _embedding_batch_cache.values():
                rows.append(
                    (
                        data.get("engine_id", "legacy"),
                        data["note_id"],
                        data.get("chunk_index"),
                        data["content_hash"],
                        _embedding_to_blob(data["embedding"]),
                        data.get("timestamp") or datetime.datetime.now().isoformat(),
                    )
                )
            delete_keys = sorted({(row[0], row[1]) for row in rows})
            conn.executemany(
                "DELETE FROM embeddings WHERE engine_id = ? AND note_id = ?",
                delete_keys,
            )
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (engine_id, note_id, chunk_index, content_hash, embedding_blob, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            batch_size = len(_embedding_batch_cache)
            _embedding_batch_cache = {}
            _embeddings_file_cache = None
            _embeddings_file_cache_path = None
            _embeddings_file_cache_time = 0
            log_debug(f"Flushed {batch_size} embeddings to DB")
        finally:
            conn.close()
        return True
    except Exception as exc:
        log_debug(f"Error flushing embedding batch: {exc}")
        return False


# ============================================================================
# Embedding Checkpoints And Error Classification


def load_embedding_exact(note_id, content_hash, db_path=None, engine_id=None):
    try:
        import sqlite3

        engine_candidates = get_embedding_engine_candidates(engine_id=engine_id)
        db_path = db_path or get_embeddings_db_path()

        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path, timeout=10)
            try:
                for candidate in engine_candidates:
                    row = conn.execute(
                        "SELECT embedding_blob FROM embeddings WHERE engine_id = ? AND note_id = ? AND content_hash = ?",
                        (candidate, note_id, content_hash),
                    ).fetchone()
                    if row:
                        return _blob_to_embedding(row[0])
            finally:
                conn.close()

        return None
    except Exception as exc:
        if "Expecting" not in str(exc) and "delimiter" not in str(exc):
            log_debug(f"Error loading exact embedding: {exc}")
        return None


def note_has_embedding(note_id, db_path=None, engine_id=None, any_engine=False):
    try:
        import sqlite3

        engine_candidates = get_embedding_engine_candidates(engine_id=engine_id)
        db_path = db_path or get_embeddings_db_path()

        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path, timeout=10)
            try:
                if any_engine:
                    row = conn.execute(
                        "SELECT 1 FROM embeddings WHERE note_id = ? LIMIT 1",
                        (note_id,),
                    ).fetchone()
                    if row:
                        return True
                else:
                    for candidate in engine_candidates:
                        row = conn.execute(
                            "SELECT 1 FROM embeddings WHERE engine_id = ? AND note_id = ? LIMIT 1",
                            (candidate, note_id),
                        ).fetchone()
                        if row:
                            return True
            finally:
                conn.close()
        return False
    except Exception as exc:
        log_debug(f"Error checking existing embedding: {exc}")
        return False


def load_embedding_key_index(engine_id=None, db_path=None, include_note_timestamps=False):
    """Return existing exact embedding keys and note ids for fast index refreshes."""
    try:
        import sqlite3

        engine_candidates = get_embedding_engine_candidates(engine_id=engine_id)
        db_path = db_path or get_embeddings_db_path()
        exact_keys = set()
        note_ids_any_engine = set()
        note_timestamps = {}

        if not os.path.exists(db_path):
            if include_note_timestamps:
                return exact_keys, note_ids_any_engine, note_timestamps
            return exact_keys, note_ids_any_engine

        conn = sqlite3.connect(db_path, timeout=30)
        try:
            _embeddings_db_ensure_table(conn)
            placeholders = ",".join("?" for _ in engine_candidates)
            for note_id, content_hash, timestamp in conn.execute(
                f"SELECT note_id, content_hash, timestamp FROM embeddings WHERE engine_id IN ({placeholders})",
                engine_candidates,
            ):
                note_id = int(note_id)
                exact_keys.add((note_id, str(content_hash)))
                if include_note_timestamps:
                    ts = str(timestamp or "")
                    if ts and ts > note_timestamps.get(note_id, ""):
                        note_timestamps[note_id] = ts
            for (note_id,) in conn.execute("SELECT DISTINCT note_id FROM embeddings"):
                note_ids_any_engine.add(int(note_id))
        finally:
            conn.close()

        if include_note_timestamps:
            return exact_keys, note_ids_any_engine, note_timestamps
        return exact_keys, note_ids_any_engine
    except Exception as exc:
        log_debug(f"Error loading embedding key index: {exc}")
        if include_note_timestamps:
            return set(), set(), {}
        return set(), set()


def load_embedding(note_id, content_hash, db_path=None, engine_id=None):
    try:
        import sqlite3

        engine_candidates = get_embedding_engine_candidates(engine_id=engine_id)
        db_path = db_path or get_embeddings_db_path()

        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path, timeout=10)
            try:
                for candidate in engine_candidates:
                    row = conn.execute(
                        "SELECT embedding_blob FROM embeddings WHERE engine_id = ? AND note_id = ? AND content_hash = ?",
                        (candidate, note_id, content_hash),
                    ).fetchone()
                    if row:
                        return _blob_to_embedding(row[0])

                for candidate in engine_candidates:
                    row = conn.execute(
                        "SELECT embedding_blob FROM embeddings WHERE engine_id = ? AND note_id = ? LIMIT 1",
                        (candidate, note_id),
                    ).fetchone()
                    if row:
                        return _blob_to_embedding(row[0])
            finally:
                conn.close()

        return None
    except Exception as exc:
        if "Expecting" not in str(exc) and "delimiter" not in str(exc):
            log_debug(f"Error loading embedding: {exc}")
        return None


def load_embeddings_bulk(db_path=None, engine_id=None):
    """Load embedding rows for vector search in one SQLite pass.

    Includes SQLite rowid so a future sqlite-vec backend can map virtual-table
    matches back to the same row metadata.
    """
    try:
        import sqlite3

        engine_candidates = get_embedding_engine_candidates(engine_id=engine_id)
        db_path = db_path or get_embeddings_db_path()

        if not os.path.exists(db_path):
            return []

        conn = sqlite3.connect(db_path, timeout=30)
        try:
            _embeddings_db_ensure_table(conn)
            placeholders = ",".join("?" for _ in engine_candidates)
            rows = conn.execute(
                f"""
                SELECT rowid, engine_id, note_id, chunk_index, content_hash, embedding_blob
                FROM embeddings
                WHERE engine_id IN ({placeholders})
                """,
                engine_candidates,
            ).fetchall()
            return [
                {
                    "rowid": row[0],
                    "engine_id": row[1],
                    "note_id": row[2],
                    "chunk_index": row[3],
                    "content_hash": row[4],
                    "embedding_blob": row[5],
                }
                for row in rows
            ]
        finally:
            conn.close()
    except Exception as exc:
        log_debug(f"Error loading embeddings in bulk: {exc}")
        return []


def load_embedding_engine_counts(db_path=None):
    """Return row counts by stored embedding engine id for diagnostics."""
    try:
        import sqlite3

        db_path = db_path or get_embeddings_db_path()
        if not os.path.exists(db_path):
            return {}

        conn = sqlite3.connect(db_path, timeout=30)
        try:
            _embeddings_db_ensure_table(conn)
            rows = conn.execute(
                "SELECT engine_id, COUNT(*) FROM embeddings GROUP BY engine_id"
            ).fetchall()
            return {str(engine_id or "legacy"): int(count or 0) for engine_id, count in rows}
        finally:
            conn.close()
    except Exception as exc:
        log_debug(f"Error loading embedding engine counts: {exc}")
        return {}


# ============================================================================
# Embedding Checkpoints And Error Classification
# ============================================================================

def make_embedding_scope_id(ntf):
    """Stable fingerprint for the note/deck/field scope used by an embedding run."""
    ntf = ntf or {}
    fields = ntf.get("note_type_fields") or {}
    payload = {
        # Bump when the text sent to the embedding model changes. This prevents
        # stale checkpoints from marking old full-note hashes as current after
        # search/index chunking behavior changes.
        "embedding_content_version": EMBEDDING_CONTENT_VERSION,
        "enabled_note_types": None
        if ntf.get("enabled_note_types") is None
        else sorted(str(name) for name in (ntf.get("enabled_note_types") or [])),
        "enabled_decks": None
        if ntf.get("enabled_decks") is None
        else sorted(str(name) for name in (ntf.get("enabled_decks") or [])),
        "note_type_fields": {
            str(model): sorted(str(field) for field in (field_names or []))
            for model, field_names in sorted(fields.items())
        },
        "search_all_fields": bool(ntf.get("search_all_fields", False)),
        "use_first_field_fallback": bool(ntf.get("use_first_field_fallback", True)),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_checkpoint(processed_note_ids, total_notes, errors=0, engine_id=None, scope_id=None):
    try:
        if engine_id is None:
            engine_id = get_embedding_engine_id()
        checkpoint_path = get_checkpoint_path()
        checkpoint_data = {
            "processed_note_ids": list(processed_note_ids),
            "total_notes": total_notes,
            "processed_count": len(processed_note_ids),
            "errors": errors,
            "engine_id": engine_id,
            "scope_id": scope_id,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        temp_path = checkpoint_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(checkpoint_data, fh, indent=2)
        if os.path.exists(checkpoint_path):
            os.replace(temp_path, checkpoint_path)
        else:
            os.rename(temp_path, checkpoint_path)
        log_debug(f"Saved checkpoint: {len(processed_note_ids)}/{total_notes} notes processed")
        return True
    except Exception as exc:
        log_debug(f"Error saving checkpoint: {exc}")
        return False


def load_checkpoint():
    try:
        checkpoint_path = get_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            return None
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        log_debug(f"Error loading checkpoint: {exc}")
        return None


def clear_checkpoint():
    try:
        checkpoint_path = get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        return True
    except Exception as exc:
        log_debug(f"Error clearing checkpoint: {exc}")
        return False
