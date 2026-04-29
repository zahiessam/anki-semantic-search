"""Embedding, collection helpers, and persistence for the addon UI/workers."""

# ============================================================================
# Imports
# ============================================================================

import datetime
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

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
    compute_tfidf_scores,
    extract_keywords_improved,
    get_extended_stop_words,
)

# ============================================================================
# Module Constants And Caches
# ============================================================================

HTML_TAG_RE = re.compile(r"<.*?>", re.DOTALL)
OLLAMA_EMBED_CHUNK_SIZE = 64

_embedding_batch_cache = {}
_embeddings_file_cache = None
_embeddings_file_cache_path = None
_embeddings_file_cache_time = 0
_corrupted_files = set()
MAX_EMBEDDING_CACHE_SIZE = 100


# ============================================================================
# Embedding Provider Clients
# ============================================================================

# --- Shared Request Helpers ---

def estimate_tokens(text):
    return len(text or "") // 4


def _request_json(url, payload=None, headers=None, timeout=30):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, body, headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


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
        data = _request_json(f"{base_url}/api/tags", timeout=10)
        models = data.get("models") or []
        names = []
        for item in models:
            if isinstance(item, dict) and item.get("name"):
                names.append(item["name"].strip())
            elif isinstance(item, str):
                names.append(item.strip())
        return sorted(names)
    except Exception as exc:
        log_debug(f"Ollama list models failed: {exc}")
        return []


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
    return candidates


def get_embedding_for_query(text, config=None):
    # explanation: resolve same-provider and independent embedding choices into backend fields.
    from ..utils.config import get_effective_embedding_config
    config = get_effective_embedding_config(config or load_config())
    sc = config.get("search_config") or {}
    engine = (sc.get("embedding_engine") or "voyage").strip().lower()

    if engine == "local_openai":
        return get_embedding_via_local_openai(
            text,
            base_url=(sc.get("local_llm_url") or "http://localhost:1234/v1").strip(),
            model=(sc.get("local_llm_model") or "text-embedding-3-small").strip(),
        )
    if engine == "ollama":
        return get_embedding_via_ollama(
            text,
            base_url=(sc.get("ollama_base_url") or "http://localhost:11434").strip(),
            model=(sc.get("ollama_embed_model") or "nomic-embed-text").strip(),
        )
    if engine == "openai":
        return get_embedding_via_openai(
            text,
            api_key=(sc.get("openai_embedding_api_key") or "").strip(),
            model=(sc.get("openai_embedding_model") or "text-embedding-3-small").strip(),
        )
    if engine == "cohere":
        return get_embedding_via_cohere(
            text,
            is_query=True,
            api_key=(sc.get("cohere_api_key") or "").strip(),
            model=(sc.get("cohere_embedding_model") or "embed-english-v3.0").strip(),
        )
    return get_embedding_via_voyage(
        text,
        is_query=True,
        api_key=(sc.get("voyage_api_key") or "").strip(),
        model=(sc.get("voyage_embedding_model") or "voyage-3.5-lite").strip(),
    )


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
    try:
        if not mw or not mw.col:
            return {}
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


def analyze_note_eligibility(ntf):
    try:
        if not mw or not mw.col:
            return {
                "total_notes": 0,
                "eligible_count": 0,
                "filtered_out_note_type_count": 0,
                "no_selected_fields_count": 0,
                "empty_selected_fields_count": 0,
                "ineligible_notes": [],
            }
        deck_q = _build_deck_query(ntf.get("enabled_decks"))
        note_ids = mw.col.find_notes(deck_q) if deck_q else mw.col.find_notes("")
        if not note_ids:
            return {
                "total_notes": 0,
                "eligible_count": 0,
                "filtered_out_note_type_count": 0,
                "no_selected_fields_count": 0,
                "empty_selected_fields_count": 0,
                "ineligible_notes": [],
            }

        enabled = ntf.get("enabled_note_types")
        enabled_set = set(enabled) if enabled else None
        search_all = bool(ntf.get("search_all_fields", False))
        ntf_fields = ntf.get("note_type_fields") or {}
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
                wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))
                if not wanted and use_first and fields:
                    wanted = {fields[0]["name"].lower()}
                indices = [i for i, field in enumerate(fields) if field["name"].lower() in wanted]
            model_map[model["id"]] = {
                "indices": indices,
                "model_name": model_name,
                "field_names": [fields[i]["name"] for i in indices if i < len(fields)],
            }

        eligible_count = 0
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
                "SELECT ?, note_id, chunk_index, content_hash, embedding_blob, timestamp FROM embeddings",
                ("legacy",),
            )
            conn.execute("DROP TABLE embeddings")
            conn.execute("ALTER TABLE embeddings_new RENAME TO embeddings")
            log_debug("Migrated embeddings table to multi-engine schema (existing rows as engine_id='legacy')")
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
    import numpy as np

    return np.array(embedding, dtype=np.float32).tobytes()


def _blob_to_embedding(blob):
    import numpy as np

    return np.frombuffer(blob, dtype=np.float32)


def save_embedding(note_id, content_hash, embedding, batch_mode=True, storage_path=None, engine_id=None):
    try:
        import numpy as np
        import sqlite3

        if engine_id is None:
            engine_id = get_embedding_engine_id()
        db_path = storage_path if storage_path is not None else get_embeddings_db_path()
        key = f"{engine_id}_{note_id}_{content_hash}"
        emb_arr = (
            np.array(embedding, dtype=np.float32)
            if not isinstance(embedding, np.ndarray)
            else embedding.astype(np.float32)
        )
        embedding_data = {
            "engine_id": engine_id,
            "note_id": note_id,
            "content_hash": content_hash,
            "embedding": emb_arr,
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
                "INSERT OR REPLACE INTO embeddings (engine_id, note_id, chunk_index, content_hash, embedding_blob, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    engine_id,
                    note_id,
                    None,
                    content_hash,
                    _embedding_to_blob(emb_arr),
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
                        None,
                        data["content_hash"],
                        _embedding_to_blob(data["embedding"]),
                        data.get("timestamp") or datetime.datetime.now().isoformat(),
                    )
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


def _load_embedding_from_json_legacy(note_id, content_hash):
    global _embeddings_file_cache, _embeddings_file_cache_path, _embeddings_file_cache_time, _corrupted_files
    try:
        storage_path = get_embeddings_storage_path_for_read()
        if not os.path.exists(storage_path) or storage_path in _corrupted_files:
            return None

        file_mtime = os.path.getmtime(storage_path)
        if (
            _embeddings_file_cache is not None
            and _embeddings_file_cache_path == storage_path
            and _embeddings_file_cache_time == file_mtime
        ):
            embeddings_data = _embeddings_file_cache
        else:
            with open(storage_path, "r", encoding="utf-8") as fh:
                embeddings_data = json.load(fh)
            _embeddings_file_cache = embeddings_data
            _embeddings_file_cache_path = storage_path
            _embeddings_file_cache_time = file_mtime
            _corrupted_files.discard(storage_path)

        key = f"{note_id}_{content_hash}"
        entry = embeddings_data.get(key)
        if isinstance(entry, dict):
            return entry.get("embedding")
        return None
    except json.JSONDecodeError as exc:
        _corrupted_files.add(storage_path)
        log_debug(f"JSON decode error in embeddings file: {exc}")
        return None
    except Exception as exc:
        log_debug(f"Legacy embedding load error: {exc}")
        return None


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

        return _load_embedding_from_json_legacy(note_id, content_hash)
    except Exception as exc:
        if "Expecting" not in str(exc) and "delimiter" not in str(exc):
            log_debug(f"Error loading embedding: {exc}")
        return None


def migrate_embeddings_json_to_db():
    try:
        import sqlite3

        json_path = get_embeddings_storage_path_for_read()
        if not os.path.exists(json_path) or not json_path.endswith(".json"):
            return 0, "No legacy JSON embeddings file found."

        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return 0, "Invalid JSON format (expected object)."

        db_path = get_embeddings_db_path()
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            _embeddings_db_ensure_table(conn)
            batch = []
            migrated = 0
            for entry in data.values():
                if not isinstance(entry, dict) or "embedding" not in entry:
                    continue
                try:
                    note_id = int(entry.get("note_id", 0))
                    content_hash = str(entry.get("content_hash", ""))
                    blob = _embedding_to_blob(entry["embedding"])
                    ts = entry.get("timestamp") or datetime.datetime.now().isoformat()
                    batch.append(("legacy", note_id, None, content_hash, blob, ts))
                    if len(batch) >= 500:
                        conn.executemany(
                            "INSERT OR REPLACE INTO embeddings (engine_id, note_id, chunk_index, content_hash, embedding_blob, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                            batch,
                        )
                        conn.commit()
                        migrated += len(batch)
                        batch = []
                except Exception:
                    continue

            if batch:
                conn.executemany(
                    "INSERT OR REPLACE INTO embeddings (engine_id, note_id, chunk_index, content_hash, embedding_blob, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    batch,
                )
                conn.commit()
                migrated += len(batch)
        finally:
            conn.close()

        return migrated, None
    except Exception as exc:
        log_debug(f"Migration error: {exc}")
        return 0, str(exc)


# ============================================================================
# Embedding Checkpoints And Error Classification
# ============================================================================

def save_checkpoint(processed_note_ids, total_notes, errors=0, engine_id=None):
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
