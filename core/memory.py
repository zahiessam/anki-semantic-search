"""Local profile and snippet memory for agentic retrieval.

Durable memory stores preferences, scope state, and compact local fact snippets
from final answer context. It does not store full note text, answers, or raw
chat history.
"""

import hashlib
import datetime
import json
import math
import os
import re
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field

from .agentic_planner import has_entity_signal


SCHEMA_VERSION = 4
DEFAULT_PROFILE_ID = "default"
SNAPSHOT_CHAR_LIMIT = 1500
MEMORY_SNIPPET_CHARS = 150
MEMORY_RETENTION_DAYS = 30
MAX_SNIPPETS_PER_SAVE = 24
MAX_SNAPSHOT_SNIPPETS = 5
MEMORY_LAZY_EMBED_LIMIT = 16
MEMORY_LAZY_EMBED_SECONDS = 3.0
_BACKGROUND_EMBED_LOCK = threading.Lock()
_BACKGROUND_EMBED_IN_FLIGHT = set()
_BACKGROUND_EMBED_TRACKED_IDS = set()
PROFILE_ALLOWED_KEYS = {
    "enabled_decks",
    "enabled_note_types",
    "selected_fields",
    "agentic_planner_mode",
    "enable_agentic_rag",
    "answer_style",
}
FORBIDDEN_PROFILE_KEYS = {
    "note_text",
    "note_content",
    "content",
    "extracted_facts",
    "facts",
    "answer",
    "answers",
    "question",
    "query",
    "clinical_entities",
}


def _now_iso():
    return datetime.datetime.now().isoformat()


def _user_files_dir():
    try:
        from ..utils.paths import _user_files_dir as path_user_files_dir
    except Exception:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "user_files")
        os.makedirs(path, exist_ok=True)
        return path
    return path_user_files_dir()


def memory_db_path():
    return os.path.join(_user_files_dir(), "agent_memory.db")


def _backup_and_remove(path):
    if not os.path.exists(path):
        return
    backup = path + ".bak"
    try:
        if os.path.exists(backup):
            os.remove(backup)
        shutil.copy2(path, backup)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def _connect(path=None, check_same_thread=True):
    path = path or memory_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=check_same_thread)
    con.row_factory = sqlite3.Row
    return con


def _init_schema(con):
    con.execute("create table if not exists meta (key text primary key, value text not null)")
    con.execute(
        """
        create table if not exists profiles (
            profile_id text primary key,
            data_json text not null,
            created_at text not null,
            updated_at text not null
        )
        """
    )
    con.execute(
        """
        create table if not exists memory_snippets (
            id integer primary key autoincrement,
            profile_id text not null,
            note_id integer not null,
            chunk_index integer not null default 0,
            snippet_hash text not null,
            snippet_text text not null,
            normalized_text text not null,
            source_query text not null default '',
            subquery_label text not null default '',
            citable integer not null default 0,
            scope_hash text not null default '',
            created_at text not null,
            last_seen_at text not null,
            hit_count integer not null default 0,
            meta_json text not null default '{}',
            unique(profile_id, note_id, chunk_index, snippet_hash)
        )
        """
    )
    if not _column_exists(con, "memory_snippets", "scope_hash"):
        try:
            con.execute("alter table memory_snippets add column scope_hash text not null default ''")
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
    con.execute(
        "create index if not exists idx_memory_snippets_profile_seen on memory_snippets(profile_id, last_seen_at)"
    )
    con.execute(
        "create index if not exists idx_memory_snippets_profile_scope_seen on memory_snippets(profile_id, scope_hash, last_seen_at)"
    )
    con.execute(
        """
        create table if not exists memory_snippet_embeddings (
            snippet_id integer not null,
            engine_id text not null,
            embedding_blob blob not null,
            embedding_dim integer not null,
            content_hash text not null,
            created_at text not null,
            updated_at text not null,
            primary key (snippet_id, engine_id),
            foreign key (snippet_id) references memory_snippets(id) on delete cascade
        )
        """
    )
    con.execute(
        "insert or replace into meta (key, value) values ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    con.commit()


def _migrate_v1_to_v2(con):
    _init_schema(con)


def _migrate_v2_to_v3(con):
    _init_schema(con)


def _column_exists(con, table, column):
    try:
        return any(row[1] == column for row in con.execute(f"pragma table_info({table})").fetchall())
    except sqlite3.Error:
        return False


def _migrate_to_v4(con):
    _init_schema(con)
    try:
        con.execute("begin")
        if not _column_exists(con, "memory_snippets", "scope_hash"):
            try:
                con.execute("alter table memory_snippets add column scope_hash text not null default ''")
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
        con.execute(
            "create index if not exists idx_memory_snippets_profile_scope_seen on memory_snippets(profile_id, scope_hash, last_seen_at)"
        )
        con.execute(
            "insert or replace into meta (key, value) values ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        con.commit()
    except Exception:
        con.rollback()
        raise


def _ensure_schema(path=None):
    path = path or memory_db_path()
    con = _connect(path)
    try:
        row = None
        try:
            row = con.execute("select value from meta where key='schema_version'").fetchone()
        except sqlite3.Error:
            row = None
        version = str(row["value"]) if row is not None else ""
        if version and version != str(SCHEMA_VERSION):
            if version == "1":
                _migrate_v1_to_v2(con)
                _migrate_to_v4(con)
                return con
            if version == "2":
                _migrate_v2_to_v3(con)
                _migrate_to_v4(con)
                return con
            if version == "3":
                _migrate_to_v4(con)
                return con
            con.close()
            _backup_and_remove(path)
            con = _connect(path)
        _init_schema(con)
        if not _column_exists(con, "memory_snippets", "scope_hash"):
            _migrate_to_v4(con)
        return con
    except Exception:
        try:
            con.close()
        except Exception:
            pass
        raise


@dataclass
class MemoryProfile:
    profile_id: str = DEFAULT_PROFILE_ID
    data: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def safe_data(self):
        return {k: self.data.get(k) for k in PROFILE_ALLOWED_KEYS if k in self.data}


@dataclass
class SessionMemory:
    resolved_followup_target: str = ""
    summary: str = ""
    fact_cache: list = field(default_factory=list)

    def clear(self):
        self.resolved_followup_target = ""
        self.summary = ""
        self.fact_cache = []

    def maybe_clear_followup_for_query(self, query, search_config=None):
        if has_entity_signal(query or "", search_config):
            self.resolved_followup_target = ""


@dataclass
class MemorySnapshot:
    profile_id: str = DEFAULT_PROFILE_ID
    active_scope: dict = field(default_factory=dict)
    answer_style: str = ""
    prior_query: str = ""
    resolved_followup_target: str = ""
    review_context_hint: str = ""
    selected_text_hint: str = ""
    retrieval_budget: dict = field(default_factory=dict)
    session_summary: str = ""
    fact_snippets: list = field(default_factory=list)

    def to_dict(self):
        return {
            "profile_id": self.profile_id,
            "active_scope": self.active_scope,
            "answer_style": self.answer_style,
            "prior_query": self.prior_query,
            "resolved_followup_target": self.resolved_followup_target,
            "review_context_hint": self.review_context_hint,
            "selected_text_hint": self.selected_text_hint,
            "retrieval_budget": self.retrieval_budget,
            "session_summary": self.session_summary,
            "fact_snippets": self.fact_snippets,
        }


def load_memory_profile(profile_id=DEFAULT_PROFILE_ID, db_path=None):
    con = _ensure_schema(db_path)
    try:
        row = con.execute(
            "select profile_id, data_json, created_at, updated_at from profiles where profile_id=?",
            (profile_id,),
        ).fetchone()
        if not row:
            return MemoryProfile(profile_id=profile_id)
        try:
            data = json.loads(row["data_json"])
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data = {k: data.get(k) for k in PROFILE_ALLOWED_KEYS if k in data}
        return MemoryProfile(
            profile_id=row["profile_id"],
            data=data,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
    finally:
        con.close()


def save_memory_profile(profile, db_path=None):
    if not isinstance(profile, MemoryProfile):
        profile = MemoryProfile(data=dict(profile or {}))
    data = profile.safe_data()
    for key in FORBIDDEN_PROFILE_KEYS:
        data.pop(key, None)
    now = _now_iso()
    created = profile.created_at or now
    con = _ensure_schema(db_path)
    try:
        con.execute(
            """
            insert into profiles (profile_id, data_json, created_at, updated_at)
            values (?, ?, ?, ?)
            on conflict(profile_id) do update set
                data_json=excluded.data_json,
                updated_at=excluded.updated_at
            """,
            (profile.profile_id or DEFAULT_PROFILE_ID, json.dumps(data, ensure_ascii=False), created, now),
        )
        con.commit()
        profile.data = data
        profile.created_at = created
        profile.updated_at = now
        return True
    finally:
        con.close()


def clear_memory_profile(profile_id=DEFAULT_PROFILE_ID, db_path=None):
    con = _ensure_schema(db_path)
    try:
        con.execute("delete from profiles where profile_id=?", (profile_id,))
        con.commit()
        return True
    finally:
        con.close()


def clear_memory(profile_id=DEFAULT_PROFILE_ID, db_path=None):
    con = _ensure_schema(db_path)
    try:
        con.execute("delete from profiles where profile_id=?", (profile_id,))
        con.execute("delete from memory_snippets where profile_id=?", (profile_id,))
        con.commit()
        return True
    finally:
        con.close()


def _snippet_stats(profile_id=DEFAULT_PROFILE_ID, db_path=None):
    con = _ensure_schema(db_path)
    try:
        row = con.execute(
            """
            select count(*) as count,
                   min(created_at) as oldest_created_at,
                   max(last_seen_at) as newest_seen_at,
                   coalesce(sum(hit_count), 0) as total_hit_count
            from memory_snippets
            where profile_id=?
            """,
            (profile_id,),
        ).fetchone()
        embedded = con.execute(
            """
            select count(distinct e.snippet_id) as count
            from memory_snippet_embeddings e
            join memory_snippets s on s.id=e.snippet_id
            where s.profile_id=?
            """,
            (profile_id,),
        ).fetchone()
        return {
            "snippet_count": int(row["count"] or 0) if row else 0,
            "embedded_snippet_count": int(embedded["count"] or 0) if embedded else 0,
            "oldest_snippet_at": row["oldest_created_at"] if row else None,
            "newest_snippet_at": row["newest_seen_at"] if row else None,
            "recent_hit_count": int(row["total_hit_count"] or 0) if row else 0,
            "retention_days": MEMORY_RETENTION_DAYS,
        }
    finally:
        con.close()


def memory_profile_summary(profile=None, db_path=None):
    profile = profile or load_memory_profile(db_path=db_path)
    data = profile.safe_data()
    summary = {
        "profile_id": profile.profile_id,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
        "keys": sorted(data.keys()),
        "enabled_deck_count": len(data.get("enabled_decks") or []),
        "enabled_note_type_count": len(data.get("enabled_note_types") or []),
        "selected_field_count": len(data.get("selected_fields") or []),
        "agentic_planner_mode": data.get("agentic_planner_mode") or "",
        "enable_agentic_rag": bool(data.get("enable_agentic_rag", False)),
    }
    summary.update(_snippet_stats(profile.profile_id, db_path=db_path))
    return summary


def _compact_text(text, max_chars):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:max_chars].strip()


def _normalize_snippet_text(text, max_chars=MEMORY_SNIPPET_CHARS):
    text = str(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\{c\d+::(.*?)(?:::[^}]*)?\}\}", r"\1", text)
    text = re.sub(r"\[sound:[^\]]+\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars].strip()


def _memory_tokens(text):
    return [
        token.lower()
        for token in re.findall(r"\b[\w+-]+\b", text or "")
        if len(token) > 2
    ]


def _snippet_hash(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:32]


def _content_hash(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _normalized_scope_payload(config):
    ntf = (config or {}).get("note_type_filter") or {}
    scope_fields = ntf.get("scope_fields") or {}
    fields = []
    if isinstance(scope_fields, dict):
        for deck_map in scope_fields.values():
            if isinstance(deck_map, dict):
                for names in deck_map.values():
                    fields.extend(names or [])
    for names in (ntf.get("note_type_fields") or {}).values():
        fields.extend(names or [])

    def norm_list(values):
        return sorted({str(value).strip().lower() for value in (values or []) if str(value).strip()})

    return {
        "scope_mode": str(ntf.get("scope_mode") or "").strip().lower(),
        "enabled_decks": norm_list(ntf.get("enabled_decks")),
        "enabled_tags": norm_list(ntf.get("enabled_tags")),
        "enabled_note_types": norm_list(ntf.get("enabled_note_types")),
        "fields": norm_list(fields),
    }


def scope_hash_from_config(config):
    payload = _normalized_scope_payload(config)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _embedding_to_blob(embedding):
    try:
        import numpy as np
        return np.array([float(x) for x in embedding], dtype=np.float32).tobytes()
    except Exception:
        import array
        return array.array("f", (float(value) for value in embedding)).tobytes()


def _blob_to_embedding(blob):
    try:
        import numpy as np
        return [float(x) for x in np.frombuffer(blob, dtype=np.float32)]
    except Exception:
        import array
        values = array.array("f")
        values.frombytes(bytes(blob))
        return [float(x) for x in values]


def _cosine_similarity(left, right):
    left = [float(x) for x in (left or [])]
    right = [float(x) for x in (right or [])]
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    if ln <= 0 or rn <= 0:
        return None
    return dot / (ln * rn)


def memory_embedding_engine_id(config=None):
    try:
        from ..core.engine import get_embedding_engine_id
    except Exception:
        try:
            from .engine import get_embedding_engine_id
        except Exception:
            get_embedding_engine_id = None
    try:
        from ..utils.config import get_effective_embedding_config
    except Exception:
        try:
            from utils.config import get_effective_embedding_config
        except Exception:
            get_effective_embedding_config = None
    cfg = get_effective_embedding_config(config) if get_effective_embedding_config else (config or {})
    sc = (cfg or {}).get("search_config") or {}
    base_id = get_embedding_engine_id(cfg) if get_embedding_engine_id else (sc.get("embedding_engine") or "default")
    provider = str(base_id).split(":", 1)[0].strip().lower()
    if provider in {"ollama", "local_openai"}:
        url = (
            sc.get("ollama_base_url")
            if provider == "ollama"
            else (sc.get("embedding_local_url") or sc.get("local_llm_url"))
        ) or ""
        normalized_url = str(url).strip().rstrip("/").lower()
        model = str(base_id).split(":", 1)[1] if ":" in str(base_id) else (sc.get("embedding_local_model") or "default")
        return f"{provider}:{normalized_url}:{model}"
    return str(base_id)


def _note_primary_text(note):
    if not isinstance(note, dict):
        return ""
    for key in ("display_content", "content_preview", "content"):
        if note.get(key):
            return note.get(key)
    return ""


def _cutoff_iso(retention_days=MEMORY_RETENTION_DAYS):
    try:
        days = max(1, int(retention_days))
    except Exception:
        days = MEMORY_RETENTION_DAYS
    return (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()


def prune_memory_snippets(profile_id=DEFAULT_PROFILE_ID, retention_days=MEMORY_RETENTION_DAYS, db_path=None):
    con = _ensure_schema(db_path)
    try:
        cur = con.execute(
            "delete from memory_snippets where profile_id=? and last_seen_at < ?",
            (profile_id, _cutoff_iso(retention_days)),
        )
        con.commit()
        return int(cur.rowcount or 0)
    finally:
        con.close()


def save_fact_snippets_from_context(
    context_notes,
    source_query="",
    subquery_label="main",
    profile_id=DEFAULT_PROFILE_ID,
    db_path=None,
    max_snippets=MAX_SNIPPETS_PER_SAVE,
    retention_days=MEMORY_RETENTION_DAYS,
    config=None,
):
    now = _now_iso()
    saved = 0
    skipped = 0
    seen = set()
    saved_ids = []
    scope_hash = scope_hash_from_config(config) if config else ""
    con = _ensure_schema(db_path)
    try:
        for note in list(context_notes or [])[: max(1, int(max_snippets or MAX_SNIPPETS_PER_SAVE))]:
            if not isinstance(note, dict):
                skipped += 1
                continue
            note_id = note.get("id")
            if note_id is None:
                skipped += 1
                continue
            snippet_text = _normalize_snippet_text(_note_primary_text(note), MEMORY_SNIPPET_CHARS)
            if not snippet_text:
                skipped += 1
                continue
            chunk_index = 0
            snip_hash = _snippet_hash(snippet_text)
            key = (profile_id, int(note_id), chunk_index, snip_hash)
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            normalized = " ".join(_memory_tokens(snippet_text))
            meta = {
                "source": "final_answer_context",
                "original_chunk_index": note.get("chunk_index"),
                "citable": False,
            }
            con.execute(
                """
                insert into memory_snippets (
                    profile_id, note_id, chunk_index, snippet_hash, snippet_text,
                    normalized_text, source_query, subquery_label, citable, scope_hash,
                    created_at, last_seen_at, hit_count, meta_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, 0, ?)
                on conflict(profile_id, note_id, chunk_index, snippet_hash) do update set
                    snippet_text=excluded.snippet_text,
                    normalized_text=excluded.normalized_text,
                    source_query=excluded.source_query,
                    subquery_label=excluded.subquery_label,
                    citable=0,
                    scope_hash=excluded.scope_hash,
                    last_seen_at=excluded.last_seen_at,
                    meta_json=excluded.meta_json
                """,
                (
                    profile_id,
                    int(note_id),
                    chunk_index,
                    snip_hash,
                    snippet_text,
                    normalized,
                    _compact_text(source_query, 220),
                    _compact_text(subquery_label, 120),
                    scope_hash,
                    now,
                    now,
                    json.dumps(meta, ensure_ascii=False),
                ),
            )
            row = con.execute(
                """
                select id from memory_snippets
                where profile_id=? and note_id=? and chunk_index=? and snippet_hash=?
                """,
                (profile_id, int(note_id), chunk_index, snip_hash),
            ).fetchone()
            if row:
                saved_ids.append(int(row["id"]))
            saved += 1
        con.commit()
    finally:
        con.close()
    pruned = prune_memory_snippets(profile_id=profile_id, retention_days=retention_days, db_path=db_path)
    return {"saved": saved, "skipped": skipped, "pruned": pruned, "saved_ids": saved_ids}


def _text_rank_memory_rows(query, rows, limit, preferred_scope_hash=""):
    query_tokens = _memory_tokens(query)
    if not query_tokens:
        return []
    query_set = set(query_tokens)
    scored = []
    for row in rows:
        text = " ".join([
            row["normalized_text"] or "",
            row["source_query"] or "",
            row["subquery_label"] or "",
        ])
        tokens = _memory_tokens(text)
        if not tokens:
            continue
        token_set = set(tokens)
        overlap = len(query_set & token_set)
        if overlap <= 0:
            continue
        score = (overlap * 10.0) + min(5, int(row["hit_count"] or 0))
        if preferred_scope_hash and row["scope_hash"] == preferred_scope_hash:
            score += 2.0
        scored.append((score, row))
    scored.sort(key=lambda item: (item[0], item[1]["last_seen_at"] or ""), reverse=True)
    return scored[: max(1, int(limit or MAX_SNAPSHOT_SNIPPETS))]


def _load_memory_rows(con, profile_id, retention_days=MEMORY_RETENTION_DAYS, scope_hash=""):
    cutoff = _cutoff_iso(retention_days)
    if scope_hash:
        return con.execute(
            """
            select id, note_id, chunk_index, snippet_hash, snippet_text, normalized_text,
                   source_query, subquery_label, scope_hash, last_seen_at, hit_count
            from memory_snippets
            where profile_id=? and last_seen_at >= ? and (scope_hash=? or scope_hash='')
            """,
            (profile_id, cutoff, scope_hash),
        ).fetchall()
    return con.execute(
        """
        select id, note_id, chunk_index, snippet_hash, snippet_text, normalized_text,
               source_query, subquery_label, scope_hash, last_seen_at, hit_count
        from memory_snippets
        where profile_id=? and last_seen_at >= ?
        """,
        (profile_id, cutoff),
    ).fetchall()


def _load_embedding_row(con, snippet_id, engine_id):
    return con.execute(
        """
        select embedding_blob, embedding_dim, content_hash
        from memory_snippet_embeddings
        where snippet_id=? and engine_id=?
        """,
        (snippet_id, engine_id),
    ).fetchone()


def _embedding_is_fresh(row, snippet_text, expected_dim=None):
    if not row:
        return False
    if row["content_hash"] != _content_hash(snippet_text):
        return False
    if expected_dim is not None and int(row["embedding_dim"] or 0) != int(expected_dim):
        return False
    return True


def _save_snippet_embedding(con, snippet_id, engine_id, snippet_text, embedding):
    values = [float(x) for x in embedding or []]
    if not values:
        return False
    now = _now_iso()
    con.execute(
        """
        insert into memory_snippet_embeddings (
            snippet_id, engine_id, embedding_blob, embedding_dim, content_hash,
            created_at, updated_at
        )
        values (?, ?, ?, ?, ?, ?, ?)
        on conflict(snippet_id, engine_id) do update set
            embedding_blob=excluded.embedding_blob,
            embedding_dim=excluded.embedding_dim,
            content_hash=excluded.content_hash,
            updated_at=excluded.updated_at
        """,
        (
            int(snippet_id),
            engine_id,
            _embedding_to_blob(values),
            len(values),
            _content_hash(snippet_text),
            now,
            now,
        ),
    )
    return True


def _log_debug(message):
    try:
        from ..utils import log_debug
    except Exception:
        try:
            from utils import log_debug
        except Exception:
            log_debug = None
    if log_debug:
        try:
            log_debug(message)
        except Exception:
            pass


def _row_payload(row):
    return {
        "id": int(row["id"]),
        "snippet_text": row["snippet_text"],
    }


def _fresh_embedding_count(con, rows, engine_id):
    count = 0
    for row in rows or []:
        emb_row = _load_embedding_row(con, row["id"], engine_id)
        if _embedding_is_fresh(emb_row, row["snippet_text"]):
            count += 1
    return count


def _tracked_background_completed(con, engine_id):
    with _BACKGROUND_EMBED_LOCK:
        tracked = set(_BACKGROUND_EMBED_TRACKED_IDS)
    if not tracked:
        return False
    placeholders = ",".join("?" for _ in tracked)
    rows = con.execute(
        f"""
        select id, snippet_text
        from memory_snippets
        where id in ({placeholders})
        """,
        list(tracked),
    ).fetchall()
    completed_ids = []
    for row in rows:
        emb_row = _load_embedding_row(con, row["id"], engine_id)
        if _embedding_is_fresh(emb_row, row["snippet_text"]):
            completed_ids.append(int(row["id"]))
    if completed_ids:
        with _BACKGROUND_EMBED_LOCK:
            _BACKGROUND_EMBED_TRACKED_IDS.difference_update(completed_ids)
        return True
    return False


def _background_embed_worker(rows, config=None, db_path=None, engine_id=""):
    try:
        from .engine import get_embedding_for_query
    except Exception as exc:
        _log_debug(f"Memory background embedding unavailable: {exc}")
        return
    con = None
    try:
        con = _connect(db_path, check_same_thread=False)
        for row in rows or []:
            snippet_id = int(row.get("id") or 0)
            snippet_text = row.get("snippet_text") or ""
            if not snippet_id or not snippet_text:
                continue
            try:
                embedding = get_embedding_for_query(snippet_text, config)
                if embedding:
                    _save_snippet_embedding(con, snippet_id, engine_id, snippet_text, embedding)
                    con.commit()
            except Exception as exc:
                _log_debug(f"Memory background embedding skipped snippet {snippet_id}: {exc}")
    except Exception as exc:
        _log_debug(f"Memory background embedding failed: {exc}")
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
        with _BACKGROUND_EMBED_LOCK:
            for row in rows or []:
                _BACKGROUND_EMBED_IN_FLIGHT.discard(int(row.get("id") or 0))


def schedule_memory_snippet_embeddings(rows, config=None, db_path=None, max_snippets=MEMORY_LAZY_EMBED_LIMIT):
    if not rows or config is None:
        return {"scheduled": False, "count": 0}
    engine_id = memory_embedding_engine_id(config)
    payload = []
    with _BACKGROUND_EMBED_LOCK:
        for row in list(rows or [])[: max(1, int(max_snippets or MEMORY_LAZY_EMBED_LIMIT))]:
            snippet_id = int(row["id"])
            if snippet_id in _BACKGROUND_EMBED_IN_FLIGHT:
                continue
            _BACKGROUND_EMBED_IN_FLIGHT.add(snippet_id)
            _BACKGROUND_EMBED_TRACKED_IDS.add(snippet_id)
            payload.append(_row_payload(row))
    if not payload:
        return {"scheduled": False, "count": 0}
    thread = threading.Thread(
        target=_background_embed_worker,
        args=(payload, config, db_path, engine_id),
        daemon=True,
        name="agent-memory-embedder",
    )
    thread.start()
    return {"scheduled": True, "count": len(payload)}


def schedule_memory_embeddings_for_ids(snippet_ids, config=None, db_path=None):
    ids = [int(item) for item in (snippet_ids or []) if str(item).strip()]
    if not ids:
        return {"scheduled": False, "count": 0}
    con = _ensure_schema(db_path)
    try:
        placeholders = ",".join("?" for _ in ids)
        rows = con.execute(
            f"select id, snippet_text from memory_snippets where id in ({placeholders})",
            ids,
        ).fetchall()
    finally:
        con.close()
    return schedule_memory_snippet_embeddings(rows, config=config, db_path=db_path, max_snippets=len(rows))


def ensure_memory_snippet_embeddings(rows, config=None, db_path=None, max_snippets=MEMORY_LAZY_EMBED_LIMIT, timeout_seconds=MEMORY_LAZY_EMBED_SECONDS):
    diagnostics = {"embedded": 0, "stale": 0, "missing": 0, "errors": 0, "timeout": False}
    try:
        from .engine import get_embedding_for_query
    except Exception:
        diagnostics["errors"] += 1
        return diagnostics
    engine_id = memory_embedding_engine_id(config)
    deadline = time.monotonic() + float(timeout_seconds or MEMORY_LAZY_EMBED_SECONDS)
    con = _ensure_schema(db_path)
    try:
        checked = 0
        for row in rows or []:
            if checked >= int(max_snippets or MEMORY_LAZY_EMBED_LIMIT):
                break
            if time.monotonic() >= deadline:
                diagnostics["timeout"] = True
                break
            existing = _load_embedding_row(con, row["id"], engine_id)
            if existing and _embedding_is_fresh(existing, row["snippet_text"]):
                continue
            if existing:
                diagnostics["stale"] += 1
            else:
                diagnostics["missing"] += 1
            checked += 1
            try:
                embedding = get_embedding_for_query(row["snippet_text"], config)
                if embedding:
                    _save_snippet_embedding(con, row["id"], engine_id, row["snippet_text"], embedding)
                    diagnostics["embedded"] += 1
                    con.commit()
            except Exception:
                diagnostics["errors"] += 1
        return diagnostics
    finally:
        con.close()


def rebuild_memory_embeddings(profile_id=DEFAULT_PROFILE_ID, config=None, db_path=None, limit=None):
    con = _ensure_schema(db_path)
    try:
        rows = _load_memory_rows(con, profile_id, retention_days=36500)
    finally:
        con.close()
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return ensure_memory_snippet_embeddings(rows, config=config, db_path=db_path, max_snippets=len(rows), timeout_seconds=3600)


def delete_memory_snippets(snippet_ids, profile_id=DEFAULT_PROFILE_ID, db_path=None):
    ids = [int(item) for item in (snippet_ids or []) if str(item).strip()]
    if not ids:
        return 0
    con = _ensure_schema(db_path)
    try:
        placeholders = ",".join("?" for _ in ids)
        con.execute(f"delete from memory_snippet_embeddings where snippet_id in ({placeholders})", ids)
        cur = con.execute(
            f"delete from memory_snippets where profile_id=? and id in ({placeholders})",
            [profile_id] + ids,
        )
        con.commit()
        return int(cur.rowcount or 0)
    finally:
        con.close()


def list_memory_snippets(profile_id=DEFAULT_PROFILE_ID, query="", limit=200, db_path=None, config=None):
    con = _ensure_schema(db_path)
    try:
        rows = _load_memory_rows(con, profile_id, retention_days=36500)
        if query:
            ranked = [row for _score, row in _text_rank_memory_rows(query, rows, limit)]
        else:
            ranked = sorted(rows, key=lambda row: row["last_seen_at"] or "", reverse=True)[: int(limit or 200)]
        engine_id = memory_embedding_engine_id(config) if config else ""
        out = []
        for row in ranked:
            status = "not_checked"
            if engine_id:
                emb_row = _load_embedding_row(con, row["id"], engine_id)
                status = "fresh" if _embedding_is_fresh(emb_row, row["snippet_text"]) else ("stale" if emb_row else "missing")
            out.append({
                "id": row["id"],
                "note_id": row["note_id"],
                "chunk_index": row["chunk_index"],
                "snippet": row["snippet_text"],
                "source_query": row["source_query"],
                "subquery_label": row["subquery_label"],
                "last_seen_at": row["last_seen_at"],
                "hit_count": row["hit_count"],
                "citable": False,
                "embedding_status": status,
            })
        return out
    finally:
        con.close()


def retrieve_memory_snippets(query, profile_id=DEFAULT_PROFILE_ID, limit=MAX_SNAPSHOT_SNIPPETS, db_path=None, mode="text", config=None, diagnostics=None):
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    diagnostics.setdefault("mode_requested", mode)
    diagnostics.setdefault("embedding_background_scheduled", False)
    diagnostics.setdefault("embedding_background_completed", False)
    con = _ensure_schema(db_path)
    try:
        current_scope_hash = scope_hash_from_config(config) if config else ""
        engine_id = memory_embedding_engine_id(config) if config else ""
        if engine_id:
            diagnostics["embedding_background_completed"] = _tracked_background_completed(con, engine_id)
        rows = _load_memory_rows(con, profile_id, scope_hash=current_scope_hash)
        text_ranked = _text_rank_memory_rows(
            query,
            rows,
            max(int(limit or MAX_SNAPSHOT_SNIPPETS) * 4, MAX_SNAPSHOT_SNIPPETS),
            preferred_scope_hash=current_scope_hash,
        )
        diagnostics["text_hits"] = len(text_ranked)
        selected_pairs = text_ranked[: max(1, int(limit or MAX_SNAPSHOT_SNIPPETS))]
        selected_mode = "text"
        if str(mode or "text").lower() in {"auto", "auto_hybrid", "hybrid"} and config is not None and text_ranked:
            candidates = [row for _score, row in text_ranked[: max(MEMORY_LAZY_EMBED_LIMIT, int(limit or MAX_SNAPSHOT_SNIPPETS))]]
            try:
                from .engine import get_embedding_for_query
                query_embedding = get_embedding_for_query(query, config)
            except Exception:
                query_embedding = None
            missing_or_stale = []
            diagnostics["embedding_missing"] = 0
            diagnostics["embedding_stale"] = 0
            diagnostics["embedding_fresh"] = 0
            if engine_id:
                for row in candidates:
                    emb_row = _load_embedding_row(con, row["id"], engine_id)
                    if _embedding_is_fresh(emb_row, row["snippet_text"]):
                        diagnostics["embedding_fresh"] += 1
                    else:
                        missing_or_stale.append(row)
                        if emb_row:
                            diagnostics["embedding_stale"] += 1
                        else:
                            diagnostics["embedding_missing"] += 1
            scheduled = schedule_memory_snippet_embeddings(
                missing_or_stale,
                config=config,
                db_path=db_path,
                max_snippets=MEMORY_LAZY_EMBED_LIMIT,
            )
            diagnostics["embedding_background_scheduled"] = bool(scheduled.get("scheduled"))
            diagnostics["embedding_background_scheduled_count"] = int(scheduled.get("count") or 0)
            if query_embedding:
                vector_scores = {}
                for row in candidates:
                    emb_row = _load_embedding_row(con, row["id"], engine_id)
                    if not _embedding_is_fresh(emb_row, row["snippet_text"], expected_dim=len(query_embedding)):
                        continue
                    similarity = _cosine_similarity(query_embedding, _blob_to_embedding(emb_row["embedding_blob"]))
                    if similarity is not None:
                        vector_scores[row["id"]] = float(similarity)
                diagnostics["vector_hits"] = len(vector_scores)
                if vector_scores:
                    text_rr = {row["id"]: 1.0 / (rank + 1) for rank, (_score, row) in enumerate(text_ranked)}
                    fused = []
                    for row in candidates:
                        score = text_rr.get(row["id"], 0.0) + (1.0 / (1.0 + sorted(vector_scores, key=vector_scores.get, reverse=True).index(row["id"])) if row["id"] in vector_scores else 0.0)
                        fused.append((score, row))
                    fused.sort(key=lambda item: item[0], reverse=True)
                    selected_pairs = fused[: max(1, int(limit or MAX_SNAPSHOT_SNIPPETS))]
                    selected_mode = "hybrid"
        diagnostics["mode_used"] = selected_mode
        selected = [row for _score, row in selected_pairs]
        if selected:
            now = _now_iso()
            ids = [row["id"] for row in selected]
            placeholders = ",".join("?" for _ in ids)
            con.execute(
                f"update memory_snippets set hit_count=hit_count+1, last_seen_at=? where id in ({placeholders})",
                [now] + ids,
            )
            con.commit()
        return [
            {
                "id": row["id"],
                "note_id": row["note_id"],
                "chunk_index": row["chunk_index"],
                "snippet": _compact_text(row["snippet_text"], MEMORY_SNIPPET_CHARS),
                "source_query": _compact_text(row["source_query"], 120),
                "subquery_label": _compact_text(row["subquery_label"], 80),
                "citable": False,
            }
            for row in selected
        ]
    finally:
        con.close()


def _scope_from_config(config):
    ntf = (config or {}).get("note_type_filter") or {}
    scope_fields = ntf.get("scope_fields") or {}
    fields = []
    if isinstance(scope_fields, dict):
        for deck_map in scope_fields.values():
            if isinstance(deck_map, dict):
                for names in deck_map.values():
                    fields.extend(names or [])
    for names in (ntf.get("note_type_fields") or {}).values():
        fields.extend(names or [])
    return {
        "decks": list(ntf.get("enabled_decks") or [])[:10],
        "note_types": list(ntf.get("enabled_note_types") or [])[:10],
        "fields": list(dict.fromkeys(str(f) for f in fields if str(f).strip()))[:20],
        "scope_mode": ntf.get("scope_mode") or "",
    }


def _latest_prior_query(chat_history):
    for item in reversed(chat_history or []):
        if item.get("role") == "user" and item.get("mode") == "notes":
            value = _compact_text(item.get("content"), 220)
            if value:
                return value
    return ""


def _derive_followup_target(prior_query, review_context, selected_text):
    for source, limit in ((selected_text, 160), (review_context, 160), (prior_query, 120)):
        value = _compact_text(source, limit)
        if value:
            return value
    return ""


def build_memory_snapshot(
    config,
    chat_history=None,
    review_context="",
    selected_text="",
    session_memory=None,
    profile=None,
    fact_snippets=None,
):
    profile = profile or load_memory_profile()
    session_memory = session_memory or SessionMemory()
    prior_query = _latest_prior_query(chat_history or [])
    if not session_memory.resolved_followup_target:
        session_memory.resolved_followup_target = _derive_followup_target(
            prior_query,
            review_context,
            selected_text,
        )
    sc = (config or {}).get("search_config") or {}
    data = profile.safe_data()
    snapshot = MemorySnapshot(
        profile_id=profile.profile_id,
        active_scope=_scope_from_config(config),
        answer_style=str(data.get("answer_style") or ""),
        prior_query=prior_query,
        resolved_followup_target=_compact_text(session_memory.resolved_followup_target, 180),
        review_context_hint=_compact_text(review_context, 220),
        selected_text_hint=_compact_text(selected_text, 220),
        retrieval_budget={
            "max_results": sc.get("max_results"),
            "planner_mode": sc.get("agentic_planner_mode"),
            "smart_retrieval": bool(sc.get("enable_agentic_rag")),
        },
        session_summary=_compact_text(session_memory.summary, 260),
        fact_snippets=_sanitize_fact_snippets(fact_snippets),
    )
    payload = snapshot.to_dict()
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= SNAPSHOT_CHAR_LIMIT:
        return payload
    while len(text) > SNAPSHOT_CHAR_LIMIT and payload.get("fact_snippets"):
        payload["fact_snippets"] = payload["fact_snippets"][:-1]
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= SNAPSHOT_CHAR_LIMIT:
            return payload
    for key in ("selected_text_hint", "review_context_hint", "session_summary", "resolved_followup_target", "prior_query"):
        payload[key] = _compact_text(payload.get(key), max(20, len(str(payload.get(key) or "")) // 2))
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= SNAPSHOT_CHAR_LIMIT:
            return payload
    while len(text) > SNAPSHOT_CHAR_LIMIT and payload.get("active_scope", {}).get("fields"):
        payload["active_scope"]["fields"] = payload["active_scope"]["fields"][:-1]
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return payload


def _sanitize_fact_snippets(snippets):
    out = []
    for item in list(snippets or [])[:MAX_SNAPSHOT_SNIPPETS]:
        if not isinstance(item, dict):
            continue
        text = _compact_text(item.get("snippet") or item.get("snippet_text") or item.get("excerpt"), MEMORY_SNIPPET_CHARS)
        if not text:
            continue
        out.append({
            "note_id": item.get("note_id"),
            "chunk_index": int(item.get("chunk_index") or 0),
            "snippet": text,
            "source_query": _compact_text(item.get("source_query"), 120),
            "subquery_label": _compact_text(item.get("subquery_label"), 80),
            "citable": False,
        })
    return out


def update_profile_from_search(config, plan=None, result_summary=None, db_path=None):
    sc = (config or {}).get("search_config") or {}
    ntf = (config or {}).get("note_type_filter") or {}
    data = {
        "enabled_decks": list(ntf.get("enabled_decks") or [])[:50],
        "enabled_note_types": list(ntf.get("enabled_note_types") or [])[:50],
        "agentic_planner_mode": sc.get("agentic_planner_mode") or "deterministic_v1",
        "enable_agentic_rag": bool(sc.get("enable_agentic_rag", False)),
    }
    fields = []
    for names in (ntf.get("note_type_fields") or {}).values():
        fields.extend(names or [])
    scope_fields = ntf.get("scope_fields") or {}
    if isinstance(scope_fields, dict):
        for deck_map in scope_fields.values():
            if isinstance(deck_map, dict):
                for names in deck_map.values():
                    fields.extend(names or [])
    data["selected_fields"] = list(dict.fromkeys(str(f) for f in fields if str(f).strip()))[:80]
    if sc.get("answer_style"):
        data["answer_style"] = str(sc.get("answer_style"))[:80]

    profile = load_memory_profile(db_path=db_path)
    merged = profile.safe_data()
    merged.update({k: data[k] for k in PROFILE_ALLOWED_KEYS if k in data})
    profile.data = merged
    return save_memory_profile(profile, db_path=db_path)
