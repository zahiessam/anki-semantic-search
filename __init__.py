# Anki Semantic Search - Entry point, hooks, menu, dialogs

# Colorama/transformers compat (call before sentence_transformers import)
from .core.compat import _patch_colorama_early, _ensure_stderr_patched

# Embedding and API logic (no Qt)
from .core.engine import (
    get_embedding_via_voyage,
    get_embeddings_via_voyage_batch,
    get_embedding_via_openai,
    get_embeddings_via_openai_batch,
    get_embedding_via_cohere,
    get_embeddings_via_cohere_batch,
    get_embedding_via_ollama,
    get_embeddings_via_ollama_batch,
    get_embedding_for_query,
    get_embeddings_batch,
    get_ollama_models,
    estimate_tokens,
    OLLAMA_EMBED_CHUNK_SIZE,
)

# Utils package (config, paths, logging)
from .utils import (
    log_debug,
    get_addon_name,
    get_config_file_path,
    load_config,
    save_config,
    get_config_value,
    VOYAGE_EMBEDDING_MODELS,
    get_embeddings_storage_path,
    get_embeddings_storage_path_for_read,
    get_embeddings_db_path,
    get_checkpoint_path,
    EmbeddingsTabMessages,
    ErrorAndEngineMessages,
    format_partial_failure_progress,
    format_partial_failure_completion,
    format_dimension_mismatch_hint,
)

def get_safe_config(config=None):
    """Return a copy of config with any key containing 'api_key' replaced by '********' for safe logging."""
    if config is None:
        try:
            config = load_config()
        except Exception:
            return {}
    if not isinstance(config, dict):
        return config
    safe = {}
    for k, v in config.items():
        if isinstance(k, str) and 'api_key' in k.lower():
            safe[k] = '********'
        elif isinstance(v, dict):
            safe[k] = get_safe_config(v)
        else:
            safe[k] = v
    return safe


from aqt import mw, gui_hooks, dialogs
from aqt.qt import *
from aqt.utils import showInfo, tooltip
from anki.hooks import addHook
import json
import urllib.request
import urllib.error
import os
import datetime
import aqt
import json as _json_for_agent_logs
import time as _time_for_agent_logs

def get_all_note_types():
    """Get sorted list of all note type names in the collection."""
    try:
        if not mw or not mw.col:
            return []
        return sorted([m['name'] for m in mw.col.models.all()])
    except Exception as e:
        log_debug(f"get_all_note_types error: {e}")
        return []


# region agent log
def _agent_debug_log(run_id, hypothesis_id, location, message, data=None):
    """Lightweight debug logger for agent-driven investigations (writes NDJSON)."""
    try:
        entry = {
            "id": f"log_{int(_time_for_agent_logs.time() * 1000)}",
            "timestamp": int(_time_for_agent_logs.time() * 1000),
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
        }
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cursor", "debug.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(_json_for_agent_logs.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Never break the add-on if logging fails
        pass


def _session_debug_log(hypothesis_id, location, message, data=None):
    """Write NDJSON to session log file for debug (path: debug-85902e.log)."""
    try:
        entry = {
            "sessionId": "85902e",
            "timestamp": int(_time_for_agent_logs.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
        }
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug-85902e.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(_json_for_agent_logs.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
# endregion


def get_notes_count_per_model():
    """Return dict mapping model name -> note count."""
    try:
        if not mw or not mw.col:
            return {}
        rows = mw.col.db.execute("SELECT mid, COUNT(*) FROM notes GROUP BY mid")
        mid_to_count = {r[0]: r[1] for r in rows}
        out = {}
        for m in mw.col.models.all():
            out[m['name']] = mid_to_count.get(m['id'], 0)
        return out
    except Exception as e:
        log_debug(f"get_notes_count_per_model error: {e}")
        return {}


def get_models_with_fields():
    """Return list of (model_name, note_count, [field_names])."""
    try:
        if not mw or not mw.col:
            return []
        counts = get_notes_count_per_model()
        return [(m['name'], counts.get(m['name'], 0), [f['name'] for f in m['flds']]) for m in mw.col.models.all()]
    except Exception as e:
        log_debug(f"get_models_with_fields error: {e}")
        return []


def get_deck_names():
    """Return sorted list of deck names (excluding filtered)."""
    try:
        if not mw or not mw.col:
            return []
        names = []
        for d in mw.col.decks.all():
            n = d.get('name', '')
            if n and not d.get('dyn', False):  # exclude dynamic (filtered) decks
                names.append(n)
        return sorted(names)
    except Exception as e:
        log_debug(f"get_deck_names error: {e}")
        return []


def get_notes_count_per_deck():
    """Return dict mapping deck name -> note count. Optimized for performance."""
    try:
        if not mw or not mw.col:
            return {}
        
        out = {}
        deck_names = get_deck_names()
        
        # Optimized: Process decks in smaller batches and yield control periodically
        # This prevents UI freezing on large collections
        batch_size = 5  # Smaller batches to keep UI responsive
        for i, deck_name in enumerate(deck_names):
            try:
                # Use find_notes which is optimized in Anki
                note_ids = mw.col.find_notes(f'deck:"{deck_name}"')
                out[deck_name] = len(note_ids)
                
                # Yield control every batch to keep UI responsive
                if (i + 1) % batch_size == 0:
                    from aqt.qt import QApplication
                    QApplication.processEvents()
            except:
                out[deck_name] = 0
        
        return out
    except Exception as e:
        log_debug(f"get_notes_count_per_deck error: {e}")
        return {}


def _build_deck_query(enabled_decks):
    """Build find_notes deck query. None or [] -> '' (all)."""
    if not enabled_decks or len(enabled_decks) == 0:
        return ""
    # Quote names that may have spaces/special chars
    parts = []
    for d in enabled_decks:
        if " " in d or ":" in d or "\\" in d:
            parts.append(f'deck:"{d}"')
        else:
            parts.append(f"deck:{d}")
    return " or ".join(parts)


def _strip_html_plain(text):
    """Module-level strip HTML tags (for use in background thread)."""
    import re
    if not text:
        return ""
    return re.sub(re.compile('<.*?>'), '', text)


def get_models_with_fields_col(col):
    """Return list of (model_name, note_count, [field_names]) using col. Safe for QueryOp."""
    try:
        if not col:
            return []
        rows = col.db.execute("SELECT mid, COUNT(*) FROM notes GROUP BY mid")
        mid_to_count = {r[0]: r[1] for r in rows}
        return [(m['name'], mid_to_count.get(m['id'], 0), [f['name'] for f in m['flds']]) for m in col.models.all()]
    except Exception as e:
        log_debug(f"get_models_with_fields_col error: {e}")
        return []


def get_notes_content_with_col(col, config):
    """
    Load note content for search using col (no mw). Safe to run in QueryOp.
    Returns (notes_data, fields_description, cache_key).
    """
    import hashlib
    notes_data = []
    ntf = config.get('note_type_filter', {})
    if ntf.get('fields_to_search') and not ntf.get('note_type_fields'):
        global_flds = set(f.lower() for f in ntf['fields_to_search'])
        ntf = dict(ntf)
        ntf['note_type_fields'] = {}
        for model_name, _c, field_names in get_models_with_fields_col(col):
            ntf['note_type_fields'][model_name] = [f for f in field_names if f.lower() in global_flds]
    legacy_fields = None
    if not ntf:
        enabled_set = None
        search_all = False
        ntf_fields = {}
        use_first = False
        legacy_fields = {'text', 'extra'}
        fields_description = "Text & Extra"
    else:
        enabled = ntf.get('enabled_note_types')
        enabled_set = set(enabled) if (enabled and len(enabled) > 0) else None
        search_all = bool(ntf.get('search_all_fields', False))
        ntf_fields = ntf.get('note_type_fields') or {}
        use_first = bool(ntf.get('use_first_field_fallback', True))
        fields_description = "all fields" if search_all else "per-type"
    deck_q = _build_deck_query(ntf.get('enabled_decks') if ntf else None)
    note_ids = col.find_notes(deck_q) if deck_q else col.find_notes("")
    total_notes = len(note_ids)
    cache_key = (deck_q or '', frozenset(enabled_set) if enabled_set is not None else None, search_all, tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in (ntf_fields or {}).items())), total_notes)
    log_debug(f"get_notes_content_with_col: {total_notes} notes")
    for idx, nid in enumerate(note_ids):
        try:
            note = col.get_note(nid)
            note_type = note.note_type()
            model_name = note_type['name']
            if enabled_set is not None and model_name not in enabled_set:
                continue
            flds = note_type['flds']
            if search_all:
                indices = [i for i in range(len(note.fields)) if i < len(note.fields)]
            else:
                if legacy_fields is not None:
                    wanted = legacy_fields
                else:
                    wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))
                    if not wanted and use_first and flds:
                        wanted = {flds[0]['name'].lower()}
                indices = [i for i, f in enumerate(flds) if i < len(note.fields) and f['name'].lower() in wanted]
            if not indices:
                continue
            content_parts = []
            for i in indices:
                if i < len(note.fields) and note.fields[i].strip():
                    content_parts.append(note.fields[i].strip())
            if not content_parts:
                continue
            content = " | ".join(content_parts)
            content = _strip_html_plain(content)
            if not content.strip():
                continue
            content_parts_raw = [note.fields[i] for i in indices]
            content_for_hash = " ".join(content_parts_raw)
            CHUNK_TARGET = 500
            chunks = _semantic_chunk_text(content, CHUNK_TARGET)
            if len(chunks) <= 1:
                content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()
                notes_data.append({'id': nid, 'content': content, 'content_hash': content_hash, 'model': model_name, 'display_content': content})
            else:
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                    notes_data.append({
                        'id': nid, 'content': chunk, 'content_hash': chunk_hash, 'model': model_name,
                        'display_content': chunk, 'chunk_index': chunk_idx, '_full_content': content,
                    })
        except Exception as e:
            log_debug(f"Error processing note {nid}: {e}")
            continue
    log_debug(f"get_notes_content_with_col: loaded {len(notes_data)} items")
    return (notes_data, fields_description, cache_key)


def count_notes_matching_config(ntf):
    """Return number of notes that would be searched with the given note_type_filter."""
    try:
        if not mw or not mw.col:
            return 0
        deck_q = _build_deck_query(ntf.get('enabled_decks'))
        note_ids = mw.col.find_notes(deck_q) if deck_q else mw.col.find_notes("")
        enabled = ntf.get('enabled_note_types')
        enabled_set = set(enabled) if (enabled and len(enabled) > 0) else None
        search_all = bool(ntf.get('search_all_fields', False))
        ntf_fields = ntf.get('note_type_fields') or {}
        use_first = bool(ntf.get('use_first_field_fallback', True))
        n = 0
        for nid in note_ids:
            try:
                note = mw.col.get_note(nid)
                m = note.note_type()
                model_name = m['name']
                if enabled_set and model_name not in enabled_set:
                    continue
                flds = m['flds']
                if search_all:
                    has_content = any((i < len(note.fields) and note.fields[i].strip()) for i in range(len(flds)))
                else:
                    wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))
                    if not wanted and use_first and flds:
                        wanted = {flds[0]['name'].lower()}
                    has_content = any((i < len(note.fields) and note.fields[i].strip()) for i, f in enumerate(flds) if f['name'].lower() in wanted)
                if has_content:
                    n += 1
            except Exception:
                continue
        return n
    except Exception as e:
        log_debug(f"count_notes_matching_config error: {e}")
        return 0


def get_search_history_path():
    """Get path to search history file"""
    addon_dir = os.path.dirname(__file__)
    return os.path.join(addon_dir, "search_history.json")

def save_search_history(query, answer, relevant_note_ids, scored_notes, context_note_ids=None):
    """Save search query and results to history. context_note_ids = order of notes in AI context (Note 1, 2, ...) for clickable refs."""
    try:
        history_path = get_search_history_path()
        history = {}
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = {}
        
        # Normalize query for lookup (lowercase, strip)
        query_key = query.lower().strip()
        
        # Store search result (limit to last 100 searches)
        if 'searches' not in history:
            history['searches'] = []
        
        # Remove duplicate if exists
        history['searches'] = [s for s in history['searches'] if s.get('query_key') != query_key]
        
        ctx_ids = context_note_ids if context_note_ids is not None else []
        history['searches'].append({
            'query': query,
            'query_key': query_key,
            'answer': answer,
            'relevant_note_ids': relevant_note_ids,
            'context_note_ids': ctx_ids,
            'scored_notes': [(score, {'id': note['id'], 'content': note['content'][:200]}) for score, note in scored_notes[:50]],  # Store top 50
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Keep only last 100 searches
        history['searches'] = history['searches'][-100:]
        
        # Use atomic write to prevent corruption
        temp_path = history_path + ".tmp"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
            # Atomic rename
            if os.path.exists(history_path):
                os.replace(temp_path, history_path)
            else:
                os.rename(temp_path, history_path)
        except Exception as write_err:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise write_err
        
        log_debug(f"Search history saved for query: {query[:50]}")
        return True
    except Exception as e:
        log_debug(f"Error saving search history: {e}")
        return False

def load_search_history(query):
    """Load search result from history if exists"""
    try:
        history_path = get_search_history_path()
        if not os.path.exists(history_path):
            return None
        
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        query_key = query.lower().strip()
        
        # Find matching search
        if 'searches' in history:
            for search in reversed(history['searches']):  # Check most recent first
                if search.get('query_key') == query_key:
                    log_debug(f"Found search history for query: {query[:50]}")
                    return search
        
        return None
    except Exception as e:
        log_debug(f"Error loading search history: {e}")
        return None


def get_search_history_queries():
    """Return list of previous search queries (most recent first) for suggestions and dropdown."""
    try:
        history_path = get_search_history_path()
        if not os.path.exists(history_path):
            return []
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        searches = history.get('searches', [])
        queries = [s.get('query', '').strip() for s in reversed(searches) if s.get('query', '').strip()]
        return queries[:50]  # Limit to 50
    except Exception:
        return []


def clear_search_history():
    """Clear all search history. Uses atomic write to prevent corruption."""
    try:
        history_path = get_search_history_path()
        history = {'searches': []}
        temp_path = history_path + ".tmp"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
            if os.path.exists(history_path):
                os.replace(temp_path, history_path)
            else:
                os.rename(temp_path, history_path)
        except Exception as write_err:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise write_err
        log_debug("Search history cleared")
        return True
    except Exception as e:
        log_debug(f"Error clearing search history: {e}")
        return False


def recover_valid_entries_from_corrupted_json(file_path):
    """
    Attempt to recover valid JSON entries from a corrupted file.
    Uses incremental parsing to extract valid entries before corruption point.
    Returns dict of recovered entries and number of recovered entries.
    """
    recovered = {}
    recovered_count = 0
    
    try:
        import re
        # Read file in chunks to handle large files
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Strategy 1: Try to find valid JSON objects before corruption
        # Look for patterns like "key": { ... } and try to extract them
        # This is a best-effort recovery
        
        # Strategy 2: Try incremental JSON parsing
        # Find the last valid complete entry before corruption
        try:
            # Try to parse as much as possible
            # Use a streaming JSON parser approach
            bracket_count = 0
            brace_count = 0
            in_string = False
            escape_next = False
            last_valid_pos = 0
            current_key = None
            current_entry = None
            entry_buffer = ""
            
            i = 0
            while i < len(content):
                char = content[i]
                
                if escape_next:
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\':
                    escape_next = True
                    i += 1
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    i += 1
                    continue
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                        if brace_count == 1:
                            # Start of a new entry
                            entry_buffer = char
                    elif char == '}':
                        brace_count -= 1
                        entry_buffer += char
                        if brace_count == 0:
                            # Complete entry found, try to parse it
                            try:
                                entry = json.loads(entry_buffer)
                                if isinstance(entry, dict) and 'note_id' in entry and 'embedding' in entry:
                                    key = f"{entry.get('note_id')}_{entry.get('content_hash', '')}"
                                    recovered[key] = entry
                                    recovered_count += 1
                                    last_valid_pos = i
                            except:
                                pass
                            entry_buffer = ""
                    elif char in ['[', ']']:
                        pass  # Ignore array brackets for now
                    else:
                        if entry_buffer:
                            entry_buffer += char
                
                i += 1
                
                # Safety limit - don't process more than 100MB
                if i > 100 * 1024 * 1024:
                    break
                    
        except Exception as e:
            log_debug(f"Error in incremental parsing: {e}")
        
        # Strategy 3: Try to extract entries using regex as fallback
        if recovered_count == 0:
            try:
                # Look for patterns like "1234567890_abc123": { ... }
                pattern = r'"(\d+_[^"]+)":\s*\{[^}]*"note_id":\s*(\d+)[^}]*"embedding":\s*\[([^\]]+)\][^}]*\}'
                matches = re.finditer(pattern, content[:10*1024*1024])  # Limit to first 10MB
                for match in matches:
                    try:
                        key = match.group(1)
                        note_id = match.group(2)
                        embedding_str = match.group(3)
                        # Try to parse embedding array
                        embedding_list = json.loads(f"[{embedding_str}]")
                        if isinstance(embedding_list, list) and len(embedding_list) > 0:
                            entry = {
                                'note_id': int(note_id),
                                'content_hash': key.split('_', 1)[1] if '_' in key else '',
                                'embedding': embedding_list
                            }
                            recovered[key] = entry
                            recovered_count += 1
                    except:
                        continue
            except Exception as e:
                log_debug(f"Error in regex recovery: {e}")
        
        if recovered_count > 0:
            log_debug(f"Recovered {recovered_count} valid entries from corrupted file")
        
    except Exception as e:
        log_debug(f"Error in corruption recovery: {e}")
    
    return recovered, recovered_count

# Global cache for batch saving embeddings
_embedding_batch_cache = {}
_embedding_batch_lock = False
# Maximum cache size to prevent memory issues (default: 1000 embeddings)
MAX_EMBEDDING_CACHE_SIZE = 1000

# Global cache for loaded embeddings file (to avoid repeated file reads)
_embeddings_file_cache = None
_embeddings_file_cache_path = None
_embeddings_file_cache_time = 0
# Track corrupted files to avoid repeated error logging
_corrupted_files = set()


def get_embedding_engine_id(config=None):
    """Return a stable string identifying the current embedding engine+model for storage keying.
    E.g. 'ollama:nomic-embed-text', 'voyage:voyage-3.5-lite'. Used so one DB can store multiple engines."""
    if config is None:
        config = load_config()
    sc = config.get("search_config") or {}
    engine = (sc.get("embedding_engine") or "voyage").strip().lower()
    if engine == "ollama":
        model = (sc.get("ollama_embed_model") or "nomic-embed-text").strip()
    elif engine == "voyage":
        model = (sc.get("voyage_embedding_model") or "voyage-3.5-lite").strip()
    elif engine == "openai":
        model = (sc.get("openai_embedding_model") or "text-embedding-3-small").strip()
    elif engine == "cohere":
        model = (sc.get("cohere_embedding_model") or "embed-english-v3.0").strip()
    else:
        model = "default"
    return f"{engine}:{model}"


# --- SQLite embeddings DB: only loads what it needs (no 600MB into RAM) ---
def _embeddings_db_ensure_table(conn):
    """Create embeddings table with engine_id so one DB can store multiple engines (Ollama, Voyage, etc.)."""
    # Check if old table exists without engine_id and migrate
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
    )
    if cur.fetchone():
        cur = conn.execute("PRAGMA table_info(embeddings)")
        columns = [row[1] for row in cur.fetchall()]
        if "engine_id" not in columns:
            # Migrate: copy to new table with engine_id='legacy', then replace
            conn.execute("""
                CREATE TABLE embeddings_new (
                    engine_id TEXT NOT NULL,
                    note_id INTEGER NOT NULL,
                    chunk_index INTEGER,
                    content_hash TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    timestamp TEXT,
                    PRIMARY KEY (engine_id, note_id, content_hash)
                )
            """)
            conn.execute(
                "INSERT INTO embeddings_new (engine_id, note_id, chunk_index, content_hash, embedding_blob, timestamp) "
                "SELECT ?, note_id, chunk_index, content_hash, embedding_blob, timestamp FROM embeddings",
                ("legacy",),
            )
            conn.execute("DROP TABLE embeddings")
            conn.execute("ALTER TABLE embeddings_new RENAME TO embeddings")
            log_debug("Migrated embeddings table to multi-engine schema (existing rows as engine_id='legacy')")
    else:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                engine_id TEXT NOT NULL,
                note_id INTEGER NOT NULL,
                chunk_index INTEGER,
                content_hash TEXT NOT NULL,
                embedding_blob BLOB NOT NULL,
                timestamp TEXT,
                PRIMARY KEY (engine_id, note_id, content_hash)
            )
        """)
    conn.commit()

def _embedding_to_blob(embedding):
    """Serialize embedding (list or ndarray) to BLOB (float32)."""
    import numpy as np
    arr = np.array(embedding, dtype=np.float32)
    return arr.tobytes()

def _blob_to_embedding(blob):
    """Deserialize BLOB to numpy float32 array."""
    import numpy as np
    return np.frombuffer(blob, dtype=np.float32)

def save_embedding(note_id, content_hash, embedding, batch_mode=True, storage_path=None, engine_id=None):
    """Save embedding to SQLite DB (batch mode = in-memory cache until flush). storage_path = db path for thread-safe use.
    engine_id: e.g. 'ollama:nomic-embed-text'; if None, derived from current config so one DB can store multiple engines."""
    try:
        import numpy as np
        if engine_id is None:
            engine_id = get_embedding_engine_id()
        db_path = storage_path if storage_path is not None else get_embeddings_db_path()
        key = f"{engine_id}_{note_id}_{content_hash}"
        emb_arr = np.array(embedding, dtype=np.float32) if not isinstance(embedding, np.ndarray) else embedding.astype(np.float32)
        embedding_data = {
            'engine_id': engine_id,
            'note_id': note_id,
            'content_hash': content_hash,
            'embedding': emb_arr,
            'timestamp': datetime.datetime.now().isoformat()
        }
        if batch_mode:
            global _embedding_batch_cache, MAX_EMBEDDING_CACHE_SIZE
            if len(_embedding_batch_cache) >= MAX_EMBEDDING_CACHE_SIZE:
                flush_embedding_batch(storage_path=storage_path)
            _embedding_batch_cache[key] = embedding_data
            return True
        import sqlite3
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            _embeddings_db_ensure_table(conn)
            blob = _embedding_to_blob(emb_arr)
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (engine_id, note_id, chunk_index, content_hash, embedding_blob, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (engine_id, note_id, None, content_hash, blob, embedding_data['timestamp'])
            )
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception as e:
        log_debug(f"Error saving embedding: {e}")
        return False

def flush_embedding_batch(storage_path=None):
    """Flush accumulated embeddings to SQLite DB. storage_path = db path for thread-safe use."""
    global _embedding_batch_cache, _embeddings_file_cache, _embeddings_file_cache_path, _embeddings_file_cache_time
    if not _embedding_batch_cache:
        return True
    try:
        db_path = storage_path if storage_path is not None else get_embeddings_db_path()
        import sqlite3
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            _embeddings_db_ensure_table(conn)
            rows = []
            for key, data in _embedding_batch_cache.items():
                engine_id = data.get('engine_id', 'legacy')
                note_id, content_hash = data['note_id'], data['content_hash']
                blob = _embedding_to_blob(data['embedding'])
                ts = data.get('timestamp') or datetime.datetime.now().isoformat()
                rows.append((engine_id, note_id, None, content_hash, blob, ts))
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (engine_id, note_id, chunk_index, content_hash, embedding_blob, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                rows
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
    except Exception as e:
        log_debug(f"Error flushing embedding batch: {e}")
        return False

def migrate_embeddings_json_to_db():
    """One-time migration: copy all embeddings from the old JSON file into the SQLite DB. No re-embedding needed.
    Returns (migrated_count, None) on success or (0, error_message) on failure."""
    try:
        json_path = get_embeddings_storage_path_for_read()
        if not os.path.exists(json_path) or not json_path.endswith('.json'):
            return 0, "No legacy JSON embeddings file found."
        db_path = get_embeddings_db_path()
        import sqlite3
        file_size_mb = os.path.getsize(json_path) / (1024 * 1024)
        if file_size_mb > 400:
            log_debug(f"Migration: large JSON ({file_size_mb:.0f}MB) — this may use significant RAM temporarily.")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return 0, "Invalid JSON format (expected object)."
        conn = sqlite3.connect(db_path, timeout=60)
        try:
            _embeddings_db_ensure_table(conn)
            batch = []
            batch_size = 500
            migrated = 0
            for key, entry in data.items():
                if not isinstance(entry, dict) or 'embedding' not in entry:
                    continue
                try:
                    note_id = int(entry.get('note_id', 0))
                    content_hash = str(entry.get('content_hash', ''))
                    emb = entry['embedding']
                    if not emb:
                        continue
                    blob = _embedding_to_blob(emb)
                    ts = entry.get('timestamp') or datetime.datetime.now().isoformat()
                    batch.append(('legacy', note_id, None, content_hash, blob, ts))
                    if len(batch) >= batch_size:
                        conn.executemany(
                            "INSERT OR REPLACE INTO embeddings (engine_id, note_id, chunk_index, content_hash, embedding_blob, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                            batch
                        )
                        conn.commit()
                        migrated += len(batch)
                        batch = []
                except (TypeError, ValueError) as e:
                    log_debug(f"Migration skip entry {key}: {e}")
                    continue
            if batch:
                conn.executemany(
                    "INSERT OR REPLACE INTO embeddings (engine_id, note_id, chunk_index, content_hash, embedding_blob, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    batch
                )
                conn.commit()
                migrated += len(batch)
            return migrated, None
        finally:
            conn.close()
    except json.JSONDecodeError as e:
        return 0, f"JSON file is corrupted or invalid: {e}"
    except Exception as e:
        log_debug(f"Migration error: {e}")
        return 0, str(e)

def load_embedding(note_id, content_hash, db_path=None, engine_id=None):
    """Load embedding from SQLite DB for the given engine (single row). engine_id: e.g. 'ollama:nomic-embed-text'; if None, from config.
    db_path: when provided (e.g. from main thread), use it so background threads don't call get_embeddings_db_path()."""
    try:
        import numpy as np
        if engine_id is None:
            engine_id = get_embedding_engine_id()
        if db_path is None:
            db_path = get_embeddings_db_path()
        if os.path.exists(db_path):
            import sqlite3
            conn = sqlite3.connect(db_path, timeout=10)
            try:
                row = conn.execute(
                    "SELECT embedding_blob FROM embeddings WHERE engine_id = ? AND note_id = ? AND content_hash = ?",
                    (engine_id, note_id, content_hash)
                ).fetchone()
                if row:
                    return _blob_to_embedding(row[0])
                row = conn.execute(
                    "SELECT embedding_blob FROM embeddings WHERE engine_id = ? AND note_id = ? LIMIT 1",
                    (engine_id, note_id)
                ).fetchone()
                if row:
                    return _blob_to_embedding(row[0])
            finally:
                conn.close()
        return _load_embedding_from_json_legacy(note_id, content_hash)
    except Exception as e:
        if "Expecting" not in str(e) and "delimiter" not in str(e):
            log_debug(f"Error loading embedding: {e}")
        return None

def _load_embedding_from_json_legacy(note_id, content_hash):
    """Load embedding from legacy JSON file (fallback when not in DB). Avoids loading full file into RAM for each key by using cache."""
    global _embeddings_file_cache, _embeddings_file_cache_path, _embeddings_file_cache_time, _corrupted_files
    try:
        import numpy as np
        storage_path = get_embeddings_storage_path_for_read()
        if not os.path.exists(storage_path):
            return None
        if storage_path in _corrupted_files:
            return None
        file_size = os.path.getsize(storage_path)
        if file_size > 50 * 1024 * 1024:
            if storage_path not in _corrupted_files:
                log_debug(f"Warning: Embeddings file is very large ({file_size / (1024*1024):.1f}MB), may cause performance issues")
        file_mtime = os.path.getmtime(storage_path)
        if (_embeddings_file_cache is not None and 
            _embeddings_file_cache_path == storage_path and 
            _embeddings_file_cache_time == file_mtime):
            embeddings_data = _embeddings_file_cache
        else:
            # Load file with error recovery for corrupted JSON
            try:
                with open(storage_path, 'r', encoding='utf-8') as f:
                    embeddings_data = json.load(f)
                # Cache the loaded data
                _embeddings_file_cache = embeddings_data
                _embeddings_file_cache_path = storage_path
                _embeddings_file_cache_time = file_mtime
                # Remove from corrupted list if it was there before
                _corrupted_files.discard(storage_path)
            except json.JSONDecodeError as json_err:
                # Try to recover valid entries from corrupted file
                if storage_path not in _corrupted_files:
                    _corrupted_files.add(storage_path)
                    log_debug(f"JSON decode error in embeddings file - file may be corrupted. Error: {str(json_err)[:200]}")
                    
                    # Try to recover valid entries
                    try:
                        recovered_entries, recovered_count = recover_valid_entries_from_corrupted_json(storage_path)
                        
                        if recovered_count > 0:
                            # Save recovered entries to a new file
                            recovered_path = storage_path + ".recovered"
                            try:
                                with open(recovered_path, 'w', encoding='utf-8') as f:
                                    json.dump(recovered_entries, f, indent=2)
                                log_debug(f"Saved {recovered_count} recovered entries to {recovered_path}")
                                
                                # Try to merge recovered entries back into main file
                                # Use atomic write to prevent corruption
                                temp_path = storage_path + ".tmp"
                                try:
                                    with open(temp_path, 'w', encoding='utf-8') as f:
                                        json.dump(recovered_entries, f, indent=2)
                                    # Atomic rename
                                    if os.path.exists(storage_path):
                                        backup_path = storage_path + ".corrupted_backup"
                                        if not os.path.exists(backup_path):
                                            import shutil
                                            shutil.copy2(storage_path, backup_path)
                                            log_debug(f"Created backup of corrupted file: {backup_path}")
                                    # Replace corrupted file with recovered data
                                    if os.path.exists(temp_path):
                                        if os.path.exists(storage_path):
                                            os.replace(temp_path, storage_path)
                                        else:
                                            os.rename(temp_path, storage_path)
                                        log_debug(f"Replaced corrupted file with {recovered_count} recovered entries")
                                        # Remove from corrupted list so we can use it
                                        _corrupted_files.discard(storage_path)
                                        # Update cache
                                        _embeddings_file_cache = recovered_entries
                                        _embeddings_file_cache_path = storage_path
                                        _embeddings_file_cache_time = os.path.getmtime(storage_path)
                                except Exception as merge_err:
                                    log_debug(f"Failed to merge recovered entries: {merge_err}")
                                    # Keep recovered file for manual inspection
                            except Exception as save_err:
                                log_debug(f"Failed to save recovered entries: {save_err}")
                        else:
                            # No entries recovered, create backup
                            backup_path = storage_path + ".corrupted_backup"
                            if not os.path.exists(backup_path):
                                import shutil
                                shutil.copy2(storage_path, backup_path)
                                log_debug(f"Created backup of corrupted file: {backup_path}")
                            log_debug(f"Embeddings file marked as corrupted. Will skip loading until file is fixed or regenerated.")
                    except Exception as recovery_err:
                        log_debug(f"Failed to recover entries: {recovery_err}")
                        # Create backup as fallback
                        try:
                            backup_path = storage_path + ".corrupted_backup"
                            if not os.path.exists(backup_path):
                                import shutil
                                shutil.copy2(storage_path, backup_path)
                                log_debug(f"Created backup of corrupted file: {backup_path}")
                        except Exception as backup_err:
                            log_debug(f"Failed to create backup: {backup_err}")
                
                # If file is still marked as corrupted, return None
                if storage_path in _corrupted_files:
                    return None
                
                # If we recovered entries, try to load from cache (exact key then note_id prefix)
                if _embeddings_file_cache is not None:
                    key = f"{note_id}_{content_hash}"
                    if key in _embeddings_file_cache:
                        embedding_list = _embeddings_file_cache[key]['embedding']
                        return np.array(embedding_list)
                    prefix = f"{note_id}_"
                    for k, v in _embeddings_file_cache.items():
                        if isinstance(k, str) and k.startswith(prefix) and isinstance(v, dict) and 'embedding' in v:
                            return np.array(v['embedding'])
                
                return None
            except Exception as load_err:
                # Only log unexpected errors once
                if storage_path not in _corrupted_files:
                    log_debug(f"Error loading embeddings file: {load_err}")
                    _corrupted_files.add(storage_path)
                return None
        
        key = f"{note_id}_{content_hash}"
        if key in embeddings_data:
            embedding_list = embeddings_data[key]['embedding']
            return np.array(embedding_list)
        # Fallback: old cache or different content hash (e.g. fields changed). Use any entry for this note_id.
        prefix = f"{note_id}_"
        for k, v in embeddings_data.items():
            if isinstance(k, str) and k.startswith(prefix) and isinstance(v, dict) and 'embedding' in v:
                embedding_list = v['embedding']
                return np.array(embedding_list)
        return None
    except Exception as e:
        # Only log if it's not a common/expected error
        if "Expecting" not in str(e) and "delimiter" not in str(e):
            log_debug(f"Error loading embedding: {e}")
        return None

def save_checkpoint(processed_note_ids, total_notes, errors=0, engine_id=None):
    """Save checkpoint with list of processed note IDs using atomic write. engine_id: so resume only for same engine."""
    try:
        if engine_id is None:
            engine_id = get_embedding_engine_id()
        checkpoint_path = get_checkpoint_path()
        checkpoint_data = {
            'processed_note_ids': processed_note_ids,
            'total_notes': total_notes,
            'processed_count': len(processed_note_ids),
            'errors': errors,
            'engine_id': engine_id,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Use atomic write to prevent corruption
        temp_path = checkpoint_path + ".tmp"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            # Atomic rename
            if os.path.exists(checkpoint_path):
                os.replace(temp_path, checkpoint_path)
            else:
                os.rename(temp_path, checkpoint_path)
        except Exception as write_err:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise write_err
        
        log_debug(f"Saved checkpoint: {len(processed_note_ids)}/{total_notes} notes processed")
        return True
    except Exception as e:
        log_debug(f"Error saving checkpoint: {e}")
        return False

def load_checkpoint():
    """Load checkpoint with processed note IDs"""
    try:
        checkpoint_path = get_checkpoint_path()
        
        if not os.path.exists(checkpoint_path):
            return None
        
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        return checkpoint_data
    except Exception as e:
        log_debug(f"Error loading checkpoint: {e}")
        return None

def clear_checkpoint():
    """Clear checkpoint file"""
    try:
        checkpoint_path = get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        return True
    except Exception as e:
        log_debug(f"Error clearing checkpoint: {e}")
        return False


def _get_notes_needing_embeddings_with_col(col, config=None, limit=200):
    """Return list of (nid, content_hash, content) for notes that don't have an embedding for the current engine. Call from main thread only (e.g. inside QueryOp)."""
    import hashlib
    import re
    config = config or load_config()
    ntf = config.get('note_type_filter', {})
    if not ntf:
        return []
    deck_q = _build_deck_query(ntf.get('enabled_decks'))
    note_ids = col.find_notes(deck_q) if deck_q else col.find_notes("")
    enabled = ntf.get('enabled_note_types')
    enabled_set = set(enabled) if (enabled and len(enabled) > 0) else None
    search_all = bool(ntf.get('search_all_fields', False))
    ntf_fields = ntf.get('note_type_fields') or {}
    use_first = bool(ntf.get('use_first_field_fallback', True))
    out = []
    for nid in note_ids:
        if len(out) >= limit:
            break
        try:
            note = col.get_note(nid)
            model_name = note.note_type()['name']
            if enabled_set is not None and model_name not in enabled_set:
                continue
            flds = note.note_type()['flds']
            if search_all:
                indices = [i for i in range(len(note.fields)) if i < len(note.fields)]
            else:
                wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))
                if not wanted and use_first and flds:
                    wanted = {flds[0]['name'].lower()}
                indices = [i for i, f in enumerate(flds) if i < len(note.fields) and f['name'].lower() in wanted]
            if not indices:
                continue
            content_parts_raw = [note.fields[i] for i in indices]
            content_for_hash = " ".join(content_parts_raw)
            if not content_for_hash.strip():
                continue
            content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()
            eid = get_embedding_engine_id(config)
            if load_embedding(nid, content_hash, engine_id=eid) is not None:
                continue
            content = " | ".join(p.strip() for p in content_parts_raw)
            content = _strip_html_plain(content)
            if not content.strip():
                continue
            out.append((nid, content_hash, content))
        except Exception as e:
            log_debug(f"Background indexer skip note {nid}: {e}")
    return out


def _run_indexer_batch(todo, config, storage_path):
    """Run embedding API + save for a pre-fetched todo list. No collection access — safe to call from a worker thread.
    On batch failure, retries with smaller batch_size (16, 8, 4) to avoid Anki 'Not Responding' freeze."""
    if not todo or not config or not storage_path:
        return
    try:
        sc = config.get('search_config') or {}
        engine = sc.get('embedding_engine') or 'voyage'
        if engine != 'ollama':
            import os
            if engine == 'voyage':
                key = (sc.get('voyage_api_key') or '').strip() or os.environ.get('VOYAGE_API_KEY', '')
            elif engine == 'openai':
                key = (sc.get('openai_embedding_api_key') or '').strip() or os.environ.get('OPENAI_API_KEY', '')
            elif engine == 'cohere':
                key = (sc.get('cohere_api_key') or '').strip() or os.environ.get('COHERE_API_KEY', '')
            else:
                key = (config.get('api_key') or '').strip()
            if not key:
                return
        batch_size = min(32, len(todo))
        i = 0
        while i < len(todo):
            batch = todo[i:i + batch_size]
            texts = [item[2] for item in batch]
            try:
                embeddings_list = get_embeddings_batch(texts, input_type="document", config=config)
                if embeddings_list and len(embeddings_list) == len(batch):
                    import numpy as np
                    eid = get_embedding_engine_id(config)
                    for (nid, ch, _), emb in zip(batch, embeddings_list):
                        save_embedding(nid, ch, np.array(emb), batch_mode=True, storage_path=storage_path, engine_id=eid)
                    flush_embedding_batch(storage_path=storage_path)
                    i += len(batch)
                    batch_size = min(32, batch_size)
                else:
                    batch_size = max(4, batch_size // 2)
                    log_debug(f"Background indexer: reducing batch size to {batch_size}")
            except Exception as e:
                log_debug(f"Background indexer batch error: {e}")
                batch_size = max(4, batch_size // 2)
                if batch_size < len(batch):
                    log_debug(f"Background indexer: retrying with batch size {batch_size}")
                else:
                    i += len(batch)
        log_debug("Background indexer: finished")
    except Exception as e:
        log_debug(f"Background indexer error: {e}")


class BackgroundIndexer(QThread):
    """Runs only API + save; receives todo and storage_path from main thread. Never touches mw.col."""
    def __init__(self, todo, config, storage_path):
        super().__init__()
        self._todo = todo
        self._config = config
        self._storage_path = storage_path

    def run(self):
        try:
            _run_indexer_batch(self._todo, self._config, self._storage_path)
        except Exception as e:
            log_debug(f"BackgroundIndexer run error: {e}")


def _start_background_indexer():
    """Start the background indexer: QueryOp gets todo + path on main thread, then worker runs API + save (no col access)."""
    # Disabled: background indexer worker thread causes crash on startup (logs show crash after worker.start(), no worker-thread logs).
    # Re-enable when indexer is made safe (e.g. run in subprocess or fix thread/Qt usage).
    return
    try:
        from aqt import mw
        from aqt.operations import QueryOp
        if not mw or not mw.col:
            return

        def on_success(pair):
            todo, storage_path = pair
            if not todo or not storage_path:
                log_debug("Background indexer: no notes needing embeddings")
                return
            log_debug(f"Background indexer: indexing {len(todo)} notes")
            worker = BackgroundIndexer(todo, load_config(), storage_path)
            worker.start()

        def op(col):
            config = load_config()
            if not config:
                return [], None
            todo = _get_notes_needing_embeddings_with_col(col, config, limit=200)
            storage_path = get_embeddings_db_path()
            return (todo, storage_path)

        op_run = QueryOp(parent=mw, op=op, success=on_success)
        op_run.with_progress(label="Preparing background indexing...").run_in_background()
    except Exception as e:
        log_debug(f"Could not start background indexer: {e}")


ADDON_NAME = get_addon_name()
log_debug(f"Detected addon folder name: {ADDON_NAME}")

# Table delegates for search results (from ui.search_dialog)
from .ui.search_dialog import RelevanceBarDelegate


def _get_spell_checker():
    """Return SpellChecker instance if pyspellchecker is available, else None.
    Checks Anki's Python first, then external 'Python for Cross-Encoder' site-packages if configured."""
    try:
        from spellchecker import SpellChecker
        return SpellChecker()
    except ImportError:
        pass
    # Try external Python's site-packages (same env as Cross-Encoder rerank)
    try:
        config = load_config()
        sc = config.get('search_config') or {}
        ext_path = (sc.get('rerank_python_path') or '').strip()
        if ext_path:
            if os.path.isfile(ext_path):
                py_dir = os.path.dirname(ext_path)
            elif os.path.isdir(ext_path):
                py_dir = ext_path
            else:
                py_dir = None
            if py_dir:
                site_packages = os.path.join(py_dir, "Lib", "site-packages")
                if os.path.isdir(site_packages) and site_packages not in sys.path:
                    sys.path.insert(0, site_packages)
                    try:
                        from spellchecker import SpellChecker
                        return SpellChecker()
                    finally:
                        if site_packages in sys.path:
                            sys.path.remove(site_packages)
    except Exception:
        pass
    return None


class SpellCheckHighlighter(QSyntaxHighlighter):
    """Underlines misspelled words in the search box. Optional, requires pyspellchecker."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._spell = _get_spell_checker()
        self._format = QTextCharFormat()
        self._format.setUnderlineColor(QColor("#e74c3c"))
        # Qt5: QTextCharFormat.SpellCheckUnderline; Qt6: QTextCharFormat.UnderlineStyle.SpellCheckUnderline
        ustyle = getattr(QTextCharFormat.UnderlineStyle, 'SpellCheckUnderline', None) or getattr(QTextCharFormat, 'SpellCheckUnderline', None)
        if ustyle is not None:
            self._format.setUnderlineStyle(ustyle)
        else:
            self._format.setUnderlineStyle(QTextCharFormat.SingleUnderline)
        self._custom_words = set()  # User-added words (e.g. medical terms)

    def highlightBlock(self, text):
        if not self._spell or not text:
            return
        import re
        txt = text if isinstance(text, str) else str(text)
        for m in re.finditer(r'\b[a-zA-Z]{2,}\b', txt):
            word = m.group()
            word_lower = word.lower()
            if word_lower in self._custom_words:
                continue
            if word_lower in self._spell.unknown([word_lower]):
                self.setFormat(m.start(), m.end() - m.start(), self._format)

    def add_custom_word(self, word):
        if word:
            self._custom_words.add(word.lower())
            self.rehighlight()


class SpellCheckPlainTextEdit(QPlainTextEdit):
    """QPlainTextEdit with optional spell check (underline + context menu suggestions)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._spell_highlighter = SpellCheckHighlighter(self.document())
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def _on_context_menu(self, pos):
        menu = self.createStandardContextMenu()
        spell = _get_spell_checker()
        if spell:
            cursor = self.textCursor()
            if cursor.hasSelection():
                word = cursor.selectedText().strip()
            else:
                try:
                    cursor.select(QTextCursor.SelectionType.WordUnderCursor)
                except AttributeError:
                    cursor.select(QTextCursor.WordUnderCursor)
                word = cursor.selectedText().strip()
            if word and len(word) >= 2 and word.replace("'", "").replace("-", "").isalpha():
                word_lower = word.lower()
                if word_lower in spell.unknown([word_lower]):
                    first = menu.actions()[0] if menu.actions() else None
                    if first:
                        menu.insertSeparator(first)
                    candidates = spell.candidates(word_lower)
                    candidates = (list(candidates)[:5] if candidates else [])
                    if candidates:
                        for corr in candidates:
                            act = QAction(f"✓ {corr}", self)
                            act.triggered.connect(lambda checked, c=corr, cur=cursor: self._replace_word_at_cursor(cur, word, c))
                            menu.insertAction(first, act)
                    add_act = QAction(f"Add \"{word}\" to dictionary", self)
                    add_act.triggered.connect(lambda: self._add_to_dict(word_lower))
                    menu.insertAction(first, add_act)
        menu.exec(self.mapToGlobal(pos))

    def _replace_word_at_cursor(self, cursor, old_word, new_word):
        self.setTextCursor(QTextCursor(cursor))
        cur = self.textCursor()
        cur.beginEditBlock()
        try:
            cur.select(QTextCursor.SelectionType.WordUnderCursor)
        except AttributeError:
            cur.select(QTextCursor.WordUnderCursor)
        if cur.selectedText().strip().lower() == old_word.lower():
            cur.insertText(new_word)
        cur.endEditBlock()

    def _add_to_dict(self, word):
        if self._spell_highlighter:
            self._spell_highlighter.add_custom_word(word)
            tooltip(f"Added \"{word}\" to dictionary")


class EmbeddingWorker(QThread):
    """Worker thread for embedding generation (prevents UI blocking)"""
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal(int, int, int, int)  # processed, errors, skipped, still_failed_count
    error_signal = pyqtSignal(str)
    
    def __init__(self, ntf, note_count, checkpoint, resume_available):
        super().__init__()
        self.ntf = ntf
        self.note_count = note_count
        self.checkpoint = checkpoint
        self.resume_available = resume_available
        self._is_paused = False
        # Load config for timeout and retry settings
        config = load_config()
        self.http_timeout = get_config_value(config, 'http_timeout', 10)  # Default 10 seconds
        self.max_retries = get_config_value(config, 'http_max_retries', 3)  # Default 3 retries
        self.progress_update_interval = get_config_value(config, 'progress_update_interval', 10)  # Update every N notes
        self.last_progress_update = 0
        # Batch size for embeddings (number of notes per HTTP call). Read as int (config may have int or float).
        sc = config.get("search_config") or {}
        engine = sc.get("embedding_engine") or "voyage"
        try:
            if engine == "ollama":
                self.effective_batch_size = max(1, min(256, int(sc.get("ollama_batch_size") or 64)))
            else:
                self.effective_batch_size = max(1, min(256, int(sc.get("voyage_batch_size") or 64)))
        except (TypeError, ValueError):
            self.effective_batch_size = 64
        # Dynamic batch size: start at configured size, adapt from response time and notes/sec
        self._dynamic_batch_size = self.effective_batch_size
        self._use_dynamic_batch = bool(sc.get("use_dynamic_batch_size", True))
        self._batch_count = 0  # for occasional log
        # Track timing for ETA estimates
        self._start_time = None
    
    def run(self):
        """Run embedding generation in background thread"""
        try:
            import numpy as np
            import hashlib
            import time
            
            # Check for interruption before starting
            if self.isInterruptionRequested():
                return
            
            # Quick embeddings API test before long run (uses configured engine: Voyage, OpenAI, Cohere, or Ollama)
            try:
                self.status_update.emit("Testing embeddings API connection...")
                test_emb = get_embedding_for_query("Test connection")
                self.log_message.emit("✅ Embeddings API connection verified")
            except Exception as e:
                raise Exception(f"Embeddings API test failed: {str(e)[:100]}")
            # Expected dimension for current engine; re-embed when stored embedding has different dimension (e.g. switched engine)
            expected_dim = len(test_emb) if test_emb else None
            config = load_config()
            engine_id = get_embedding_engine_id(config)
            
            # Check for interruption after connection test
            if self.isInterruptionRequested():
                return
            
            # Start timing after successful connection test
            self._start_time = time.time()
            
            # Only use checkpoint if it was for this engine (so switching Voyage<->Ollama doesn't resume wrong run)
            if self.checkpoint and self.resume_available and self.checkpoint.get('engine_id') != engine_id:
                self.checkpoint = None
                self.resume_available = False
                self.log_message.emit(f"Checkpoint was for a different engine; starting fresh for {engine_id}")
            
            # Get notes to process
            deck_q = _build_deck_query(self.ntf.get('enabled_decks'))
            note_ids = mw.col.find_notes(deck_q) if deck_q else mw.col.find_notes("")
            
            enabled = self.ntf.get('enabled_note_types')
            enabled_set = set(enabled) if (enabled and len(enabled) > 0) else None
            search_all = bool(self.ntf.get('search_all_fields', False))
            ntf_fields = self.ntf.get('note_type_fields') or {}
            use_first = bool(self.ntf.get('use_first_field_fallback', True))
            
            # Load checkpoint data if resuming
            processed_note_ids = set()
            if self.checkpoint and self.resume_available:
                processed_note_ids = set(self.checkpoint.get('processed_note_ids', []))
                self.log_message.emit(f"📋 Starting from previous checkpoint: {len(processed_note_ids):,} notes (latest checkpoint will update below as we save)")
            
            processed = len(processed_note_ids)
            errors = self.checkpoint.get('errors', 0) if self.checkpoint else 0
            skipped = 0
            # Track last saved checkpoint so we never miss a multiple of 500 when processed jumps by batch size
            last_checkpoint_at = (processed // 500) * 500
            
            # Initialize batch cache
            global _embedding_batch_cache
            _embedding_batch_cache = {}
            # Optimize batch size: larger batches for better performance (100 instead of 50)
            batch_save_interval = 100
            last_batch_save = 0
            # Pending notes for Voyage batch API: list of (nid, content_hash, note_content)
            pending_notes = []
            # Failed notes for proofing retry: nid -> (content_hash, note_content)
            failed_notes = {}

            def maybe_save_checkpoint():
                """Save checkpoint if we've crossed a 500 boundary (handles jumps from batch processing)."""
                nonlocal last_checkpoint_at
                next_checkpoint = (processed // 500) * 500
                if next_checkpoint <= last_checkpoint_at or self.isInterruptionRequested():
                    return
                save_checkpoint(list(processed_note_ids), self.note_count, errors, engine_id=engine_id)
                last_checkpoint_at = next_checkpoint
                if self._start_time and processed > 0 and self.note_count > 0:
                    elapsed = time.time() - self._start_time
                    rate = processed / max(elapsed, 1e-6)
                    remaining = max(self.note_count - processed, 0)
                    eta_seconds = remaining / max(rate, 1e-6)
                    eta_minutes = int(eta_seconds // 60)
                    eta_secs = int(eta_seconds % 60)
                    self.status_update.emit(
                        f"Processed {processed:,} / {self.note_count:,} notes • Checkpoint: {next_checkpoint:,} • ETA ~ {eta_minutes}m {eta_secs}s"
                    )
                else:
                    self.status_update.emit(f"Processed {processed:,} / {self.note_count:,} notes • Checkpoint: {next_checkpoint:,}")
                self.log_message.emit(f"📋 Checkpoint saved: {next_checkpoint:,} / {self.note_count:,} notes")

            def process_pending_batch():
                """Send accumulated notes in a single batch request; adapt batch size from response time."""
                nonlocal pending_notes, processed, errors, last_batch_save, failed_notes
                if not pending_notes:
                    return
                # Check interruption before making HTTP request
                if self.isInterruptionRequested():
                    return
                batch_n = len(pending_notes)
                texts = [item[2] for item in pending_notes]
                t0 = time.time()
                try:
                    embeddings_list = get_embeddings_batch(texts, input_type="document")
                    t1 = time.time()
                except Exception as batch_err:
                    # If batch fails, fall back to per-note calls so we don't lose all progress
                    errors += 1
                    if errors <= 10:
                        self.log_message.emit(f"⚠️ Batch embedding error: {str(batch_err)[:50]} - falling back to per-note")
                    for nid, content_hash, note_content in pending_notes:
                        if self.isInterruptionRequested():
                            break
                        try:
                            single_embedding_list = get_embedding_for_query(note_content)
                            if not single_embedding_list:
                                raise ValueError("Empty embedding array from API")
                            embedding = np.array(single_embedding_list)
                            save_embedding(nid, content_hash, embedding, batch_mode=True, engine_id=engine_id)
                            processed_note_ids.add(nid)
                            processed += 1
                            if processed - self.last_progress_update >= self.progress_update_interval:
                                self.progress_update.emit(processed)
                                self.last_progress_update = processed
                            if processed - last_batch_save >= batch_save_interval:
                                flush_embedding_batch()
                                last_batch_save = processed
                        except Exception as embed_error:
                            errors += 1
                            failed_notes[nid] = (content_hash, note_content)
                            if errors <= 10:
                                self.log_message.emit(f"⚠️ Error generating embedding for note {nid}: {str(embed_error)[:50]}")
                    pending_notes = []
                    return

                # Successful batch response
                if not embeddings_list or len(embeddings_list) != len(pending_notes):
                    errors += 1
                    if errors <= 10:
                        self.log_message.emit("⚠️ Batch embedding error: mismatched embeddings count")
                    for nid, content_hash, note_content in pending_notes:
                        failed_notes[nid] = (content_hash, note_content)
                    pending_notes = []
                    return

                for (nid, content_hash, _note_content), embedding_list in zip(pending_notes, embeddings_list):
                    if self.isInterruptionRequested():
                        break
                    try:
                        if not embedding_list:
                            raise ValueError("Empty embedding array from API")
                        embedding = np.array(embedding_list)
                        save_embedding(nid, content_hash, embedding, batch_mode=True, engine_id=engine_id)
                        processed_note_ids.add(nid)
                        processed += 1
                        # Throttle progress updates to reduce UI overhead
                        if processed - self.last_progress_update >= self.progress_update_interval:
                            self.progress_update.emit(processed)
                            self.last_progress_update = processed
                            # Update status label with simple ETA
                            if self._start_time and processed > 0 and self.note_count > 0:
                                elapsed = time.time() - self._start_time
                                rate = processed / max(elapsed, 1e-6)
                                remaining = max(self.note_count - processed, 0)
                                eta_seconds = remaining / max(rate, 1e-6)
                                eta_minutes = int(eta_seconds // 60)
                                eta_secs = int(eta_seconds % 60)
                                self.status_update.emit(
                                    f"Processed {processed:,} / {self.note_count:,} notes "
                                    f"(ETA ~ {eta_minutes}m {eta_secs}s)"
                                )
                        if processed - last_batch_save >= batch_save_interval:
                            flush_embedding_batch()
                            last_batch_save = processed
                    except Exception as embed_error:
                        errors += 1
                        failed_notes[nid] = (content_hash, _note_content)
                        if errors <= 10:
                            self.log_message.emit(f"⚠️ Error generating embedding for note {nid}: {str(embed_error)[:50]}")

                # Dynamic batch size: increase or decrease from actual response time to balance speed and responsiveness
                if self._use_dynamic_batch and batch_n > 0 and t1 > t0:
                    batch_duration = t1 - t0
                    notes_per_sec = batch_n / max(batch_duration, 0.001)
                    old_size = self._dynamic_batch_size
                    if batch_duration > 15:
                        # Slow: reduce batch for more frequent progress and stability
                        self._dynamic_batch_size = max(8, int(self._dynamic_batch_size * 0.6))
                    elif batch_duration < 6 and self._dynamic_batch_size < 256:
                        # Fast: increase batch to improve throughput (fewer round-trips)
                        self._dynamic_batch_size = min(256, int(self._dynamic_batch_size * 1.2) + 1)
                    if self._dynamic_batch_size != old_size:
                        self.log_message.emit(
                            f"Batch size {old_size} → {self._dynamic_batch_size} "
                            f"({batch_duration:.1f}s, {notes_per_sec:.1f} notes/s)"
                        )
                    self._batch_count += 1

                pending_notes = []
            
            for nid in note_ids:
                # Check for thread interruption request
                if self.isInterruptionRequested():
                    self.log_message.emit("⚠️ Thread interruption requested. Stopping...")
                    break
                
                # Check for pause (with interruption check)
                while self._is_paused:
                    if self.isInterruptionRequested():
                        self.log_message.emit("⚠️ Thread interruption requested during pause. Stopping...")
                        break
                    time.sleep(0.1)
                
                # Check again after pause loop
                if self.isInterruptionRequested():
                    break
                
                try:
                    note = mw.col.get_note(nid)
                    m = note.note_type()
                    model_name = m['name']
                    
                    if enabled_set and model_name not in enabled_set:
                        continue
                    
                    # Get note content (back-only: exclude first field / question for AI)
                    flds = m['flds']
                    if search_all:
                        indices = [i for i in range(min(len(note.fields), len(flds))) if note.fields[i].strip()]
                    else:
                        wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))
                        if not wanted and use_first and flds:
                            wanted = {flds[0]['name'].lower()}
                        indices = [i for i, f in enumerate(flds) if i < len(note.fields) and f['name'].lower() in wanted and note.fields[i].strip()]
                    if not indices:
                        continue
                    content_parts = [note.fields[i] for i in indices]
                    
                    if not content_parts:
                        continue
                    
                    note_content = " ".join(content_parts)
                    content_hash = hashlib.md5(note_content.encode()).hexdigest()
                    
                    # Check if embedding already exists for this engine (one DB stores multiple engines)
                    existing_embedding = load_embedding(nid, content_hash, engine_id=engine_id)
                    skip_existing = False
                    if existing_embedding is not None and expected_dim is not None:
                        try:
                            skip_existing = (len(existing_embedding) == expected_dim)
                        except (TypeError, AttributeError):
                            pass
                    elif existing_embedding is not None:
                        skip_existing = True  # unknown expected_dim: keep old behavior (skip if exists)
                    if skip_existing:
                        if nid not in processed_note_ids:
                            processed_note_ids.add(nid)
                            processed += 1
                            skipped += 1
                            # Throttle progress updates
                            if processed - self.last_progress_update >= self.progress_update_interval:
                                self.progress_update.emit(processed)
                                self.last_progress_update = processed
                                if self._start_time and processed > 0 and self.note_count > 0:
                                    elapsed = time.time() - self._start_time
                                    rate = processed / max(elapsed, 1e-6)
                                    remaining = max(self.note_count - processed, 0)
                                    eta_seconds = remaining / max(rate, 1e-6)
                                    eta_minutes = int(eta_seconds // 60)
                                    eta_secs = int(eta_seconds % 60)
                                    self.status_update.emit(
                                        f"Processed {processed:,} / {self.note_count:,} notes "
                                        f"(ETA ~ {eta_minutes}m {eta_secs}s)"
                                    )
                            # Save checkpoint when we cross a 500 boundary (even when only skipping)
                            maybe_save_checkpoint()
                        continue
                    
                    if nid in processed_note_ids:
                        continue
                    
                    # Check for interruption before making HTTP request
                    if self.isInterruptionRequested():
                        break
                    
                    # Emit progress update for skipped notes (throttled)
                    if processed - self.last_progress_update >= self.progress_update_interval:
                        self.progress_update.emit(processed)
                        self.last_progress_update = processed
                        if self._start_time and processed > 0 and self.note_count > 0:
                            elapsed = time.time() - self._start_time
                            rate = processed / max(elapsed, 1e-6)
                            remaining = max(self.note_count - processed, 0)
                            eta_seconds = remaining / max(rate, 1e-6)
                            eta_minutes = int(eta_seconds // 60)
                            eta_secs = int(eta_seconds % 60)
                            self.status_update.emit(
                                f"Processed {processed:,} / {self.note_count:,} notes "
                                f"(ETA ~ {eta_minutes}m {eta_secs}s)"
                            )
                    
                    # Queue this note for batch embedding (batch size may adapt for speed/responsiveness)
                    pending_notes.append((nid, content_hash, note_content))
                    if len(pending_notes) >= self._dynamic_batch_size:
                        process_pending_batch()
                        # Save checkpoint when we cross a 500 boundary (catches jumps from batch)
                        maybe_save_checkpoint()
                    
                except Exception as e:
                    errors += 1
                    if errors <= 10:
                        self.log_message.emit(f"⚠️ Error processing note {nid}: {str(e)[:50]}")
                    continue
            
            # Final flush and checkpoint save (only if not interrupted)
            if not self.isInterruptionRequested():
                # Process any remaining pending notes
                process_pending_batch()
                flush_embedding_batch()
                save_checkpoint(list(processed_note_ids), self.note_count, errors, engine_id=engine_id)
                # Final checkpoint status in log
                self.log_message.emit(f"📋 Final checkpoint saved: {processed:,} / {self.note_count:,} notes")
            
            # Proofing: retry failed notes once (so missed ones get another chance)
            if failed_notes and not self.isInterruptionRequested():
                self.log_message.emit(f"🔄 Retrying {len(failed_notes)} failed note(s)...")
                self.status_update.emit(f"Retrying {len(failed_notes)} failed note(s)...")
                try:
                    from aqt.qt import QApplication
                    qapp = QApplication.instance()
                except Exception:
                    qapp = None
                retry_list = list(failed_notes.items())
                for idx, (nid, (content_hash, note_content)) in enumerate(retry_list):
                    if self.isInterruptionRequested():
                        break
                    if qapp and idx % 5 == 0:
                        qapp.processEvents()
                    try:
                        single_embedding_list = get_embedding_for_query(note_content)
                        if not single_embedding_list:
                            raise ValueError("Empty embedding array from API")
                        embedding = np.array(single_embedding_list)
                        save_embedding(nid, content_hash, embedding, batch_mode=True, engine_id=engine_id)
                        processed_note_ids.add(nid)
                        processed += 1
                        del failed_notes[nid]
                        if processed - self.last_progress_update >= self.progress_update_interval:
                            self.progress_update.emit(processed)
                            self.last_progress_update = processed
                    except Exception as retry_err:
                        if nid in failed_notes and errors <= 15:
                            self.log_message.emit(f"⚠️ Retry failed for note {nid}: {str(retry_err)[:50]}")
                flush_embedding_batch()
                if failed_notes:
                    self.log_message.emit(f"⚠️ {format_partial_failure_progress(len(failed_notes))}")
                    save_checkpoint(list(processed_note_ids), self.note_count, errors, engine_id=engine_id)
                else:
                    self.log_message.emit("✅ All failed notes succeeded on retry.")
            
            still_failed_count = len(failed_notes)
            
            # Emit final progress update (always, even if throttled)
            if processed != self.last_progress_update:
                self.progress_update.emit(processed)
            
            # Emit finished signal (processed, errors, skipped, still_failed_count)
            self.finished_signal.emit(processed, errors, skipped, still_failed_count)
            
        except Exception as e:
            self.error_signal.emit(str(e))


class SettingsDialog(QDialog):
    def __init__(self, parent=None, open_to_embeddings=False):
        import time
        _t0 = time.time()
        super().__init__(parent)
        self.open_to_embeddings = open_to_embeddings
        self.setWindowTitle("Anki Semantic Search — Settings")
        self._rerank_check_done = False  # defer rerank check until after show
        # Size: allow small minimum, no max so user can maximize/resize to expand and reduce cramming
        self.setMinimumWidth(750)
        self.setMinimumHeight(550)
        # Open large by default so Search Settings content is less crammed; user can maximize or resize
        screen = QApplication.primaryScreen().geometry()
        w = min(1200, int(screen.width() * 0.96))
        h = min(960, int(screen.height() * 0.92))
        self.resize(w, h)
        # Behave like a normal top-level window so minimize/maximize work
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        
        palette = QApplication.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
        
        if is_dark:
            self.setStyleSheet("""
                QDialog { background-color: #1e1e1e; }
                QLabel { color: #e0e0e0; }
                QLineEdit { padding: 8px; border: 2px solid #3498db; border-radius: 6px; background-color: #2d2d2d; color: #e0e0e0; font-size: 12px; }
                QPushButton { padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 12px; color: white; }
                QPushButton#saveBtn { background-color: #27ae60; border: none; }
                QPushButton#saveBtn:hover { background-color: #2ecc71; }
                QPushButton#cancelBtn { background-color: #555555; border: none; }
                QPushButton#cancelBtn:hover { background-color: #777777; }
                QComboBox { padding: 8px; border: 2px solid #3498db; border-radius: 6px; background-color: #2d2d2d; color: #e0e0e0; }
                QGroupBox { font-weight: bold; border: 2px solid #3498db; border-radius: 5px; margin-top: 10px; padding-top: 10px; color: #ffffff; }
                QGroupBox:disabled { color: #666666; border-color: #555555; }
                QSpinBox { padding: 5px; border: 2px solid #3498db; border-radius: 4px; background-color: #2d2d2d; color: #ffffff; }
                QTabWidget::pane { border: 1px solid #555555; background-color: #1e1e1e; }
                QTabBar::tab { background-color: #2d2d2d; color: #ffffff; padding: 8px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
                QTabBar::tab:selected { background-color: #3498db; color: #ffffff; }
            """)
        else:
            self.setStyleSheet("""
                QDialog { background-color: #f5f5f5; }
                QLabel { color: #2c3e50; }
                QLineEdit { padding: 8px; border: 2px solid #3498db; border-radius: 6px; background-color: white; font-size: 12px; }
                QPushButton { padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 12px; }
                QPushButton#saveBtn { background-color: #27ae60; color: white; border: none; }
                QPushButton#saveBtn:hover { background-color: #229954; }
                QPushButton#cancelBtn { background-color: #95a5a6; color: white; border: none; }
                QPushButton#cancelBtn:hover { background-color: #7f8c8d; }
                QComboBox { padding: 8px; border: 2px solid #3498db; border-radius: 6px; background-color: white; }
                QGroupBox { font-weight: bold; border: 2px solid #3498db; border-radius: 5px; margin-top: 10px; padding-top: 10px; color: #2c3e50; }
                QGroupBox:disabled { color: #95a5a6; border-color: #bdc3c7; }
            """)
        log_debug("=== Settings Dialog Opened ===")
        # Store reference to service process
        self.service_process = None
        self.setup_ui()

    def showEvent(self, event):
        """Defer rerank availability check until after window is shown so opening Settings doesn't freeze."""
        super().showEvent(event)
        if not getattr(self, "_rerank_check_scheduled", False):
            self._rerank_check_scheduled = True
            from aqt.qt import QTimer
            QTimer.singleShot(80, self._deferred_check_rerank)

    def _deferred_check_rerank(self):
        """Run _check_rerank_available in a worker thread so Settings never freezes."""
        import time
        config = load_config()
        sc = (config or {}).get("search_config") or {}
        rerank_python = (sc.get("rerank_python_path") or "").strip() or None
        self._rerank_check_worker = RerankCheckWorker(self, rerank_python)
        self._rerank_check_start = time.time()

        def _on_rerank_check_done(available):
            self._rerank_available = available
            if hasattr(self, "enable_rerank_cb") and self.enable_rerank_cb is not None:
                self.enable_rerank_cb.setEnabled(available)
            if hasattr(self, "_update_rerank_tooltip"):
                self._update_rerank_tooltip()
            self._rerank_check_worker = None

        self._rerank_check_worker.finished_signal.connect(_on_rerank_check_done)
        self._rerank_check_worker.start()

    def setup_ui(self):
        # Debug: Track timing for performance analysis
        import time
        start_time = time.time()
        log_debug("=== Settings Dialog UI Setup Started ===")
        
        # Main layout with proper spacing - fix layout issues
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        elapsed = time.time() - start_time
        log_debug(f"  [Timing] Layout setup: {elapsed:.3f}s")
        
        tabs = QTabWidget()
        # Size to content; scroll_content minimum and scroll area give enough height for scrolling when needed
        tabs.setMinimumHeight(0)
        
        # API Settings Tab
        api_tab = QWidget()
        api_layout = QVBoxLayout(api_tab)
        api_layout.setSpacing(15)
        api_layout.setContentsMargins(20, 20, 20, 20)
        
        info = QLabel("🔑 API & AI provider")
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 15px; font-weight: bold; color: #2c3e50; margin-bottom: 4px;")
        api_layout.addWidget(info)
        
        subtitle = QLabel(
            "Choose Ollama (local, no key) or an API key — provider is detected from the key. Works with OpenAI, Anthropic, Google, OpenRouter, or custom endpoints."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 11px; color: #7f8c8d; margin-bottom: 10px;")
        api_layout.addWidget(subtitle)
        
        privacy_note = QLabel(
            "⚠️ AI answers may send selected note content to the chosen provider. Use Ollama (local) for both embeddings and answers to keep everything on your machine. Cloud providers require explicit opt-in — your note content will be sent to external APIs."
        )
        privacy_note.setWordWrap(True)
        privacy_note.setStyleSheet("font-size: 11px; color: #e67e22; margin-bottom: 10px; padding: 8px; background-color: rgba(230, 126, 34, 0.1); border-radius: 4px;")
        api_layout.addWidget(privacy_note)
        
        # Answer with: API key or Ollama
        answer_provider_row = QFormLayout()
        self.answer_provider_combo = QComboBox()
        self.answer_provider_combo.addItem("Ollama (local, no API key) — recommended", "ollama")
        self.answer_provider_combo.addItem("API key (cloud: OpenAI, Anthropic, Google, etc.)", "api_key")
        self.answer_provider_combo.setToolTip(
            "Answer and simple LLM steps (HyDE, query expansion, generic-term detection) use this provider. "
            "Retrieval (embeddings) uses the Search & Embeddings tab. Set to Ollama for local generation and minimal cloud tokens; use RAG-optimized button there for best retrieval + local answer."
        )
        self.answer_provider_combo.currentIndexChanged.connect(self._on_answer_provider_changed)
        answer_provider_row.addRow("Answer with:", self.answer_provider_combo)
        api_embed_note = QLabel("Embeddings (for semantic search) are configured in the Search & Embeddings tab.")
        api_embed_note.setWordWrap(True)
        api_embed_note.setStyleSheet("font-size: 10px; color: #7f8c8d; margin-top: 2px;")
        answer_provider_row.addRow("", api_embed_note)
        api_layout.addLayout(answer_provider_row)
        
        # API Key section (hidden when Ollama is selected)
        self.api_key_section = QWidget()
        api_key_section_layout = QVBoxLayout(self.api_key_section)
        api_key_section_layout.setContentsMargins(0, 0, 0, 0)
        key_layout = QHBoxLayout()
        key_label = QLabel("API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Paste your API key here...")
        self.api_key_input.textChanged.connect(self.detect_provider)
        
        key_layout.addWidget(key_label)
        key_layout.addWidget(self.api_key_input)
        
        self.show_key_btn = QPushButton("Show")
        self.show_key_btn.setMaximumWidth(80)
        self.show_key_btn.clicked.connect(self.toggle_key_visibility)
        key_layout.addWidget(self.show_key_btn)
        api_key_section_layout.addLayout(key_layout)
        
        self.provider_label = QLabel()
        self.provider_label.setStyleSheet("background-color: #e8f4f8; color: #00529b; padding: 10px; border-radius: 5px; font-weight: bold;")
        self.provider_label.hide()
        api_key_section_layout.addWidget(self.provider_label)
        
        url_layout = QHBoxLayout()
        url_label = QLabel("API URL:")
        self.api_url_input = QLineEdit()
        self.api_url_input.setPlaceholderText("https://api.example.com/v1/chat/completions")
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.api_url_input)
        self.url_widget = QWidget()
        self.url_widget.setLayout(url_layout)
        self.url_widget.hide()
        api_key_section_layout.addWidget(self.url_widget)
        api_layout.addWidget(self.api_key_section)
        
        # Ollama answer section (visible only when Answer with = Ollama)
        self.ollama_answer_section = QWidget()
        ollama_answer_layout = QFormLayout(self.ollama_answer_section)
        ollama_answer_layout.addRow(QLabel("Ollama URL:"), QLabel("Uses URL from Search & Embeddings tab (Embeddings → Ollama)."))
        chat_model_row = QHBoxLayout()
        self.ollama_chat_model_combo = QComboBox()
        self.ollama_chat_model_combo.setEditable(True)
        self.ollama_chat_model_combo.setMinimumWidth(220)
        self.ollama_chat_model_combo.setToolTip("Model for AI answers. Click Refresh to load from Ollama.")
        chat_model_row.addWidget(self.ollama_chat_model_combo)
        self.ollama_chat_refresh_btn = QPushButton("🔄 Refresh models")
        self.ollama_chat_refresh_btn.setToolTip("Load available models from Ollama (uses URL from Search & Embeddings tab)")
        self.ollama_chat_refresh_btn.clicked.connect(self._refresh_ollama_chat_models)
        chat_model_row.addWidget(self.ollama_chat_refresh_btn)
        ollama_answer_layout.addRow("Chat model:", chat_model_row)
        test_ollama_btn = QPushButton("🔌 Test Ollama connection")
        test_ollama_btn.setToolTip("Test connection to Ollama (uses URL from Search & Embeddings tab). Shows pass/fail with latency.")
        test_ollama_btn.clicked.connect(self._test_ollama_connection)
        ollama_answer_layout.addRow("", test_ollama_btn)
        self.ollama_answer_section.hide()
        api_layout.addWidget(self.ollama_answer_section)
        
        # Collapsible help: "Need help?" toggles visibility
        self._api_help_visible = False
        api_help_btn = QPushButton("Need help? (providers, free options)")
        api_help_btn.setToolTip("Click to show or hide provider links and free options")
        self.info_text = QLabel()
        self.info_text.setWordWrap(True)
        self.info_text.setStyleSheet("background-color: #f0f0f0; color: #333333; padding: 10px; border-radius: 5px; font-size: 11px;")
        self.info_text.setText(
            "Answer with Ollama (local): no API key — choose 'Ollama (local)' above and set Chat model. Uses Ollama URL from Search & Embeddings tab.\n\n"
            "API Key (cloud):\n"
            "• Anthropic (Claude): sk-ant-... → console.anthropic.com\n"
            "• OpenAI (GPT): sk-... → platform.openai.com/api-keys\n"
            "• Google (Gemini): AI... → aistudio.google.com/app/apikey (FREE!)\n"
            "• OpenRouter: sk-or-... → openrouter.ai/keys\n\n"
            "💡 Free options: Google Gemini or Ollama (local)"
        )
        self.info_text.setVisible(False)
        def _toggle_api_help():
            self._api_help_visible = not self._api_help_visible
            self.info_text.setVisible(self._api_help_visible)
            api_help_btn.setText("▲ Hide help" if self._api_help_visible else "Need help? (providers, free options)")
        api_help_btn.clicked.connect(_toggle_api_help)
        api_layout.addWidget(api_help_btn)
        api_layout.addWidget(self.info_text)
        api_layout.addStretch()
        
        tabs.addTab(api_tab, "🔑 API Settings")
        
        # Styling Tab
        style_tab = QWidget()
        style_layout = QVBoxLayout(style_tab)
        style_layout.setSpacing(15)
        style_layout.setContentsMargins(20, 20, 20, 20)
        
        style_info = QLabel("🎨 Appearance")
        style_info.setStyleSheet("font-size: 15px; font-weight: bold; color: #2c3e50; margin-bottom: 4px;")
        style_layout.addWidget(style_info)
        style_sub = QLabel("Font sizes, window size, and layout. Optional; defaults work for most users.")
        style_sub.setWordWrap(True)
        style_sub.setStyleSheet("font-size: 11px; color: #7f8c8d; margin-bottom: 10px;")
        style_layout.addWidget(style_sub)
        
        # Font Size Settings
        font_group = QGroupBox("Font Sizes")
        font_layout = QVBoxLayout(font_group)
        
        question_font_layout = QHBoxLayout()
        question_font_label = QLabel("Question Input Font Size:")
        self.question_font_spin = QSpinBox()
        self.question_font_spin.setRange(10, 20)
        self.question_font_spin.setValue(13)
        self.question_font_spin.setSuffix(" px")
        question_font_layout.addWidget(question_font_label)
        question_font_layout.addStretch()
        question_font_layout.addWidget(self.question_font_spin)
        font_layout.addLayout(question_font_layout)
        
        answer_font_layout = QHBoxLayout()
        answer_font_label = QLabel("AI Answer Font Size:")
        self.answer_font_spin = QSpinBox()
        self.answer_font_spin.setRange(10, 20)
        self.answer_font_spin.setValue(13)
        self.answer_font_spin.setSuffix(" px")
        answer_font_layout.addWidget(answer_font_label)
        answer_font_layout.addStretch()
        answer_font_layout.addWidget(self.answer_font_spin)
        font_layout.addLayout(answer_font_layout)
        
        notes_font_layout = QHBoxLayout()
        notes_font_label = QLabel("Notes List Font Size:")
        self.notes_font_spin = QSpinBox()
        self.notes_font_spin.setRange(10, 18)
        self.notes_font_spin.setValue(12)
        self.notes_font_spin.setSuffix(" px")
        notes_font_layout.addWidget(notes_font_label)
        notes_font_layout.addStretch()
        notes_font_layout.addWidget(self.notes_font_spin)
        font_layout.addLayout(notes_font_layout)
        
        label_font_layout = QHBoxLayout()
        label_font_label = QLabel("Label Font Size:")
        self.label_font_spin = QSpinBox()
        self.label_font_spin.setRange(11, 18)
        self.label_font_spin.setValue(14)
        self.label_font_spin.setSuffix(" px")
        label_font_layout.addWidget(label_font_label)
        label_font_layout.addStretch()
        label_font_layout.addWidget(self.label_font_spin)
        font_layout.addLayout(label_font_layout)
        
        style_layout.addWidget(font_group)
        
        # Window Size Settings
        window_group = QGroupBox("Window Size")
        window_layout = QVBoxLayout(window_group)
        
        width_layout = QHBoxLayout()
        width_label = QLabel("Default Window Width:")
        self.width_spin = QSpinBox()
        self.width_spin.setRange(800, 1600)
        self.width_spin.setValue(1100)
        self.width_spin.setSuffix(" px")
        width_layout.addWidget(width_label)
        width_layout.addStretch()
        width_layout.addWidget(self.width_spin)
        window_layout.addLayout(width_layout)
        
        height_layout = QHBoxLayout()
        height_label = QLabel("Default Window Height:")
        self.height_spin = QSpinBox()
        self.height_spin.setRange(600, 1200)
        self.height_spin.setValue(800)
        self.height_spin.setSuffix(" px")
        height_layout.addWidget(height_label)
        height_layout.addStretch()
        height_layout.addWidget(self.height_spin)
        window_layout.addLayout(height_layout)
        
        style_layout.addWidget(window_group)
        
        # Layout
        layout_group = QGroupBox("Layout")
        layout_layout = QVBoxLayout(layout_group)
        layout_row = QHBoxLayout()
        layout_label = QLabel("Answer & Notes:")
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Side-by-side (answer | notes)", "side_by_side")
        self.layout_combo.addItem("Stacked (answer above notes)", "stacked")
        layout_row.addWidget(layout_label)
        layout_row.addStretch()
        layout_row.addWidget(self.layout_combo)
        layout_layout.addLayout(layout_row)
        style_layout.addWidget(layout_group)
        
        # Spacing Settings
        spacing_group = QGroupBox("Spacing & Padding")
        spacing_layout = QVBoxLayout(spacing_group)
        
        section_spacing_layout = QHBoxLayout()
        section_spacing_label = QLabel("Section Spacing:")
        self.section_spacing_spin = QSpinBox()
        self.section_spacing_spin.setRange(5, 20)
        self.section_spacing_spin.setValue(12)
        self.section_spacing_spin.setSuffix(" px")
        section_spacing_layout.addWidget(section_spacing_label)
        section_spacing_layout.addStretch()
        section_spacing_layout.addWidget(self.section_spacing_spin)
        spacing_layout.addLayout(section_spacing_layout)
        
        answer_spacing_layout = QHBoxLayout()
        answer_spacing_label = QLabel("Answer line spacing:")
        self.answer_spacing_combo = QComboBox()
        self.answer_spacing_combo.addItem("Compact", "compact")
        self.answer_spacing_combo.addItem("Normal", "normal")
        self.answer_spacing_combo.addItem("Comfortable", "comfortable")
        answer_spacing_layout.addWidget(answer_spacing_label)
        answer_spacing_layout.addStretch()
        answer_spacing_layout.addWidget(self.answer_spacing_combo)
        spacing_layout.addLayout(answer_spacing_layout)
        
        style_layout.addWidget(spacing_group)
        style_layout.addStretch()
        
        tabs.addTab(style_tab, "🎨 Styling")
        
        # --- Note Types & Fields Tab ---
        nt_tab = QWidget()
        nt_main = QVBoxLayout(nt_tab)
        nt_main.setSpacing(10)
        nt_main.setContentsMargins(20, 20, 20, 20)
        
        nt_info = QLabel(
            "📋 Choose which note types and decks to search. Select fields to search per note type. "
            "Tip: For shared or public decks, select the note types and decks you want; you can leave all selected to search everything."
        )
        nt_info.setWordWrap(True)
        nt_info.setStyleSheet("font-size: 11px; color: #7f8c8d; margin-bottom: 6px;")
        nt_info.setStyleSheet("font-size: 15px; font-weight: bold; color: #2c3e50; margin-bottom: 6px;")
        nt_info.setWordWrap(True)
        nt_main.addWidget(nt_info)
        
        # Side-by-side: left = Note types + Decks (stacked), right = Fields by note type
        main_h_split = QSplitter(Qt.Orientation.Horizontal)
        left_v_split = QSplitter(Qt.Orientation.Vertical)
        
        # ---- Left column: Note types (top) ----
        nt_group = QGroupBox("Note types to include")
        nt_gl = QVBoxLayout(nt_group)
        nt_btn_row = QHBoxLayout()
        nt_select_btn = QPushButton("Select All")
        nt_select_btn.clicked.connect(lambda: self._set_note_types_checked(True))
        nt_deselect_btn = QPushButton("Deselect All")
        nt_deselect_btn.clicked.connect(lambda: self._set_note_types_checked(False))
        nt_btn_row.addWidget(nt_select_btn)
        nt_btn_row.addWidget(nt_deselect_btn)
        nt_btn_row.addStretch()
        # Sort options
        sort_label = QLabel("Sort by:")
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Note Count (Desc)", "count_desc")
        self.sort_combo.addItem("Note Count (Asc)", "count_asc")
        self.sort_combo.addItem("Name (A-Z)", "name_asc")
        self.sort_combo.addItem("Name (Z-A)", "name_desc")
        self.sort_combo.currentIndexChanged.connect(self._on_sort_note_types_changed)
        nt_btn_row.addWidget(sort_label)
        nt_btn_row.addWidget(self.sort_combo)
        nt_gl.addLayout(nt_btn_row)
        self.include_all_note_types_cb = QCheckBox("Include all note types")
        self.include_all_note_types_cb.setChecked(True)
        self.include_all_note_types_cb.stateChanged.connect(self._on_include_all_note_types_toggled)
        nt_gl.addWidget(self.include_all_note_types_cb)
        self.note_types_table = QTableWidget()
        self.note_types_table.setColumnCount(2)
        self.note_types_table.setHorizontalHeaderLabels(["Note Type", "Note Count"])
        self.note_types_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.note_types_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.note_types_table.setColumnWidth(1, 120)  # Set minimum width for count column
        self.note_types_table.setMinimumHeight(80)
        self.note_types_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.note_types_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.note_types_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.note_types_table.itemChanged.connect(self._update_field_groups_enabled)
        self.note_types_table.setSortingEnabled(True)
        self.note_types_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)  # Sort by count descending by default
        # Add spacing between columns
        self.note_types_table.setColumnWidth(0, 200)
        nt_gl.addWidget(self.note_types_table)
        left_v_split.addWidget(nt_group)
        
        # ---- Left column: Decks (bottom) ----
        deck_group = QGroupBox("Decks to search")
        deck_gl = QVBoxLayout(deck_group)
        self.include_all_decks_cb = QCheckBox("Include all decks")
        self.include_all_decks_cb.setChecked(True)
        self.include_all_decks_cb.stateChanged.connect(self._on_include_all_decks_toggled)
        deck_gl.addWidget(self.include_all_decks_cb)
        deck_btn_row = QHBoxLayout()
        deck_select_btn = QPushButton("Select All")
        deck_select_btn.clicked.connect(lambda: self._set_decks_checked(True))
        deck_deselect_btn = QPushButton("Deselect All")
        deck_deselect_btn.clicked.connect(lambda: self._set_decks_checked(False))
        deck_btn_row.addWidget(deck_select_btn)
        deck_btn_row.addWidget(deck_deselect_btn)
        deck_btn_row.addStretch()
        deck_gl.addLayout(deck_btn_row)
        # Use QTreeWidget for hierarchical deck display (like main Anki interface)
        self.decks_list = QTreeWidget()
        self.decks_list.setMinimumHeight(80)
        self.decks_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Simplified header: only show deck name and total notes
        self.decks_list.setHeaderLabels(["Deck", "Notes"])
        self.decks_list.setRootIsDecorated(True)  # Show expand/collapse arrows
        self.decks_list.setAlternatingRowColors(True)
        # Set column widths: deck name and notes
        self.decks_list.setColumnWidth(0, 260)  # Deck name
        self.decks_list.setColumnWidth(1, 70)   # Notes
        self.decks_list.itemChanged.connect(self._on_deck_item_changed)
        deck_gl.addWidget(self.decks_list)
        left_v_split.addWidget(deck_group)
        
        left_v_split.setSizes([220, 180])
        left_v_split.setChildrenCollapsible(False)
        left_v_split.setHandleWidth(6)
        main_h_split.addWidget(left_v_split)
        
        # ---- Right column: Fields by note type ----
        fld_outer = QGroupBox("Fields to search per note type (greyed if note type unchecked)")
        fld_outer_l = QVBoxLayout(fld_outer)
        self.search_all_fields_cb = QCheckBox("Search in all fields (ignore selections below)")
        self.search_all_fields_cb.setChecked(False)
        self.search_all_fields_cb.stateChanged.connect(self._on_search_all_fields_toggled)
        fld_outer_l.addWidget(self.search_all_fields_cb)
        self.use_first_field_cb = QCheckBox("Use first field when no fields selected for a note type")
        self.use_first_field_cb.setChecked(True)
        self.use_first_field_cb.setToolTip("If a note type has no checked fields, use its first field instead of skipping.")
        fld_outer_l.addWidget(self.use_first_field_cb)
        self.fields_by_note_type_scroll = QScrollArea()
        self.fields_by_note_type_scroll.setMinimumHeight(120)
        self.fields_by_note_type_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.fields_by_note_type_scroll.setWidgetResizable(True)
        self.fields_by_note_type_inner = QWidget()
        self.fields_by_note_type_layout = QVBoxLayout(self.fields_by_note_type_inner)
        self.fields_by_note_type_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.fields_by_note_type_scroll.setWidget(self.fields_by_note_type_inner)
        fld_outer_l.addWidget(self.fields_by_note_type_scroll)
        main_h_split.addWidget(fld_outer)
        self._field_cbs = {}  # model_name -> { field_name: QCheckBox }
        self._field_groupboxes = {}  # model_name -> QGroupBox (for greying when note type unchecked)
        
        main_h_split.setSizes([380, 420])
        main_h_split.setChildrenCollapsible(False)
        main_h_split.setHandleWidth(6)
        nt_main.addWidget(main_h_split)
        
        # ---- Count notes, Save/Load/Delete, Refresh ----
        action_row = QHBoxLayout()
        self.count_notes_btn = QPushButton("📊 Count notes (with current settings)")
        self.count_notes_btn.clicked.connect(self._on_count_notes)
        action_row.addWidget(self.count_notes_btn)
        action_row.addStretch()
        nt_main.addLayout(action_row)
        
        save_pl = QHBoxLayout()
        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("e.g. Default, Shared deck")
        self.preset_name_edit.setMaximumWidth(200)
        self.preset_name_edit.setToolTip("Name for this set of note types and decks (for quick switching)")
        save_pl.addWidget(QLabel("Save as preset:"))
        save_pl.addWidget(self.preset_name_edit)
        save_preset_btn = QPushButton("💾 Save preset")
        save_preset_btn.clicked.connect(self._on_save_preset)
        save_pl.addWidget(save_preset_btn)
        save_pl.addStretch()
        save_pl.addWidget(QLabel("Load preset:"))
        self.load_preset_combo = QComboBox()
        self.load_preset_combo.setMaximumWidth(180)
        self.load_preset_combo.setEditable(False)
        save_pl.addWidget(self.load_preset_combo)
        load_preset_btn = QPushButton("Load")
        load_preset_btn.clicked.connect(self._on_load_preset)
        save_pl.addWidget(load_preset_btn)
        self.delete_preset_combo = QComboBox()
        self.delete_preset_combo.setMaximumWidth(160)
        save_pl.addWidget(QLabel("Delete:"))
        save_pl.addWidget(self.delete_preset_combo)
        delete_preset_btn = QPushButton("Delete preset")
        delete_preset_btn.clicked.connect(self._on_delete_preset)
        save_pl.addWidget(delete_preset_btn)
        nt_main.addLayout(save_pl)
        
        refresh_btn = QPushButton("🔄 Refresh lists (detect new note types/fields/decks)")
        refresh_btn.clicked.connect(self._refresh_note_type_lists)
        nt_main.addWidget(refresh_btn)
        
        # Defer heavy operations to avoid lag (with timing logs)
        QTimer.singleShot(50, lambda: self._populate_note_type_lists_with_timing())
        QTimer.singleShot(100, lambda: self._populate_fields_by_note_type_with_timing())
        QTimer.singleShot(150, lambda: self._populate_decks_list_with_timing())
        QTimer.singleShot(200, lambda: self._refresh_preset_combos_with_timing())
        
        nt_main.addStretch()
        
        tabs.addTab(nt_tab, "📋 Note Types & Fields")
        
        # Search Settings Tab (compact layout)
        search_tab = QWidget()
        search_layout = QVBoxLayout(search_tab)
        search_layout.setSpacing(12)
        search_layout.setContentsMargins(24, 16, 24, 16)
        
        search_info = QLabel("🔍 Search & embeddings")
        search_info.setStyleSheet("font-size: 15px; font-weight: bold; color: #2c3e50; margin-bottom: 4px;")
        search_layout.addWidget(search_info)
        search_sub = QLabel("Choose how notes are matched (keyword, hybrid, or embedding). Optional: tune result count and relevance. Works with any deck.")
        search_sub.setWordWrap(True)
        search_sub.setStyleSheet("font-size: 11px; color: #7f8c8d; margin-bottom: 10px;")
        search_layout.addWidget(search_sub)
        
        # Search Method
        method_group = QGroupBox("Search Method")
        method_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        method_layout = QVBoxLayout(method_group)
        method_layout.setSpacing(6)

        method_label = QLabel("Choose how notes are matched to your query:")
        method_label.setWordWrap(True)
        method_layout.addWidget(method_label)
        
        self.search_method_combo = QComboBox()
        self.search_method_combo.addItem("Keyword Only - Fast, no dependencies", "keyword")
        self.search_method_combo.addItem("Keyword + Re-rank - Keywords then cross-encoder rerank", "keyword_rerank")
        self.search_method_combo.addItem("Hybrid (RRF) - Keywords + embeddings, best recall", "hybrid")
        self.search_method_combo.addItem("Embedding Only - Best semantic understanding", "embedding")
        self.search_method_combo.setToolTip(
            "Keyword: TF-IDF scoring, no dependencies.\n"
            "Hybrid: Keyword/TF-IDF first, then embedding only on top candidates — minimizes embedding API usage (least token). Combines via RRF for best recall.\n"
            "Embedding: Set in Embeddings section below."
        )
        method_layout.addWidget(self.search_method_combo)
        self.search_method_combo.currentIndexChanged.connect(self._on_search_method_changed)
        
        # Advanced Options (shown in "Advanced" panel below)
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout(advanced_group)
        
        self.enable_query_expansion_cb = QCheckBox("Enable Query Expansion (uses AI to add synonyms)")
        self.enable_query_expansion_cb.setToolTip(
            "Expands your query with related terms using AI. Uses your configured provider "
            "(including local Ollama when selected) and may use extra API tokens for cloud models."
        )
        advanced_layout.addWidget(self.enable_query_expansion_cb)
        
        self.use_ai_generic_term_detection_cb = QCheckBox("Use AI to detect generic query terms")
        self.use_ai_generic_term_detection_cb.setToolTip(
            "One short LLM call per search to find words that are too generic to help find notes "
            "(e.g. 'difference', 'between'). Those terms are excluded from keywords and from 'Matching terms'. "
            "Uses the same provider as answers (Ollama or API key)."
        )
        advanced_layout.addWidget(self.use_ai_generic_term_detection_cb)
        
        self.enable_hyde_cb = QCheckBox("Enable HyDE (Hypothetical Document Embeddings)")
        self.enable_hyde_cb.setToolTip("Generates a hypothetical answer before retrieval to improve semantic search. Adds one extra API call (typically 5–30 s). If you see \"Generating HyDE...\" for several minutes, the API may be slow or rate-limited. Best for abstract queries.")
        advanced_layout.addWidget(self.enable_hyde_cb)
        
        # Optional Python executable for Cross-Encoder (e.g. Python 3.11 when Anki's Python 3.13 fails)
        python_path_row = QFormLayout()
        self.rerank_python_path_input = QLineEdit()
        self.rerank_python_path_input.setMinimumWidth(420)
        self.rerank_python_path_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.rerank_python_path_input.setPlaceholderText(r"C:\...\Python311\python.exe  (optional)")
        self.rerank_python_path_input.setToolTip(
            "If Anki's Python fails with sentence-transformers (e.g. Python 3.13), set this to a Python that has it installed "
            "(e.g. Python 3.11). Use the full path to python.exe. Reranking will run in that Python. "
            "Install there first: that_python -m pip install sentence-transformers"
        )
        python_path_row.addRow("Python for Cross-Encoder (optional):", self.rerank_python_path_input)
        advanced_layout.addLayout(python_path_row)
        
        self.enable_rerank_cb = QCheckBox("Enable Cross-Encoder Re-Ranking")
        # Defer _check_rerank_available to after show (see _deferred_check_rerank) so opening Settings doesn't freeze
        self._rerank_available = False
        self._rerank_check_scheduled = False
        self.enable_rerank_cb.setEnabled(False)
        self.enable_rerank_cb.setToolTip("Checking availability…")
        self._update_rerank_tooltip()
        advanced_layout.addWidget(self.enable_rerank_cb)
        # Row: Check again + Install Dependencies + Install into external Python
        rerank_btn_row = QHBoxLayout()
        rerank_btn_row.addStretch()
        check_rerank_btn = QPushButton("Check again")
        check_rerank_btn.setToolTip("Re-check if sentence-transformers is available in Anki's Python or the external Python (e.g. after installing).")
        check_rerank_btn.clicked.connect(self._on_check_rerank_again)
        rerank_btn_row.addWidget(check_rerank_btn)
        install_deps_btn = QPushButton("Install Dependencies")
        install_deps_btn.setToolTip("Install sentence-transformers into Anki's Python. If Anki's Python doesn't support it, set 'Python for Cross-Encoder' above and use 'Install into external Python'.")
        install_deps_btn.clicked.connect(lambda: install_dependencies(python_exe=None))
        rerank_btn_row.addWidget(install_deps_btn)
        self.install_external_btn = QPushButton("Install into external Python")
        self.install_external_btn.setToolTip("Install sentence-transformers into the Python set in 'Python for Cross-Encoder' above. Use when Anki's Python doesn't support sentence-transformers (e.g. Python 3.13).")
        self.install_external_btn.clicked.connect(self._on_install_into_external_python)
        rerank_btn_row.addWidget(self.install_external_btn)
        def _update_install_external_btn():
            self.install_external_btn.setEnabled(bool((self.rerank_python_path_input.text() or "").strip()))
        self.rerank_python_path_input.textChanged.connect(_update_install_external_btn)
        _update_install_external_btn()  # initial state (enabled only when path is set)
        advanced_layout.addLayout(rerank_btn_row)
        
        self.use_context_boost_cb = QCheckBox("Context-Aware Ranking (boost notes from same deck/type)")
        self.use_context_boost_cb.setChecked(True)
        self.use_context_boost_cb.setToolTip("Boosts notes that are in the same deck or note type as previously selected notes.")
        advanced_layout.addWidget(self.use_context_boost_cb)
        
        # Result tuning (optional)
        tuning_group = QGroupBox("Result quality (optional)")
        tuning_layout = QFormLayout(tuning_group)
        
        self.min_relevance_spin = QSpinBox()
        self.min_relevance_spin.setRange(15, 75)
        self.min_relevance_spin.setValue(55)
        self.min_relevance_spin.setSuffix(" %")
        self.min_relevance_spin.setToolTip("Only notes with at least this relevance (0–100) are kept. Higher = fewer, more focused results (AnkiHub-style). 55–65% reduces side notes.")
        tuning_layout.addRow("Min relevance %:", self.min_relevance_spin)
        
        self.strict_relevance_cb = QCheckBox("Strict relevance (fewer, more on-topic notes)")
        self.strict_relevance_cb.setChecked(True)
        self.strict_relevance_cb.setToolTip("Requires notes to match more keywords and have a higher score to be included. Reduces tangentially related cards (recommended for context-first).")
        tuning_layout.addRow("", self.strict_relevance_cb)
        
        self.verbose_search_debug_cb = QCheckBox("Verbose search debug (log query analysis to debug_log.txt)")
        self.verbose_search_debug_cb.setToolTip(
            "When enabled, writes detailed per-query diagnostics to debug_log.txt: expanded query, keywords, "
            "auto-detected high-frequency filler words, and top-note scores. Useful when tuning search behavior."
        )
        tuning_layout.addRow("", self.verbose_search_debug_cb)
        
        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(5, 50)
        self.max_results_spin.setValue(12)
        self.max_results_spin.setToolTip("Maximum number of relevant notes returned. Lower = more concise, only highly related notes (AnkiHub-style: try 8–15).")
        tuning_layout.addRow("Max results:", self.max_results_spin)

        self.extra_stop_words_input = QLineEdit()
        self.extra_stop_words_input.setPlaceholderText("pediatrics, medicine, clinical")
        self.extra_stop_words_input.setToolTip(
            "Extra generic words to ignore as primary keywords and in 'Matching terms'. "
            "Comma-separated list, e.g. pediatrics, medicine, clinical"
        )
        tuning_layout.addRow("Extra stop-words (queries):", self.extra_stop_words_input)
        
        self.context_chars_per_note_spin = QSpinBox()
        self.context_chars_per_note_spin.setRange(0, 5000)
        self.context_chars_per_note_spin.setValue(0)
        self.context_chars_per_note_spin.setSuffix(" (0 = full)")
        self.context_chars_per_note_spin.setToolTip("Max characters per note sent to the AI. 0 = send full content of selected fields. Set 400–800 to cap context size (faster, cheaper) if notes are very long.")
        tuning_layout.addRow("Max chars per note (AI context):", self.context_chars_per_note_spin)
        
        self.relevance_from_answer_cb = QCheckBox("Relevance from answer")
        self.relevance_from_answer_cb.setToolTip("After the AI answers, re-rank matching notes by similarity to the answer text. Relevance % then reflects \"how much this note supports the answer\" instead of \"relevance to the query\". Uses the same embedding engine (Voyage/Ollama).")
        tuning_layout.addRow("", self.relevance_from_answer_cb)
        
        self.hybrid_weight_label = QLabel("Hybrid: Embedding weight %:")
        self.hybrid_weight_spin = QSpinBox()
        self.hybrid_weight_spin.setRange(0, 100)
        self.hybrid_weight_spin.setValue(30)
        self.hybrid_weight_spin.setSuffix(" %")
        self.hybrid_weight_spin.setToolTip(
            "Weight for semantic (embedding) vs lexical (keyword) in hybrid search. "
            "Medical/factual (doses, lab values): use ~30% embedding (70% lexical). "
            "Conceptual/synonym-heavy: 60–80%. 50% = balanced. Industry default for factual QA: 0.3–0.4 dense."
        )
        tuning_layout.addRow(self.hybrid_weight_label, self.hybrid_weight_spin)
        
        # Embedding engine: Voyage, OpenAI, Cohere (cloud) or Ollama (local)
        self.embedding_group = QGroupBox("Embeddings (for semantic search)")
        embedding_layout = QVBoxLayout(self.embedding_group)
        
        engine_row = QFormLayout()
        self.embedding_engine_combo = QComboBox()
        self.embedding_engine_combo.addItem("Ollama (local) — recommended", "ollama")
        self.embedding_engine_combo.addItem("Voyage AI (cloud)", "voyage")
        self.embedding_engine_combo.addItem("OpenAI (cloud)", "openai")
        self.embedding_engine_combo.addItem("Cohere (cloud)", "cohere")
        self.embedding_engine_combo.setToolTip(
            "Retrieval (embeddings): use Voyage or OpenAI for best results. "
            "Answer and HyDE/expansion use 'Answer with' (API tab) — set to Ollama to keep generation local and minimize cloud tokens. "
            "Cloud providers send note text to external APIs; opt-in only."
        )
        self.embedding_engine_combo.currentIndexChanged.connect(self._on_embedding_engine_changed)
        engine_row.addRow("Engine:", self.embedding_engine_combo)
        embedding_layout.addLayout(engine_row)
        self.embedding_hybrid_hint = QLabel(
            "RAG split: Retrieval (here) = AI embedding for best quality. Answer (API tab) = Ollama for local generation and minimal cloud tokens."
        )
        self.embedding_hybrid_hint.setWordWrap(True)
        self.embedding_hybrid_hint.setStyleSheet("font-size: 11px; color: #7f8c8d; margin-top: 4px;")
        embedding_layout.addWidget(self.embedding_hybrid_hint)
        self.apply_hybrid_btn = QPushButton("Apply: RAG-optimized (Cloud retrieval + local answer)")
        self.apply_hybrid_btn.setToolTip(
            "RAG-optimized: AI embedding for retrieval, Ollama for answer (best quality, minimal cloud tokens). "
            "Sets Embeddings = Voyage, Answer with = Ollama, Search method = Hybrid, Re-rank = on. Click Save to apply."
        )
        self.apply_hybrid_btn.clicked.connect(self._on_apply_hybrid_retrieval)
        embedding_layout.addWidget(self.apply_hybrid_btn)
        
        self.voyage_options = QWidget()
        voyage_form = QFormLayout(self.voyage_options)
        voyage_key_row = QHBoxLayout()
        self.voyage_api_key_input = QLineEdit()
        self.voyage_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.voyage_api_key_input.setPlaceholderText("Paste your Voyage API key here...")
        self.voyage_api_key_input.setMinimumWidth(280)
        self.voyage_api_key_input.setToolTip("Voyage AI API key. Get one at voyageai.com. Leave empty to use VOYAGE_API_KEY.")
        voyage_key_row.addWidget(self.voyage_api_key_input)
        self.voyage_show_key_btn = QPushButton("Show")
        self.voyage_show_key_btn.setMaximumWidth(80)
        self.voyage_show_key_btn.clicked.connect(self._toggle_voyage_key_visibility)
        voyage_key_row.addWidget(self.voyage_show_key_btn)
        voyage_form.addRow("Voyage API key:", voyage_key_row)
        self.voyage_embedding_model_combo = QComboBox()
        for m in VOYAGE_EMBEDDING_MODELS:
            self.voyage_embedding_model_combo.addItem(m, m)
        self.voyage_embedding_model_combo.setToolTip("voyage-3-lite: faster, fewer dimensions. voyage-3.5-lite: higher quality.")
        voyage_form.addRow("Model:", self.voyage_embedding_model_combo)
        embedding_layout.addWidget(self.voyage_options)
        
        self.openai_options = QWidget()
        openai_form = QFormLayout(self.openai_options)
        openai_key_row = QHBoxLayout()
        self.openai_embedding_api_key_input = QLineEdit()
        self.openai_embedding_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_embedding_api_key_input.setPlaceholderText("Paste your OpenAI API key here...")
        self.openai_embedding_api_key_input.setMinimumWidth(280)
        self.openai_embedding_api_key_input.setToolTip("OpenAI API key for embeddings. Leave empty to use OPENAI_API_KEY.")
        openai_key_row.addWidget(self.openai_embedding_api_key_input)
        self.openai_show_key_btn = QPushButton("Show")
        self.openai_show_key_btn.setMaximumWidth(80)
        self.openai_show_key_btn.clicked.connect(self._toggle_openai_key_visibility)
        openai_key_row.addWidget(self.openai_show_key_btn)
        openai_form.addRow("OpenAI API key:", openai_key_row)
        self.openai_embedding_model_input = QLineEdit()
        self.openai_embedding_model_input.setPlaceholderText("text-embedding-3-small")
        self.openai_embedding_model_input.setToolTip("Embedding model, e.g. text-embedding-3-small, text-embedding-3-large.")
        openai_form.addRow("Model:", self.openai_embedding_model_input)
        embedding_layout.addWidget(self.openai_options)
        
        self.cohere_options = QWidget()
        cohere_form = QFormLayout(self.cohere_options)
        cohere_key_row = QHBoxLayout()
        self.cohere_api_key_input = QLineEdit()
        self.cohere_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.cohere_api_key_input.setPlaceholderText("Paste your Cohere API key here...")
        self.cohere_api_key_input.setMinimumWidth(280)
        self.cohere_api_key_input.setToolTip("Cohere API key for embeddings. Leave empty to use COHERE_API_KEY.")
        cohere_key_row.addWidget(self.cohere_api_key_input)
        self.cohere_show_key_btn = QPushButton("Show")
        self.cohere_show_key_btn.setMaximumWidth(80)
        self.cohere_show_key_btn.clicked.connect(self._toggle_cohere_key_visibility)
        cohere_key_row.addWidget(self.cohere_show_key_btn)
        cohere_form.addRow("Cohere API key:", cohere_key_row)
        self.cohere_embedding_model_input = QLineEdit()
        self.cohere_embedding_model_input.setPlaceholderText("embed-english-v3.0")
        self.cohere_embedding_model_input.setToolTip("Embedding model, e.g. embed-english-v3.0, embed-multilingual-v3.0.")
        cohere_form.addRow("Model:", self.cohere_embedding_model_input)
        embedding_layout.addWidget(self.cohere_options)
        
        self.cloud_batch_widget = QWidget()
        cloud_batch_layout = QFormLayout(self.cloud_batch_widget)
        self.voyage_batch_size_spin = QSpinBox()
        self.voyage_batch_size_spin.setRange(8, 256)
        self.voyage_batch_size_spin.setValue(64)
        self.voyage_batch_size_spin.setSuffix(" notes/batch")
        self.voyage_batch_size_spin.setToolTip("Batch size for cloud APIs (Voyage, OpenAI, Cohere). With dynamic batch size on, this adapts from response time.")
        cloud_batch_layout.addRow("Batch size:", self.voyage_batch_size_spin)
        embedding_layout.addWidget(self.cloud_batch_widget)
        
        self.ollama_options = QWidget()
        ollama_form = QFormLayout(self.ollama_options)
        self.ollama_base_url_input = QLineEdit()
        self.ollama_base_url_input.setMinimumWidth(380)
        self.ollama_base_url_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.ollama_base_url_input.setPlaceholderText("http://localhost:11434")
        self.ollama_base_url_input.setToolTip("Ollama server URL. Default: http://localhost:11434")
        ollama_form.addRow("Ollama URL:", self.ollama_base_url_input)
        # Model: editable combo so user can pick from detected list or type a custom model
        model_row = QHBoxLayout()
        self.ollama_embed_model_combo = QComboBox()
        self.ollama_embed_model_combo.setEditable(True)
        self.ollama_embed_model_combo.setMinimumWidth(280)
        self.ollama_embed_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.ollama_embed_model_combo.setToolTip("Embedding model. Click 'Refresh models' to load from Ollama, or type a model name (e.g. nomic-embed-text).")
        model_row.addWidget(self.ollama_embed_model_combo)
        self.ollama_refresh_models_btn = QPushButton("🔄 Refresh models")
        self.ollama_refresh_models_btn.setToolTip("Fetch available models from Ollama (requires Ollama to be running)")
        self.ollama_refresh_models_btn.clicked.connect(self._refresh_ollama_models)
        model_row.addWidget(self.ollama_refresh_models_btn)
        ollama_form.addRow("Embed model:", model_row)
        self.ollama_batch_size_spin = QSpinBox()
        self.ollama_batch_size_spin.setRange(8, 256)
        self.ollama_batch_size_spin.setValue(64)
        self.ollama_batch_size_spin.setSuffix(" notes/batch")
        self.ollama_batch_size_spin.setToolTip("Starting batch size. With 'Use dynamic batch size' on, this adapts automatically from response time and notes/sec for best speed.")
        ollama_form.addRow("Batch size:", self.ollama_batch_size_spin)
        embedding_layout.addWidget(self.ollama_options)
        
        self.use_dynamic_batch_size_cb = QCheckBox("Use dynamic batch size (adapt to response time for best speed)")
        self.use_dynamic_batch_size_cb.setChecked(True)
        self.use_dynamic_batch_size_cb.setToolTip("When enabled, batch size adapts both ways from response time: decrease if a batch is slow (>15s), increase if fast (<6s), to balance total time and responsiveness.")
        embedding_layout.addWidget(self.use_dynamic_batch_size_cb)
        
        self.embedding_status_label = QLabel()
        self.embedding_status_label.setWordWrap(True)
        embedding_layout.addWidget(self.embedding_status_label)
        
        # Buttons for embedding operations
        embedding_btn_layout = QHBoxLayout()
        
        test_connection_btn = QPushButton("🔌 Test Connection")
        test_connection_btn.setToolTip(EmbeddingsTabMessages.TEST_CONNECTION_TOOLTIP)
        test_connection_btn.clicked.connect(self._test_embedding_connection)
        embedding_btn_layout.addWidget(test_connection_btn)
        
        legacy_json_path = get_embeddings_storage_path_for_read()
        has_legacy_json = bool(
            legacy_json_path
            and isinstance(legacy_json_path, str)
            and legacy_json_path.endswith(".json")
            and os.path.exists(legacy_json_path)
        )
        
        create_embedding_btn = QPushButton("🔄 Create/Update Embeddings")
        create_embedding_btn.setToolTip(EmbeddingsTabMessages.CREATE_UPDATE_TOOLTIP)
        create_embedding_btn.clicked.connect(self._create_or_update_embeddings)
        embedding_btn_layout.addWidget(create_embedding_btn)
        
        if has_legacy_json:
            migrate_json_btn = QPushButton("📦 Legacy migration: JSON → DB")
            migrate_json_btn.setToolTip(
                "One-time legacy migration for users upgrading from older versions that stored embeddings in a JSON "
                "file. Copies existing embeddings from the old JSON cache into the SQLite database so you don't need "
                "to re-embed. Most users can ignore this."
            )
            migrate_json_btn.clicked.connect(self._migrate_json_to_db)
            embedding_btn_layout.addWidget(migrate_json_btn)
        
        embedding_layout.addLayout(embedding_btn_layout)
        
        # Accordion: Search method & results | Embeddings | Result quality & tuning
        search_toolbox = QToolBox()
        page1 = QWidget()
        page1_layout = QVBoxLayout(page1)
        page1_layout.setContentsMargins(0, 4, 0, 0)
        page1_layout.addWidget(method_group)
        search_toolbox.addItem(page1, "Search method & results")
        page2 = QWidget()
        page2_layout = QVBoxLayout(page2)
        page2_layout.setContentsMargins(0, 4, 0, 0)
        page2_layout.addWidget(self.embedding_group)
        search_toolbox.addItem(page2, "Embeddings")
        page3 = QWidget()
        page3_layout = QVBoxLayout(page3)
        page3_layout.setContentsMargins(0, 4, 0, 0)
        page3_layout.addWidget(tuning_group)
        page3_layout.addSpacing(8)
        page3_layout.addWidget(advanced_group)
        search_toolbox.addItem(page3, "Result quality & tuning")
        search_toolbox.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        search_layout.addWidget(search_toolbox)
        search_layout.addStretch()
        self._search_toolbox = search_toolbox
        self._search_tab = search_tab
        self._tabs = tabs
        
        tabs.addTab(search_tab, "🔍 Search & Embeddings")
        # Tab order: API → Search & Embeddings → Note Types & Fields → Styling
        tabs.removeTab(tabs.indexOf(search_tab))
        tabs.insertTab(1, search_tab, "🔍 Search & Embeddings")
        tabs.removeTab(tabs.indexOf(nt_tab))
        tabs.insertTab(2, nt_tab, "📋 Note Types & Fields")
        
        if self.open_to_embeddings:
            tabs.setCurrentWidget(search_tab)
            search_toolbox.setCurrentIndex(1)
        
        tabs_elapsed = time.time() - start_time
        log_debug(f"  [Timing] All tabs created: {tabs_elapsed:.3f}s")
        
        # Initialize embedding status (lazy load to avoid blocking)
        QTimer.singleShot(100, self._refresh_embedding_status)  # Slight delay to avoid race conditions
        
        # Scroll area: wrap tabs; compact min height to avoid excessive empty space
        scroll_content = QWidget()
        scroll_content.setMinimumHeight(380)
        scroll_content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(0, 0, 0, 0)
        scroll_content_layout.addWidget(tabs)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_content)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setMinimumHeight(450)
        scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll_area.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self._settings_scroll_area = scroll_area
        self._settings_scroll_content = scroll_content
        self._settings_tabs = tabs
        scroll_content.installEventFilter(self)
        tabs.installEventFilter(self)
        scroll_area.viewport().installEventFilter(self)
        main_layout.addWidget(scroll_area, 1)
        
        # Final timing
        total_elapsed = time.time() - start_time
        log_debug(f"=== Settings Dialog UI Setup Completed: {total_elapsed:.3f}s total ===")
        
        # Buttons at bottom (always visible, not inside scroll)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()  # Push buttons to the right
        save_btn = QPushButton("💾 Save")
        save_btn.setObjectName("saveBtn")
        save_btn.clicked.connect(self.save_settings)
        
        cancel_btn = QPushButton("✖ Cancel")
        cancel_btn.setObjectName("cancelBtn")
        cancel_btn.clicked.connect(self.close)
        
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)
        
        # Load existing config
        config = load_config()
        if 'api_key' in config:
            self.api_key_input.setText(config['api_key'])
        # Answer provider: API key vs Ollama
        provider = config.get('provider', 'openai')
        if provider == 'ollama':
            idx = self.answer_provider_combo.findData('ollama')
            if idx >= 0:
                self.answer_provider_combo.setCurrentIndex(idx)
            sc = config.get('search_config') or {}
            self.ollama_chat_model_combo.setCurrentText((sc.get('ollama_chat_model') or 'llama3.2').strip())
        self._on_answer_provider_changed()
        
        if 'api_url' in config:
            self.api_url_input.setText(config['api_url'])
        
        # Apply config to UI (this might involve slow operations)
        apply_start = time.time()
        if 'styling' in config:
            styling = config['styling']
            self.question_font_spin.setValue(styling.get('question_font_size', 13))
            self.answer_font_spin.setValue(styling.get('answer_font_size', 13))
            self.notes_font_spin.setValue(styling.get('notes_font_size', 12))
            self.label_font_spin.setValue(styling.get('label_font_size', 14))
            self.width_spin.setValue(styling.get('window_width', 1100))
            self.height_spin.setValue(styling.get('window_height', 800))
            self.section_spacing_spin.setValue(styling.get('section_spacing', 12))
            mode = styling.get('layout_mode', 'side_by_side')
            idx = self.layout_combo.findData(mode)
            if idx >= 0:
                self.layout_combo.setCurrentIndex(idx)
            spacing_mode = styling.get('answer_spacing', 'normal')
            idx = self.answer_spacing_combo.findData(spacing_mode)
            if idx >= 0:
                self.answer_spacing_combo.setCurrentIndex(idx)
        ntf = config.get('note_type_filter', {})
        ntf_start = time.time()
        self._apply_note_type_config(ntf)
        ntf_elapsed = time.time() - ntf_start
        log_debug(f"  [Timing] _apply_note_type_config() in __init__: {ntf_elapsed:.3f}s")
        
        # Load search config - default to keyword
        search_config = config.get('search_config', {})
        method = search_config.get('search_method', 'hybrid')
        idx = self.search_method_combo.findData(method)
        if idx >= 0:
            self.search_method_combo.setCurrentIndex(idx)
        else:
            self.search_method_combo.setCurrentIndex(2)  # Hybrid by default (0=Keyword, 1=Keyword+Re-rank, 2=Hybrid, 3=Embedding)
        self._on_search_method_changed()  # Hide/show Cloud Embeddings and Hybrid row by method
        self.enable_query_expansion_cb.setChecked(search_config.get('enable_query_expansion', False))
        self.use_ai_generic_term_detection_cb.setChecked(bool(search_config.get('use_ai_generic_term_detection', False)))
        self.enable_hyde_cb.setChecked(search_config.get('enable_hyde', False))
        self.enable_rerank_cb.setChecked(search_config.get('enable_rerank', False))
        self.rerank_python_path_input.setText((search_config.get('rerank_python_path') or '').strip())
        self.use_context_boost_cb.setChecked(search_config.get('use_context_boost', True))
        self.min_relevance_spin.setValue(max(15, min(75, search_config.get('min_relevance_percent', 55))))
        self.strict_relevance_cb.setChecked(bool(search_config.get('strict_relevance', True)))
        self.verbose_search_debug_cb.setChecked(bool(search_config.get('verbose_search_debug', False)))
        self.relevance_from_answer_cb.setChecked(bool(search_config.get('relevance_from_answer', False)))
        self.max_results_spin.setValue(max(5, min(50, search_config.get('max_results', 12))))
        self.context_chars_per_note_spin.setValue(max(0, min(5000, search_config.get('context_chars_per_note', 0))))
        self.hybrid_weight_spin.setValue(max(0, min(100, search_config.get('hybrid_embedding_weight', 30))))
        # Extra stop-words for query keyword extraction / matching-terms (comma-separated)
        extra_stop_words = search_config.get('extra_stop_words') or []
        if isinstance(extra_stop_words, str):
            # Accept a single comma-separated string directly from config
            self.extra_stop_words_input.setText(extra_stop_words)
        else:
            self.extra_stop_words_input.setText(", ".join(extra_stop_words))
        # Embedding engine: Voyage, OpenAI, Cohere, or Ollama (load keys, models, batch size)
        engine = search_config.get('embedding_engine') or 'voyage'
        self.voyage_api_key_input.setText((search_config.get('voyage_api_key') or '').strip())
        voyage_model = (search_config.get('voyage_embedding_model') or 'voyage-3.5-lite').strip()
        idx_v = self.voyage_embedding_model_combo.findData(voyage_model)
        if idx_v >= 0:
            self.voyage_embedding_model_combo.setCurrentIndex(idx_v)
        self.openai_embedding_api_key_input.setText((search_config.get('openai_embedding_api_key') or '').strip())
        self.openai_embedding_model_input.setText((search_config.get('openai_embedding_model') or 'text-embedding-3-small').strip())
        self.cohere_api_key_input.setText((search_config.get('cohere_api_key') or '').strip())
        self.cohere_embedding_model_input.setText((search_config.get('cohere_embedding_model') or 'embed-english-v3.0').strip())
        try:
            vb = int(search_config.get('voyage_batch_size', 64))
            self.voyage_batch_size_spin.setValue(max(8, min(256, vb)))
        except (TypeError, ValueError):
            self.voyage_batch_size_spin.setValue(64)
        idx = self.embedding_engine_combo.findData(engine)
        if idx >= 0:
            self.embedding_engine_combo.setCurrentIndex(idx)
        self.ollama_base_url_input.setText((search_config.get('ollama_base_url') or "http://localhost:11434").strip())
        self.ollama_embed_model_combo.setCurrentText((search_config.get('ollama_embed_model') or "nomic-embed-text").strip())
        try:
            ob = int(search_config.get('ollama_batch_size', 64))
            self.ollama_batch_size_spin.setValue(max(8, min(256, ob)))
        except (TypeError, ValueError):
            self.ollama_batch_size_spin.setValue(64)
        self.use_dynamic_batch_size_cb.setChecked(bool(search_config.get('use_dynamic_batch_size', True)))
        self._on_embedding_engine_changed()
    
    def _on_apply_hybrid_retrieval(self):
        """One-click RAG-optimized: AI embedding for retrieval, Ollama for answer; Hybrid + Re-rank for best quality with minimal cloud tokens."""
        idx_emb = self.embedding_engine_combo.findData("voyage")
        if idx_emb >= 0:
            self.embedding_engine_combo.setCurrentIndex(idx_emb)
        idx_ans = self.answer_provider_combo.findData("ollama")
        if idx_ans >= 0:
            self.answer_provider_combo.setCurrentIndex(idx_ans)
        idx_hybrid = self.search_method_combo.findData("hybrid")
        if idx_hybrid >= 0:
            self.search_method_combo.setCurrentIndex(idx_hybrid)
        if hasattr(self, 'enable_rerank_cb') and self.enable_rerank_cb is not None:
            self.enable_rerank_cb.setChecked(True)
        self._on_embedding_engine_changed()
        self._on_answer_provider_changed()
        if hasattr(self, '_on_search_method_changed'):
            self._on_search_method_changed()
        showInfo("RAG-optimized applied: Embeddings = Voyage, Answer = Ollama, Hybrid, Re-rank on. Click Save to apply and run Create/Update Embeddings if needed.")
    
    def _on_embedding_engine_changed(self):
        """Show/hide provider-specific options and cloud batch size by engine."""
        engine = self.embedding_engine_combo.currentData() or "voyage"
        self.voyage_options.setVisible(engine == "voyage")
        self.openai_options.setVisible(engine == "openai")
        self.cohere_options.setVisible(engine == "cohere")
        self.cloud_batch_widget.setVisible(engine in ("voyage", "openai", "cohere"))
        self.ollama_options.setVisible(engine == "ollama")
    
    def eventFilter(self, obj, event):
        """Forward mouse wheel over tab content (and any child widget) to scroll area."""
        if event.type() != QEvent.Type.Wheel:
            return super().eventFilter(obj, event)
        scroll_area = getattr(self, "_settings_scroll_area", None)
        tabs = getattr(self, "_settings_tabs", None)
        scroll_content = getattr(self, "_settings_scroll_content", None)
        if not scroll_area or not tabs or scroll_content is None:
            return super().eventFilter(obj, event)
        # Allow scroll when wheel is over scroll content, tabs, viewport, or any descendant
        target = obj
        while target:
            if target == scroll_content or target == tabs or target == scroll_area.viewport():
                break
            target = target.parentWidget() if hasattr(target, "parentWidget") else None
        else:
            return super().eventFilter(obj, event)
        if scroll_area.verticalScrollBar().isVisible():
            delta = event.angleDelta().y() if hasattr(event, "angleDelta") else getattr(event, "delta", 0)
            sb = scroll_area.verticalScrollBar()
            sb.setValue(sb.value() - delta)
            return True
        return super().eventFilter(obj, event)
    
    def _refresh_ollama_models(self):
        """Fetch model list from Ollama and populate the embed model combo."""
        base_url = (self.ollama_base_url_input.text() or "http://localhost:11434").strip()
        current = self.ollama_embed_model_combo.currentText().strip() or "nomic-embed-text"
        try:
            names = get_ollama_models(base_url)
            self.ollama_embed_model_combo.clear()
            self.ollama_embed_model_combo.addItems(names)
            if current and current not in names:
                self.ollama_embed_model_combo.insertItem(0, current)
                self.ollama_embed_model_combo.setCurrentIndex(0)
            elif current in names:
                idx = self.ollama_embed_model_combo.findText(current)
                if idx >= 0:
                    self.ollama_embed_model_combo.setCurrentIndex(idx)
            if not names:
                self.ollama_embed_model_combo.setCurrentText(current or "nomic-embed-text")
            if names:
                showInfo(f"Found {len(names)} model(s) at {base_url}. Choose an embedding model (e.g. nomic-embed-text).")
            else:
                showInfo(
                    "No models returned from Ollama. Make sure Ollama is running (ollama serve) and you have pulled at least one model.\n\n"
                    "You can still type an embedding model name manually (e.g. nomic-embed-text)."
                )
        except Exception as e:
            showInfo(
                f"Could not fetch models from {base_url}.\n\n"
                f"Error: {e}\n\n"
                "Check that Ollama is running (ollama serve). You can type a model name manually (e.g. nomic-embed-text)."
            )
    
    def _populate_note_type_lists_with_timing(self):
        """Fill note types table with name and count columns (with timing)."""
        # Check if widget still exists (dialog might have been closed)
        try:
            if not hasattr(self, 'note_types_table') or self.note_types_table is None:
                return
            # Check if the C++ object is still valid
            if not sip.isdeleted(self.note_types_table) if hasattr(sip, 'isdeleted') else True:
                import time
                start = time.time()
                self._populate_note_type_lists()
                elapsed = time.time() - start
                log_debug(f"  [Timing] _populate_note_type_lists(): {elapsed:.3f}s")
                # After table is populated, apply saved note/deck/field config
                try:
                    cfg = load_config()
                    ntf = cfg.get('note_type_filter', {})
                    self._apply_note_type_config(ntf)
                except Exception as e:
                    log_debug(f"Error re-applying note_type_filter after note type populate: {e}")
        except RuntimeError:
            # Widget was deleted, ignore
            pass
        except Exception as e:
            log_debug(f"Error in _populate_note_type_lists_with_timing: {e}")
    
    def _populate_note_type_lists(self):
        """Fill note types table with name and count columns."""
        # Check if widget still exists
        if not hasattr(self, 'note_types_table') or self.note_types_table is None:
            return
        try:
            self.note_types_table.setRowCount(0)
            counts = get_notes_count_per_model()
            for name in sorted(counts.keys()):
                c = counts.get(name, 0)
                row = self.note_types_table.rowCount()
                self.note_types_table.insertRow(row)
                
                # Name column with checkbox
                name_item = QTableWidgetItem(name)
                name_item.setData(Qt.ItemDataRole.UserRole, name)
                name_item.setFlags(name_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                name_item.setCheckState(Qt.CheckState.Unchecked)
                self.note_types_table.setItem(row, 0, name_item)
                
                # Count column
                count_item = QTableWidgetItem()
                count_item.setData(Qt.ItemDataRole.DisplayRole, c)  # Store numeric value for proper sorting
                count_item.setText(str(c))
                count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make read-only
                self.note_types_table.setItem(row, 1, count_item)
            
            # Sort by count descending by default
            self.note_types_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)
        except RuntimeError:
            # Widget was deleted, ignore
            pass
        except Exception as e:
            log_debug(f"Error in _populate_note_type_lists: {e}")
    
    def _populate_fields_by_note_type(self):
        """Build GroupBox per note type with field checkboxes. Clears and repopulates."""
        # Check if widget still exists
        if not hasattr(self, 'fields_by_note_type_layout') or self.fields_by_note_type_layout is None:
            return
        try:
            while self.fields_by_note_type_layout.count() > 0:
                it = self.fields_by_note_type_layout.takeAt(0)
                if it and it.widget():
                    it.widget().deleteLater()
            self._field_cbs.clear()
            self._field_groupboxes.clear()
            for model_name, count, field_names in get_models_with_fields():
                gb = QGroupBox(f"{model_name}  ({count} notes)")
                lay = QVBoxLayout(gb)
                cbs = {}
                for fn in field_names:
                    cb = QCheckBox(fn)
                    cbs[fn] = cb
                    lay.addWidget(cb)
                self._field_cbs[model_name] = cbs
                self._field_groupboxes[model_name] = gb
                self.fields_by_note_type_layout.addWidget(gb)
            self._update_field_groups_enabled()
        except RuntimeError:
            # Widget was deleted, ignore
            pass
        except Exception as e:
            log_debug(f"Error in _populate_fields_by_note_type: {e}")
    
    def _populate_fields_by_note_type_with_timing(self):
        """Populate fields by note type (with timing)."""
        import time
        start = time.time()
        self._populate_fields_by_note_type()
        elapsed = time.time() - start
        if elapsed > 0.1:  # Only log if it takes significant time
            log_debug(f"  [Timing] _populate_fields_by_note_type(): {elapsed:.3f}s")
        # Ensure field checkboxes match saved configuration
        try:
            cfg = load_config()
            ntf = cfg.get('note_type_filter', {})
            self._apply_note_type_config(ntf)
        except Exception as e:
            log_debug(f"Error re-applying note_type_filter after field populate: {e}")
    
    def _populate_decks_list_with_timing(self):
        """Populate decks list (with timing)."""
        # Check if widget still exists (dialog might have been closed)
        try:
            if not hasattr(self, 'decks_list') or self.decks_list is None:
                return
            import time
            start = time.time()
            self._populate_decks_list()
            elapsed = time.time() - start
            log_debug(f"  [Timing] _populate_decks_list(): {elapsed:.3f}s")
            # Ensure deck checkboxes match saved configuration
            try:
                cfg = load_config()
                ntf = cfg.get('note_type_filter', {})
                self._apply_note_type_config(ntf)
            except Exception as e:
                log_debug(f"Error re-applying note_type_filter after deck populate: {e}")
        except RuntimeError:
            # Widget was deleted, ignore
            pass
        except Exception as e:
            log_debug(f"Error in _populate_decks_list_with_timing: {e}")
    
    def _populate_decks_list(self):
        import time
        deck_start = time.time()
        
        # Check if widget still exists
        if not hasattr(self, 'decks_list') or self.decks_list is None:
            return
        try:
            self.decks_list.clear()
            
            if not mw or not mw.col:
                return
            
            counts_start = time.time()
            counts = get_notes_count_per_deck()
            counts_elapsed = time.time() - counts_start
            log_debug(f"  [Timing] get_notes_count_per_deck(): {counts_elapsed:.3f}s")
            
            # Get deck hierarchy and card counts
            deck_names_start = time.time()
            deck_names = get_deck_names()
            deck_names_elapsed = time.time() - deck_names_start
            log_debug(f"  [Timing] get_deck_names(): {deck_names_elapsed:.3f}s")
            
            # Build hierarchical deck structure
            deck_tree = {}  # parent_name -> [child_decks]
            top_level_decks = []
            
            for name in deck_names:
                if '::' in name:
                    # Sub-deck
                    parts = name.split('::')
                    parent = '::'.join(parts[:-1])
                    if parent not in deck_tree:
                        deck_tree[parent] = []
                    deck_tree[parent].append(name)
                else:
                    # Top-level deck
                    top_level_decks.append(name)
            
            # Get card counts for each deck (new, learn, due)
            card_counts = {}
            try:
                from anki.scheduler import Scheduler
                sched = mw.col.sched
                deck_ids = {name: mw.col.decks.id(name) for name in deck_names if mw.col.decks.by_name(name)}
                
                for name, deck_id in deck_ids.items():
                    try:
                        # Get counts from scheduler
                        counts_info = sched.counts(deck_id)
                        card_counts[name] = {
                            'new': counts_info.new,
                            'learn': counts_info.learn,
                            'due': counts_info.review
                        }
                    except:
                        card_counts[name] = {'new': 0, 'learn': 0, 'due': 0}
            except Exception as e:
                log_debug(f"Error getting card counts: {e}")
                # Fallback: set all to 0
                for name in deck_names:
                    card_counts[name] = {'new': 0, 'learn': 0, 'due': 0}
            
            # Sort decks
            top_level_decks.sort()
            for parent in deck_tree:
                deck_tree[parent].sort()
            
            # Create tree items
            def create_deck_item(name, is_parent=False):
                """Create a tree item for a deck (showing only name + total notes)."""
                note_count = counts.get(name, 0)
                # Hide the built-in empty 'Default' deck, which many users don't use
                if name == "Default" and note_count == 0:
                    return None
                
                # Extract display name (without parent prefix for sub-decks)
                display_name = name.split('::')[-1] if '::' in name else name
                
                item = QTreeWidgetItem([display_name, str(note_count)])
                
                # Store full deck name in item data for later retrieval
                item.setData(0, Qt.ItemDataRole.UserRole, name)
                
                # Make checkable
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Unchecked)
                
                # Style parent decks (bold)
                if is_parent:
                    font = item.font(0)
                    font.setBold(True)
                    item.setFont(0, font)
                
                return item
            
            # Add top-level decks
            for deck_name in top_level_decks:
                parent_item = create_deck_item(deck_name, is_parent=(deck_name in deck_tree))
                if parent_item is None:
                    continue
                self.decks_list.addTopLevelItem(parent_item)
                
                # Add children if any
                if deck_name in deck_tree:
                    for child_name in deck_tree[deck_name]:
                        child_item = create_deck_item(child_name)
                        parent_item.addChild(child_item)
                    
                    # Collapsed by default; user expands on click
                    parent_item.setExpanded(False)
            
            # Also add decks that are parents but not top-level (nested hierarchies)
            for parent_name in sorted(deck_tree.keys()):
                if parent_name not in top_level_decks:
                    # This is a nested parent, find its position in hierarchy
                    parts = parent_name.split('::')
                    if len(parts) > 1:
                        # Find parent item
                        grandparent_name = '::'.join(parts[:-1])
                        # Search for grandparent in tree
                        for i in range(self.decks_list.topLevelItemCount()):
                            parent_item = self._find_deck_item_recursive(self.decks_list.topLevelItem(i), grandparent_name)
                            if parent_item:
                                child_item = create_deck_item(parent_name, is_parent=True)
                                if child_item is None:
                                    break
                                parent_item.addChild(child_item)
                                # Add its children
                                for child_name in deck_tree[parent_name]:
                                    grandchild_item = create_deck_item(child_name)
                                    child_item.addChild(grandchild_item)
                                child_item.setExpanded(False)
                                break
            
            deck_elapsed = time.time() - deck_start
            log_debug(f"  [Timing] _populate_decks_list(): {deck_elapsed:.3f}s")
            
        except RuntimeError:
            # Widget was deleted, ignore
            pass
        except Exception as e:
            log_debug(f"Error in _populate_decks_list: {e}")
            import traceback
            log_debug(traceback.format_exc())
    
    def _find_deck_item_recursive(self, item, deck_name):
        """Recursively find a deck item by its full name"""
        if item is None:
            return None
        if item.data(0, Qt.ItemDataRole.UserRole) == deck_name:
            return item
        for i in range(item.childCount()):
            found = self._find_deck_item_recursive(item.child(i), deck_name)
            if found:
                return found
        return None
    
    def _iterate_all_deck_items(self):
        """Generator that yields all deck items (top-level and children)"""
        for i in range(self.decks_list.topLevelItemCount()):
            item = self.decks_list.topLevelItem(i)
            yield item
            # Recursively yield children
            for j in range(item.childCount()):
                yield from self._iterate_all_deck_items_recursive(item.child(j))
    
    def _iterate_all_deck_items_recursive(self, item):
        """Recursively yield item and all its children"""
        if item:
            yield item
            for i in range(item.childCount()):
                yield from self._iterate_all_deck_items_recursive(item.child(i))
    
    def _refresh_preset_combos_with_timing(self):
        """Refresh preset combos (with timing)."""
        # Check if widget still exists (dialog might have been closed)
        try:
            if not hasattr(self, 'load_preset_combo') or self.load_preset_combo is None:
                return
            import time
            start = time.time()
            self._refresh_preset_combos()
            elapsed = time.time() - start
            log_debug(f"  [Timing] _refresh_preset_combos(): {elapsed:.3f}s")
        except RuntimeError:
            # Widget was deleted, ignore
            pass
        except Exception as e:
            log_debug(f"Error in _refresh_preset_combos_with_timing: {e}")
    
    def _refresh_preset_combos(self):
        # Check if widgets still exist
        if not hasattr(self, 'load_preset_combo') or self.load_preset_combo is None:
            return
        try:
            config = load_config()
            presets = config.get('saved_presets') or {}
            names = sorted(presets.keys())
            self.load_preset_combo.clear()
            self.load_preset_combo.addItem("-- Select --", None)
            for n in names:
                self.load_preset_combo.addItem(n, n)
            if hasattr(self, 'delete_preset_combo') and self.delete_preset_combo is not None:
                try:
                    self.delete_preset_combo.clear()
                    self.delete_preset_combo.addItem("-- Select --", None)
                    for n in names:
                        self.delete_preset_combo.addItem(n, n)
                except RuntimeError:
                    # Widget was deleted, ignore
                    pass
        except RuntimeError:
            # Widget was deleted, ignore
            pass
        except Exception as e:
            log_debug(f"Error in _refresh_preset_combos: {e}")
    
    def _apply_note_type_config(self, ntf):
        """Apply note_type_filter config. Migrate fields_to_search -> note_type_fields if needed."""
        self._applying_note_type_config = True
        try:
            self._apply_note_type_config_impl(ntf)
        finally:
            self._applying_note_type_config = False
    
    def _apply_note_type_config_impl(self, ntf):
        # Migrate: if fields_to_search exists but not note_type_fields, build note_type_fields
        ntf = dict(ntf)
        if ntf.get('fields_to_search') and not ntf.get('note_type_fields'):
            global_flds = set(f.lower() for f in ntf['fields_to_search'])
            ntf['note_type_fields'] = {}
            for model_name, _c, field_names in get_models_with_fields():
                ntf['note_type_fields'][model_name] = [f for f in field_names if f.lower() in global_flds]
        # Note types
        enabled = ntf.get('enabled_note_types')
        # Interpretation:
        #   None        -> include all note types
        #   [] (empty)  -> user has not chosen any specific types yet
        #                  (start with none selected to reduce workload)
        include_all_nt = (enabled is None)
        self.include_all_note_types_cb.setChecked(include_all_nt)
        self._on_include_all_note_types_toggled()
        if not include_all_nt and enabled:
            enabled_set = set(enabled)
            for i in range(self.note_types_table.rowCount()):
                it = self.note_types_table.item(i, 0)
                if it:
                    name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()
                    it.setCheckState(Qt.CheckState.Checked if (name in enabled_set) else Qt.CheckState.Unchecked)
        else:
            self._set_note_types_checked(True)
        # Search all / use first field
        self.search_all_fields_cb.setChecked(bool(ntf.get('search_all_fields', False)))
        self._on_search_all_fields_toggled()
        self.use_first_field_cb.setChecked(bool(ntf.get('use_first_field_fallback', True)))
        # Fields by note type (default to Text+Extra when neither note_type_fields nor fields_to_search)
        ntf_fields = ntf.get('note_type_fields') or {}
        default_flds = None
        if not ntf_fields and not ntf.get('fields_to_search'):
            default_flds = {'text', 'extra'}
        for model_name, cbs in self._field_cbs.items():
            wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))
            if not wanted and default_flds:
                wanted = default_flds
            for fn, cb in cbs.items():
                cb.setChecked(fn.lower() in wanted)
        # Decks (block signals so programmatic setCheckState doesn't trigger persist)
        deck_list = ntf.get('enabled_decks')
        # Interpretation:
        #   None        -> include all decks
        #   [] (empty)  -> no decks selected (all unchecked)
        #   [names]     -> only these decks checked
        include_all_d = (deck_list is None)
        self.include_all_decks_cb.blockSignals(True)
        self.include_all_decks_cb.setChecked(include_all_d)
        self.include_all_decks_cb.blockSignals(False)
        self._on_include_all_decks_toggled()
        if hasattr(self, 'decks_list') and self.decks_list:
            self.decks_list.blockSignals(True)
        try:
            if include_all_d:
                self._set_decks_checked(True)
            elif deck_list:
                ds = set(deck_list)
                for it in self._iterate_all_deck_items():
                    deck_name = it.data(0, Qt.ItemDataRole.UserRole) or it.text(0)
                    it.setCheckState(0, Qt.CheckState.Checked if (deck_name in ds) else Qt.CheckState.Unchecked)
            else:
                # Empty list: user chose no decks (all unchecked)
                self._set_decks_checked(False)
        finally:
            if hasattr(self, 'decks_list') and self.decks_list:
                self.decks_list.blockSignals(False)
        self._update_field_groups_enabled()
    
    def _update_field_groups_enabled(self):
        """Grey out field GroupBoxes whose note type is unchecked in the note types table."""
        if not getattr(self, '_field_groupboxes', None):
            return
        include_all = self.include_all_note_types_cb.isChecked()
        if include_all:
            included = None
        else:
            included = set()
            for i in range(self.note_types_table.rowCount()):
                it = self.note_types_table.item(i, 0)
                if it and it.checkState() == Qt.CheckState.Checked:
                    name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()
                    included.add(name)
        for model_name, gb in self._field_groupboxes.items():
            gb.setEnabled(included is None or model_name in included)
    
    def _on_include_all_note_types_toggled(self):
        self.note_types_table.setEnabled(not self.include_all_note_types_cb.isChecked())
        self._update_field_groups_enabled()
    
    def _on_sort_note_types_changed(self, index):
        """Handle sort combo box change."""
        data = self.sort_combo.itemData(index)
        if data == "count_desc":
            self.note_types_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)
        elif data == "count_asc":
            self.note_types_table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
        elif data == "name_asc":
            self.note_types_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        elif data == "name_desc":
            self.note_types_table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    
    def _on_search_all_fields_toggled(self):
        en = not self.search_all_fields_cb.isChecked()
        self.fields_by_note_type_scroll.setEnabled(en)
        for cbs in self._field_cbs.values():
            for cb in cbs.values():
                cb.setEnabled(en)
        self._update_field_groups_enabled()
    
    def _on_include_all_decks_toggled(self):
        self.decks_list.setEnabled(not self.include_all_decks_cb.isChecked())
        # Also disable/enable header if needed
        header = self.decks_list.header()
        if header:
            header.setEnabled(not self.include_all_decks_cb.isChecked())
        self._persist_note_type_filter()
    
    def _persist_note_type_filter(self):
        """Save current Note Types & Fields (decks, note types, fields) to config so changes persist without clicking Save."""
        if getattr(self, '_applying_note_type_config', False):
            return
        try:
            config = load_config()
            config['note_type_filter'] = self._build_ntf_from_ui()
            save_config(config)
        except Exception as e:
            log_debug(f"Error persisting note_type_filter: {e}")
    
    def _on_deck_item_changed(self, item, column):
        """When user toggles a deck checkbox, persist so settings are saved."""
        if column == 0 and item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            self._persist_note_type_filter()
    
    def _set_note_types_checked(self, checked):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.note_types_table.rowCount()):
            it = self.note_types_table.item(i, 0)
            if it:
                it.setCheckState(state)
    
    def _set_decks_checked(self, checked):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for item in self._iterate_all_deck_items():
            item.setCheckState(0, state)
    
    def _get_note_type_fields_from_ui(self):
        out = {}
        for model_name, cbs in self._field_cbs.items():
            sel = [fn for fn, cb in cbs.items() if cb.isChecked()]
            if sel:
                out[model_name] = sel
        return out
    
    def _get_decks_from_ui(self):
        if self.include_all_decks_cb.isChecked():
            return None
        # Get checked deck names from tree widget
        checked_decks = []
        for item in self._iterate_all_deck_items():
            if item.checkState(0) == Qt.CheckState.Checked:
                # Get full deck name from item data
                deck_name = item.data(0, Qt.ItemDataRole.UserRole)
                if deck_name:
                    checked_decks.append(deck_name)
        return checked_decks
    
    def _build_ntf_from_ui(self):
        include_all_nt = self.include_all_note_types_cb.isChecked()
        enabled_nt = None if include_all_nt else [self.note_types_table.item(i, 0).data(Qt.ItemDataRole.UserRole) or self.note_types_table.item(i, 0).text().strip() for i in range(self.note_types_table.rowCount()) if self.note_types_table.item(i, 0) and self.note_types_table.item(i, 0).checkState() == Qt.CheckState.Checked]
        # Preserve enabled_decks from config if deck list not yet populated (async load at 150ms)
        if hasattr(self, 'decks_list') and self.decks_list and self.decks_list.topLevelItemCount() > 0:
            enabled_decks = self._get_decks_from_ui()
        else:
            enabled_decks = load_config().get('note_type_filter', {}).get('enabled_decks')
        return {
            'enabled_note_types': enabled_nt,
            'search_all_fields': self.search_all_fields_cb.isChecked(),
            'note_type_fields': self._get_note_type_fields_from_ui() if not self.search_all_fields_cb.isChecked() else {},
            'use_first_field_fallback': self.use_first_field_cb.isChecked(),
            'enabled_decks': enabled_decks,
        }
    
    def _on_count_notes(self):
        ntf = self._build_ntf_from_ui()
        c = count_notes_matching_config(ntf)
        showInfo(f"With current settings, about {c} notes would be searched.")
    
    def _on_save_preset(self):
        name = self.preset_name_edit.text().strip()
        if not name:
            showInfo("Enter a preset name.")
            return
        config = load_config()
        presets = config.get('saved_presets') or {}
        presets[name] = self._build_ntf_from_ui()
        config['saved_presets'] = presets
        if save_config(config):
            self.preset_name_edit.clear()
            self._refresh_preset_combos()
            showInfo(f"Preset '{name}' saved.")
    
    def _on_load_preset(self):
        name = self.load_preset_combo.currentData()
        if not name:
            showInfo("Select a preset to load.")
            return
        config = load_config()
        presets = config.get('saved_presets') or {}
        if name not in presets:
            showInfo("Preset not found.")
            return
        self._apply_note_type_config(presets[name])
        showInfo(f"Loaded preset '{name}'.")
    
    def _on_delete_preset(self):
        name = self.delete_preset_combo.currentData()
        if not name:
            showInfo("Select a preset to delete.")
            return
        config = load_config()
        presets = config.get('saved_presets') or {}
        if name in presets:
            del presets[name]
            config['saved_presets'] = presets
            save_config(config)
            self._refresh_preset_combos()
            showInfo(f"Preset '{name}' deleted.")
    
    def _refresh_note_type_lists(self):
        """Repopulate all lists and preserve checked state where possible."""
        checked_nt = [self.note_types_table.item(i, 0).data(Qt.ItemDataRole.UserRole) or self.note_types_table.item(i, 0).text().strip() for i in range(self.note_types_table.rowCount()) if self.note_types_table.item(i, 0) and self.note_types_table.item(i, 0).checkState() == Qt.CheckState.Checked]
        ntf_prev = self._get_note_type_fields_from_ui()
        checked_decks = self._get_decks_from_ui()
        self._populate_note_type_lists()
        for i in range(self.note_types_table.rowCount()):
            it = self.note_types_table.item(i, 0)
            if it:
                name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()
                if name in checked_nt:
                    it.setCheckState(Qt.CheckState.Checked)
        self._populate_fields_by_note_type()
        for model_name, cbs in self._field_cbs.items():
            wanted = set(f.lower() for f in (ntf_prev.get(model_name) or []))
            for fn, cb in cbs.items():
                if fn.lower() in wanted:
                    cb.setChecked(True)
        self._populate_decks_list()
        if checked_decks:
            ds = set(checked_decks)
            for it in self._iterate_all_deck_items():
                deck_name = it.data(0, Qt.ItemDataRole.UserRole) or it.text(0)
                it.setCheckState(0, Qt.CheckState.Checked if (deck_name in ds) else Qt.CheckState.Unchecked)
        else:
            self._set_decks_checked(True)
        self._refresh_preset_combos()
        showInfo("Lists refreshed.")
    
    def detect_provider(self):
        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.provider_label.hide()
            self.url_widget.hide()
            return ""
        
        if api_key.startswith("sk-ant-"):
            provider = "Anthropic (Claude)"
            self.url_widget.hide()
        elif api_key.startswith("sk-or-"):
            provider = "OpenRouter"
            self.url_widget.hide()
        elif api_key.startswith("sk-"):
            provider = "OpenAI (GPT)"
            self.url_widget.hide()
        elif api_key.startswith("AI"):
            provider = "Google (Gemini)"
            self.url_widget.hide()
        else:
            provider = "Custom/Unknown Provider"
            self.url_widget.show()
        
        self.provider_label.setText(f"✓ Detected: {provider}")
        self.provider_label.show()
        return provider
    
    def toggle_key_visibility(self):
        if self.api_key_input.echoMode() == QLineEdit.EchoMode.Password:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_key_btn.setText("Hide")
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_key_btn.setText("Show")
    
    def _toggle_voyage_key_visibility(self):
        if self.voyage_api_key_input.echoMode() == QLineEdit.EchoMode.Password:
            self.voyage_api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.voyage_show_key_btn.setText("Hide")
        else:
            self.voyage_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.voyage_show_key_btn.setText("Show")
    
    def _toggle_openai_key_visibility(self):
        if self.openai_embedding_api_key_input.echoMode() == QLineEdit.EchoMode.Password:
            self.openai_embedding_api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.openai_show_key_btn.setText("Hide")
        else:
            self.openai_embedding_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.openai_show_key_btn.setText("Show")
    
    def _toggle_cohere_key_visibility(self):
        if self.cohere_api_key_input.echoMode() == QLineEdit.EchoMode.Password:
            self.cohere_api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.cohere_show_key_btn.setText("Hide")
        else:
            self.cohere_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.cohere_show_key_btn.setText("Show")
    
    def save_settings(self):
        existing = load_config()
        answer_with = self.answer_provider_combo.currentData() or "api_key"
        if answer_with == "ollama":
            provider_type = "ollama"
            api_key = existing.get("api_key", "") or ""
        else:
            api_key = self.api_key_input.text().strip()
            if not api_key:
                showInfo("Please enter an API key, or choose 'Ollama (local)' for answers.")
                return
            provider_type = self.detect_provider_type(api_key)
            if provider_type == "custom":
                api_url = self.api_url_input.text().strip()
                if not api_url:
                    showInfo("Please enter an API URL for custom provider")
                    return
        
        note_type_filter = self._build_ntf_from_ui()
        
        config = {
            'api_key': api_key,
            'provider': provider_type,
            'styling': {
                'question_font_size': self.question_font_spin.value(),
                'answer_font_size': self.answer_font_spin.value(),
                'notes_font_size': self.notes_font_spin.value(),
                'label_font_size': self.label_font_spin.value(),
                'window_width': self.width_spin.value(),
                'window_height': self.height_spin.value(),
                'section_spacing': self.section_spacing_spin.value(),
                'layout_mode': self.layout_combo.currentData() or 'side_by_side',
                'answer_spacing': self.answer_spacing_combo.currentData() or 'normal'
            },
            'note_type_filter': note_type_filter,
            'search_config': {
                'search_method': self.search_method_combo.currentData() or 'hybrid',  # Default to hybrid for semantic search
                'enable_query_expansion': self.enable_query_expansion_cb.isChecked(),
                'use_ai_generic_term_detection': self.use_ai_generic_term_detection_cb.isChecked(),
                'enable_hyde': self.enable_hyde_cb.isChecked(),
                'enable_rerank': self.enable_rerank_cb.isChecked(),
                'use_context_boost': self.use_context_boost_cb.isChecked(),
                'min_relevance_percent': self.min_relevance_spin.value(),
                'strict_relevance': self.strict_relevance_cb.isChecked(),
                'max_results': self.max_results_spin.value(),
                'context_chars_per_note': self.context_chars_per_note_spin.value(),
                'relevance_from_answer': self.relevance_from_answer_cb.isChecked(),
                'hybrid_embedding_weight': self.hybrid_weight_spin.value(),
                # Extra generic words to ignore as primary query keywords and in matching-term tooltips
                # Stored as a list of lowercased strings in config
                'extra_stop_words': [
                    w.strip().lower()
                    for w in (self.extra_stop_words_input.text() or "").split(",")
                    if w.strip()
                ],
                'verbose_search_debug': self.verbose_search_debug_cb.isChecked(),
                'embedding_engine': self.embedding_engine_combo.currentData() or 'voyage',
                'voyage_api_key': (self.voyage_api_key_input.text() or '').strip(),
                'voyage_embedding_model': (self.voyage_embedding_model_combo.currentData() or 'voyage-3.5-lite').strip(),
                'openai_embedding_api_key': (self.openai_embedding_api_key_input.text() or '').strip(),
                'openai_embedding_model': (self.openai_embedding_model_input.text() or 'text-embedding-3-small').strip(),
                'cohere_api_key': (self.cohere_api_key_input.text() or '').strip(),
                'cohere_embedding_model': (self.cohere_embedding_model_input.text() or 'embed-english-v3.0').strip(),
                'voyage_batch_size': int(self.voyage_batch_size_spin.value()),
                'ollama_base_url': (self.ollama_base_url_input.text() or "http://localhost:11434").strip(),
                'ollama_embed_model': (self.ollama_embed_model_combo.currentText() or "nomic-embed-text").strip(),
                'ollama_batch_size': int(self.ollama_batch_size_spin.value()),
                'use_dynamic_batch_size': self.use_dynamic_batch_size_cb.isChecked(),
                'ollama_chat_model': (self.ollama_chat_model_combo.currentText() or "llama3.2").strip(),
                'rerank_python_path': (self.rerank_python_path_input.text() or '').strip() or None,
            }
        }
        # Preserve search_config keys not edited in Settings (e.g. sensitivity_percent)
        if 'search_config' in existing:
            for k, v in existing['search_config'].items():
                if k not in config['search_config']:
                    config['search_config'][k] = v
        if 'saved_presets' in existing:
            config['saved_presets'] = existing['saved_presets']
        
        if provider_type == "custom":
            config['api_url'] = self.api_url_input.text().strip()
        
        if save_config(config):
            provider_label = "Ollama (local)" if provider_type == "ollama" else self.detect_provider()
            showInfo(f"Settings saved!\nProvider: {provider_label}")
            self.accept()
        else:
            showInfo("Error saving settings")
    
    def _on_answer_provider_changed(self):
        """Show/hide API key vs Ollama answer options."""
        use_ollama = (self.answer_provider_combo.currentData() or "") == "ollama"
        self.api_key_section.setVisible(not use_ollama)
        self.ollama_answer_section.setVisible(use_ollama)
    
    def _refresh_ollama_chat_models(self):
        """Fetch chat models from Ollama (uses URL from Search & Embeddings tab) and populate combo."""
        base_url = (self.ollama_base_url_input.text() or "http://localhost:11434").strip()
        current = self.ollama_chat_model_combo.currentText().strip() or "llama3.2"
        try:
            names = get_ollama_models(base_url)
            self.ollama_chat_model_combo.clear()
            self.ollama_chat_model_combo.addItems(names)
            if current and current not in names:
                self.ollama_chat_model_combo.insertItem(0, current)
                self.ollama_chat_model_combo.setCurrentIndex(0)
            elif current in names:
                idx = self.ollama_chat_model_combo.findText(current)
                if idx >= 0:
                    self.ollama_chat_model_combo.setCurrentIndex(idx)
            if not names:
                self.ollama_chat_model_combo.setCurrentText(current or "llama3.2")
            if names:
                showInfo(f"Found {len(names)} model(s). Choose a chat model for AI answers (e.g. llama3.2, mistral).")
            else:
                showInfo("No models from Ollama. Ensure Ollama is running and the URL in Search & Embeddings tab is correct. You can type a model name (e.g. llama3.2).")
        except Exception as e:
            showInfo(f"Could not fetch models: {e}\n\nCheck that Ollama is running. You can type a model name (e.g. llama3.2).")
    
    def _test_ollama_connection(self):
        """Test Ollama connection (answer provider). Shows pass/fail with latency."""
        import time
        base_url = (self.ollama_base_url_input.text() or "http://localhost:11434").strip()
        try:
            t0 = time.perf_counter()
            names = get_ollama_models(base_url)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            if names:
                showInfo(f"✅ Ollama connection OK\n\nLatency: {elapsed_ms} ms\nModels: {len(names)} available")
            else:
                showInfo("⚠️ Ollama responded but no models found. Run 'ollama pull <model>' to install.")
        except Exception as e:
            showInfo(f"❌ Ollama test failed\n\nError: {e}\n\nMake sure Ollama is running (ollama serve) and the URL is correct.")
    
    def detect_provider_type(self, api_key):
        if api_key.startswith("sk-ant-"):
            return "anthropic"
        elif api_key.startswith("sk-or-"):
            return "openrouter"
        elif api_key.startswith("sk-"):
            return "openai"
        elif api_key.startswith("AI"):
            return "google"
        else:
            return "custom"
    
    def _check_rerank_available(self, extra_path=None, python_path=None):
        """Check if sentence-transformers CrossEncoder is available.
        If python_path is set (path to python.exe), use that Python for the check.
        Else if extra_path is set, run Anki's Python with that folder on sys.path.
        Else run Anki's Python."""
        try:
            import os
            import subprocess
            import sys
            # Prefer user's Python (e.g. Python 3.11) when set
            if python_path:
                python_path = python_path.strip()
                # Allow folder or executable: if folder, append python.exe on Windows
                if os.path.isdir(python_path):
                    python_exe = os.path.join(python_path, "python.exe")
                    if not os.path.isfile(python_exe):
                        python_exe = os.path.join(python_path, "python")
                    python_path = python_exe if os.path.isfile(python_exe) else python_path
                if not os.path.isfile(python_path):
                    return False
                result = subprocess.run(
                    [python_path, "-c", "from sentence_transformers import CrossEncoder; print('ok')"],
                    capture_output=True, text=True, timeout=30,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
                return result.returncode == 0 and 'ok' in (result.stdout or '')
            env = os.environ.copy()
            if extra_path and os.path.isdir(extra_path):
                check_script = (
                    "import sys, os; "
                    "p = os.environ.get('AI_SEARCH_ST_PATH', ''); "
                    "p and sys.path.insert(0, p); "
                    "from sentence_transformers import CrossEncoder; "
                    "print('ok')"
                )
                env['AI_SEARCH_ST_PATH'] = extra_path
            else:
                check_script = "from sentence_transformers import CrossEncoder; print('ok')"
            result = subprocess.run(
                [sys.executable, "-c", check_script],
                capture_output=True, text=True, timeout=15, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            return result.returncode == 0 and 'ok' in (result.stdout or '')
        except Exception:
            return False
    
    def _update_rerank_tooltip(self):
        """Update Cross-Encoder checkbox tooltip with status and (if unavailable) Python path."""
        import sys
        base = "Re-ranks top 15 results with a cross-encoder for 10-30% better relevance.\n"
        if self._rerank_available:
            self.enable_rerank_cb.setToolTip(base + "Ready to use.")
        else:
            self.enable_rerank_cb.setToolTip(
                base + "Not installed. Set 'Python for Cross-Encoder' to your Python (e.g. Python 3.11) that has "
                "sentence-transformers, or click 'Install Dependencies' to install into Anki's Python:\n" + sys.executable
            )
    
    def _on_check_rerank_again(self):
        """Re-check sentence-transformers and update Cross-Encoder checkbox state and tooltip."""
        import sys
        import time
        t0 = time.time()
        python_path = (self.rerank_python_path_input.text() or '').strip() or None
        self._rerank_available = self._check_rerank_available(python_path=python_path)
        self.enable_rerank_cb.setEnabled(self._rerank_available)
        self._update_rerank_tooltip()
        if self._rerank_available:
            showInfo("sentence-transformers is available. Cross-Encoder Re-Ranking can be enabled.")
        else:
            msg = (
                "sentence-transformers not found.\n\n"
                "Option A — Use your own Python (e.g. Python 3.11):\n"
                "1. Set 'Python for Cross-Encoder' above to that python.exe (or its folder).\n"
                "2. Click 'Install into external Python' to install sentence-transformers there.\n"
                "3. Click 'Check again'.\n\n"
                "Option B — Use Anki's Python:\n"
                "Clear the optional path, click 'Install Dependencies', then 'Check again'.\n\n"
                "Anki's Python: " + sys.executable
            )
            showInfo(msg)
    
    def _on_install_into_external_python(self):
        """Install sentence-transformers into the Python set in 'Python for Cross-Encoder'."""
        path = (self.rerank_python_path_input.text() or '').strip()
        if not path:
            showInfo("Enter a path in 'Python for Cross-Encoder' (python.exe or its folder), then try again.")
            return
        python_exe = _resolve_external_python_exe(path)
        if not python_exe:
            showInfo(f"Path not found or not a valid Python:\n{path}\n\nEnter the path to python.exe or the folder containing it.")
            return
        install_dependencies(python_exe=python_exe)
    
    def _on_search_method_changed(self):
        """Show/hide Cloud Embeddings and options based on search method."""
        method = self.search_method_combo.currentData() or "hybrid"
        self.embedding_group.setVisible(method in ("embedding", "hybrid"))
        # HyDE only applies to embedding/hybrid
        self.enable_hyde_cb.setVisible(method in ("embedding", "hybrid"))
        # Hybrid weight: used in weighted RRF (α_lexical * 1/(k+r_kw) + α_dense * 1/(k+r_emb))
        self.hybrid_weight_label.setVisible(method == "hybrid")
        self.hybrid_weight_spin.setVisible(method == "hybrid")
    
    def _refresh_embedding_status(self):
        """Check and display embedding status for the configured engine (Voyage, OpenAI, Cohere, or Ollama)."""
        try:
            config = load_config()
            sc = config.get('search_config') or {}
            engine = sc.get('embedding_engine') or 'voyage'
            if engine == 'ollama':
                base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()
                model = (sc.get('ollama_embed_model') or 'nomic-embed-text').strip()
                status_text = (
                    f"✅ Embeddings: Ollama (local)\n\n"
                    f"URL: {base_url}\n"
                    f"Model: {model}\n\n"
                    f"Ensure Ollama is running (ollama serve) and the model is pulled (ollama pull {model})."
                )
            elif engine == "voyage":
                import os
                api_key = (sc.get("voyage_api_key") or "").strip() or os.environ.get("VOYAGE_API_KEY", "").strip()
                if not api_key:
                    status_text = (
                        "❌ Voyage embeddings disabled\n\n"
                        "Enter your Voyage API key above, or set VOYAGE_API_KEY, or switch provider."
                    )
                else:
                    source = "settings" if (sc.get("voyage_api_key") or "").strip() else "VOYAGE_API_KEY"
                    status_text = (
                        "✅ Embeddings: Voyage AI (cloud)\n\n"
                        f"Using API key from {source}. Create/update embeddings from this dialog."
                    )
            elif engine == "openai":
                import os
                api_key = (sc.get("openai_embedding_api_key") or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
                model = (sc.get("openai_embedding_model") or "text-embedding-3-small").strip()
                if not api_key:
                    status_text = (
                        "❌ OpenAI embeddings disabled\n\n"
                        "Enter your OpenAI API key above, or set OPENAI_API_KEY, or switch provider."
                    )
                else:
                    status_text = (
                        f"✅ Embeddings: OpenAI (cloud)\n\n"
                        f"Model: {model}. Create/update embeddings from this dialog."
                    )
            else:  # cohere
                import os
                api_key = (sc.get("cohere_api_key") or "").strip() or os.environ.get("COHERE_API_KEY", "").strip()
                model = (sc.get("cohere_embedding_model") or "embed-english-v3.0").strip()
                if not api_key:
                    status_text = (
                        "❌ Cohere embeddings disabled\n\n"
                        "Enter your Cohere API key above, or set COHERE_API_KEY, or switch provider."
                    )
                else:
                    status_text = (
                        f"✅ Embeddings: Cohere (cloud)\n\n"
                        f"Model: {model}. Create/update embeddings from this dialog."
                    )
        except Exception as e:
            status_text = f"❌ Error checking status: {str(e)[:100]}"
        
        self.embedding_status_label.setText(status_text)
        palette = QApplication.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
        if is_dark:
            self.embedding_status_label.setStyleSheet("padding: 10px; background-color: #2d2d2d; color: #e0e0e0; border-radius: 5px;")
        else:
            self.embedding_status_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; color: #333333; border-radius: 5px;")
    
    def _start_embedding_service(self):
        """Start the embedding service in a separate process"""
        import subprocess
        import sys
        import os
        import urllib.request
        import json
        import time
        
        # Local embedding service is no longer supported.
        # This method is kept only to avoid breaking older configs.
        showInfo(
            "The local embedding service has been removed.\n\n"
            "This addon now uses only the cloud embeddings API (Voyage) for semantic search.\n"
            "You can generate embeddings via the 'Create/Update Embeddings' button."
        )
        return
        
        # Check if process is already running
        if self.service_process is not None:
            if sys.platform == 'win32':
                # On Windows, we can't easily check if process is running
                # Just check via HTTP below
                pass
            elif hasattr(self.service_process, 'poll') and self.service_process.poll() is None:
                showInfo("Service is already starting. Please wait...")
                return
            else:
                # Process has ended, reset reference
                self.service_process = None
        
        # Get addon directory
        addon_dir = os.path.dirname(__file__)
        
        # Try to start embedding_service.py first (real service)
        service_file = os.path.join(addon_dir, "embedding_service.py")
        fallback_file = os.path.join(addon_dir, "simple_embedding_server.py")
        
        service_script = None
        service_name = None
        
        if os.path.exists(service_file):
            service_script = service_file
            service_name = "embedding_service.py (Real Service)"
        elif os.path.exists(fallback_file):
            service_script = fallback_file
            service_name = "simple_embedding_server.py (Test Server)"
        else:
            showInfo(
                f"❌ Cannot find embedding service files!\n\n"
                f"Expected files:\n"
                f"- {service_file}\n"
                f"- {fallback_file}\n\n"
                f"Please make sure the service files are in the addon directory."
            )
            return
        
        try:
            # Start the service in a new process
            # On Windows, use cmd.exe start to open a new window
            if sys.platform == 'win32':
                # Create a batch file to run the service (handles paths with spaces better)
                import tempfile
                batch_content = f'''@echo off
title Embedding Service
cd /d "{addon_dir}"
echo Starting embedding service...
echo.
"{sys.executable}" "{service_script}"
if errorlevel 1 (
    echo.
    echo Service exited with an error.
    echo Press any key to close this window...
    pause >nul
)
'''
                # Write batch file
                batch_file = os.path.join(addon_dir, "start_embedding_service.bat")
                try:
                    # Ensure directory exists and file can be written
                    os.makedirs(addon_dir, exist_ok=True)
                    with open(batch_file, 'w', encoding='utf-8', newline='\r\n') as f:  # Windows line endings
                        f.write(batch_content)
                    log_debug(f"Created batch file: {batch_file}")
                    
                    # Try multiple methods to start the service, prioritizing simpler ones
                    service_started = False
                    
                    # Method 1: Use VBScript wrapper (handles paths with spaces perfectly)
                    try:
                        vbs_script = os.path.join(addon_dir, "start_service.vbs")
                        vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "{addon_dir}"
WshShell.Run "cmd /k ""{batch_file}""", 1, False
Set WshShell = Nothing
'''
                        with open(vbs_script, 'w', encoding='utf-8') as f:
                            f.write(vbs_content)
                        
                        # Execute VBScript - this handles paths with spaces automatically
                        subprocess.Popen(['wscript', vbs_script], shell=False)
                        log_debug(f"Started service via VBScript wrapper: {vbs_script}")
                        service_started = True
                        self.service_process = True
                    except Exception as vbs_err:
                        log_debug(f"Method 1 (VBScript) failed: {vbs_err}")
                    
                    # Method 2: Direct batch file execution using subprocess with shell=True
                    if not service_started:
                        try:
                            # Use shell=True which handles path quoting automatically
                            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                                subprocess.Popen(
                                    f'cmd /c start cmd /k "{batch_file}"',
                                    shell=True,
                                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                                    cwd=addon_dir
                                )
                            else:
                                subprocess.Popen(
                                    f'cmd /c start cmd /k "{batch_file}"',
                                    shell=True,
                                    cwd=addon_dir
                                )
                            log_debug(f"Started service via batch file (shell=True): {batch_file}")
                            service_started = True
                            self.service_process = True
                        except Exception as batch_err:
                            log_debug(f"Method 2 (batch file shell) failed: {batch_err}")
                    
                    # Method 3: Direct Python execution with shell=True
                    if not service_started:
                        try:
                            # Use shell=True for automatic path handling
                            cmd_str = f'cmd /c start cmd /k "cd /d "{addon_dir}" && "{sys.executable}" "{service_script}""'
                            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                                subprocess.Popen(
                                    cmd_str,
                                    shell=True,
                                    creationflags=subprocess.CREATE_NEW_CONSOLE
                                )
                            else:
                                subprocess.Popen(cmd_str, shell=True)
                            log_debug(f"Started service via direct Python execution (shell=True)")
                            service_started = True
                            self.service_process = True
                        except Exception as direct_err:
                            log_debug(f"Method 3 (direct Python shell) failed: {direct_err}")
                    
                    # Method 4: os.startfile (Windows only, simplest but no console)
                    if not service_started and sys.platform == 'win32':
                        try:
                            os.startfile(batch_file)
                            log_debug(f"Started service via os.startfile")
                            service_started = True
                            self.service_process = True
                        except Exception as startfile_err:
                            log_debug(f"Method 4 (os.startfile) failed: {startfile_err}")
                    
                    # Method 5: PowerShell as last resort (only if admin needed)
                    if not service_started:
                        try:
                            # Create PowerShell script with proper escaping
                            ps_script = os.path.join(addon_dir, "start_embedding_service_admin.ps1")
                            # Escape backslashes and quotes properly
                            batch_file_escaped = batch_file.replace('\\', '\\\\').replace("'", "''")
                            addon_dir_escaped = addon_dir.replace('\\', '\\\\').replace("'", "''")
                            python_exe_escaped = sys.executable.replace('\\', '\\\\').replace("'", "''")
                            service_script_escaped = service_script.replace('\\', '\\\\').replace("'", "''")
                            
                            # Use $PSScriptRoot for dynamic path (works regardless of folder name)
                            ps_content = f'''# PowerShell script to start embedding service with admin privileges if needed
# This script uses $PSScriptRoot to get the directory where the script is located (works regardless of folder name)
$ErrorActionPreference = "Continue"
$scriptDir = $PSScriptRoot
$batchFile = Join-Path $scriptDir "start_embedding_service.bat"
$pythonExe = '{python_exe_escaped}'
$serviceScript = Join-Path $scriptDir "embedding_service.py"

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {{
    Write-Host "Requesting administrator privileges..."
    $cmd = "Set-Location -LiteralPath '$scriptDir'; & '$pythonExe' '$serviceScript'"
    Start-Process powershell -Verb RunAs -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $cmd
}} else {{
    Write-Host "Running with administrator privileges..."
    Set-Location -LiteralPath $scriptDir
    & $pythonExe $serviceScript
}}
'''
                            with open(ps_script, 'w', encoding='utf-8', newline='\r\n') as f:
                                f.write(ps_content)
                            log_debug(f"Created PowerShell script: {ps_script}")
                            
                            # Execute with shell=True for proper path handling
                            subprocess.Popen(
                                f'powershell -ExecutionPolicy Bypass -File "{ps_script}"',
                                shell=True,
                                cwd=addon_dir
                            )
                            log_debug(f"Started service via PowerShell script")
                            service_started = True
                            self.service_process = True
                        except Exception as ps_err:
                            log_debug(f"Method 5 (PowerShell) failed: {ps_err}")
                    
                    if not service_started:
                        raise Exception("All service startup methods failed. Check debug_log.txt for details.")
                except Exception as batch_error:
                    log_debug(f"Failed to create batch file, trying direct method: {batch_error}")
                    # Fallback: try direct method with explicit window
                    try:
                        # Try direct Python execution in new console window
                        # Use full path to Python and service script
                        python_exe = sys.executable
                        service_path = service_script
                        # Escape paths with spaces properly
                        if ' ' in python_exe:
                            python_exe = f'"{python_exe}"'
                        if ' ' in service_path:
                            service_path = f'"{service_path}"'
                        
                        # Create a command that changes directory and runs Python
                        cmd_str = f'cd /d "{addon_dir}" && {python_exe} {service_path}'
                        log_debug(f"Starting service with command: {cmd_str}")
                        
                        # Use CREATE_NEW_CONSOLE flag to ensure new window
                        if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                            subprocess.Popen(
                                ['cmd', '/c', 'start', 'cmd', '/k', cmd_str],
                                shell=False,
                                creationflags=subprocess.CREATE_NEW_CONSOLE
                            )
                        else:
                            subprocess.Popen(
                                ['cmd', '/c', 'start', 'cmd', '/k', cmd_str],
                                shell=True
                            )
                        self.service_process = True
                        log_debug("Service started via direct command method")
                    except Exception as direct_error:
                        log_debug(f"Direct method also failed: {direct_error}")
                        # Last resort: try PowerShell
                        try:
                            ps_cmd = f'Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd \'{addon_dir}\'; & \'{sys.executable}\' \'{service_script}\'"'
                            subprocess.Popen(['powershell', '-Command', ps_cmd], shell=False)
                            self.service_process = True
                            log_debug("Service started via PowerShell method")
                        except Exception as ps_error:
                            log_debug(f"PowerShell method also failed: {ps_error}")
                            raise Exception(f"All service startup methods failed. Last error: {ps_error}")
            else:
                self.service_process = subprocess.Popen(
                    [sys.executable, service_script],
                    cwd=addon_dir
                )
            
            # Wait a moment for the service to start
            time.sleep(3)
            
            # Check if service is responding (more reliable than checking process)
            # On Windows, we can't easily track the detached process, so we check HTTP
            if sys.platform == 'win32':
                # For Windows, we check via HTTP instead of process polling
                pass  # Will check below
            elif self.service_process.poll() is not None:
                # Process has already terminated (error starting) - only for non-Windows
                showInfo(
                    f"❌ Failed to start service!\n\n"
                    f"Service: {service_name}\n\n"
                    f"The service process exited immediately. Check the console window for error messages.\n\n"
                    f"Common issues:\n"
                    f"- Missing dependencies (pip install flask sentence-transformers)\n"
                    f"- Port 9000 already in use\n"
                    f"- Python path issues"
                )
                self.service_process = None
                return
            
            # Test if service is responding
            try:
                test_data = json.dumps({"text": "test"}).encode('utf-8')
                test_req = urllib.request.Request(url, test_data, {"Content-Type": "application/json"})
                urllib.request.urlopen(test_req, timeout=3)
                
                showInfo(
                    f"✅ Service started successfully!\n\n"
                    f"Service: {service_name}\n"
                    f"URL: {url}\n\n"
                    f"A console window has been opened showing the service output.\n"
                    f"Keep this window open while using the embedding service."
                )
                # Refresh status
                QTimer.singleShot(500, self._refresh_embedding_status)
            except Exception as e:
                # Service started but not responding yet
                showInfo(
                    f"⚠️ Service process started but not responding yet.\n\n"
                    f"Service: {service_name}\n"
                    f"URL: {url}\n\n"
                    f"Please wait a few seconds and click '🔌 Test Connection' to verify.\n\n"
                    f"If the service doesn't start, check the console window for errors."
                )
                # Refresh status after a delay
                QTimer.singleShot(3000, self._refresh_embedding_status)
                
        except Exception as e:
            showInfo(
                f"❌ Error starting service!\n\n"
                f"Service: {service_name}\n"
                f"Error: {str(e)}\n\n"
                f"Please check:\n"
                f"- Python is installed and in PATH\n"
                f"- Service file exists: {service_script}\n"
                f"- Required dependencies are installed"
            )
            self.service_process = None
    
    def _test_embedding_connection(self):
        """Test connection to the selected embedding engine (Voyage, OpenAI, Cohere, or Ollama). Shows pass/fail with latency."""
        import time
        test_text = "Test connection"
        engine = self.embedding_engine_combo.currentData() or "voyage"
        sc = {
            "embedding_engine": engine,
            "voyage_api_key": (self.voyage_api_key_input.text() or "").strip(),
            "voyage_embedding_model": (self.voyage_embedding_model_combo.currentData() or "voyage-3.5-lite"),
            "openai_embedding_api_key": (self.openai_embedding_api_key_input.text() or "").strip(),
            "openai_embedding_model": (self.openai_embedding_model_input.text() or "text-embedding-3-small").strip(),
            "cohere_api_key": (self.cohere_api_key_input.text() or "").strip(),
            "cohere_embedding_model": (self.cohere_embedding_model_input.text() or "embed-english-v3.0").strip(),
            "voyage_batch_size": self.voyage_batch_size_spin.value(),
            "ollama_base_url": (self.ollama_base_url_input.text() or "http://localhost:11434").strip(),
            "ollama_embed_model": (self.ollama_embed_model_combo.currentText() or "nomic-embed-text").strip(),
            "ollama_batch_size": self.ollama_batch_size_spin.value(),
        }
        config = {"search_config": sc}
        if hasattr(self, 'embedding_status_label') and self.embedding_status_label:
            self.embedding_status_label.setText("Testing connection...")
            QApplication.processEvents()
        try:
            t0 = time.perf_counter()
            embedding = get_embedding_for_query(test_text, config=config)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            dim = len(embedding) if embedding else 0
            if embedding and dim > 0:
                engine_names = {"ollama": "Ollama", "voyage": "Voyage AI", "openai": "OpenAI", "cohere": "Cohere"}
                engine_name = engine_names.get(engine, engine)
                showInfo(
                    f"✅ Embedding connection OK — {engine_name}\n\n"
                    f"Dimension: {dim} | Latency: {elapsed_ms} ms"
                )
                if hasattr(self, 'embedding_status_label') and self.embedding_status_label:
                    self.embedding_status_label.setText(f"✅ {engine_name} OK ({elapsed_ms} ms)")
            else:
                showInfo(
                    "⚠️ Connection succeeded but received an empty embedding.\n\n"
                    "Check your engine settings (URL/model or API key) and try again."
                )
                if hasattr(self, 'embedding_status_label') and self.embedding_status_label:
                    self.embedding_status_label.setText("⚠️ Empty embedding")
        except Exception as e:
            if hasattr(self, 'embedding_status_label') and self.embedding_status_label:
                self.embedding_status_label.setText("❌ Test failed")
            if engine == "ollama":
                hint = "Make sure Ollama is running (ollama serve) and the model is pulled (e.g. ollama pull nomic-embed-text)."
            elif engine == "openai":
                hint = "Enter your OpenAI API key above (or set OPENAI_API_KEY) and check internet access."
            elif engine == "cohere":
                hint = "Enter your Cohere API key above (or set COHERE_API_KEY) and check internet access."
            else:
                hint = "Enter your API key above (or set the provider's env var) and check internet access."
            showInfo(
                f"❌ Embedding test failed!\n\n"
                f"Error: {e}\n\n"
                f"{hint}"
            )
    
    def _migrate_json_to_db(self):
        """Copy embeddings from legacy JSON file into SQLite DB (no re-embedding)."""
        if hasattr(self, 'embedding_status_label') and self.embedding_status_label:
            self.embedding_status_label.setText("Migrating JSON → database...")
            QApplication.processEvents()
        try:
            count, err = migrate_embeddings_json_to_db()
            if err:
                showInfo(f"Migration could not complete.\n\n{err}")
                if hasattr(self, 'embedding_status_label') and self.embedding_status_label:
                    self.embedding_status_label.setText("")
                return
            showInfo(f"Migrated {count} embeddings from the old JSON file into the database.\n\nYou can keep or delete the old .json file; new data is now in the .db file.")
            if hasattr(self, 'embedding_status_label') and self.embedding_status_label:
                self.embedding_status_label.setText(f"Migrated {count} embeddings to database.")
            try:
                QTimer.singleShot(100, self._refresh_embedding_status)
            except Exception:
                pass
        except Exception as e:
            showInfo(f"Migration failed: {e}")
            if hasattr(self, 'embedding_status_label') and self.embedding_status_label:
                self.embedding_status_label.setText("")
    
    def _create_or_update_embeddings(self):
        """Create or update embeddings for all notes using the selected engine (Voyage, OpenAI, Cohere, or Ollama)."""
        # Persist current UI engine/URL/model so worker uses them (user may have changed without saving dialog)
        config = load_config()
        sc = dict(config.get('search_config') or {})
        sc['embedding_engine'] = self.embedding_engine_combo.currentData() or 'voyage'
        sc['voyage_api_key'] = (self.voyage_api_key_input.text() or '').strip()
        sc['voyage_embedding_model'] = (self.voyage_embedding_model_combo.currentData() or 'voyage-3.5-lite')
        sc['openai_embedding_api_key'] = (self.openai_embedding_api_key_input.text() or '').strip()
        sc['openai_embedding_model'] = (self.openai_embedding_model_input.text() or 'text-embedding-3-small').strip()
        sc['cohere_api_key'] = (self.cohere_api_key_input.text() or '').strip()
        sc['cohere_embedding_model'] = (self.cohere_embedding_model_input.text() or 'embed-english-v3.0').strip()
        sc['voyage_batch_size'] = int(self.voyage_batch_size_spin.value())
        sc['ollama_base_url'] = (self.ollama_base_url_input.text() or "http://localhost:11434").strip()
        sc['ollama_embed_model'] = (self.ollama_embed_model_combo.currentText() or "nomic-embed-text").strip()
        sc['ollama_batch_size'] = int(self.ollama_batch_size_spin.value())
        sc['use_dynamic_batch_size'] = self.use_dynamic_batch_size_cb.isChecked()
        config['search_config'] = sc
        save_config(config)  # persist batch size and embedding settings
        engine = sc.get('embedding_engine') or 'voyage'
        # For Ollama: verify model is available before starting (Ollama loads models on first request)
        if engine == 'ollama':
            base_url = sc.get('ollama_base_url') or 'http://localhost:11434'
            model = sc.get('ollama_embed_model') or 'nomic-embed-text'
            model_base = model.split(':')[0]  # nomic-embed-text:latest -> nomic-embed-text
            try:
                names = get_ollama_models(base_url.strip())
                if not names:
                    showInfo(
                        "❌ Ollama returned no models.\n\n"
                        "Make sure Ollama is running (ollama serve) and you have pulled at least one model.\n\n"
                        f"For embeddings, run: ollama pull {model_base}"
                    )
                    return
                # Check if our model (or base name) is available
                if not any(model_base in n or n.startswith(model_base) for n in names):
                    showInfo(
                        f"❌ Ollama embedding model '{model_base}' not found.\n\n"
                        f"Available models: {', '.join(names[:8])}{'...' if len(names) > 8 else ''}\n\n"
                        f"Run: ollama pull {model_base}"
                    )
                    return
            except Exception as e:
                showInfo(
                    f"❌ Cannot reach Ollama at {base_url}\n\n"
                    f"Error: {e}\n\n"
                    "Make sure Ollama is running (ollama serve)."
                )
                return
        # Quick API check first to avoid running a long job with bad config
        try:
            test_embedding = get_embedding_for_query("Test connection")
            if not test_embedding:
                showInfo(
                    "❌ Embedding engine returned an empty result.\n\n"
                    "Check your engine (URL/model or API key) and try again."
                )
                return
        except Exception as e:
            if engine == 'ollama':
                showInfo(
                    f"❌ Ollama embedding test failed.\n\n"
                    f"Error: {e}\n\n"
                    "Make sure Ollama is running (ollama serve) and the model is pulled "
                    f"(e.g. ollama pull {model_base})."
                )
            else:
                showInfo(
                    f"❌ Cannot use embeddings API!\n\n"
                    f"Error: {e}\n\n"
                    "Enter your API key for the selected engine above and check internet access."
                )
            return
        
        # Get note type filter config
        # Always base this on the *current* UI selections so user choices
        # (note types, decks, fields) are remembered between sessions,
        # even if they didn't click the main "Save Settings" button.
        current_ntf = self._build_ntf_from_ui()
        config = load_config()
        config['note_type_filter'] = current_ntf
        # Persist immediately so next Anki restart / addon open uses the same
        # note/deck/field selection.
        save_config(config)
        ntf = current_ntf
        
        # Count notes that will be processed
        note_count = count_notes_matching_config(ntf)
        if note_count == 0:
            showInfo("No notes found to process. Check your note type and deck filters.")
            return
        
        # Check for existing checkpoint (only resume if it was for the same embedding engine)
        checkpoint = load_checkpoint()
        resume_available = False
        current_engine_id = get_embedding_engine_id(config)
        if checkpoint and checkpoint.get('engine_id') != current_engine_id:
            checkpoint = None  # different engine: start fresh, don't offer resume
        if checkpoint:
            processed_count = checkpoint.get('processed_count', 0)
            total_notes = checkpoint.get('total_notes', 0)
            if processed_count > 0 and processed_count < total_notes:
                resume_available = True
                reply = QMessageBox.question(
                    self,
                    "Resume Embedding Generation?",
                    f"Found a previous checkpoint:\n\n"
                    f"Processed: {processed_count:,} / {total_notes:,} notes\n"
                    f"Timestamp: {checkpoint.get('timestamp', 'unknown')}\n\n"
                    f"Would you like to resume from where you left off?\n\n"
                    f"(Click 'No' to start over)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Cancel:
                    return
                elif reply == QMessageBox.StandardButton.No:
                    # Clear checkpoint and start fresh
                    clear_checkpoint()
                    checkpoint = None
                    resume_available = False
        
        if not resume_available:
            reply = QMessageBox.question(
                self,
                "Create/Update Embeddings",
                f"This will generate embeddings for approximately {note_count:,} notes.\n\n"
                f"This may take a while depending on the number of notes.\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Create progress dialog (non-modal so Anki stays responsive)
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Creating Embeddings")
        progress_dialog.setMinimumWidth(500)
        progress_dialog.setMinimumHeight(350)
        progress_dialog.setModal(False)  # Non-modal so user can continue using Anki
        # Add minimize and maximize buttons
        flags = progress_dialog.windowFlags()
        progress_dialog.setWindowFlags(flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)
        progress_layout = QVBoxLayout(progress_dialog)
        
        # Track pause state
        progress_dialog._is_paused = False
        progress_dialog._pause_lock = False
        
        status_label = QLabel("Initializing embedding model...")
        status_label.setWordWrap(True)
        progress_layout.addWidget(status_label)
        
        progress_bar = QProgressBar()
        progress_bar.setRange(0, note_count)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        progress_bar.setFormat("%p%")
        progress_layout.addWidget(progress_bar)
        
        log_text = QTextEdit()
        log_text.setReadOnly(True)
        log_text.setMaximumHeight(200)
        log_text.setFont(QFont("Courier", 9))
        progress_layout.addWidget(log_text)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        pause_button = QPushButton("⏸ Pause")
        pause_button.clicked.connect(lambda: self._toggle_pause(progress_dialog, pause_button, log_text))
        button_layout.addWidget(pause_button)
        
        button_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.setEnabled(False)
        close_button.clicked.connect(progress_dialog.close)
        button_layout.addWidget(close_button)
        
        progress_layout.addLayout(button_layout)
        
        # Store references for worker thread
        progress_dialog._status_label = status_label
        progress_dialog._progress_bar = progress_bar
        progress_dialog._log_text = log_text
        progress_dialog._close_button = close_button
        progress_dialog._pause_button = pause_button
        
        progress_dialog.show()
        QApplication.processEvents()
        
        # Create and start worker thread for embedding (prevents blocking)
        worker = EmbeddingWorker(
            ntf, note_count, checkpoint, resume_available
        )
        
        # Connect worker signals to UI updates
        worker.status_update.connect(status_label.setText)
        worker.progress_update.connect(progress_bar.setValue)
        worker.log_message.connect(log_text.append)
        worker.finished_signal.connect(lambda processed, errors, skipped, still_failed: self._on_embedding_finished(
            progress_dialog, processed, errors, skipped, still_failed, note_count
        ))
        worker.error_signal.connect(lambda msg: self._on_embedding_error(progress_dialog, msg))
        
        # Store worker reference
        progress_dialog._worker = worker
        
        # Start worker thread
        worker.start()
    
    def _toggle_pause(self, progress_dialog, pause_button, log_text):
        """Toggle pause/resume for embedding process"""
        if progress_dialog._is_paused:
            # Resume
            progress_dialog._is_paused = False
            pause_button.setText("⏸ Pause")
            if hasattr(progress_dialog, '_worker') and progress_dialog._worker:
                progress_dialog._worker._is_paused = False
            log_text.append("▶ Resumed processing...")
        else:
            # Pause
            progress_dialog._is_paused = True
            pause_button.setText("▶ Resume")
            if hasattr(progress_dialog, '_worker') and progress_dialog._worker:
                progress_dialog._worker._is_paused = True
            log_text.append("⏸ Paused - Click 'Resume' to continue...")
    
    def _on_embedding_finished(self, progress_dialog, processed, errors, skipped, still_failed_count, note_count):
        """Handle embedding completion"""
        # Invalidate embeddings file cache so next search loads the updated file
        global _embeddings_file_cache, _embeddings_file_cache_path, _embeddings_file_cache_time
        _embeddings_file_cache = None
        _embeddings_file_cache_path = None
        _embeddings_file_cache_time = 0
        status_label = progress_dialog._status_label
        log_text = progress_dialog._log_text
        close_button = progress_dialog._close_button
        
        status_label.setText(f"✅ Completed! Processed {processed:,} notes ({errors} errors)")
        log_text.append(f"\n✅ Embedding generation complete!")
        log_text.append(f"Processed: {processed:,} notes")
        if skipped > 0:
            log_text.append(f"Skipped (already had embeddings): {skipped:,} notes")
        if errors > 0:
            log_text.append(f"Errors: {errors}")
        if still_failed_count > 0:
            log_text.append(f"⚠️ {format_partial_failure_progress(still_failed_count)}")
        
        # Clear checkpoint only when no notes are still missing (so next run is full; missed ones get retried)
        if still_failed_count == 0:
            clear_checkpoint()
        
        close_button.setEnabled(True)
        message = f"Embedding generation complete!\nProcessed: {processed:,} notes\nErrors: {errors}"
        if skipped > 0:
            message += f"\nSkipped (already had embeddings): {skipped:,} notes"
        if still_failed_count > 0:
            message += f"\n\n⚠️ {format_partial_failure_completion(still_failed_count)}"
        showInfo(message)
    
    def _on_embedding_error(self, progress_dialog, error_msg):
        """Handle embedding error"""
        status_label = progress_dialog._status_label
        log_text = progress_dialog._log_text
        close_button = progress_dialog._close_button
        
        status_label.setText(f"❌ Error: {error_msg}")
        log_text.append(f"❌ Error: {error_msg}")
        close_button.setEnabled(True)
        showInfo(f"Error during embedding generation: {error_msg}")

# END OF PART 1 - Continue to PART 2
# PART 2 OF 3 - Continue from PART 1

class AISearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Anki Semantic Search")
        
        config = load_config()
        styling = config.get('styling', {})
        default_width = styling.get('window_width', 1100)
        default_height = styling.get('window_height', 800)
        
        self.setMinimumWidth(1000)
        self.setMinimumHeight(750)
        self.resize(default_width, default_height)
        
        # Behave like a normal window so minimize/maximize work (don't use dialog-only flags)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        
        self.styling_config = styling
        self.sensitivity_slider = None
        
        palette = QApplication.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
        
        if is_dark:
            self.setStyleSheet("""
                QDialog { background-color: #1e1e1e; }
                QLabel { color: #ffffff; }
                QLineEdit, QTextEdit, QPlainTextEdit { padding: 8px; border: 2px solid #3498db; border-radius: 6px; background-color: #2d2d2d; color: #ffffff; font-size: 13px; }
                QPushButton { padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 12px; color: white; }
                QPushButton#searchBtn { background-color: #3498db; border: none; }
                QPushButton#searchBtn:hover { background-color: #5dade2; }
                QPushButton#settingsBtn { background-color: #555555; border: none; padding: 6px 12px; }
                QPushButton#viewBtn { background-color: #27ae60; border: 2px solid #1e8449; color: white; }
                QPushButton#viewBtn:hover { background-color: #2ecc71; border-color: #27ae60; }
                QPushButton#viewBtn:disabled { background-color: #555555; border-color: #444444; color: #888888; }
                QPushButton#viewAllBtn { background-color: #16a085; border: 2px solid #117a65; color: white; }
                QPushButton#viewAllBtn:hover { background-color: #1abc9c; border-color: #16a085; }
                QPushButton#viewAllBtn:disabled { background-color: #555555; border-color: #444444; color: #888888; }
                QPushButton#closeBtn { background-color: #c0392b; border: 2px solid #922b21; color: white; }
                QPushButton#closeBtn:hover { background-color: #e74c3c; border-color: #c0392b; }
                QPushButton#toggleSelectBtn { background-color: #3498db; border: 2px solid #2980b9; color: white; padding: 4px 8px; font-size: 11px; }
                QPushButton#toggleSelectBtn:hover { background-color: #5dade2; border-color: #3498db; }
                QPushButton#toggleSelectBtn:disabled { background-color: #555555; border-color: #444444; color: #888888; }
                QTableWidget { border: 2px solid #3498db; border-radius: 6px; background-color: #2d2d2d; color: #e0e0e0; gridline-color: #444444; alternate-background-color: #252525; }
                QTableWidget::item { padding: 8px; border: none; }
                QTableWidget::item:hover { background-color: #3d3d3d; }
                QTableWidget::item:selected { background-color: #3498db; color: white; }
                QTableWidget::item:selected:hover { background-color: #5dade2; }
                QHeaderView::section { background-color: #1e1e1e; color: #ffffff; padding: 8px; border: 1px solid #444444; font-weight: bold; }
                QSlider::groove:horizontal { border: 1px solid #555555; height: 6px; background: #2d2d2d; border-radius: 3px; }
                QSlider::handle:horizontal { background: #3498db; border: 1px solid #5dade2; width: 14px; margin: -4px 0; border-radius: 7px; }
                QSlider::groove:horizontal { border: 1px solid #555555; height: 8px; background: #2d2d2d; border-radius: 4px; }
                QSlider::handle:horizontal { background: #3498db; border: 1px solid #5dade2; width: 18px; margin: -5px 0; border-radius: 9px; }
                QProgressBar { text-align: center; color: palette(window-text); font-weight: bold; }
                QProgressBar::chunk { background-color: #3498db; border-radius: 3px; }
            """)
        else:
            self.setStyleSheet("""
                QDialog { background-color: #f5f5f5; }
                QLabel { color: #2c3e50; }
                QLineEdit, QTextEdit, QPlainTextEdit { padding: 8px; border: 2px solid #3498db; border-radius: 6px; background-color: white; color: #1a1a1a; font-size: 13px; }
                QPushButton { padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 12px; }
                QPushButton#searchBtn { background-color: #3498db; color: white; border: none; }
                QPushButton#settingsBtn { background-color: #95a5a6; color: white; border: none; }
                QPushButton#viewBtn { background-color: #27ae60; color: white; border: 2px solid #1e8449; }
                QPushButton#viewBtn:hover { background-color: #229954; border-color: #1e8449; }
                QPushButton#viewBtn:disabled { background-color: #bdc3c7; border-color: #95a5a6; color: #7f8c8d; }
                QPushButton#viewAllBtn { background-color: #16a085; color: white; border: 2px solid #117a65; }
                QPushButton#viewAllBtn:hover { background-color: #138d75; border-color: #117a65; }
                QPushButton#viewAllBtn:disabled { background-color: #bdc3c7; border-color: #95a5a6; color: #7f8c8d; }
                QPushButton#closeBtn { background-color: #c0392b; color: white; border: 2px solid #922b21; }
                QPushButton#closeBtn:hover { background-color: #e74c3c; border-color: #c0392b; }
                QPushButton#toggleSelectBtn { background-color: #3498db; color: white; border: 2px solid #2980b9; padding: 4px 8px; font-size: 11px; }
                QPushButton#toggleSelectBtn:hover { background-color: #5dade2; border-color: #3498db; }
                QPushButton#toggleSelectBtn:disabled { background-color: #bdc3c7; border-color: #95a5a6; color: #7f8c8d; }
                QTableWidget { border: 2px solid #3498db; border-radius: 6px; background-color: white; gridline-color: #e0e0e0; alternate-background-color: #f8f9fa; }
                QTableWidget::item { padding: 8px; border: none; }
                QTableWidget::item:hover { background-color: #ebf5fb; }
                QTableWidget::item:selected { background-color: #3498db; color: white; }
                QTableWidget::item:selected:hover { background-color: #5dade2; }
                QHeaderView::section { background-color: #ecf0f1; color: #2c3e50; padding: 8px; border: 1px solid #bdc3c7; font-weight: bold; }
                QSlider::groove:horizontal { border: 1px solid #bdc3c7; height: 6px; background: #ecf0f1; border-radius: 3px; }
                QSlider::handle:horizontal { background: #3498db; border: 1px solid #2980b9; width: 14px; margin: -4px 0; border-radius: 7px; }
                QSlider::groove:horizontal { border: 1px solid #bdc3c7; height: 8px; background: #ecf0f1; border-radius: 4px; }
                QSlider::handle:horizontal { background: #3498db; border: 1px solid #2980b9; width: 18px; margin: -5px 0; border-radius: 9px; }
                QProgressBar { text-align: center; color: palette(window-text); font-weight: bold; }
                QProgressBar::chunk { background-color: #3498db; border-radius: 3px; }
            """)
        self.setup_ui()
    
    def perform_search(self):
        """
        Bridge method so AISearchDialog always has perform_search.
        Delegates to the implementation that was originally defined on EmbeddingSearchWorker.
        """
        return EmbeddingSearchWorker.perform_search(self)

    def toggle_select_all(self):
        """Bridge to shared toggle_select_all implementation."""
        return EmbeddingSearchWorker.toggle_select_all(self)

    def select_all_notes(self):
        """Bridge to shared select_all_notes implementation."""
        return EmbeddingSearchWorker.select_all_notes(self)

    def deselect_all_notes(self):
        """Bridge to shared deselect_all_notes implementation."""
        return EmbeddingSearchWorker.deselect_all_notes(self)
        
    def setup_ui(self):
        layout = QVBoxLayout()
        section_spacing = self.styling_config.get('section_spacing', 8)
        layout.setSpacing(section_spacing)
        layout.setContentsMargins(12, 10, 12, 10)
        
        palette = QApplication.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
        
        # Compact header: title + hint + scope banner + settings (shrunk)
        header_layout = QHBoxLayout()
        title_label = QLabel("🔍 Anki Semantic Search")
        title_label.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {'#ffffff' if is_dark else '#2c3e50'};")
        header_layout.addWidget(title_label)
        hint_label = QLabel("Ask a question → AI answer from your notes")
        hint_label.setStyleSheet(f"font-size: 10px; color: {'#95a5a6' if is_dark else '#7f8c8d'};")
        header_layout.addWidget(hint_label)
        self.scope_banner = QLabel("")
        self.scope_banner.setStyleSheet(f"font-size: 10px; color: {'#95a5a6' if is_dark else '#7f8c8d'}; padding: 2px 8px;")
        self.scope_banner.setToolTip("Search scope. Click Settings to change note types, fields, decks.")
        self.scope_banner.setTextFormat(Qt.TextFormat.RichText)
        self.scope_banner.setOpenExternalLinks(False)
        self.scope_banner.linkActivated.connect(lambda _: self.open_settings())
        header_layout.addWidget(self.scope_banner)
        header_layout.addStretch()
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.setObjectName("settingsBtn")
        settings_btn.setToolTip("Configure API key, note types, decks, and search behavior")
        settings_btn.clicked.connect(self.open_settings)
        header_layout.addWidget(settings_btn)
        layout.addLayout(header_layout)
        
        # Search input (shrunk)
        search_container = QWidget()
        search_container.setStyleSheet("background-color: rgba(52, 152, 219, 0.12); border-radius: 4px; padding: 4px 6px;")
        search_layout = QVBoxLayout(search_container)
        search_layout.setSpacing(2)
        search_layout.setContentsMargins(4, 4, 4, 4)
        
        label_font_size = max(11, self.styling_config.get('label_font_size', 13) - 2)
        search_label = QLabel("Search:")
        search_label.setToolTip("Type a question; matching notes will be found and the AI will answer using them. Ctrl+Enter to search.")
        search_label.setStyleSheet(f"font-weight: bold; font-size: {label_font_size}px; color: {'#ffffff' if is_dark else '#2c3e50'};")
        search_layout.addWidget(search_label)
        
        search_input_layout = QHBoxLayout()
        try:
            self.search_input = SpellCheckPlainTextEdit()
        except Exception:
            self.search_input = QPlainTextEdit()
        self.search_input.setPlaceholderText("e.g., hypertension  or  causes of heart failure — Ctrl+Enter to search")
        self.search_input.setMinimumHeight(44)
        self.search_input.setMaximumHeight(100)
        spell_hint = " Right-click misspelled words for corrections." if _get_spell_checker() else ""
        self.search_input.setToolTip(f"Type your question. Ctrl+Enter to search.{spell_hint}")
        question_font_size = max(11, self.styling_config.get('question_font_size', 13) - 1)
        self.search_input.setStyleSheet(f"font-size: {question_font_size}px;")
        # History dropdown for recent searches
        self._search_history_model = QStringListModel(get_search_history_queries())
        self.search_history_combo = QComboBox()
        self.search_history_combo.setModel(self._search_history_model)
        self.search_history_combo.setEditable(False)
        self.search_history_combo.setMinimumWidth(120)
        self.search_history_combo.setToolTip("Recent searches — select to fill the search box, then click Search.")
        self.search_history_combo.activated.connect(self._on_search_history_selected)
        
        clear_history_btn = QPushButton("Clear history")
        clear_history_btn.setToolTip("Delete all search history")
        clear_history_btn.setMaximumWidth(100)
        clear_history_btn.clicked.connect(self._on_clear_search_history)
        
        self.search_btn = QPushButton("🔍 Search")
        self.search_btn.setObjectName("searchBtn")
        self.search_btn.setMinimumHeight(42)
        self.search_btn.setMinimumWidth(100)
        self.search_btn.clicked.connect(self.perform_search)
        
        search_input_layout.addWidget(self.search_input, 1)
        search_input_layout.addWidget(self.search_history_combo)
        search_input_layout.addWidget(clear_history_btn)
        search_input_layout.addWidget(self.search_btn)
        search_layout.addLayout(search_input_layout)
        
        # Ctrl+Enter to search
        search_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        search_shortcut.activated.connect(self.perform_search)
        
        # Keyboard shortcuts for select/deselect
        select_all_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
        select_all_shortcut.activated.connect(self.select_all_notes)
        deselect_all_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        deselect_all_shortcut.activated.connect(self.deselect_all_notes)
        
        layout.addWidget(search_container)
        
        # Splitter for resizable sections (side-by-side or stacked)
        layout_mode = self.styling_config.get('layout_mode', 'side_by_side')
        self.use_side_by_side = layout_mode == 'side_by_side'
        split_orientation = Qt.Orientation.Horizontal if self.use_side_by_side else Qt.Orientation.Vertical
        main_splitter = QSplitter(split_orientation)
        
        # AI Answer section
        answer_container = QWidget()
        answer_container.setStyleSheet("background-color: rgba(46, 204, 113, 0.12); border-radius: 6px; padding: 8px;")
        answer_layout = QVBoxLayout(answer_container)
        
        answer_header = QHBoxLayout()
        answer_label = QLabel("💡 AI Answer:")
        answer_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #27ae60;")
        answer_header.addWidget(answer_label)
        answer_header.addStretch()
        self.copy_answer_btn = QPushButton("📋 Copy")
        self.copy_answer_btn.setMaximumWidth(80)
        self.copy_answer_btn.setToolTip("Copy AI answer (paste into Word for bullets and formatting)")
        self.copy_answer_btn.clicked.connect(self.copy_answer_to_clipboard)
        self.copy_answer_btn.setEnabled(False)
        answer_header.addWidget(self.copy_answer_btn)
        answer_layout.addLayout(answer_header)
        
        try:
            self.answer_box = QTextBrowser()
        except NameError:
            self.answer_box = QTextEdit()
        self.answer_box.setReadOnly(True)
        self.answer_box.setOpenExternalLinks(False)
        # Ensure link clicks are emitted (not opened) so citation links work (Ollama and all providers)
        if hasattr(self.answer_box, 'setOpenLinks'):
            self.answer_box.setOpenLinks(False)
        if hasattr(self.answer_box, 'setOpenExternalLinks'):
            self.answer_box.setOpenExternalLinks(False)
        # Explicitly enable link interaction so [1], [2] citation links are clickable
        if hasattr(Qt, 'TextInteractionFlag') and hasattr(Qt.TextInteractionFlag, 'TextBrowserInteraction'):
            self.answer_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        elif hasattr(Qt, 'TextBrowserInteraction'):
            self.answer_box.setTextInteractionFlags(Qt.TextBrowserInteraction)
        if hasattr(self.answer_box, 'setPlaceholderText'):
            self.answer_box.setPlaceholderText("Enter a question above and click Search to see an AI answer based on your notes.")
        self.answer_box.setMinimumHeight(100)
        self.answer_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        answer_font_size = self.styling_config.get('answer_font_size', 13)
        self.answer_box.setStyleSheet(
            f"background-color: {'#2d2d2d' if is_dark else '#ffffff'}; "
            f"border: 2px solid #27ae60; color: {'#ffffff' if is_dark else '#1a1a1a'}; "
            f"font-size: {answer_font_size}px; padding: 10px;"
            f"a {{ color: #3498db; text-decoration: underline; }} "
            f"a:hover {{ color: #5dade2; }} "
        )
        # Connect link click: anchorClicked (Qt6) or linkActivated (PyQt5) so citation links work with Ollama and all providers
        if hasattr(self.answer_box, 'anchorClicked'):
            self.answer_box.anchorClicked.connect(self._on_answer_link_clicked)
        elif hasattr(self.answer_box, 'linkActivated'):
            self.answer_box.linkActivated.connect(self._on_answer_link_clicked)
        answer_layout.addWidget(self.answer_box)
        
        # Hint: where the answer came from (API name or local model)
        self.answer_source_label = QLabel("")
        self.answer_source_label.setStyleSheet(f"font-size: 11px; color: {'#95a5a6' if is_dark else '#7f8c8d'}; font-style: italic; margin-top: 4px;")
        self.answer_source_label.setWordWrap(True)
        self.answer_source_label.setToolTip("Shows whether the answer came from an online API or a local model (Ollama).")
        answer_layout.addWidget(self.answer_source_label)
        
        main_splitter.addWidget(answer_container)
        
        # Results section
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        
        results_header = QHBoxLayout()
        results_label = QLabel("📋 Matching notes:")
        results_label.setToolTip("Notes that match your question. Check the ones to send to the AI for the answer.")
        results_label.setStyleSheet(f"font-weight: bold; font-size: {label_font_size}px; color: {'#ffffff' if is_dark else '#2c3e50'};")
        results_header.addWidget(results_label)
        self.selected_count_label = QLabel("(0 selected)")
        self.selected_count_label.setStyleSheet(f"font-size: {label_font_size - 2}px; color: {'#95a5a6' if is_dark else '#7f8c8d'}; font-style: italic;")
        self.selected_count_label.setToolTip("Notes you checked (for \"View Selected\"). Not the same as \"cited\" in the AI answer—use \"Show only cited notes\" for that.")
        results_header.addWidget(self.selected_count_label)
        results_header.addStretch()
        # Compact secondary controls
        preview_label = QLabel("Preview:")
        preview_label.setStyleSheet(f"font-size: 10px; color: {'#95a5a6' if is_dark else '#7f8c8d'};")
        preview_label.setToolTip("Preview length (characters)")
        results_header.addWidget(preview_label)
        self.preview_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_slider.setMinimum(50)
        self.preview_slider.setMaximum(500)
        self.preview_slider.setValue(150)
        self.preview_slider.setMaximumWidth(80)
        self.preview_slider.setToolTip("Preview length (characters)")
        self.preview_slider.valueChanged.connect(self.on_preview_length_changed)
        results_header.addWidget(self.preview_slider)
        self.preview_length_label = QLabel("150 chars")
        self.preview_length_label.setMinimumWidth(40)
        self.preview_length_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_length_label.setStyleSheet(f"font-size: 10px; color: {'#95a5a6' if is_dark else '#7f8c8d'};")
        self.preview_length_label.setToolTip("Preview length in characters")
        results_header.addWidget(self.preview_length_label)
        self.toggle_select_btn = QPushButton("✓ Select All")
        self.toggle_select_btn.setObjectName("toggleSelectBtn")
        self.toggle_select_btn.setMaximumWidth(95)
        self.toggle_select_btn.setToolTip("Toggle select/deselect all (Ctrl+A / Ctrl+D)")
        self.toggle_select_btn.clicked.connect(self.toggle_select_all)
        self.toggle_select_btn.setEnabled(False)
        results_header.addWidget(self.toggle_select_btn)
        results_layout.addLayout(results_header)
        
        # Create table: Ref (citation [1],[2]…) | Content | Note ID | Relevance
        self.results_list = QTableWidget()
        self.results_list.setColumnCount(4)
        self.results_list.setHorizontalHeaderLabels(["Ref", "Content", "Note ID", "Relevance"])
        self.results_list.setMinimumHeight(120)
        self.results_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        notes_font_size = self.styling_config.get('notes_font_size', 12)
        self.results_list.setStyleSheet(f"font-size: {notes_font_size}px;")
        self.results_list.setWordWrap(True)
        
        # Configure columns
        self.results_list.setColumnWidth(0, 42)   # Ref (citation number matching [1], [2] in answer)
        self.results_list.setColumnWidth(1, 400)  # Content
        self.results_list.setColumnWidth(2, 80)   # Note ID (hidden by default)
        self.results_list.setColumnWidth(3, 100)  # Relevance (bar + %)
        self.results_list.setColumnHidden(2, True)  # Hide Note ID column (right-click header to show)
        self.results_list.horizontalHeader().setStretchLastSection(False)
        self.results_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.results_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.results_list.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.results_list.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        # Make column headers readable on dark and light themes (was hard to see on dark)
        header_color = "#ecf0f1" if is_dark else "#2c3e50"
        self.results_list.horizontalHeader().setStyleSheet(
            f"color: {header_color}; font-weight: bold; font-size: {max(11, notes_font_size - 1)}px;"
        )
        
        self.results_list.setSortingEnabled(True)
        self.results_list.setItemDelegateForColumn(3, RelevanceBarDelegate(self.results_list))  # Relevance bar + %
        self.results_list.sortItems(3, Qt.SortOrder.DescendingOrder)  # Sort by Relevance
        
        # Enable double-click on rows
        self.results_list.itemDoubleClicked.connect(self.open_in_browser)
        
        # Hide vertical header (row numbers)
        self.results_list.verticalHeader().setVisible(False)
        
        # Set selection behavior to select entire rows
        self.results_list.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        
        # Enable alternating row colors (zebra striping)
        self.results_list.setAlternatingRowColors(True)
        
        # Store preview length setting
        self.preview_length = 150  # Default preview length
        
        results_layout.addWidget(self.results_list, 1)
        
        # Label at bottom of notes area: which search mechanism yielded these results
        self.search_method_result_label = QLabel("")
        self.search_method_result_label.setStyleSheet(
            "font-size: 11px; color: #7f8c8d; padding: 4px 0; margin-top: 4px;"
        )
        self.search_method_result_label.setWordWrap(True)
        results_layout.addWidget(self.search_method_result_label)
        
        main_splitter.addWidget(results_container)
        main_splitter.setSizes([450, 550] if self.use_side_by_side else [350, 450])
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(8)
        
        layout.addWidget(main_splitter, 1)
        
        # Result tuning: relevance mode controls (Focused / Balanced / Broad); no slider
        sensitivity_container = QWidget()
        sensitivity_container.setStyleSheet("background-color: rgba(241, 196, 15, 0.1); border-radius: 6px; padding: 6px;")
        sensitivity_layout = QVBoxLayout(sensitivity_container)
        sensitivity_layout.setSpacing(4)
        sensitivity_layout.setContentsMargins(4, 2, 4, 2)
        self.sensitivity_slider = None  # Slider removed; mode alone controls how many notes are shown
        self.sensitivity_value_label = None

        # Relevance modes + "Show only cited notes" aligned on one line
        mode_and_filter_row = QHBoxLayout()
        mode_and_filter_row.setSpacing(8)
        mode_and_filter_row.setContentsMargins(0, 0, 0, 0)
        
        # Relevance mode selector (Focused / Balanced / Broad)
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(0)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        self.relevance_mode_group = QButtonGroup(self)
        self.relevance_mode_group.setExclusive(True)
        
        # Determine initial mode from config (fallback to strict_relevance when missing)
        try:
            sc_mode = load_config().get("search_config", {}).get("relevance_mode", "")
        except Exception:
            sc_mode = ""
        if not sc_mode:
            try:
                sc = load_config().get("search_config", {})
                sc_mode = "focused" if sc.get("strict_relevance", True) else "balanced"
            except Exception:
                sc_mode = "balanced"
        sc_mode = (sc_mode or "balanced").lower()
        if sc_mode not in ("focused", "balanced", "broad"):
            sc_mode = "balanced"
        self.relevance_mode = sc_mode
        
        # Shared segmented-control style for relevance mode buttons
        mode_border_color = "#f1c40f"
        mode_text_color = "#f1c40f" if is_dark else "#7d6608"
        mode_checked_bg = "#f1c40f" if is_dark else "#f9e79f"
        mode_checked_text = "#1e1e1e" if is_dark else "#7d6608"
        mode_hover_bg = "rgba(241, 196, 15, 0.18)"
        mode_btn_style = (
            "QRadioButton {"
            "  font-size: 11px;"
            "  padding: 4px 10px;"
            "  border: 1px solid " + mode_border_color + ";"
            "  color: " + mode_text_color + ";"
            "  background-color: transparent;"
            "  margin: 0;"
            "}"
            "QRadioButton::indicator {"
            "  width: 0px;"
            "  height: 0px;"
            "}"
            "QRadioButton:hover {"
            "  background-color: " + mode_hover_bg + ";"
            "}"
            "QRadioButton:checked {"
            "  background-color: " + mode_checked_bg + ";"
            "  color: " + mode_checked_text + ";"
            "}"
        )
        
        def _add_mode_button(label, mode_key, tooltip):
            btn = QRadioButton(label)
            btn.setToolTip(tooltip)
            btn.setProperty("mode_key", mode_key)
            btn.setMinimumWidth(72)
            btn.setStyleSheet(mode_btn_style)
            self.relevance_mode_group.addButton(btn)
            mode_layout.addWidget(btn)
            if mode_key == self.relevance_mode:
                btn.setChecked(True)
        
        _add_mode_button(
            "Focused",
            "focused",
            "Fewer notes, most on-topic only. Same search results; only which notes are shown changes.",
        )
        _add_mode_button(
            "Balanced",
            "balanced",
            "Moderate set. Same search results; only which notes are shown changes.",
        )
        _add_mode_button(
            "Broad",
            "broad",
            "More notes, including tangential. Same search results; only which notes are shown changes.",
        )
        
        self.relevance_mode_group.buttonToggled.connect(self._on_relevance_mode_changed)
        mode_and_filter_row.addLayout(mode_layout)
        
        self.show_only_cited_cb = QCheckBox("Show only cited notes")
        self.show_only_cited_cb.setToolTip(
            "Show only notes that the AI explicitly cited in its answer."
        )
        try:
            sc = load_config().get('search_config', {})
            self.show_only_cited_cb.setChecked(bool(sc.get('show_only_cited', False)))
        except Exception:
            self.show_only_cited_cb.setChecked(False)
        self.show_only_cited_cb.stateChanged.connect(self._on_show_only_cited_changed)
        mode_and_filter_row.addWidget(self.show_only_cited_cb)
        mode_and_filter_row.addStretch()
        
        sensitivity_layout.addLayout(mode_and_filter_row)
        
        layout.addWidget(sensitivity_container)
        
        # Action buttons
        btn_container = QWidget()
        btn_container.setObjectName("actionBar")
        btn_bar_bg = "#2d2d2d" if is_dark else "#d5d8dc"
        btn_bar_border = "#555555" if is_dark else "#95a5a6"
        btn_container.setStyleSheet(f"""
            QWidget#actionBar {{
                background-color: {btn_bar_bg};
                border: 1px solid {btn_bar_border};
                border-radius: 6px;
                padding: 6px;
            }}
        """)
        btn_layout = QHBoxLayout(btn_container)
        
        self.view_btn = QPushButton("👁️ View Selected")
        self.view_btn.setObjectName("viewBtn")
        self.view_btn.setToolTip("Open selected notes in the Anki browser")
        self.view_btn.setMinimumHeight(32)
        self.view_btn.clicked.connect(self.open_selected_in_browser)
        self.view_btn.setEnabled(False)
        
        self.view_all_btn = QPushButton("📚 View All")
        self.view_all_btn.setObjectName("viewAllBtn")
        self.view_all_btn.setToolTip(
            "Open all visible notes in the Anki browser. "
            "No notes in the list — run a search first."
        )
        self.view_all_btn.setMinimumHeight(32)
        self.view_all_btn.clicked.connect(self.open_all_in_browser)
        self.view_all_btn.setEnabled(False)
        
        btn_layout.addWidget(self.view_btn)
        btn_layout.addWidget(self.view_all_btn)
        btn_layout.addStretch()
        
        close_btn = QPushButton("✖ Close")
        close_btn.setObjectName("closeBtn")
        close_btn.setMinimumHeight(32)
        close_btn.setMinimumWidth(80)
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addWidget(btn_container)
        
        # Status bar
        status_container = QWidget()
        status_container.setStyleSheet("background-color: rgba(52, 152, 219, 0.15); border-radius: 4px; padding: 4px;")
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(8, 4, 8, 4)
        
        status_icon = QLabel("ℹ️")
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {'#ffffff' if is_dark else '#2c3e50'}; font-size: 12px; font-weight: 600;")
        self.status_label.setToolTip(
            "Showing X of Y: X = notes passing the Sensitivity filter, Y = notes in this result set (from Min relevance % and Max results in Settings). "
            "Move the Sensitivity slider to change X. Raise Min relevance % in Settings to get a smaller result set (smaller Y)."
        )
        
        status_layout.addWidget(status_icon)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        # Progress bar and % for embedding search (hidden when idle)
        self.search_progress_bar = QProgressBar()
        self.search_progress_bar.setMinimum(0)
        self.search_progress_bar.setMaximum(100)
        self.search_progress_bar.setValue(0)
        self.search_progress_bar.setMaximumWidth(120)
        self.search_progress_bar.setMinimumWidth(80)
        self.search_progress_bar.setTextVisible(True)
        self.search_progress_bar.setFormat("%p%")
        self.search_progress_bar.setStyleSheet("QProgressBar { text-align: center; color: palette(window-text); }")
        self.search_progress_bar.setVisible(False)
        status_layout.addWidget(self.search_progress_bar)
        self.search_progress_label = QLabel("")
        self.search_progress_label.setStyleSheet(f"font-size: 11px; color: {'#95a5a6' if is_dark else '#7f8c8d'};")
        self.search_progress_label.setVisible(False)
        status_layout.addWidget(self.search_progress_label)
        
        layout.addWidget(status_container)
        
        self.setLayout(layout)
        
        # Enable buttons based on selections
        self.results_list.itemSelectionChanged.connect(
            lambda: self.view_btn.setEnabled(bool(self.results_list.selectedItems()))
        )
        
        self.results_list.itemSelectionChanged.connect(self._update_view_all_button_state)

        # Track checkbox state changes to update count and button text
        # Only track changes in column 1 (content column with checkbox)
        self.results_list.itemChanged.connect(self.on_item_changed)

        # Store selected note IDs for persistence
        self.selected_note_ids = set()
        self._pinned_note_ids = set()  # note IDs from clicked [N] refs in AI answer
        self._cited_note_ids = set()   # note IDs cited in AI answer ([1], [2], …) for "Show only cited" filter

        QTimer.singleShot(100, self._refresh_scope_banner)

    def _update_view_all_button_state(self):
        """Update View All button enabled state and tooltip based on whether the results list has rows."""
        if not hasattr(self, 'view_all_btn') or not self.view_all_btn:
            return
        has_rows = self.results_list.rowCount() > 0 if hasattr(self, 'results_list') and self.results_list else False
        self.view_all_btn.setEnabled(has_rows)
        self.view_all_btn.setToolTip(
            "Open all visible notes in the Anki browser"
            if has_rows
            else "No notes in the list — run a search first."
        )

    def showEvent(self, event):
        """Refresh scope banner when dialog is shown (e.g. after Settings changed)."""
        super().showEvent(event)
        self._refresh_scope_banner()
    
    def _refresh_scope_banner(self):
        """Update scope banner: X note types, Y fields, Z decks with shortcut to Settings."""
        if not hasattr(self, 'scope_banner') or not self.scope_banner:
            return
        try:
            config = load_config()
            ntf = config.get('note_type_filter') or {}
            enabled_types = ntf.get('enabled_note_types') or []
            ntf_fields = ntf.get('note_type_fields') or {}
            enabled_decks = ntf.get('enabled_decks') or []
            search_all = bool(ntf.get('search_all_fields', False))
            n_types = len(enabled_types) if enabled_types else len(get_all_note_types())
            n_decks = len(enabled_decks) if enabled_decks else len(get_deck_names())
            fields_set = set()
            if search_all:
                fields_set = set(f['name'] for m in (mw.col.models.all() if mw and mw.col else []) for f in m.get('flds', []))
            else:
                for flist in ntf_fields.values():
                    fields_set.update(flist or [])
            n_fields = len(fields_set) if fields_set else 1
            txt = f"Searching: {n_types} note types, {n_fields} fields, {n_decks} decks — <a href='#settings' style='color:#3498db;'>Settings</a>"
            self.scope_banner.setText(txt)
        except Exception as e:
            log_debug(f"Scope banner refresh error: {e}")
            self.scope_banner.setText("")

    def _on_search_history_selected(self, index):
        """When user selects a recent search from the dropdown, populate the input."""
        if index >= 0 and hasattr(self, 'search_input') and hasattr(self, 'search_history_combo'):
            text = self.search_history_combo.currentText()
            if text:
                self.search_input.setPlainText(text)
                self.search_input.setFocus()
                if hasattr(self, 'status_label') and self.status_label:
                    self.status_label.setText("Query loaded — press Ctrl+Enter to search.")
                    QTimer.singleShot(3000, self._clear_query_loaded_status)

    def _clear_query_loaded_status(self):
        """Clear the 'Query loaded' status after a delay if it wasn't replaced by search results."""
        if hasattr(self, 'status_label') and self.status_label:
            if self.status_label.text().startswith("Query loaded —"):
                self.status_label.setText("Ready")

    def _on_clear_search_history(self):
        """Clear all search history and refresh the dropdown."""
        if clear_search_history():
            self._refresh_search_history()

    def _refresh_search_history(self):
        """Reload the previous-searches list from search_history.json."""
        try:
            if hasattr(self, '_search_history_model'):
                self._search_history_model.setStringList(get_search_history_queries())
        except Exception:
            pass

    def on_item_changed(self, item):
        """Handle item changes - only update count if checkbox column changed"""
        if item.column() == 1:  # Only process changes in content column (checkbox)
            self.update_selection_count()
    
    def _on_relevance_mode_changed(self, _btn, checked):
        """Persist relevance_mode (Focused/Balanced/Broad) and refresh current view."""
        # #region agent log
        _session_debug_log(
            "H1",
            "__init__._on_relevance_mode_changed.entry",
            "mode change handler",
            data={"checked": checked, "relevance_mode_before": getattr(self, "relevance_mode", None), "_effective_relevance_mode": getattr(self, "_effective_relevance_mode", None)},
        )
        # #endregion
        if not checked:
            return
        try:
            btn = self.relevance_mode_group.checkedButton()
            mode_key = (btn.property("mode_key") or "balanced").lower()
        except Exception:
            mode_key = "balanced"
        if mode_key not in ("focused", "balanced", "broad"):
            mode_key = "balanced"
        self.relevance_mode = mode_key
        self._effective_relevance_mode = mode_key  # so status bar and "Results from" label show the selected mode
        # #region agent log
        _session_debug_log(
            "H1",
            "__init__._on_relevance_mode_changed.after_assign",
            "relevance_mode set",
            data={"mode_key": mode_key, "relevance_mode": self.relevance_mode, "has_all_scored_notes": hasattr(self, "all_scored_notes")},
        )
        # #endregion
        # Persist last-used mode and keep strict_relevance in sync for compatibility
        try:
            config = load_config()
            sc = dict(config.get("search_config") or {})
            sc["relevance_mode"] = mode_key
            sc["strict_relevance"] = True if mode_key == "focused" else False
            config["search_config"] = sc
            save_config(config)
        except Exception:
            pass
        # Update "Results from: ... · Mode · Embeddings: ..." so it shows the new mode
        if hasattr(self, "search_method_result_label") and getattr(self, "_last_search_method", None):
            try:
                sc = load_config().get("search_config") or {}
                mode_display = {"focused": "Focused", "balanced": "Balanced", "broad": "Broad"}.get(mode_key, "Balanced")
                engine = (sc.get("embedding_engine") or "ollama").strip().lower()
                engine_display = {"ollama": "Ollama (local)", "voyage": "Voyage AI", "openai": "OpenAI", "cohere": "Cohere"}.get(engine, engine or "unknown")
                self.search_method_result_label.setText(f"Results from: {self._last_search_method} · {mode_display} · Embeddings: {engine_display}")
                self.search_method_result_label.setVisible(True)
            except Exception:
                pass
        # Update any existing results with the new mode (will use effective flags)
        if hasattr(self, "all_scored_notes"):
            self.filter_and_display_notes()
    
    def on_sensitivity_changed(self, value):
        if getattr(self, 'sensitivity_value_label', None) is not None:
            self.sensitivity_value_label.setText(f"{value}%")
        # Persist so next time the add-on opens with the same choice (keep full search_config)
        try:
            config = load_config()
            sc = dict(config.get('search_config') or {})  # copy so we don't lose other keys
            sc['sensitivity_percent'] = value
            config['search_config'] = sc
            save_config(config)
        except Exception:
            pass
        if hasattr(self, 'all_scored_notes'):
            self.filter_and_display_notes()
    
    def _on_show_only_cited_changed(self, _state):
        """Persist 'Show only cited notes' and refresh the table."""
        try:
            config = load_config()
            sc = dict(config.get('search_config') or {})
            sc['show_only_cited'] = getattr(self, 'show_only_cited_cb', None) and self.show_only_cited_cb.isChecked()
            config['search_config'] = sc
            save_config(config)
        except Exception:
            pass
        if hasattr(self, 'all_scored_notes'):
            self.filter_and_display_notes()
    
    def _restore_answer_html(self, html):
        """Restore the answer box HTML (used after link click so the AI answer does not disappear)."""
        if html and hasattr(self, 'answer_box'):
            self.answer_box.setHtml(html)
    
    def _on_answer_link_clicked(self, url):
        """Citation links: single-click highlights note in Matching notes; double-click opens in Anki Browser (over add-on). Supports #cite-N, anki:goto_note:{note_id}, and legacy note:N."""
        import time
        saved_html = getattr(self, '_last_formatted_answer', None) or (self.answer_box.toHtml() if hasattr(self.answer_box, 'toHtml') else None)
        s = url.toString() if hasattr(url, 'toString') else str(url)
        if not s:
            if saved_html:
                self.answer_box.setHtml(saved_html)
            return
        ctx = getattr(self, '_context_note_ids', None) or []
        note_id = None
        num = None
        if s.startswith('#cite-'):
            try:
                num = int(s.replace('#cite-', '').strip())
                if 1 <= num <= len(ctx):
                    note_id = ctx[num - 1]
            except (ValueError, TypeError):
                pass
        elif s.startswith('anki:goto_note:'):
            try:
                note_id = int(s.split(':', 2)[2].strip())
                if ctx and note_id in ctx:
                    num = ctx.index(note_id) + 1
                else:
                    num = note_id
            except (ValueError, IndexError):
                pass
        elif s.startswith('note:'):
            try:
                num = int(s.split(':', 1)[1].strip())
                if 1 <= num <= len(ctx):
                    note_id = ctx[num - 1]
            except (ValueError, IndexError):
                pass
        if note_id is None:
            if saved_html:
                self.answer_box.setHtml(saved_html)
            return
        if num is None:
            num = (ctx.index(note_id) + 1) if note_id in ctx else note_id

        # Single vs double click: second click on same link within 400ms = open browser; else only highlight
        now = time.time()
        last = getattr(self, '_citation_last_click', None)
        is_double = last is not None and last[0] == s and (now - last[1]) < 0.4
        self._citation_last_click = None if is_double else (s, now)

        if is_double:
            self._open_note_in_browser(note_id, num)

        # Always highlight the corresponding row in the results list
        if hasattr(self, 'all_scored_notes') and self.all_scored_notes:
            self._pinned_note_ids.add(note_id)
            if hasattr(self, 'selected_note_ids'):
                self.selected_note_ids.add(note_id)
            max_score = self.all_scored_notes[0][0]
            thresh = self.sensitivity_slider.value() if self.sensitivity_slider else 0
            min_score = (thresh / 100.0) * max_score if max_score > 0 else 0
            id_to_score = {n['id']: s for s, n in self.all_scored_notes}
            pinned_orig_scores = [id_to_score.get(nid, 0) for nid in self._pinned_note_ids]
            any_filtered = any(orig < min_score for orig in pinned_orig_scores)
            if any_filtered and self.sensitivity_slider is not None:
                self.sensitivity_slider.blockSignals(True)
                self.sensitivity_slider.setValue(0)
                if self.sensitivity_value_label is not None:
                    self.sensitivity_value_label.setText("0%")
                self.sensitivity_slider.blockSignals(False)
            order = {nid: i for i, nid in enumerate(ctx)}
            pinned = []
            rest = []
            for score, note in self.all_scored_notes:
                if note['id'] in self._pinned_note_ids:
                    pinned.append((max_score, note))
                else:
                    rest.append((score, note))
            pinned.sort(key=lambda x: order.get(x[1]['id'], 999))
            self.all_scored_notes = pinned + rest
            self.filter_and_display_notes()
            
            # Scroll to and highlight the note row (match by Ref when available so chunk display scrolls to cited ref)
            for row in range(self.results_list.rowCount()):
                ref_item = self.results_list.item(row, 0)
                content_item = self.results_list.item(row, 1)
                if num is not None and ref_item and str(ref_item.text()) == str(num):
                    if content_item:
                        self.results_list.selectRow(row)
                        self.results_list.scrollToItem(content_item)
                    break
                if num is None and content_item and content_item.data(Qt.ItemDataRole.UserRole) == note_id:
                    self.results_list.selectRow(row)
                    self.results_list.scrollToItem(content_item)
                    break
        
        if saved_html:
            QTimer.singleShot(0, lambda h=saved_html: self._restore_answer_html(h))
    
    def _citation_timer_clear(self):
        self._citation_last_click = None
    
    def _bring_browser_to_front(self, browser):
        """Raise browser window after a short delay so it stays on top of the add-on dialog."""
        if browser and hasattr(browser, 'activateWindow'):
            browser.activateWindow()
            browser.raise_()

    def _open_note_in_browser(self, note_id, num):
        """Open note in Anki Browser (used when user double-clicks a citation link). Brings browser to front over add-on."""
        try:
            browser = aqt.dialogs.open("Browser", mw)
            if browser:
                browser.form.searchEdit.lineEdit().setText(f"nid:{note_id}")
                browser.onSearchActivated()
                QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))
                tooltip(f"Opened note [{num}] (ID: {note_id}) in browser")
        except Exception as e:
            log_debug(f"Error opening note in browser: {e}")
            tooltip(f"Could not open note [{num}] in browser")
    
    def _spacing_styles(self):
        mode = self.styling_config.get('answer_spacing', 'normal')
        if mode == 'compact':
            return {'lh': '1.2', 'p': '0.15em 0 0.3em 0', 'ul': '0.15em 0 0.3em 0', 'li': '0.08em 0'}
        if mode == 'comfortable':
            return {'lh': '1.5', 'p': '0.3em 0 0.5em 0', 'ul': '0.3em 0 0.5em 0', 'li': '0.15em 0'}
        return {'lh': '1.35', 'p': '0.2em 0 0.4em 0', 'ul': '0.2em 0 0.4em 0', 'li': '0.1em 0'}
    
    def format_answer(self, answer):
        import re
        import html
        
        if not answer:
            return ""
        
        s = self._spacing_styles()
        # Escape HTML first, then we'll allow <strong> etc. via placeholders
        def rich_escape(text):
            escaped = html.escape(text)
            # Bold: **...** (allow multiline) and **... at end of line (missing closing **)
            escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong style="font-weight: bold;">\1</strong>', escaped, flags=re.DOTALL)
            escaped = re.sub(r'\*\*([^*]+)$', r'<strong style="font-weight: bold;">\1</strong>', escaped, flags=re.MULTILINE)
            # __...__ → <strong>
            escaped = re.sub(r'__(.+?)__', r'<strong style="font-weight: bold;">\1</strong>', escaped)
            # Short "Label:" lines (e.g. Locations:, Risk Factors:) → bold the label (max 50 chars)
            escaped = re.sub(r'^(.{1,50}):(\s*)$', r'<strong style="font-weight: bold;">\1</strong>:\2', escaped, flags=re.MULTILINE)
            # Highlight exam-style: ~~...~~ → yellow background
            escaped = re.sub(r'~~(.+?)~~', r'<span style="background-color: rgba(255,235,59,0.45); padding: 0 2px;">\1</span>', escaped)
            # Normalize bracket-like characters so [1], [47] etc. always match (some models output ［］【】)
            for _open, _close in (('\uFF3B', '\uFF3D'), ('\u3010', '\u3011'), ('\u301A', '\u301B')):
                escaped = escaped.replace(_open, '[').replace(_close, ']')
            # Convert [1], [2], [N40] to clickable links.
            ctx = getattr(self, '_context_note_ids', None) or []
            ctx_len = len(ctx)
            def cite_link(m):
                raw = m.group(1)
                pairs = []  # (note_id or None, display text, 1-based pos)
                for part in raw.split(','):
                    d = part.strip()
                    n = d.lstrip('N').strip()
                    if n.isdigit():
                        pos = int(n)
                        if 1 <= pos <= ctx_len:
                            note_id = ctx[pos - 1]
                            pairs.append((note_id, d, pos))
                        else:
                            pairs.append((None, d, 0))
                if not pairs:
                    return m.group(0)
                links = []
                for note_id, disp, pos in pairs:
                    if note_id is not None:
                        # Use #cite-N fragment so links are reliably clickable in QTextBrowser (anki: scheme can be blocked)
                        links.append(f'<a href="#cite-{pos}" style="color:#3498db;text-decoration:underline;cursor:pointer;" title="Single-click: highlight in list. Double-click: open in browser.">[{disp}]</a>')
                    else:
                        links.append(f'<span title="Citation out of range (max {ctx_len})" style="color:#95a5a6;">[{disp}]</span>')
                return '[' + ','.join(links) + ']'
            # [N2, N4, N8] or [N40]: allow N and digits inside brackets
            escaped = re.sub(r'\[N([\d,\sN]+)\]', cite_link, escaped)
            escaped = re.sub(r'\[([\d,\s]+)\]', cite_link, escaped)  # [1], [2], [38,43]
            return escaped
        
        lines = answer.split('\n')
        result_lines = []
        in_list = False
        list_depth = 0  # 0 = none, 1 = top-level ul, 2 = nested ul
        
        for raw in lines:
            line = raw.rstrip()
            if not line:
                if in_list:
                    if list_depth == 2:
                        result_lines.append('</ul></li></ul>')
                    elif list_depth == 1:
                        result_lines.append('</ul>')
                    in_list = False
                    list_depth = 0
                result_lines.append('<br>')
                continue
            
            # Detect indent for sub-bullets (2+ spaces or tab before bullet)
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            is_sub = indent >= 2 and (stripped.startswith('•') or stripped.startswith('-') or stripped.startswith('*'))
            
            # Section header: ## Something → bold with ring (●), no hashes
            if stripped.startswith('##'):
                if in_list:
                    if list_depth == 2:
                        result_lines.append('</ul></li></ul>')
                    elif list_depth == 1:
                        result_lines.append('</ul>')
                    in_list = False
                    list_depth = 0
                title = stripped.lstrip('#').strip()
                title = rich_escape(title)
                result_lines.append(f'<p style="margin: 0.75em 0 0.4em 0; font-weight: bold; font-size: 1.22em;">● {title}</p>')
                continue
            
            # Single # heading (treat as section too)
            if stripped.startswith('#') and not stripped.startswith('##'):
                if in_list:
                    if list_depth == 2:
                        result_lines.append('</ul></li></ul>')
                    elif list_depth == 1:
                        result_lines.append('</ul>')
                    in_list = False
                    list_depth = 0
                title = stripped.lstrip('#').strip()
                title = rich_escape(title)
                result_lines.append(f'<p style="margin: 0.75em 0 0.4em 0; font-weight: bold; font-size: 1.22em;">● {title}</p>')
                continue
            
            # Bullet (•, -, *)
            if stripped.startswith('•') or stripped.startswith('-') or stripped.startswith('*'):
                content = stripped.lstrip('•-*').strip()
                # Strip any remaining leading bullets (e.g. "• • Types..." or "· Types...")
                bullet_chars = '•-*·●◦∙\u2022\u2023\u00b7'
                while content and content[0] in bullet_chars:
                    content = content[1:].lstrip()
                content = rich_escape(content)
                # Skip empty bullets (e.g. "- " or trailing bullet) so answer doesn't look incomplete
                if not content or not content.strip():
                    continue
                if is_sub and in_list and list_depth == 1:
                    # Start nested list inside the previous <li> (replace last </li> with <ul><li>...)
                    if result_lines and result_lines[-1].strip().endswith('</li>'):
                        result_lines[-1] = result_lines[-1].rstrip().rstrip('</li>').rstrip() + '<ul style="margin: 0.2em 0 0.2em 0.8em; padding-left: 1.2em; list-style-type: circle;">'
                    else:
                        result_lines.append('<ul style="margin: 0.2em 0 0.2em 0.8em; padding-left: 1.2em; list-style-type: circle;">')
                    result_lines.append(f'<li style="margin: {s["li"]};">{content}</li>')
                    list_depth = 2
                elif is_sub and in_list and list_depth == 2:
                    result_lines.append(f'<li style="margin: {s["li"]};">{content}</li>')
                elif not in_list:
                    result_lines.append(f'<ul style="margin: {s["ul"]}; padding-left: 1.3em; list-style-type: disc;">')
                    result_lines.append(f'<li style="margin: {s["li"]};">{content}</li>')
                    in_list = True
                    list_depth = 1
                else:
                    if list_depth == 2:
                        result_lines.append('</ul></li>')
                        list_depth = 1
                    result_lines.append(f'<li style="margin: {s["li"]};">{content}</li>')
                continue
            
            # Plain paragraph
            if in_list:
                if list_depth == 2:
                    result_lines.append('</ul></li></ul>')
                elif list_depth == 1:
                    result_lines.append('</ul>')
                in_list = False
                list_depth = 0
            result_lines.append(f'<p style="margin: {s["p"]};">{rich_escape(line)}</p>')
        
        if in_list:
            if list_depth == 2:
                result_lines.append('</ul></li></ul>')
            elif list_depth == 1:
                result_lines.append('</ul>')
        
        html_content = ''.join(result_lines)
        # Force link styling inside the document (QTextBrowser may not apply widget stylesheet to content)
        link_style = (
            "<style>a { color: #3498db !important; text-decoration: underline !important; } "
            "a:hover { color: #5dade2 !important; }</style>"
        )
        # Add invisible targets for #cite-N so QTextBrowser doesn't navigate away when link is clicked
        ctx_len = len(getattr(self, '_context_note_ids', None) or [])
        anchors = ''.join(f'<span id="cite-{i}" style="position:absolute;width:0;height:0;"></span>' for i in range(1, ctx_len + 1))
        return f'{link_style}<div style="line-height: {s["lh"]}; margin: 0;">{html_content}</div>{anchors}'
    
    def copy_answer_to_clipboard(self):
        html = getattr(self, '_last_formatted_answer', None) or ""
        plain = self.answer_box.toPlainText().strip()
        if html or plain:
            cb = QApplication.clipboard()
            if cb:
                mime = QMimeData()
                if html:
                    mime.setHtml(html)
                mime.setText(plain if plain else "")
                cb.setMimeData(mime)
                tooltip("Copied (paste into Word for bullets and formatting)")
        else:
            tooltip("No answer to copy")
    
    def open_settings(self):
        """Open the settings dialog in a non-modal window so Anki stays usable."""
        dialog = SettingsDialog(self)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        
    def get_config(self):
        config = load_config()
        if not config or 'api_key' not in config:
            return None
        return config
        
    def get_all_notes_content(self):
        log_debug("Starting to load notes from collection...")
        notes_data = []
        config = load_config()
        ntf = config.get('note_type_filter', {})
        
        # Migrate: fields_to_search -> note_type_fields
        if ntf.get('fields_to_search') and not ntf.get('note_type_fields'):
            global_flds = set(f.lower() for f in ntf['fields_to_search'])
            ntf = dict(ntf)
            ntf['note_type_fields'] = {}
            for model_name, _c, field_names in get_models_with_fields():
                ntf['note_type_fields'][model_name] = [f for f in field_names if f.lower() in global_flds]
        
        # Backward compat: empty ntf -> legacy Text & Extra globally
        legacy_fields = None
        if not ntf:
            enabled_set = None
            search_all = False
            ntf_fields = {}
            use_first = False
            legacy_fields = {'text', 'extra'}
            self.fields_description = "Text & Extra"
        else:
            enabled = ntf.get('enabled_note_types')
            enabled_set = set(enabled) if (enabled and len(enabled) > 0) else None
            search_all = bool(ntf.get('search_all_fields', False))
            ntf_fields = ntf.get('note_type_fields') or {}
            use_first = bool(ntf.get('use_first_field_fallback', True))
            self.fields_description = "all fields" if search_all else "per-type"
        deck_q = _build_deck_query(ntf.get('enabled_decks') if ntf else None)
        note_ids = mw.col.find_notes(deck_q) if deck_q else mw.col.find_notes("")
        total_notes = len(note_ids)
        cache_key = (deck_q or '', frozenset(enabled_set) if enabled_set is not None else None, search_all, tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in (ntf_fields or {}).items())), total_notes)
        if getattr(self, '_cached_notes_key', None) == cache_key and getattr(self, '_cached_notes', None) is not None:
            log_debug("Using cached notes (singleton search engine)")
            return self._cached_notes
        log_debug(f"Found {total_notes} total notes (decks: {'all' if not deck_q else 'filtered'}, note types: {'all' if enabled_set is None else 'filtered'}, fields: {self.fields_description})")
        
        for idx, nid in enumerate(note_ids):
            # Allow UI to update periodically so Anki doesn't show "Not Responding" during large collections
            if idx > 0 and idx % 500 == 0:
                try:
                    QApplication.processEvents()
                except Exception:
                    pass
            try:
                note = mw.col.get_note(nid)
                note_type = note.note_type()
                model_name = note_type['name']
                if enabled_set is not None and model_name not in enabled_set:
                    continue
                
                flds = note_type['flds']
                if search_all:
                    indices = [i for i in range(len(note.fields)) if i < len(note.fields)]
                else:
                    if legacy_fields is not None:
                        wanted = legacy_fields
                    else:
                        wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))
                        if not wanted and use_first and flds:
                            wanted = {flds[0]['name'].lower()}
                    indices = [i for i, f in enumerate(flds) if i < len(note.fields) and f['name'].lower() in wanted]
                
                if not indices:
                    continue
                
                content_parts = []
                for i in indices:
                    if i < len(note.fields) and note.fields[i].strip():
                        content_parts.append(note.fields[i].strip())
                if not content_parts:
                    continue
                
                content = " | ".join(content_parts)
                content = self.strip_html(content)
                if not content.strip():
                    continue
                content_parts_raw = [note.fields[i] for i in indices]
                content_for_hash = " ".join(content_parts_raw)
                import hashlib
                # Semantic chunking: split long notes at sentence boundaries (~500 chars), same Note ID per chunk for section citation
                CHUNK_TARGET = 500
                chunks = _semantic_chunk_text(content, CHUNK_TARGET)
                if len(chunks) <= 1:
                    content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()
                    notes_data.append({'id': nid, 'content': content, 'content_hash': content_hash, 'model': model_name, 'display_content': content})
                else:
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                        notes_data.append({
                            'id': nid, 'content': chunk, 'content_hash': chunk_hash, 'model': model_name,
                            'display_content': chunk, 'chunk_index': chunk_idx, '_full_content': content,
                        })
                
                if (idx + 1) % 1000 == 0:
                    progress = ((idx + 1) / total_notes) * 100
                    log_debug(f"Processed {idx + 1}/{total_notes} notes ({progress:.1f}%)")
                    try:
                        if hasattr(self, 'status_label') and self.status_label:
                            self.status_label.setText(f"Loading notes... {idx + 1}/{total_notes} ({progress:.1f}%)")
                            QApplication.processEvents()
                    except RuntimeError:
                        pass
            except Exception as e:
                log_debug(f"Error processing note {nid}: {str(e)}")
                continue
        
        if not hasattr(self, 'fields_description'):
            self.fields_description = "Text & Extra"
        log_debug(f"Loaded {len(notes_data)} notes from collection")
        self._cached_notes = notes_data
        self._cached_notes_key = cache_key
        return notes_data
    
    def _aggregate_scored_notes_by_note_id(self, scored_notes):
        """After semantic chunking, collapse multiple chunks per note to one entry per note (best score, full content)."""
        if not scored_notes:
            return scored_notes
        by_id = {}
        for score, note in scored_notes:
            nid = note.get('id')
            if nid is None:
                by_id[id(note)] = (score, note)
                continue
            if nid not in by_id or score > by_id[nid][0]:
                rep = dict(note)
                if rep.get('_full_content'):
                    rep['display_content'] = rep['content'] = rep['_full_content']
                by_id[nid] = (score, rep)
        return sorted(by_id.values(), key=lambda x: -x[0])
    
    def strip_html(self, text):
        return _strip_html_plain(text)
    
    def reveal_cloze_for_display(self, text):
        """Reveal cloze deletions for display: {{cN::answer}} or {{cN::answer::hint}} -> answer (plain text with answer shown)."""
        if not text:
            return text
        import re
        # {{c1::answer}} or {{c1::answer::hint}} -> answer (capture until }} or :: so answer can contain colons)
        return re.sub(r'\{\{c\d+::(.*?)(?=}}|::)(?:::[^}]*)?\}\}', r'\1', text)
    
    # ========== SEMANTIC SEARCH IMPROVEMENTS ==========
    
    def _simple_stem(self, word):
        """Simple stemming: remove common suffixes"""
        if len(word) <= 3:
            return word
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's', 'es', 'tion', 'sion', 'ness', 'ment']
        word_lower = word.lower()
        for suffix in suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                return word_lower[:-len(suffix)]
        return word_lower
    
    def _get_extended_stop_words(self):
        """Return extended stop words, including optional domain-specific extras from config."""
        builtin_stop_words = {
            'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who',
            'does', 'do', 'can', 'could', 'would', 'should', 'tell', 'me', 'about', 'explain',
            'describe', 'define', 'list', 'show', 'give', 'provide', 'this', 'that', 'these',
            'those', 'with', 'from', 'for', 'and', 'or', 'but', 'not', 'have', 'has', 'had',
            'been', 'being', 'was', 'were', 'will', 'would', 'may', 'might', 'must', 'shall',
            'which', 'work', 'works', 'working', 'use', 'uses', 'used', 'using',
            'cause', 'causes',
            'overview', 'introduction', 'review', 'study', 'case', 'cases',
            'difference', 'between', 'compared', 'comparison', 'similar', 'similarity',
            'different', 'same', 'other', 'another', 'each', 'every', 'both', 'either', 'neither',
            'like', 'such', 'same', 'common', 'generally', 'usually', 'often', 'typically',
            'example', 'examples', 'including', 'involves', 'involve', 'related', 'association',
        }
        try:
            config = load_config()
            sc = (config or {}).get('search_config') or {}
            stop_words = set(builtin_stop_words)
            extra = sc.get('extra_stop_words') or []
            if isinstance(extra, str):
                extra = [extra]
            extra_set = {
                (w or '').strip().lower()
                for w in extra
                if isinstance(w, str) and (w or '').strip()
            }
            stop_words.update(extra_set)
        except Exception:
            stop_words = set(builtin_stop_words)
        return stop_words
    
    def _extract_keywords_improved(self, query):
        """Improved keyword extraction with stemming and better stop word handling"""
        import re
        
        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)
        
        # Extended stop words (generic question words + structural fillers + optional domain extras)
        stop_words = self._get_extended_stop_words()
        ai_excluded = getattr(self, '_query_ai_excluded_terms', None) or set()
        if not isinstance(ai_excluded, set):
            ai_excluded = set(ai_excluded) if ai_excluded else set()
        
        # Extract keywords with stemming
        keywords = []
        stems = {}
        for w in query_words:
            if w not in stop_words and w not in ai_excluded and len(w) > 2:
                stem = self._simple_stem(w)
                keywords.append(w)
                if stem != w:
                    stems[stem] = w
        
        # Generate n-grams (bigrams and trigrams)
        phrases = []
        if len(keywords) > 1:
            for i in range(len(keywords) - 1):
                phrases.append(keywords[i] + " " + keywords[i + 1])
        if len(keywords) > 2:
            for i in range(len(keywords) - 2):
                phrases.append(keywords[i] + " " + keywords[i + 1] + " " + keywords[i + 2])
        # region agent log
        try:
            if "trisom" in query_lower:
                _agent_debug_log(
                    run_id="pre-fix",
                    hypothesis_id="H1",
                    location="__init__._extract_keywords_improved",
                    message="keywords_extracted",
                    data={
                        "query": query,
                        "keywords": keywords,
                        "stems": stems,
                        "phrases": phrases[:10],
                    },
                )
        except Exception:
            pass
        # endregion
        return keywords, stems, phrases
    
    def _compute_tfidf_scores(self, notes, query_keywords):
        """Compute TF-IDF scores for keywords across notes"""
        import math
        
        # Handle trivial cases up front
        if not notes or not query_keywords:
            # Reset per-query high-frequency keywords cache
            try:
                self._query_high_freq_keywords = set()
            except Exception:
                pass
            return {}
        
        # Term frequency in each note
        note_tfs = {}
        # Document frequency (how many notes contain each keyword)
        doc_freq = {}
        
        for note in notes:
            content_lower = note['content'].lower()
            note_tfs[note['id']] = {}
            for keyword in query_keywords:
                count = content_lower.count(keyword)
                if count > 0:
                    note_tfs[note['id']][keyword] = count
                    doc_freq[keyword] = doc_freq.get(keyword, 0) + 1
        
        # Compute TF-IDF scores
        total_notes = max(1, len(notes))
        # Automatically down-weight very common keywords so you don't have to
        # manually add every generic word as a stop word. Any query keyword that
        # appears in a large fraction of notes for this search is treated as a
        # high-frequency term and ignored by the later keyword scorer.
        high_freq_threshold = 0.65  # e.g. appears in >=65% of candidate notes
        high_freq_keywords = {
            kw for kw, df in doc_freq.items()
            if df / total_notes >= high_freq_threshold
        }
        try:
            # Stash on the instance so scoring functions can see it for this query
            self._query_high_freq_keywords = high_freq_keywords
        except Exception:
            self._query_high_freq_keywords = high_freq_keywords
        
        tfidf_scores = {}
        for note_id, tfs in note_tfs.items():
            score = 0
            for keyword, tf in tfs.items():
                idf = math.log(total_notes / (doc_freq.get(keyword, 1) + 1))
                tfidf = tf * idf
                score += tfidf
            tfidf_scores[note_id] = score
        
        return tfidf_scores
    
    def _get_note_embedding(self, note_content, note_id=None):
        """Get embedding for a note (checks persistent storage first, then cache).
        
        IMPORTANT:
        - Embeddings are keyed by a hash of the *generation-time* note content,
          not necessarily the string we display in the search UI.
        - Originally we hashed the UI content, which differed (HTML stripped,
          different separators), so lookups always missed and embeddings were
          treated as "not available".
        
        This helper now reconstructs the same content that was used during the
        Create/Update Embeddings workflow so the hashes line up.
        """
        import hashlib
        
        # Lazily initialise in‑memory cache
        if not hasattr(self, "_embedding_cache"):
            self._embedding_cache = {}
        
        # Try fast path: hash of the UI content (in case future versions align)
        ui_hash = hashlib.md5(note_content.encode()).hexdigest()
        if ui_hash in self._embedding_cache:
            return self._embedding_cache.get(ui_hash)
        
        # If we don't have a note id we can't reconstruct the generation
        # content reliably, so bail out early.
        if note_id is None:
            return None
        
        # Reconstruct the content string the embedding generator used so that
        # we derive the exact same MD5 and key.
        try:
            from aqt import mw  # Local import to avoid issues at import time
            note = mw.col.get_note(note_id)
            m = note.note_type()
            model_name = m["name"]
            flds = m["flds"]
            
            config = load_config()
            ntf = config.get("note_type_filter", {})
            enabled = set(ntf.get("enabled_note_types") or [])
            search_all = ntf.get("search_all_fields", False)
            ntf_fields = ntf.get("note_type_fields", {})
            use_first = ntf.get("use_first_field_fallback", True)
            
            # If embeddings were originally restricted to certain note types,
            # mirror that logic here. If the model wasn't enabled at generation
            # time we almost certainly won't have an embedding anyway.
            if enabled and model_name not in enabled:
                return None
            
            if search_all:
                indices = [
                    i for i in range(min(len(note.fields), len(flds)))
                    if note.fields[i].strip()
                ]
            else:
                wanted = set(
                    f.lower() for f in (ntf_fields.get(model_name) or [])
                )
                if not wanted and use_first and flds:
                    wanted = {flds[0]["name"].lower()}
                indices = [
                    i for i, f in enumerate(flds)
                    if i < len(note.fields)
                    and f["name"].lower() in wanted
                    and note.fields[i].strip()
                ]
            if not indices:
                return None
            content_parts = [note.fields[i] for i in indices]
            
            # This mirrors the generator: join with spaces and *do not* strip HTML.
            generation_content = " ".join(content_parts)
            content_hash = hashlib.md5(generation_content.encode()).hexdigest()
        except Exception:
            # Fall back to the UI hash; if that doesn't exist in storage,
            # load_embedding will just return None.
            content_hash = ui_hash
        
        # Check cache again with the (possibly different) generation hash
        if content_hash in self._embedding_cache:
            return self._embedding_cache.get(content_hash)
        
        # Then check persistent storage using the reconstructed hash
        persistent_embedding = load_embedding(note_id, content_hash)
        if persistent_embedding is not None:
            # Cache using the generation hash; we also alias under the UI hash
            # so repeated lookups for the same text are fast.
            self._embedding_cache[content_hash] = persistent_embedding
            self._embedding_cache[ui_hash] = persistent_embedding
            return persistent_embedding
        
        # If not found, we skip generating on-the-fly to avoid extra API usage.
        return None
    
    def _embedding_search(self, query, notes):
        """Semantic search using precomputed embeddings (Voyage).
        
        Uses the configured embedding engine (Voyage or Ollama) for the query
        embedding, and expects note embeddings to have been generated ahead of
        time via the Create/Update Embeddings workflow.
        """
        try:
            import numpy as np
        except ImportError:
            log_debug("numpy not available, embedding search disabled")
            return None
        
        try:
            # Get query embedding via configured engine (Voyage or Ollama)
            embedding_list = get_embedding_for_query(query)
            if not embedding_list:
                log_debug("Empty query embedding from embedding engine")
                return None
            query_embedding = np.array(embedding_list)
            
            # Compute similarities
            scored_notes = []
            
            # For very large collections, limit how many notes we run full
            # embedding search over to keep things responsive.
            max_notes_for_embedding = 5000
            total_notes = len(notes)
            if total_notes > max_notes_for_embedding:
                log_debug(f"Embedding search: limiting to first {max_notes_for_embedding} of {total_notes} notes for performance.")
                notes_iter = notes[:max_notes_for_embedding]
            else:
                notes_iter = notes
            
            # Keep UI responsive during long embedding loops
            try:
                from aqt.qt import QApplication
            except Exception:
                QApplication = None
            
            for idx, note in enumerate(notes_iter):
                if QApplication is not None and idx % 500 == 0:
                    QApplication.processEvents()
                    log_debug(f"Embedding search progress: processed {idx}/{len(notes_iter)} notes")
                
                note_embedding = self._get_note_embedding(note['content'], note.get('id'))
                if note_embedding is not None:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, note_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(note_embedding)
                    )
                    # Convert to 0-100 score
                    score = (similarity + 1) * 50  # Normalize from [-1,1] to [0,100]
                    scored_notes.append((score, note))
            
            scored_notes.sort(reverse=True, key=lambda x: x[0])
            return scored_notes[:80]
        except Exception as e:
            log_debug(f"Error in embedding search: {e}")
            return None
    
    def _get_note_metadata(self, note_id):
        """Get metadata for context-aware ranking"""
        try:
            note = mw.col.get_note(note_id)
            card = note.cards()[0] if note.cards() else None
            
            metadata = {
                'deck_id': card.did if card else None,
                'note_type': note.note_type()['name'] if note else None,
                'mod_time': note.mod if hasattr(note, 'mod') else 0,
                'review_count': card.reps if card else 0,
                'last_review': card.rep if card else 0,
            }
            return metadata
        except:
            return {}
    
    def _context_aware_boost(self, note, base_score, selected_note_ids=None):
        """Apply context-aware ranking boosts"""
        if not hasattr(self, 'selected_note_ids') or not self.selected_note_ids:
            selected_note_ids = set()
        else:
            selected_note_ids = self.selected_note_ids
        
        boost = 1.0
        metadata = self._get_note_metadata(note['id'])
        
        # Boost notes from same deck/note type as previously selected
        if selected_note_ids:
            try:
                selected_metadata = [self._get_note_metadata(nid) for nid in selected_note_ids]
                selected_decks = {m.get('deck_id') for m in selected_metadata if m.get('deck_id')}
                selected_types = {m.get('note_type') for m in selected_metadata if m.get('note_type')}
                
                if metadata.get('deck_id') in selected_decks:
                    boost *= 1.2
                if metadata.get('note_type') in selected_types:
                    boost *= 1.15
            except:
                pass
        
        # Boost recent notes (last 30 days)
        if metadata.get('mod_time'):
            try:
                from datetime import datetime, timedelta
                mod_date = datetime.fromtimestamp(metadata['mod_time'] / 1000)
                days_ago = (datetime.now() - mod_date).days
                if days_ago < 30:
                    boost *= 1.1
            except:
                pass
        
        # Slight boost for well-reviewed notes
        if metadata.get('review_count', 0) > 10:
            boost *= 1.05
        
        return base_score * boost
    
    def _expand_query(self, query, config):
        """Expand query with synonyms using AI (optional). Supports Ollama and cloud providers."""
        search_config = load_config().get('search_config', {})
        
        # Always apply a small built‑in synonym map for very common medical
        # variants so you don't have to remember which spelling your deck
        # used (e.g. adrenaline vs epinephrine). This runs even when the
        # AI-based expansion setting is disabled.
        try:
            q_lower = (query or "").lower()
            extra_terms = []
            # Pairs and small groups of common aliases / spelling variants
            synonym_groups = [
                # Catecholamines
                ["adrenaline", "epinephrine"],
                ["noradrenaline", "norepinephrine"],
                # Analgesics
                ["acetaminophen", "paracetamol"],
                # Hormones / vitamins (common exam phrasing variants)
                ["pth", "parathyroid hormone"],
                ["vitamin d", "cholecalciferol", "ergocalciferol"],
            ]
            for group in synonym_groups:
                present = [term for term in group if term in q_lower]
                if present:
                    for term in group:
                        if term not in q_lower:
                            extra_terms.append(term)
            # Config-driven synonym overrides (same logic; no UI, edit config.json if needed)
            overrides = search_config.get("synonym_overrides") or []
            for item in overrides:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    group = [str(t).strip().lower() for t in item if t and str(t).strip()]
                    if len(group) < 2:
                        continue
                    present = [term for term in group if term in q_lower]
                    if present:
                        for term in group:
                            if term not in q_lower:
                                extra_terms.append(term)
            if extra_terms:
                query = f"{query} " + " ".join(extra_terms)
        except Exception:
            # If anything goes wrong here, just fall back to the original query
            pass
        
        # Optional AI-based expansion (controlled from settings)
        if not search_config.get('enable_query_expansion', False):
            return query
        
        try:
            # Use a simple prompt to get synonyms / closely related terms
            prompt = (
                "Given this search query, list 2–4 key synonyms or closely related medical terms "
                "that would help find the same content. Return only the terms, comma-separated, "
                "no explanations or labels.\n\n"
                f"Query: {query}\n\n"
                "Synonyms:"
            )
            
            provider = config.get('provider', 'openai')
            import urllib.request
            import json
            
            # Ollama: fully local HTTP API
            if provider == 'ollama':
                sc = config.get('search_config') or search_config
                base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()
                # Allow a dedicated expansion model, falling back to chat model
                model = (
                    sc.get('ollama_query_expansion_model')
                    or sc.get('ollama_chat_model')
                    or 'llama3.2'
                )
                model = str(model).strip()
                url = base_url.rstrip("/") + "/api/generate"
                data = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 64},
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                resp = urllib.request.urlopen(req, timeout=8)
                result = json.loads(resp.read())
                expanded = (result.get("response") or "").strip()
            
            else:
                api_key = config.get('api_key', '')
                if not api_key:
                    return query
                
                model = self.get_best_model(provider)
                
                # Quick API call for expansion (cloud providers)
                if provider == "openai":
                    url = "https://api.openai.com/v1/chat/completions"
                    data = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 50,
                        "temperature": 0.3,
                    }
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }
                elif provider == "anthropic":
                    url = "https://api.anthropic.com/v1/messages"
                    data = {
                        "model": model,
                        "max_tokens": 50,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    headers = {
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    }
                elif provider == "google":
                    url = (
                        "https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{model}:generateContent?key={api_key}"
                    )
                    data = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"maxOutputTokens": 50, "temperature": 0.3},
                    }
                    headers = {"Content-Type": "application/json"}
                else:
                    # Skip expansion for unsupported providers
                    return query
                
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers=headers,
                )
                resp = urllib.request.urlopen(req, timeout=5)
                result = json.loads(resp.read())
                
                if provider == "openai":
                    expanded = result["choices"][0]["message"]["content"].strip()
                elif provider == "anthropic":
                    expanded = result["content"][0]["text"].strip()
                elif provider == "google":
                    expanded = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                else:
                    return query
            
            # Clean up and combine
            expanded = (expanded or "").replace("Synonyms:", "").strip()
            if not expanded:
                return query
            
            # Parse comma-separated terms, trim, and drop empties
            terms = [t.strip() for t in expanded.split(",") if t.strip()]
            if not terms:
                return query
            
            # Append terms to the original query so keyword extraction sees them
            return f"{query} " + " ".join(terms)
        except Exception as e:
            log_debug(f"Query expansion failed: {e}")
        
        return query
    
    def _get_ai_excluded_terms(self, query, config):
        """One short LLM call to detect generic query terms to exclude. Returns set of lowercased terms, or empty set on failure."""
        prompt = (
            "List only words that are too generic to help find specific study notes "
            "(e.g. 'difference', 'between', 'what', 'the', 'mechanism', 'treatment'). "
            "Include question filler and words that appear in most notes. "
            "Return comma-separated words only, or exactly 'none' if all words are useful. "
            f"Query: {query}"
        )
        try:
            import urllib.request
            import json
            provider = config.get('provider', 'openai')
            search_config = config.get('search_config') or {}
            response_text = ""
            if provider == 'ollama':
                sc = config.get('search_config') or search_config
                base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()
                model = (sc.get('ollama_chat_model') or 'llama3.2').strip()
                url = base_url.rstrip("/") + "/api/generate"
                data = {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 50}}
                req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}, method="POST")
                resp = urllib.request.urlopen(req, timeout=8)
                result = json.loads(resp.read())
                response_text = (result.get("response") or "").strip()
            else:
                api_key = config.get('api_key', '')
                if not api_key:
                    return set()
                model = self.get_best_model(provider)
                if provider == "openai":
                    url = "https://api.openai.com/v1/chat/completions"
                    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 50, "temperature": 0.1}
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")
                    resp = urllib.request.urlopen(req, timeout=8)
                    result = json.loads(resp.read())
                    response_text = (result.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
                elif provider == "anthropic":
                    url = "https://api.anthropic.com/v1/messages"
                    data = {"model": model, "max_tokens": 50, "messages": [{"role": "user", "content": prompt}]}
                    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
                    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")
                    resp = urllib.request.urlopen(req, timeout=8)
                    result = json.loads(resp.read())
                    response_text = (result.get("content") or [{}])[0].get("text", "").strip()
                elif provider == "google":
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                    data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 50, "temperature": 0.1}}
                    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}, method="POST")
                    resp = urllib.request.urlopen(req, timeout=8)
                    result = json.loads(resp.read())
                    response_text = (result.get("candidates") or [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                else:
                    return set()
            raw = (response_text or "").strip().lower()
            if not raw:
                return set()
            if raw in ("none", "n/a", "no", "nil"):
                return set()
            terms = []
            for part in raw.replace(";", ",").split(","):
                t = part.strip()
                if t and t not in ("none", "n/a", "no", "nil"):
                    terms.append(t)
            result = set(terms)
            if search_config.get('verbose_search_debug') and result:
                log_debug(f"AI generic term detection excluded for this query: {result}")
            return result
        except Exception as e:
            log_debug(f"AI generic term detection failed: {e}")
            return set()

    def _generate_hyde_document(self, query, config):
        """Generate a brief hypothetical answer (HyDE) for retrieval: AI 'hallucinates' an answer, then we search on it."""
        HYDE_MAX_TOKENS = 60
        prompt = (
            "Write a brief 1–2 sentence hypothetical answer, as if from your study notes. "
            "Plain text only, no markdown.\n\nQuestion: " + query
        )
        try:
            provider = config.get('provider', 'openai')
            api_key = config.get('api_key', '')
            if provider == 'ollama':
                sc = config.get('search_config') or {}
                base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()
                model = (sc.get('ollama_chat_model') or 'llama3.2').strip()
                url = base_url.rstrip("/") + "/api/generate"
                data = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": HYDE_MAX_TOKENS}
                }
                req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}, method="POST")
                resp = urllib.request.urlopen(req, timeout=15)
                result = json.loads(resp.read())
                return (result.get("response") or "").strip()
            model = self.get_best_model(provider)
            if provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": HYDE_MAX_TOKENS, "temperature": 0.3}
                req = urllib.request.Request(url, data=json.dumps(data).encode(),
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
                resp = urllib.request.urlopen(req, timeout=60)
                result = json.loads(resp.read())
                return result['choices'][0]['message']['content'].strip()
            elif provider == "anthropic":
                url = "https://api.anthropic.com/v1/messages"
                data = {"model": model, "max_tokens": HYDE_MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]}
                req = urllib.request.Request(url, data=json.dumps(data).encode(),
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, method="POST")
                resp = urllib.request.urlopen(req, timeout=60)
                result = json.loads(resp.read())
                return result['content'][0]['text'].strip()
            elif provider == "google":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": HYDE_MAX_TOKENS, "temperature": 0.3}}
                req = urllib.request.Request(url, data=json.dumps(data).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                resp = urllib.request.urlopen(req, timeout=60)
                result = json.loads(resp.read())
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
            elif provider == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": HYDE_MAX_TOKENS, "temperature": 0.3}
                req = urllib.request.Request(url, data=json.dumps(data).encode(),
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
                resp = urllib.request.urlopen(req, timeout=60)
                result = json.loads(resp.read())
                return result['choices'][0]['message']['content'].strip()
            else:
                # Custom endpoint (OpenAI-compatible shape)
                api_url = (config.get('api_url') or '').strip()
                if not api_url:
                    return None
                data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": HYDE_MAX_TOKENS, "temperature": 0.3}
                req = urllib.request.Request(api_url, data=json.dumps(data).encode(),
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
                resp = urllib.request.urlopen(req, timeout=60)
                result = json.loads(resp.read())
                if 'choices' in result:
                    return result['choices'][0]['message']['content'].strip()
                if 'content' in result and result['content']:
                    return result['content'][0].get('text', '').strip()
                return None
        except Exception as e:
            log_debug(f"HyDE generation failed: {e}")
        return None

    def _passes_focused_balanced_broad(
        self,
        matched_keywords,
        final_score,
        emb_score,
        max_emb_score,
        keywords,
        search_method,
        embeddings_available,
        min_emb_frac=0.25,
        very_high_emb_frac=0.9,
    ):
        """Compute whether a note would pass Focused, Balanced, or Broad inclusion. Returns (passes_focused, passes_balanced, passes_broad)."""
        n_kw = len(keywords) if keywords else 0
        # Focused
        min_kw_focused = max(2, int(n_kw * 0.4)) if n_kw else 1
        if n_kw <= 2:
            min_kw_focused = 1
        min_score_focused = 18
        # Balanced
        min_kw_balanced = max(1, int(n_kw * 0.25)) if n_kw else 1
        min_score_balanced = 10
        # Broad
        min_kw_broad = max(1, int(n_kw * 0.2)) if n_kw else 1
        min_score_broad = 8

        if search_method == "embedding" and embeddings_available:
            if emb_score > 0:
                return (True, True, True)
            return (False, False, False)

        if search_method == "hybrid" and embeddings_available and max_emb_score > 0:
            very_high = emb_score >= very_high_emb_frac * max_emb_score
            decent = emb_score >= min_emb_frac * max_emb_score
            pf = (
                (decent and (matched_keywords >= min_kw_focused or final_score > min_score_focused)) or very_high
            ) and (matched_keywords > 0 or very_high)
            pb_al = (decent and (matched_keywords >= min_kw_balanced or final_score > min_score_balanced)) or very_high
            pb_br = (decent and (matched_keywords >= min_kw_broad or final_score > min_score_broad)) or very_high
            return (pf, pb_al, pb_br)

        # Keyword-only or fallback
        pf = matched_keywords >= min_kw_focused
        if n_kw <= 2:
            pb_al = matched_keywords >= 1
            pb_br = matched_keywords >= 1
        else:
            pb_al = matched_keywords >= min_kw_balanced or final_score > min_score_balanced
            pb_br = matched_keywords >= min_kw_broad or final_score > min_score_broad
        return (pf, pb_al, pb_br)


def _is_embedding_dimension_mismatch(exc):
    """Detect NumPy shape mismatch (e.g. 768 vs 1024) from embedding engine switch."""
    s = str(exc)
    return "not aligned" in s or ("shapes" in s and "dim" in s)


def _run_embedding_search_sync(embedding_query, notes, config, db_path=None):
    """Run embedding search in a background thread (for taskman). Returns scored_notes or None.
    db_path: from main thread so profile-specific path is correct in background."""
    import hashlib
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        embedding_list = get_embedding_for_query(embedding_query, config)
        if not embedding_list:
            return None
        query_embedding = np.array(embedding_list)
        query_dim = len(query_embedding)
        total = len(notes)
        scored_notes = []
        for idx, note in enumerate(notes):
            content_hash = note.get('content_hash')
            from_note = content_hash is not None
            if content_hash is None:
                content_hash = hashlib.md5(note['content'].encode()).hexdigest()
            note_embedding = load_embedding(note.get('id'), content_hash, db_path=db_path)
            if note_embedding is not None:
                emb = np.array(note_embedding)
                if len(emb) != query_dim:
                    log_debug(f"Embedding search: skipping note {note.get('id')} (dimension mismatch query={query_dim} stored={len(emb)})")
                    continue
                similarity = np.dot(query_embedding, emb) / (
                    max(np.linalg.norm(query_embedding) * np.linalg.norm(emb), 1e-9)
                )
                score = (similarity + 1) * 50
                scored_notes.append((score, note))
        scored_notes.sort(reverse=True, key=lambda x: x[0])
        return scored_notes[:80]
    except Exception as e:
        log_debug(f"Embedding search error: {e}")
        if _is_embedding_dimension_mismatch(e):
            return {"embedding_results": None, "error": "dimension_mismatch"}
        return None


class EmbeddingSearchWorker(QThread):
    """Worker thread for embedding search (fallback when taskman not used)."""
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, embedding_query, notes, config, db_path=None):
        super().__init__()
        self.embedding_query = embedding_query
        self.notes = notes
        self.config = config
        self.db_path = db_path

    def run(self):
        import hashlib
        try:
            import numpy as np
        except ImportError:
            self.error_signal.emit("numpy not available")
            self.finished_signal.emit(None)
            return
        try:
            embedding_list = get_embedding_for_query(self.embedding_query, self.config)
            if not embedding_list:
                self.finished_signal.emit(None)
                return
            query_embedding = np.array(embedding_list)
            total = len(self.notes)
            scored_notes = []
            progress_interval = max(50, total // 40)
            for idx, note in enumerate(self.notes):
                if self.isInterruptionRequested():
                    self.finished_signal.emit(None)
                    return
                if idx % progress_interval == 0 or idx == total - 1:
                    pct = int(100 * (idx + 1) / total) if total else 0
                    self.progress_signal.emit(idx + 1, total, f"Embedding search: {idx + 1}/{total} ({pct}%)")
                content_hash = note.get('content_hash')
                if content_hash is None:
                    content_hash = hashlib.md5(note['content'].encode()).hexdigest()
                note_embedding = load_embedding(note.get('id'), content_hash, db_path=getattr(self, 'db_path', None))
                if note_embedding is not None:
                    emb = np.array(note_embedding)
                    if len(emb) != len(query_embedding):
                        log_debug(f"EmbeddingSearchWorker: skipping note {note.get('id')} (dimension mismatch query={len(query_embedding)} stored={len(emb)})")
                        continue
                    similarity = np.dot(query_embedding, emb) / (
                        max(np.linalg.norm(query_embedding) * np.linalg.norm(emb), 1e-9)
                    )
                    score = (similarity + 1) * 50
                    scored_notes.append((score, note))
            scored_notes.sort(reverse=True, key=lambda x: x[0])
            self.finished_signal.emit(scored_notes[:80])
        except Exception as e:
            log_debug(f"EmbeddingSearchWorker error: {e}")
            if _is_embedding_dimension_mismatch(e):
                msg = (
                    "Embedding dimension mismatch: your notes were embedded with a different engine "
                    "(e.g. Voyage). Run Create/Update Embeddings with your current engine (Ollama) to enable hybrid search."
                )
                self.error_signal.emit(msg)
            else:
                self.error_signal.emit(str(e)[:200])
            self.finished_signal.emit(None)


MAX_RERANK_COUNT = 15  # Limit rerank to top 15 to avoid CPU bottleneck (~2s with 50 notes)
RRF_K = 60  # Reciprocal Rank Fusion constant (standard in retrieval literature; 1/(k+rank) per list)


def _semantic_chunk_text(text, target_size=500):
    """Split text into chunks of ~target_size chars at sentence boundaries. Returns list of non-empty strings."""
    import re
    text = (text or "").strip()
    if not text or len(text) <= target_size:
        return [text] if text else []
    chunks = []
    # Sentence boundary: . ! ? followed by space or end; or newline
    pattern = re.compile(r'(?<=[.!?\n])\s+|\n+')
    start = 0
    while start < len(text):
        end = min(start + target_size, len(text))
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        # Find last sentence boundary in this window
        segment = text[start:end]
        last_dot = segment.rfind('.')
        last_excl = segment.rfind('!')
        last_q = segment.rfind('?')
        last_nl = segment.rfind('\n')
        best = max(last_dot, last_excl, last_q, last_nl)
        if best >= target_size // 2:
            end = start + best + 1
            chunk = text[start:end].strip()
        else:
            chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks if chunks else [text]


def _do_rerank(query, scored_notes, top_k, search_config):
    """
    Re-rank top results using a cross-encoder (gold standard for NotebookLM-style accuracy).
    Uses top_k=15 by default to avoid CPU bottleneck. Blends cross-encoder scores with pre-rerank.
    Returns (scored_notes, success).
    """
    import json
    import os
    import subprocess
    top_k = min(top_k, MAX_RERANK_COUNT)
    top_notes = scored_notes[:top_k]
    if not top_notes:
        return scored_notes, False
    pre_scores = {note['id']: score for score, note in top_notes}
    contents = [note.get('content', '')[:512] for _, note in top_notes]
    rerank_python = (search_config.get('rerank_python_path') or '').strip()
    if rerank_python:
        python_exe = rerank_python
        if os.path.isdir(rerank_python):
            python_exe = os.path.join(rerank_python, "python.exe")
            if not os.path.isfile(python_exe):
                python_exe = os.path.join(rerank_python, "python")
        if os.path.isfile(python_exe):
            addon_dir = os.path.dirname(os.path.abspath(__file__))
            helper_path = os.path.join(addon_dir, "rerank_helper.py")
            if os.path.isfile(helper_path):
                try:
                    payload = json.dumps({"query": query, "contents": contents})
                    creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                    proc = subprocess.Popen(
                        [python_exe, helper_path],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, creationflags=creationflags
                    )
                    out, err = proc.communicate(input=payload, timeout=60)
                    if proc.returncode != 0:
                        log_debug(f"Rerank helper failed: {err or out}")
                        return scored_notes, False
                    data = json.loads(out)
                    if "error" in data:
                        log_debug(f"Rerank helper error: {data['error']}")
                        return scored_notes, False
                    scores = data.get("scores", [])
                    if len(scores) != len(top_notes):
                        return scored_notes, False
                    reranked = list(zip(scores, [note for _, note in top_notes]))
                    reranked.sort(reverse=True, key=lambda x: x[0])
                    min_s, max_s = min(s[0] for s in reranked), max(s[0] for s in reranked)
                    span = max_s - min_s if max_s > min_s else 1
                    normalized = [(50 + 50 * (s - min_s) / span, note) for s, note in reranked]
                    # Blend for ordering; display % = cross-encoder score so "answers the question" gets high, tangential gets low
                    blend = []
                    for rn, note in normalized:
                        pre = pre_scores.get(note['id'], 50)
                        blended = 0.5 * rn + 0.5 * pre
                        blend.append((blended, note))
                        # rn is 50-100; show as 0-100 so relevance reflects "answers this question" not just topic similarity
                        note['_display_relevance'] = max(0, min(100, round((rn - 50) * 2)))
                    blend.sort(reverse=True, key=lambda x: x[0])
                    max_b = blend[0][0] if blend else 1
                    scaled = [(b / max_b * 100.0, note) for b, note in blend]
                    # Soft floor: notes that were top-15 by pre-rerank don't show below 55%
                    top_pre_ids = set(nid for _, nid in sorted(pre_scores.items(), key=lambda x: -x[1])[:15])
                    scaled = [(max(pct, 55.0) if note['id'] in top_pre_ids else pct, note) for pct, note in scaled]
                    scaled.sort(reverse=True, key=lambda x: x[0])
                    # Renormalize _display_relevance so top note(s) show 100% and rest spread below
                    max_d = max((note.get('_display_relevance') or 0) for _, note in scaled)
                    if max_d > 0:
                        for _, note in scaled:
                            p = note.get('_display_relevance')
                            if p is not None:
                                note['_display_relevance'] = max(0, min(100, round(100 * p / max_d)))
                    # Rest notes weren't reranked; no _display_relevance (UI will use score)
                    rest = [(0.0, note) for _, note in scored_notes[top_k:]]
                    return scaled + rest, True
                except subprocess.TimeoutExpired:
                    proc.kill()
                    log_debug("Rerank helper timed out")
                    return scored_notes, False
                except Exception as e:
                    log_debug(f"Rerank subprocess failed: {e}")
                    return scored_notes, False
    try:
        _patch_colorama_early()
        _ensure_stderr_patched()
        from sentence_transformers import CrossEncoder
    except ImportError:
        log_debug("Cross-encoder re-ranking skipped: sentence-transformers not installed")
        try:
            from aqt.utils import showInfo
            showInfo(
                "sentence-transformers is not installed.\n\n"
                "Cross-Encoder re-ranking is disabled. To enable it, click "
                "'Install Dependencies' in the AI Search menu (or in Settings → Search & Embeddings)."
            )
        except Exception:
            pass
        return scored_notes, False
    except OSError as e:
        log_debug(f"Cross-encoder re-ranking skipped: library load failed ({e})")
        try:
            from aqt.utils import showInfo
            showInfo(
                "Reranking was skipped: sentence-transformers/torch could not be loaded "
                "(e.g. DLL error on Windows). Search results are unchanged.\n\n"
                "To use reranking, set 'Python for Cross-Encoder' in Settings → Search & Embeddings to a Python "
                "that has sentence-transformers working, then use 'Install into external Python'."
            )
        except Exception:
            pass
        return scored_notes, False
    try:
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, note['content'][:512]) for _, note in top_notes]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
        reranked = list(zip(scores, [note for _, note in top_notes]))
        reranked.sort(reverse=True, key=lambda x: x[0])
        min_s, max_s = min(s[0] for s in reranked), max(s[0] for s in reranked)
        span = max_s - min_s if max_s > min_s else 1
        normalized = [(50 + 50 * (s - min_s) / span, note) for s, note in reranked]
        # Blend for ordering; display % = cross-encoder score so "answers the question" gets high, tangential gets low
        blend = []
        for rn, note in normalized:
            pre = pre_scores.get(note['id'], 50)
            blended = 0.5 * rn + 0.5 * pre
            blend.append((blended, note))
            # rn is 50-100; show as 0-100 so relevance reflects "answers this question" not just topic similarity
            note['_display_relevance'] = max(0, min(100, round((rn - 50) * 2)))
        blend.sort(reverse=True, key=lambda x: x[0])
        max_b = blend[0][0] if blend else 1
        scaled = [(b / max_b * 100.0, note) for b, note in blend]
        # Soft floor: notes that were top-15 by pre-rerank don't show below 55%
        top_pre_ids = set(nid for _, nid in sorted(pre_scores.items(), key=lambda x: -x[1])[:15])
        scaled = [(max(pct, 55.0) if note['id'] in top_pre_ids else pct, note) for pct, note in scaled]
        scaled.sort(reverse=True, key=lambda x: x[0])
        # Renormalize _display_relevance so top note(s) show 100% and rest spread below
        max_d = max((note.get('_display_relevance') or 0) for _, note in scaled)
        if max_d > 0:
            for _, note in scaled:
                p = note.get('_display_relevance')
                if p is not None:
                    note['_display_relevance'] = max(0, min(100, round(100 * p / max_d)))
        # Rest notes weren't reranked; no _display_relevance (UI will use score)
        rest = [(0.0, note) for _, note in scored_notes[top_k:]]
        return scaled + rest, True
    except Exception as e:
        log_debug(f"Cross-encoder re-ranking failed: {e}")
        return scored_notes, False


class RerankCheckWorker(QThread):
    """Worker thread for checking sentence-transformers availability so Settings doesn't freeze."""
    finished_signal = pyqtSignal(bool)  # available

    def __init__(self, dialog, python_path=None):
        super().__init__()
        self._dialog = dialog
        self._python_path = python_path

    def run(self):
        try:
            result = self._dialog._check_rerank_available(python_path=self._python_path)
            self.finished_signal.emit(result)
        except Exception:
            self.finished_signal.emit(False)


class KeywordFilterWorker(QThread):
    """Worker thread for keyword_filter so search doesn't freeze the main thread."""
    finished_signal = pyqtSignal(object)  # result from keyword_filter

    def __init__(self, dialog, query, notes):
        super().__init__()
        self._dialog = dialog
        self._query = query
        self._notes = notes

    def run(self):
        try:
            result = self._dialog.keyword_filter(self._query, self._notes)
            self.finished_signal.emit(result)
        except Exception as e:
            log_debug(f"KeywordFilterWorker error: {e}")
            self.finished_signal.emit(None)


class KeywordFilterContinueWorker(QThread):
    """Worker thread for keyword_filter_continue so combining results doesn't freeze the UI. Emits progress."""
    progress_signal = pyqtSignal(int, int, str)  # current, total, message
    finished_signal = pyqtSignal(object)  # result

    def __init__(self, dialog, state, embedding_results):
        super().__init__()
        self._dialog = dialog
        self._state = state
        self._embedding_results = embedding_results

    def run(self):
        try:
            def progress_callback(idx, total):
                self.progress_signal.emit(idx, total, f"Combining results... {idx}/{total}")

            result = self._dialog.keyword_filter_continue(
                self._state, self._embedding_results, progress_callback=progress_callback
            )
            self.finished_signal.emit(result)
        except Exception as e:
            log_debug(f"KeywordFilterContinueWorker error: {e}")
            self.finished_signal.emit(None)


class RerankWorker(QThread):
    """Worker thread for cross-encoder reranking so the UI stays responsive."""
    finished_signal = pyqtSignal(object, bool)  # (scored_notes, success)

    def __init__(self, query, scored_notes, top_k, search_config):
        super().__init__()
        self.query = query
        self.scored_notes = scored_notes
        self.top_k = top_k
        self.search_config = search_config

    def run(self):
        try:
            scored_notes, success = _do_rerank(self.query, self.scored_notes, self.top_k, self.search_config)
            self.finished_signal.emit(scored_notes, success)
        except Exception as e:
            log_debug(f"RerankWorker error: {e}")
            self.finished_signal.emit(self.scored_notes, False)


class RelevanceRerankWorker(QThread):
    """Worker for re-ranking notes by similarity to the AI answer (relevance-from-answer). Runs embeddings off the main thread to avoid lag."""
    progress_signal = pyqtSignal(int, str)   # percent, message
    finished_signal = pyqtSignal(object)     # new all_scored_notes or None on failure

    def __init__(self, answer_text, note_texts, all_scored_notes, config):
        super().__init__()
        self.answer_text = answer_text
        self.note_texts = note_texts
        self.all_scored_notes = all_scored_notes
        self.config = config

    def run(self):
        try:
            import numpy as np
            self.progress_signal.emit(5, "Re-ranking by relevance… (embedding answer)")
            answer_emb = get_embedding_for_query(self.answer_text, self.config)
            if not answer_emb:
                self.finished_signal.emit(None)
                return
            self.progress_signal.emit(20, "Re-ranking by relevance… (embedding notes)")
            note_embs = get_embeddings_batch(self.note_texts, input_type="document", config=self.config)
            if not note_embs or len(note_embs) != len(self.all_scored_notes):
                self.finished_signal.emit(None)
                return
            self.progress_signal.emit(70, "Re-ranking by relevance… (scoring)")
            answer_vec = np.array(answer_emb, dtype=float)
            norm_a = max(np.linalg.norm(answer_vec), 1e-9)
            new_scores = []
            for i, (_, note) in enumerate(self.all_scored_notes):
                ne = np.array(note_embs[i], dtype=float)
                norm_n = max(np.linalg.norm(ne), 1e-9)
                sim = float(np.dot(answer_vec, ne) / (norm_a * norm_n))
                pct = max(0, min(100, round((sim + 1) * 50)))
                note['_display_relevance'] = pct
                new_scores.append((float(pct), note))
            new_scores.sort(reverse=True, key=lambda x: x[0])
            if new_scores:
                max_pct = new_scores[0][0]
                if max_pct > 0:
                    for score, note in new_scores:
                        note['_display_relevance'] = max(0, min(100, round(100 * (note['_display_relevance'] or 0) / max_pct)))
                    new_scores = [(100.0 if i == 0 else (note['_display_relevance'] or 0), note) for i, (_, note) in enumerate(new_scores)]
                    new_scores.sort(reverse=True, key=lambda x: x[0])
            self.progress_signal.emit(100, "Re-ranking by relevance… (done)")
            self.finished_signal.emit(new_scores)
        except Exception as e:
            log_debug(f"RelevanceRerankWorker error: {e}")
            self.finished_signal.emit(None)


class AskAIWorker(QThread):
    """Run ask_ai in a background thread so the main thread stays responsive (no 'Not Responding')."""
    success_signal = pyqtSignal(object, object)  # (answer, relevant_indices)
    error_signal = pyqtSignal(str)

    def __init__(self, dialog, query, context_notes, context, config):
        super().__init__()
        self._dialog = dialog
        self._query = query
        self._context_notes = context_notes
        self._context = context
        self._config = config

    def run(self):
        try:
            answer, relevant_indices = self._dialog.ask_ai(
                self._query, self._context_notes, self._context, self._config
            )
            self.success_signal.emit(answer, relevant_indices)
        except Exception as e:
            log_debug(f"AskAIWorker error: {e}")
            self.error_signal.emit(str(e))


def _get_query_focus_instruction(query):
    """Map query to a short focus instruction so the model structures the answer accordingly (NotebookLM-style).
    Returns None if no mapping matches."""
    if not query or not isinstance(query, str):
        return None
    q = query.lower().strip()
    # First match wins; order by specificity if needed
    focus_map = [
        ("receptor", "Focus: The question is about receptors; list receptor types/subtypes with citations for each."),
        ("mechanism", "Focus: The question is about mechanism; explain step-by-step with citations."),
        ("pathway", "Focus: The question is about pathways; describe with citations."),
        ("cause", "Focus: The question is about causes; list them with citations."),
        ("causes", "Focus: The question is about causes; list them with citations."),
        ("treatment", "Focus: The question is about treatment; list options with citations."),
        ("treat ", "Focus: The question is about treatment; list options with citations."),
        ("indication", "Focus: The question is about indications; list with citations."),
        ("side effect", "Focus: The question is about side effects; list with citations."),
        ("adverse effect", "Focus: The question is about adverse effects; list with citations."),
        ("diagnosis", "Focus: The question is about diagnosis; list criteria or steps with citations."),
        ("diagnose", "Focus: The question is about diagnosis; list criteria or steps with citations."),
        ("symptom", "Focus: The question is about symptoms; list with citations."),
    ]
    for keyword, instruction in focus_map:
        if keyword in q:
            return instruction
    return None


def _build_anthropic_prompt_parts(query, context, focus_instruction=None):
    """Build (system_blocks, user_content) for Anthropic with prompt caching. System + context use cache_control."""
    system_instruction = """You are an assistant for question-answering over provided notes. Use ONLY the numbered notes below as your factual source (you may add brief connecting logic, but no outside facts).
If the notes contain at least some relevant information, give the **best partial answer you can** based only on these notes and then briefly mention what is missing.
Only if the notes are essentially unrelated to the question, say exactly: "The provided notes do not contain enough information to answer this."

Rules:
- Base every claim strictly on these notes. One sentence or bullet per idea is fine.
- Write in a clear, exam-oriented style: use bullet points (•) for key points; use 2-space indented bullets for sub-points. Use **double asterisks** around important terms (diagnoses, drugs, criteria). Do not use ## for headings—use a single bold line with ● then bullets underneath.
- NUMBERED LISTS: When the question or notes refer to a numbered list (e.g. six childhood exanthems, 1st–6th disease, steps 1–6 of ventricular partitioning), **always** present in **strict numerical order**: 1st, then 2nd, then 3rd, then 4th, then 5th, then 6th. **Include every step or number that any note describes**—if Note X explicitly mentions "3rd step" or "third step", you MUST include it in your answer and cite it [X]. Do not skip steps; do not reorder by relevance.
- INLINE CITATIONS: Cite the supporting note(s) using [N] or [N,M] where N is between 1 and the number of notes below only. Do not use citation numbers outside that range.
- At the end, on one line, list all note numbers you cited. Format: RELEVANT_NOTES: 1,3,5"""
    num_notes = context.count("Note ")  # approximate; we pass explicit N in caller if needed
    context_block = f"""Context information is below. There are notes numbered Note 1, Note 2, ... (cite only using numbers 1 to the number of notes below).
---------------------
{context}
---------------------"""
    system_blocks = [
        {"type": "text", "text": system_instruction},
        {"type": "text", "text": context_block, "cache_control": {"type": "ephemeral"}},
    ]
    user_content = f"""Given the context information and not prior knowledge, answer the question.

Question: {query}"""
    if focus_instruction:
        user_content += "\n" + focus_instruction
    return system_blocks, user_content


class AnthropicStreamWorker(QThread):
    """Worker thread for Anthropic streaming. Emits text chunks for real-time UI updates."""
    chunk_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, api_key, model, system_blocks, user_content, notes):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.system_blocks = system_blocks
        self.user_content = user_content
        self.notes = notes

    def run(self):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            full_text = ""
            with client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=self.system_blocks,
                messages=[{"role": "user", "content": self.user_content}],
            ) as stream:
                for text in stream.text_stream:
                    full_text += text
                    self.chunk_signal.emit(text)
            self.done_signal.emit(full_text)
        except Exception as e:
            log_debug(f"AnthropicStreamWorker error: {e}")
            self.error_signal.emit(str(e))


# END OF PART 2 - PART 3: Methods below are indented under EmbeddingSearchWorker
# but are copied to AISearchDialog at module load (see _aisearch_methods_from_worker).

    def _get_answer_source_text(self, config):
        """Return a short hint: where the answer came from (online API name or local model)."""
        if not config:
            return ""
        provider = config.get("provider", "openai")
        if provider == "ollama":
            sc = config.get("search_config") or {}
            model = (sc.get("ollama_chat_model") or "llama3.2").strip()
            return f"Ollama (local) — {model}"
        names = {
            "anthropic": "Anthropic (Claude)",
            "openai": "OpenAI (GPT)",
            "google": "Google (Gemini)",
            "openrouter": "OpenRouter",
            "custom": "Custom API",
        }
        name = names.get(provider, "API")
        model = self.get_best_model(provider)
        return f"{name} — {model}"

    def _on_embedding_search_progress(self, current, total, message):
        """Update status and progress bar while embedding search runs in background."""
        try:
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText(message)
            if hasattr(self, 'search_progress_bar') and self.search_progress_bar and total > 0:
                self.search_progress_bar.setRange(0, total)
                self.search_progress_bar.setValue(current)
                self.search_progress_bar.setVisible(True)
            if hasattr(self, 'search_progress_label') and self.search_progress_label:
                self.search_progress_label.setText(f"{current}/{total}")
                self.search_progress_label.setVisible(True)
        except Exception:
            pass

    def _show_busy_progress(self, message=""):
        """Show indeterminate progress bar and optional label during long operations (re-rank, AI call, load)."""
        self._show_centile_progress(message, 0)

    def _show_centile_progress(self, message="", percent=0):
        """Show 0–100% progress bar and label. Use for estimated or real progress during long operations."""
        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:
            self.search_progress_bar.setRange(0, 100)
            self.search_progress_bar.setValue(max(0, min(100, round(percent))))
            self.search_progress_bar.setVisible(True)
        if hasattr(self, 'search_progress_label') and self.search_progress_label:
            self.search_progress_label.setText(message)
            self.search_progress_label.setVisible(True)
        self._last_progress_message = message

    def _start_estimated_progress_timer(self, duration_sec, start_pct=5, end_pct=95):
        """Advance progress bar from start_pct to end_pct over duration_sec (est. wait). Call _stop_estimated_progress_timer when done."""
        import time
        self._stop_estimated_progress_timer()
        self._progress_estimate_active = True
        self._progress_estimate_start = time.time()
        self._progress_estimate_duration = max(1, duration_sec)
        self._progress_estimate_start_pct = start_pct
        self._progress_estimate_end_pct = end_pct

        def _tick():
            if not getattr(self, '_progress_estimate_active', False):
                return
            elapsed = time.time() - getattr(self, '_progress_estimate_start', 0)
            dur = getattr(self, '_progress_estimate_duration', 30)
            s = getattr(self, '_progress_estimate_start_pct', 5)
            e = getattr(self, '_progress_estimate_end_pct', 95)
            pct = s + (elapsed / dur) * (e - s)
            pct = max(s, min(e, pct))
            msg = getattr(self, '_last_progress_message', '')
            self._show_centile_progress(msg, pct)
            if elapsed < dur:
                QTimer.singleShot(500, _tick)

        QTimer.singleShot(300, _tick)

    def _stop_estimated_progress_timer(self):
        """Stop the estimated progress timer (e.g. when the long operation finishes)."""
        self._progress_estimate_active = False

    def _hide_busy_progress(self):
        """Hide progress bar and label; reset bar to deterministic range for next use."""
        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:
            self.search_progress_bar.setRange(0, 100)
            self.search_progress_bar.setValue(0)
            self.search_progress_bar.setVisible(False)
        if hasattr(self, 'search_progress_label') and self.search_progress_label:
            self.search_progress_label.setText("")
            self.search_progress_label.setVisible(False)

    def _on_embedding_search_finished(self, embedding_results):
        """Embedding search worker finished; continue with scoring and display."""
        try:
            if hasattr(self, 'search_progress_bar') and self.search_progress_bar:
                self.search_progress_bar.setVisible(False)
            if hasattr(self, 'search_progress_label') and self.search_progress_label:
                self.search_progress_label.setVisible(False)
            state = getattr(self, '_search_pending_state', None)
            notes = getattr(self, '_search_pending_notes', None)
            if state is None or notes is None:
                self.status_label.setText("Ready")
                self.search_btn.setEnabled(True)
                return
            combine_in_background = False
            # Handle dimension-mismatch dict from _run_embedding_search_sync
            if isinstance(embedding_results, dict) and "error" in embedding_results:
                setattr(self, "_last_embedding_error", embedding_results.get("error"))
                embedding_results = embedding_results.get("embedding_results")
            else:
                setattr(self, "_last_embedding_error", None)
            # Run keyword_filter_continue in a worker so the UI shows progress and does not freeze.
            if embedding_results:
                if hasattr(self, 'status_label') and self.status_label:
                    self.status_label.setText("Combining results...")
                if hasattr(self, 'search_progress_bar') and self.search_progress_bar:
                    self.search_progress_bar.setVisible(True)
                    self.search_progress_bar.setRange(0, 100)
                    self.search_progress_bar.setValue(0)
                if hasattr(self, 'search_progress_label') and self.search_progress_label:
                    self.search_progress_label.setVisible(True)
                QApplication.processEvents()
                self._keyword_filter_continue_worker = KeywordFilterContinueWorker(self, state, embedding_results)
                self._keyword_filter_continue_worker.progress_signal.connect(self._on_embedding_search_progress)
                self._keyword_filter_continue_worker.finished_signal.connect(
                    lambda result: self._on_keyword_filter_continue_done(result, state)
                )
                self._keyword_filter_continue_worker.start()
                combine_in_background = True
            else:
                result = self.keyword_filter_continue(state, embedding_results)
                self._on_keyword_filter_continue_done(result, state)
        except Exception as e:
            log_debug(f"Embedding search finish error: {e}")
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText("Error occurred")
            self.answer_box.setText(f"❌ Error after embedding search:\n{str(e)}")
        finally:
            if not combine_in_background and not getattr(self, '_pending_rerank', False):
                self.search_btn.setEnabled(True)
            QApplication.processEvents()

    def _on_keyword_filter_continue_done(self, result, state):
        """Called when keyword_filter_continue finishes (worker or inline). Starts rerank or continues to display."""
        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:
            self.search_progress_bar.setVisible(False)
        if hasattr(self, 'search_progress_label') and self.search_progress_label:
            self.search_progress_label.setVisible(False)
        notes = state.get("notes") or getattr(self, '_search_pending_notes', None)
        if result is None or notes is None:
            self.status_label.setText("Ready")
            self.search_btn.setEnabled(True)
            return
        if isinstance(result, tuple) and result[0] == "PENDING_RERANK":
            _, notes, scored_notes, effective_method, _ = result
            self._pending_rerank = True
            search_config = state.get("search_config") or {}
            self._rerank_continue = (notes, effective_method, search_config)
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText("Re-ranking results... (may take 10–30 s)")
            self._show_centile_progress("Re-ranking…", 0)
            self._start_estimated_progress_timer(25, 5, 95)
            query = state.get("query", "")
            try:
                from aqt.operations import QueryOp
                op = QueryOp(
                    parent=mw,
                    op=lambda col: _do_rerank(query, scored_notes, MAX_RERANK_COUNT, search_config),
                    success=lambda pair: self._on_rerank_done(pair[0], pair[1]),
                )
                op.run_in_background()
            except Exception:
                self._rerank_worker = RerankWorker(query, scored_notes, MAX_RERANK_COUNT, search_config)
                self._rerank_worker.finished_signal.connect(self._on_rerank_done)
                self._rerank_worker.start()
            return
        scored_notes, effective_method, total_above_threshold = result
        self._perform_search_continue(notes, scored_notes, effective_method, total_above_threshold)
        self.search_btn.setEnabled(True)

    def _on_embedding_search_error(self, error_msg):
        """Handle embedding search worker error."""
        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:
            self.search_progress_bar.setVisible(False)
        if hasattr(self, 'search_progress_label') and self.search_progress_label:
            self.search_progress_label.setVisible(False)
        self.status_label.setText("Error occurred")
        self.answer_box.setText(f"❌ Embedding search failed:\n{error_msg}")
        self.search_btn.setEnabled(True)

    def _perform_search_continue(self, notes, scored_notes, effective_method, total_above_threshold):
        """Continue search after keyword_filter (or after embedding worker): display results, call AI, etc."""
        config = load_config()
        query = getattr(self, 'current_query', '')
        log_debug(f"Filtered to {len(scored_notes)} potentially relevant notes (method: {effective_method}, total above threshold: {total_above_threshold})")
        self.all_scored_notes = scored_notes
        self._total_above_threshold = total_above_threshold
        self._last_search_method = effective_method
        if hasattr(self, 'search_method_result_label'):
            search_config = config.get("search_config") or {}
            mode = (search_config.get("relevance_mode") or "").strip().lower()
            if mode == "focused":
                mode_display = "Focused"
            elif mode == "broad":
                mode_display = "Broad"
            elif mode:
                mode_display = mode.capitalize()
            else:
                mode_display = "Balanced"
            engine = (search_config.get("embedding_engine") or "ollama").strip().lower()
            engine_display = {
                "ollama": "Ollama (local)",
                "voyage": "Voyage AI",
                "openai": "OpenAI",
                "cohere": "Cohere",
            }.get(engine, engine or "unknown")
            label_text = f"Results from: {effective_method} · {mode_display} · Embeddings: {engine_display}"
            self.search_method_result_label.setText(label_text)
            self.search_method_result_label.setVisible(True)
            # When user chose embedding/hybrid but we fell back to keyword, add a hint
            if "Keyword only" in effective_method and getattr(self, '_last_requested_search_method', None) in ('embedding', 'hybrid'):
                err = getattr(self, "_last_embedding_error", None)
                if err == "dimension_mismatch":
                    engine = (config.get("search_config") or {}).get("embedding_engine") or "ollama"
                    engine_display = {"ollama": "Ollama", "voyage": "Voyage AI", "openai": "OpenAI", "cohere": "Cohere"}.get(engine, engine)
                    hint = format_dimension_mismatch_hint(engine_display)
                else:
                    hint = "embedding unavailable — run Create/Update Embeddings in Settings → Search & Embeddings, or check API key"
                self.search_method_result_label.setText(f"Results from: {effective_method} ({hint})")
        if not scored_notes:
            n_searched = getattr(self, 'total_notes_searched', None) or len(set(n['id'] for n in notes))
            self.answer_box.setText(f"No notes found matching keywords from your query. Searched {n_searched} notes ({getattr(self, 'fields_description', 'Text & Extra')}).")
            if hasattr(self, 'answer_source_label'):
                self.answer_source_label.setText("")
            self.status_label.setText("No matches found")
            return
        # Use un-aggregated list for AI context when we have chunks so the AI can cite specific sections
        raw_for_context = getattr(self, '_scored_notes_for_context', None)
        if raw_for_context:
            relevant_notes = [note for _, note in raw_for_context]
        else:
            relevant_notes = [note for _, note in scored_notes]
        history_result = load_search_history(query)
        used_history = False
        if history_result:
            log_debug("Using cached search result from history")
            if hasattr(self, 'search_method_result_label'):
                if getattr(self, '_last_search_method', None) == "Hybrid":
                    self.search_method_result_label.setText("Results from: cache (same query as before)")
                    self.search_method_result_label.setVisible(True)
                else:
                    self.search_method_result_label.setVisible(False)
            if not hasattr(self, '_total_above_threshold'):
                self._total_above_threshold = len(self.all_scored_notes)
            self.status_label.setText("📚 Loading from cache... (saved AI API call)")
            QApplication.processEvents()
            answer = history_result.get('answer', '')
            relevant_indices = []
            used_history = True
            self._context_note_ids = history_result.get('context_note_ids') or []
            self._context_note_id_and_chunk = None  # History has no chunk info; use ctx order for Ref
            self._display_scored_notes = None  # History uses aggregated list
            if 'scored_notes' in history_result:
                history_scored = []
                note_id_map = {note['id']: note for _, note in scored_notes}
                for score, hist_note in history_result['scored_notes']:
                    note_id = hist_note.get('id')
                    if note_id in note_id_map:
                        history_scored.append((score, note_id_map[note_id]))
                if history_scored:
                    history_note_ids = {note['id'] for _, note in history_scored}
                    for score, note in scored_notes:
                        if note['id'] not in history_note_ids:
                            history_scored.append((score, note))
                    self.all_scored_notes = sorted(history_scored, reverse=True, key=lambda x: x[0])
                relevant_note_ids = set(history_result.get('relevant_note_ids', []))
                relevant_notes = [note for _, note in self.all_scored_notes]
                for idx, note in enumerate(relevant_notes):
                    if note['id'] in relevant_note_ids:
                        relevant_indices.append(idx)
                if not self._context_note_ids and self.all_scored_notes:
                    self._context_note_ids = [n['id'] for _, n in self.all_scored_notes]
            else:
                self._context_note_ids = [n['id'] for _, n in self.all_scored_notes] if self.all_scored_notes else []
        else:
            # Cap notes/chunks sent to the AI (avoids rate limits and token overflow; chunked results can be huge)
            search_config = config.get('search_config') or {}
            max_context = max(5, min(50, search_config.get('max_results', 12)))
            selected_ids = set(getattr(self, 'selected_note_ids', set()) or [])
            pinned_ids = set(getattr(self, '_pinned_note_ids', set()) or [])
            priority_ids = selected_ids | pinned_ids
            if priority_ids:
                prioritized = [n for n in relevant_notes if n['id'] in priority_ids]
                remaining = [n for n in relevant_notes if n['id'] not in priority_ids]
                context_notes = (prioritized + remaining)[:max_context]
            else:
                context_notes = list(relevant_notes)[:max_context]
            context_note_ids = [n['id'] for n in context_notes]
            # Store (note_id, chunk_index) in context order so Ref column and citation [N] match
            self._context_note_id_and_chunk = [(n['id'], n.get('chunk_index')) for n in context_notes]
            # Store (score, note) for each context item so display can show all refs the AI can cite
            if raw_for_context:
                note_to_score = {}
                for s, n in raw_for_context:
                    key = (n['id'], n.get('chunk_index'))
                    note_to_score[key] = s
                display_pairs = []
                for n in context_notes:
                    k = (n['id'], n.get('chunk_index'))
                    display_pairs.append((note_to_score.get(k) or 0, n))
                self._display_scored_notes = display_pairs
            else:
                self._display_scored_notes = None
            # Per-note limit: 0 = full content; >0 = truncate (Settings → Search & Embeddings → Max chars per note)
            context_chars_per_note = max(0, search_config.get('context_chars_per_note', 0))
            # When we have chunks, label sections so the AI can cite specific sections (e.g. [1], [2] for Note 1 section 2)
            def _context_line(i, n):
                chunk_idx = n.get('chunk_index')
                text = self.reveal_cloze_for_display(n['content'])
                if context_chars_per_note:
                    text = text[:context_chars_per_note]
                if chunk_idx is not None:
                    return f"Note {i+1} (section {chunk_idx + 1} of note ID {n['id']}): {text}"
                return f"Note {i+1}: {text}"
            context = "\n\n".join([_context_line(i, n) for i, n in enumerate(context_notes)])
            n_notes = len(context_notes)
            self.status_label.setText(f"Asking AI... (sending top {n_notes} notes, 10–30 s)")
            self._show_centile_progress(f"Asking AI… ({n_notes} notes)", 0)
            self._start_estimated_progress_timer(30, 5, 95)
            self.answer_box.setPlainText("Thinking…")
            QApplication.processEvents()
            provider = config.get('provider', 'openai')
            if provider == "anthropic":
                try:
                    self._start_anthropic_stream(query, context_notes, context, config, relevant_notes, scored_notes, context_note_ids)
                    return
                except Exception as e:
                    log_debug(f"Anthropic streaming not available, falling back to non-streaming: {e}")
            # Run ask_ai in background so UI stays responsive (avoids "Not Responding" during 10–300 s request)
            self._ask_ai_relevant_notes = relevant_notes
            self._ask_ai_scored_notes = scored_notes
            self._ask_ai_context_note_ids = context_note_ids
            self._ask_ai_used_history = used_history
            self._ask_ai_notes = notes
            self._ask_ai_config = config
            self._ask_ai_worker = AskAIWorker(self, query, context_notes, context, config)
            self._ask_ai_worker.success_signal.connect(self._on_ask_ai_success)
            self._ask_ai_worker.error_signal.connect(self._on_ask_ai_error)
            self._ask_ai_worker.finished.connect(self._on_ask_ai_worker_finished)
            self._ask_ai_worker.start()
            return

    def _on_ask_ai_worker_finished(self):
        """Clear worker reference after thread finishes."""
        self._ask_ai_worker = None

    def _on_ask_ai_error(self, error_msg):
        """Handle AI request failure (runs on main thread)."""
        log_debug(f"Error calling AI API: {error_msg}")
        config = getattr(self, '_ask_ai_config', None) or {}
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            if hasattr(self, 'answer_source_label'):
                self.answer_source_label.setText("")
            self.answer_box.setText(
                "⏱️ Request timed out. This could mean:\n"
                "• The API service is slow or overloaded\n"
                "• Your internet connection is unstable\n\n"
                "Try again or reduce the number of notes in your collection."
            )
        elif "401" in error_msg or "403" in error_msg or "authentication" in error_msg.lower():
            if hasattr(self, 'answer_source_label'):
                self.answer_source_label.setText("")
            self.answer_box.setText("🔑 Authentication Error:\nYour API key appears to be invalid or expired.\n\nPlease check your API key in Settings.")
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            if hasattr(self, 'answer_source_label'):
                self.answer_source_label.setText("")
            self.answer_box.setText("⚠️ Rate Limit Exceeded:\nYou've made too many requests.\n\nPlease wait a few minutes and try again.")
        elif config.get('provider') == 'ollama' and any(x in error_msg.lower() for x in ('connection', 'refused', 'connect', 'unreachable')):
            if hasattr(self, 'answer_source_label'):
                self.answer_source_label.setText("")
            self.answer_box.setText(
                "❌ Cannot reach Ollama.\n\n"
                "Make sure Ollama is running (ollama serve) and the URL in Settings → Search & Embeddings (Ollama URL) is correct."
            )
        else:
            self.answer_box.setText(f"❌ Error calling AI API:\n{error_msg}\n\nPlease check your API key and internet connection.")
        if hasattr(self, 'answer_source_label'):
            self.answer_source_label.setText("")
        self._stop_estimated_progress_timer()
        self._hide_busy_progress()
        self.status_label.setText("Error occurred")

    def _on_ask_ai_success(self, answer, relevant_indices):
        """Handle AI answer from worker (runs on main thread)."""
        self._stop_estimated_progress_timer()
        log_debug(f"AI returned answer (length: {len(answer) if answer else 0})")
        relevant_notes = getattr(self, '_ask_ai_relevant_notes', [])
        scored_notes = getattr(self, '_ask_ai_scored_notes', [])
        context_note_ids = getattr(self, '_ask_ai_context_note_ids', [])
        used_history = getattr(self, '_ask_ai_used_history', False)
        notes = getattr(self, '_ask_ai_notes', [])
        config = getattr(self, '_ask_ai_config', {})
        relevant_note_ids = [relevant_notes[idx]['id'] for idx in relevant_indices if 0 <= idx < len(relevant_notes)]
        save_search_history(getattr(self, 'current_query', ''), answer, relevant_note_ids, scored_notes, context_note_ids)
        self._context_note_ids = context_note_ids
        self.current_answer = answer
        ai_relevant_note_ids = set()
        for idx in relevant_indices:
            if 0 <= idx < len(relevant_notes):
                ai_relevant_note_ids.add(relevant_notes[idx]['id'])
        self._cited_note_ids = ai_relevant_note_ids
        improved_scored_notes = []
        for score, note in self.all_scored_notes:
            if note['id'] in ai_relevant_note_ids:
                improved_score = score * 2
            else:
                improved_score = score
            improved_scored_notes.append((improved_score, note))
        improved_scored_notes.sort(reverse=True, key=lambda x: x[0])
        if improved_scored_notes:
            max_boosted = improved_scored_notes[0][0]
            if max_boosted > 0:
                self.all_scored_notes = [(score / max_boosted * 100.0, note) for score, note in improved_scored_notes]
            else:
                self.all_scored_notes = improved_scored_notes
        else:
            self.all_scored_notes = improved_scored_notes
        search_config = config.get('search_config') or {}
        if search_config.get('relevance_from_answer'):
            answer_text = (answer[:8000]).strip()
            note_texts = []
            for _, note in self.all_scored_notes:
                raw = note.get('display_content') or note.get('content', '')
                text = self.reveal_cloze_for_display(raw) if hasattr(self, 'reveal_cloze_for_display') else raw
                note_texts.append((text[:2000]) if text else "")
            if answer_text and note_texts:
                self.status_label.setText("Re-ranking by relevance to answer…")
                self._show_centile_progress("Re-ranking by relevance…", 0)
                self._relevance_rerank_worker = RelevanceRerankWorker(answer_text, note_texts, self.all_scored_notes, config)
                self._relevance_rerank_worker.progress_signal.connect(lambda p, m: self._show_centile_progress(m, p))
                self._relevance_rerank_worker.finished_signal.connect(
                    lambda res: self._on_relevance_rerank_done(res, answer, config, used_history, notes)
                )
                self._relevance_rerank_worker.start()
                return
            try:
                self._rerank_by_relevance_to_answer(answer, config)
            except Exception as e:
                log_debug(f"Relevance-from-answer rerank failed: {e}")
        self._display_answer_and_notes_after_rerank(answer, config, used_history, notes)

    def _on_relevance_rerank_done(self, result, answer, config, used_history, notes):
        """Called when RelevanceRerankWorker finishes. Apply result and display answer/notes."""
        self._hide_busy_progress()
        if getattr(self, '_relevance_rerank_worker', None):
            self._relevance_rerank_worker = None
        if result is not None:
            self.all_scored_notes = result
        self._display_answer_and_notes_after_rerank(answer, config, used_history, notes)

    def _display_answer_and_notes_after_rerank(self, answer, config, used_history, notes):
        """Display formatted answer, filter notes, and update status (shared after rerank or when no rerank)."""
        log_debug("Displaying answer and filtering notes...")
        formatted_answer = self.format_answer(answer)
        self._last_formatted_answer = formatted_answer
        self.answer_box.setHtml(formatted_answer)
        if hasattr(self, 'answer_source_label'):
            src = self._get_answer_source_text(config)
            self.answer_source_label.setText(f"Answer from: {src}" if src else "")
        for attr in ('copy_answer_btn',):
            if hasattr(self, attr):
                getattr(self, attr).setEnabled(True)
        self.filter_and_display_notes()
        threshold = self.sensitivity_slider.value() if getattr(self, 'sensitivity_slider', None) else 0
        cache_indicator = " (📚 from cache)" if used_history else ""
        if self.all_scored_notes:
            n_searched = getattr(self, 'total_notes_searched', None) or len(set(n['id'] for n in notes))
            base_text = self.status_label.text() or ""
            suffix = f" (searched {n_searched} in {getattr(self, 'fields_description', 'Text & Extra')}){cache_indicator}"
            if " (searched " in base_text:
                base_text = base_text.split(" (searched ")[0]
            self.status_label.setText(base_text + suffix)
        else:
            self.status_label.setText(f"Found {len(self.all_scored_notes)} relevant notes{cache_indicator}")
        self._refresh_search_history()
        log_debug("Search completed successfully")

    def perform_search(self):
        """Perform search with proper error handling and UI updates"""
        log_debug("=== Perform Search Called ===")
        
        # Check for config
        config = self.get_config()
        log_debug(f"Retrieved config for search: {get_safe_config(config)}")
        
        if not config:
            log_debug("ERROR: No config found")
            self.answer_box.setText("Please configure your API key first. Click the ⚙ button.")
            if hasattr(self, 'answer_source_label'):
                self.answer_source_label.setText("")
            tooltip("API not configured")
            return
        
        query = self.search_input.toPlainText().strip()
        if not query:
            tooltip("Please enter a search query")
            return
        
        # Disable search button to prevent multiple clicks
        self.search_btn.setEnabled(False)
        self.status_label.setText("Searching notes... (this may take 10-30 seconds)")
        self.answer_box.clear()
        self.results_list.setRowCount(0)  # Clear table
        self._update_view_all_button_state()
        if hasattr(self, 'search_method_result_label'):
            self.search_method_result_label.setText("")
        self.total_notes_searched = None
        self._pinned_note_ids = set()
        self._cited_note_ids = set()  # clear until new answer has citations
        # Clear selected note IDs when starting new search
        if hasattr(self, 'selected_note_ids'):
            self.selected_note_ids.clear()
        for attr in ('copy_answer_btn',):
            if hasattr(self, attr):
                getattr(self, attr).setEnabled(False)
        # Disable toggle button when list is cleared
        if hasattr(self, 'toggle_select_btn'):
            self.toggle_select_btn.setEnabled(False)
        if hasattr(self, 'selected_count_label'):
            self.selected_count_label.setText("(0 selected)")
        self._last_formatted_answer = None
        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:
            self.search_progress_bar.setVisible(False)
        if hasattr(self, 'search_progress_label') and self.search_progress_label:
            self.search_progress_label.setVisible(False)
        QApplication.processEvents()
        
        try:
            search_config = config.get('search_config', {})
            self._last_requested_search_method = search_config.get('search_method', 'hybrid')
            self._search_pending_query = query
            self._search_pending_config = config
            self._search_pending_async = True
            self.status_label.setText("Loading notes...")
            self._show_centile_progress("Loading notes…", 0)
            self._start_estimated_progress_timer(30, 5, 95)
            log_debug("Starting to load notes in background...")
            from aqt.operations import QueryOp
            op = QueryOp(
                parent=mw,
                op=lambda col: get_notes_content_with_col(col, config),
                success=self._on_get_notes_done,
            )
            op.run_in_background()
            return
        except Exception as e:
            log_debug(f"Unexpected error in perform_search: {type(e).__name__}: {str(e)}")
            import traceback
            log_debug(f"Traceback: {traceback.format_exc()}")
            self.answer_box.setText(
                f"❌ Unexpected Error:\n{str(e)}\n\n"
                "Please check the debug log for details."
            )
            self.status_label.setText("Error occurred")
            self._search_pending_async = False
        finally:
            if not getattr(self, '_search_pending_async', False) and not getattr(self, '_pending_rerank', False):
                self.search_btn.setEnabled(True)
            self._hide_busy_progress()
            QApplication.processEvents()

    def _on_get_notes_done(self, payload):
        """Called when background get_notes_content_with_col finishes. Starts keyword_filter in worker."""
        import time
        self._search_pending_async = False
        self._stop_estimated_progress_timer()
        self._hide_busy_progress()
        if payload is None or not isinstance(payload, (list, tuple)) or len(payload) != 3:
            self.search_btn.setEnabled(True)
            self.status_label.setText("Ready")
            return
        notes, fields_description, cache_key = payload
        self.fields_description = fields_description
        self._cached_notes = notes
        self._cached_notes_key = cache_key
        unique_note_count = len(set(n['id'] for n in notes)) if notes else 0
        if not notes:
            self.answer_box.setText(f"No notes with {fields_description} content found in your collection.")
            if hasattr(self, 'answer_source_label'):
                self.answer_source_label.setText("")
            self.status_label.setText("Ready")
            self.search_btn.setEnabled(True)
            return
        self.total_notes_searched = unique_note_count
        self._search_pending_notes = notes
        self.status_label.setText(f"Filtering {unique_note_count} notes...")
        QApplication.processEvents()
        self._search_pending_async = True
        self._keyword_filter_worker = KeywordFilterWorker(self, self._search_pending_query, notes)
        self._keyword_filter_worker.finished_signal.connect(self._on_keyword_filter_done)
        self._keyword_filter_worker.start()

    def _on_keyword_filter_done(self, result):
        """Called when KeywordFilterWorker finishes. Handles PENDING_EMBEDDING, PENDING_RERANK, or direct result."""
        import time
        self._search_pending_async = False
        notes = getattr(self, '_search_pending_notes', None)
        if notes is None and hasattr(self, '_cached_notes'):
            notes = self._cached_notes
        config = getattr(self, '_search_pending_config', None) or load_config()
        query = getattr(self, '_search_pending_query', '')
        if result is None:
            self.status_label.setText("Error during search")
            self.search_btn.setEnabled(True)
            return
        if isinstance(result, tuple) and result[0] == "PENDING_EMBEDDING":
            # Embedding search will run in background worker; show progress and return
            _, embedding_query, notes_for_embedding, state = result
            self._search_pending_state = state
            self._search_pending_notes = notes
            setattr(self, "_last_embedding_error", None)
            self.current_query = state["query"]
            config = state["config"]
            if hasattr(self, 'search_progress_bar') and self.search_progress_bar:
                self.search_progress_bar.setVisible(True)
                self.search_progress_bar.setRange(0, 100)
                self.search_progress_bar.setValue(0)
            if hasattr(self, 'search_progress_label') and self.search_progress_label:
                self.search_progress_label.setVisible(True)
                self.search_progress_label.setText("Starting embedding search...")
            self.status_label.setText("Embedding search: starting...")
            db_path = get_embeddings_db_path()
            self._embedding_search_worker = EmbeddingSearchWorker(embedding_query, notes_for_embedding, config, db_path=db_path)
            self._embedding_search_worker.progress_signal.connect(self._on_embedding_search_progress)
            self._embedding_search_worker.finished_signal.connect(self._on_embedding_search_finished)
            self._embedding_search_worker.error_signal.connect(self._on_embedding_search_error)
            # Use QThread worker so embedding search always runs off the main thread and does not freeze the UI.
            self._embedding_search_worker.start()
            return
        if isinstance(result, tuple) and result[0] == "PENDING_RERANK":
            # Rerank in background so UI stays responsive
            _, scored_notes, effective_method, _total_above, notes = result
            self.current_query = query
            self._pending_rerank = True
            search_config = config.get('search_config') or {}
            self._rerank_continue = (notes, effective_method, search_config)
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText("Re-ranking results... (may take 10–30 s)")
            self._show_centile_progress("Re-ranking…", 0)
            self._start_estimated_progress_timer(25, 5, 95)
            try:
                from aqt.operations import QueryOp
                op = QueryOp(
                    parent=mw,
                    op=lambda col: _do_rerank(query, scored_notes, MAX_RERANK_COUNT, search_config),
                    success=lambda pair: self._on_rerank_done(pair[0], pair[1]),
                )
                op.run_in_background()
            except Exception:
                self._rerank_worker = RerankWorker(query, scored_notes, MAX_RERANK_COUNT, search_config)
                self._rerank_worker.finished_signal.connect(self._on_rerank_done)
                self._rerank_worker.start()
            return
        scored_notes, effective_method, total_above_threshold = result
        self.current_query = query
        self._perform_search_continue(notes, scored_notes, effective_method, total_above_threshold)
        self.search_btn.setEnabled(True)

    def _on_rerank_done(self, scored_notes, success):
        """Called when RerankWorker finishes; apply min_relevance/max_results and continue search."""
        self._pending_rerank = False
        self._stop_estimated_progress_timer()
        self._hide_busy_progress()
        self.search_btn.setEnabled(True)
        notes, effective_method, search_config = getattr(self, '_rerank_continue', (None, '', {}))
        if notes is None:
            return
        MAX_STORED_FOR_MODES = 100
        min_relevance = max(15, min(75, search_config.get('min_relevance_percent', 55)))
        min_relevance_stored = min(20, min_relevance)
        all_above_stored = [x for x in scored_notes if x[0] >= min_relevance_stored]
        scored_notes = all_above_stored[:MAX_STORED_FOR_MODES] if all_above_stored else scored_notes[:5]
        total_above_threshold = sum(1 for x in scored_notes if x[0] >= min_relevance)
        self._perform_search_continue(notes, scored_notes, effective_method, total_above_threshold)
    
    def _rerank_with_cross_encoder(self, query, scored_notes, top_k=15):
        """
        Re-rank top results using a cross-encoder (delegates to _do_rerank).
        Limited to top 15 to avoid CPU bottleneck. Use RerankWorker for non-blocking UI.
        """
        config = load_config()
        sc = config.get('search_config') or {}
        top_k = min(top_k, MAX_RERANK_COUNT)
        return _do_rerank(query, scored_notes, top_k, sc)

    def _passes_focused_balanced_broad(
        self,
        matched_keywords,
        final_score,
        emb_score,
        max_emb_score,
        keywords,
        search_method,
        embeddings_available,
        min_emb_frac=0.25,
        very_high_emb_frac=0.9,
    ):
        """Compute whether a note would pass Focused, Balanced, or Broad inclusion. Returns (passes_focused, passes_balanced, passes_broad)."""
        n_kw = len(keywords) if keywords else 0
        # Focused
        min_kw_focused = max(2, int(n_kw * 0.4)) if n_kw else 1
        if n_kw <= 2:
            min_kw_focused = 1
        min_score_focused = 18
        # Balanced
        min_kw_balanced = max(1, int(n_kw * 0.25)) if n_kw else 1
        min_score_balanced = 10
        # Broad
        min_kw_broad = max(1, int(n_kw * 0.2)) if n_kw else 1
        min_score_broad = 8

        if search_method == "embedding" and embeddings_available:
            if emb_score > 0:
                return (True, True, True)
            return (False, False, False)

        if search_method == "hybrid" and embeddings_available and max_emb_score > 0:
            very_high = emb_score >= very_high_emb_frac * max_emb_score
            decent = emb_score >= min_emb_frac * max_emb_score
            # Focused: (decent + (kw or score)) or very_high; and (not strict or matched_kw>0 or very_high)
            pf = (
                (decent and (matched_keywords >= min_kw_focused or final_score > min_score_focused)) or very_high
            ) and (matched_keywords > 0 or very_high)
            # Balanced
            pb_al = (decent and (matched_keywords >= min_kw_balanced or final_score > min_score_balanced)) or very_high
            # Broad
            pb_br = (decent and (matched_keywords >= min_kw_broad or final_score > min_score_broad)) or very_high
            return (pf, pb_al, pb_br)

        # Keyword-only or fallback
        pf = matched_keywords >= min_kw_focused
        if n_kw <= 2:
            pb_al = matched_keywords >= 1
            pb_br = matched_keywords >= 1
        else:
            pb_al = matched_keywords >= min_kw_balanced or final_score > min_score_balanced
            pb_br = matched_keywords >= min_kw_broad or final_score > min_score_broad
        return (pf, pb_al, pb_br)
    
    def keyword_filter(self, query, notes):
        """
        Enhanced semantic search with multiple methods:
        - Improved keyword extraction (stemming, n-grams, TF-IDF)
        - Optional embedding-based search using cloud embeddings (Voyage)
        - Hybrid approach combining both methods
        - Context-aware ranking
        """
        import re
        
        # Get search configuration
        config = load_config()
        search_config = config.get('search_config', {}) or {}
        # Effective relevance mode for this search (Focused/Balanced/Broad)
        mode = getattr(self, 'relevance_mode', None) or search_config.get('relevance_mode') or ''
        mode = (mode or '').lower()
        if mode not in ('focused', 'balanced', 'broad'):
            # Backwards compatibility: infer from strict_relevance when mode missing
            mode = 'focused' if search_config.get('strict_relevance', True) else 'balanced'
        self._effective_relevance_mode = mode
        self._effective_strict_relevance = (mode == 'focused')
        original_search_method = search_config.get('search_method', 'hybrid')
        search_method = original_search_method  # 'keyword', 'keyword_rerank', 'embedding', 'hybrid'
        use_context_boost = search_config.get('use_context_boost', True)
        # keyword_rerank = keyword scoring then cross-encoder rerank (no embeddings)
        if search_method == 'keyword_rerank':
            search_method = 'keyword'  # use keyword path; effective_method will show "Keyword + Re-rank"
        
        # Always run synonym expansion (built-in medical aliases + config synonym_overrides).
        # Optional AI-based expansion runs inside _expand_query when enable_query_expansion is on.
        query = self._expand_query(query, config)
        
        # Optional: AI-mediated generic term exclusion (one short LLM call per search)
        if search_config.get('use_ai_generic_term_detection', False):
            try:
                self._query_ai_excluded_terms = self._get_ai_excluded_terms(query, config)
            except Exception:
                self._query_ai_excluded_terms = set()
        else:
            self._query_ai_excluded_terms = set()
        
        # Improved keyword extraction
        keywords, stems, phrases = self._extract_keywords_improved(query)
        
        if not keywords and not phrases:
            return ([(1, note) for note in notes[:50]], "Keyword only", min(50, len(notes)))
        
        # Compute TF-IDF scores
        tfidf_scores = self._compute_tfidf_scores(notes, keywords)
        
        # Get embedding scores if available and method requires it
        embedding_scores = None
        embeddings_available = False
        # HyDE: optional hypothetical document for better semantic retrieval
        embedding_query = query
        if search_method in ('embedding', 'hybrid') and search_config.get('enable_hyde', False):
            try:
                if hasattr(self, 'status_label') and self.status_label:
                    self.status_label.setText("Generating HyDE... (one short API call, usually 5–30 s)")
                    QApplication.processEvents()
            except Exception:
                pass
            hyde_doc = self._generate_hyde_document(query, config)
            try:
                if hasattr(self, 'status_label') and self.status_label and hyde_doc:
                    self.status_label.setText("Searching notes... (embedding search)")
                    QApplication.processEvents()
            except Exception:
                pass
            if hyde_doc:
                embedding_query = hyde_doc
                log_debug("Using HyDE hypothetical document for embedding search")
        
        if search_method in ('embedding', 'hybrid'):
            # For speed and better relevance, only run the slower
            # embedding-based search on the top N TF-IDF candidates.
            max_notes_for_embedding = 2000
            notes_sorted = None
            if len(notes) > max_notes_for_embedding:
                notes_sorted = sorted(
                    notes,
                    key=lambda n: tfidf_scores.get(n['id'], 0),
                    reverse=True,
                )
                notes_for_embedding = notes_sorted[:max_notes_for_embedding]
            else:
                notes_for_embedding = notes
            # Run embedding search in background worker so UI stays responsive
            state = dict(
                notes=notes, query=query, keywords=keywords, stems=stems, phrases=phrases,
                tfidf_scores=tfidf_scores, search_method=search_method,
                original_search_method=original_search_method, search_config=search_config,
                use_context_boost=use_context_boost, config=config,
                notes_sorted=notes_sorted, max_notes_for_embedding=max_notes_for_embedding,
            )
            return ("PENDING_EMBEDDING", embedding_query, notes_for_embedding, state)
        
        # Score notes using selected method (keyword-only path)
        scored_notes = []
        keyword_scored_list = []  # For RRF: (keyword_score, note) for hybrid
        max_score = 0
        max_emb_score = max(embedding_scores.values()) if embedding_scores else 0.0
        min_emb_frac = 0.25  # hybrid weighted fallback
        very_high_emb_frac = 0.9  # near-best semantic match gets through even w/ weak keywords
        use_rrf = (search_method == 'hybrid' and embedding_scores and embeddings_available)
        high_freq_keywords = getattr(self, "_query_high_freq_keywords", set()) or set()
        if not isinstance(high_freq_keywords, set):
            try:
                high_freq_keywords = set(high_freq_keywords)
            except Exception:
                high_freq_keywords = set()
        # Search cascade: when all query keywords are high-freq (generic), down-weight keyword-only score
        query_all_high_freq = bool(keywords and all(k in high_freq_keywords for k in keywords))
        
        try:
            from aqt.qt import QApplication
        except Exception:
            QApplication = None
        
        for idx, note in enumerate(notes):
            # Let the UI breathe every few hundred notes so Anki doesn't show "Not Responding"
            if QApplication is not None and idx % 500 == 0:
                QApplication.processEvents()
            content_lower = note['content'].lower()
            keyword_score = 0
            matched_keywords = 0
            
            # Improved keyword matching with stemming
            for keyword in keywords:
                # Skip auto-detected high-frequency filler terms for this query
                if keyword in high_freq_keywords:
                    continue
                # Exact whole word match
                whole_word = bool(re.search(r'\b' + re.escape(keyword) + r'\b', content_lower))
                if whole_word:
                    matched_keywords += 1
                    count = content_lower.count(keyword)
                    keyword_score += min(count * 12, 45) + 10
                elif keyword in content_lower:
                    matched_keywords += 1
                    keyword_score += 4
                
                # Check stemmed versions
                stem = self._simple_stem(keyword)
                if stem != keyword and stem in content_lower:
                    keyword_score += 3
            
            # Phrase matching (bigrams and trigrams)
            for phrase in phrases:
                # Don't give phrase bonus if all its tokens are high-frequency fillers
                tokens = phrase.split()
                if tokens and all(t in high_freq_keywords for t in tokens):
                    continue
                if phrase in content_lower:
                    keyword_score += 18
            
            # Add TF-IDF component
            tfidf_score = tfidf_scores.get(note['id'], 0) * 2  # Weight TF-IDF
            keyword_score += tfidf_score
            
            # Combine with embedding score if available
            if search_method == 'embedding':
                if embedding_scores and embeddings_available:
                    # Use only embedding score
                    final_score = embedding_scores.get(note['id'], 0)
                else:
                    # Fallback to keyword if embeddings not available
                    # This ensures the search still works even without embeddings
                    final_score = keyword_score
            elif search_method == 'hybrid':
                final_score = keyword_score
            else:
                final_score = keyword_score
            
            # Cascade: don't treat generic-keyword-only matches as highly relevant
            if query_all_high_freq and final_score == keyword_score:
                final_score = final_score * 0.3
            
            # Apply context-aware boost
            if use_context_boost:
                final_score = self._context_aware_boost(note, final_score)
            
            # For RRF hybrid: collect keyword scores; attach Focused/Balanced/Broad flags for UI filtering.
            if use_rrf:
                emb_score = embedding_scores.get(note['id'], 0) if embedding_scores else 0
                passes_focused, passes_balanced, passes_broad = self._passes_focused_balanced_broad(
                    matched_keywords, final_score, emb_score, max_emb_score, keywords,
                    search_method, embeddings_available, min_emb_frac, very_high_emb_frac,
                )
                note['_passes_focused'] = passes_focused
                note['_passes_balanced'] = passes_balanced
                note['_passes_broad'] = passes_broad
                keyword_scored_list.append((keyword_score, final_score, note))  # final_score has context boost
                continue
            
            # Inclusion criteria for non-RRF: include if passes Broad (superset); attach all three flags for UI filtering.
            emb_score = embedding_scores.get(note['id'], 0) if embedding_scores else 0
            passes_focused, passes_balanced, passes_broad = self._passes_focused_balanced_broad(
                matched_keywords, final_score, emb_score, max_emb_score, keywords,
                search_method, embeddings_available, min_emb_frac, very_high_emb_frac,
            )
            note['_passes_focused'] = passes_focused
            note['_passes_balanced'] = passes_balanced
            note['_passes_broad'] = passes_broad

            # region agent log
            try:
                q_lower = (getattr(self, "_search_pending_query", None) or query or "").lower()
                if "trisom" in q_lower:
                    _agent_debug_log(
                        run_id="post-fix_candidate",
                        hypothesis_id="H4",
                        location="__init__.keyword_filter",
                        message="candidate_inclusion_decision",
                        data={
                            "note_id": note.get("id"),
                            "matched_keywords": matched_keywords,
                            "final_score": final_score,
                            "emb_score": emb_score,
                            "search_method": search_method,
                            "len_keywords": len(keywords),
                            "passes_focused": passes_focused,
                            "passes_balanced": passes_balanced,
                            "passes_broad": passes_broad,
                        },
                    )
            except Exception:
                pass
            # endregion

            if passes_broad:
                scored_notes.append((final_score, note))
                max_score = max(max_score, final_score)
        
        # RRF (Reciprocal Rank Fusion): combine keyword and vector rankings. Standard formula 1/(k+rank); more effective than weighted averaging.
        # With chunks, use best rank per note id (min rank across chunks).
        if use_rrf and embedding_results:
            k = RRF_K
            emb_weight = max(0, min(1, (search_config.get('hybrid_embedding_weight', 50) or 50) / 100.0))
            kw_weight = 1.0 - emb_weight
            keyword_ranked = sorted(keyword_scored_list, key=lambda x: (x[0], x[1]), reverse=True)
            kw_rank = {}
            for rank, (_, _, note) in enumerate(keyword_ranked, start=1):
                nid = note['id']
                kw_rank[nid] = min(kw_rank.get(nid, rank), rank)
            emb_rank = {}
            for rank, (_, note) in enumerate(embedding_results, start=1):
                nid = note['id']
                emb_rank[nid] = min(emb_rank.get(nid, rank), rank)
            all_ids = set(kw_rank) | set(emb_rank)
            rrf_scores = []
            for nid in all_ids:
                rrf = 0
                if nid in kw_rank:
                    rrf += kw_weight * (1.0 / (k + kw_rank[nid]))
                if nid in emb_rank:
                    rrf += emb_weight * (1.0 / (k + emb_rank[nid]))
                if rrf > 0:
                    note = next((n for _, _, n in keyword_ranked if n['id'] == nid), None) or next((n for _, n in embedding_results if n['id'] == nid), None)
                    if note:
                        rrf_scores.append((rrf, note))
            scored_notes = sorted(rrf_scores, reverse=True, key=lambda x: x[0])
            # Notes that came only from embedding_results may lack _passes_*; show them in Broad/Balanced.
            for _s, note in scored_notes:
                if '_passes_broad' not in note:
                    note['_passes_broad'] = True
                    note['_passes_balanced'] = True
                    note['_passes_focused'] = False
            max_score = scored_notes[0][0] if scored_notes else 1

        # Normalize scores to 0-100 range if needed
        if max_score > 0 and max_score != 100:
            scored_notes = [(score / max_score * 100, note) for score, note in scored_notes]

        scored_notes.sort(reverse=True, key=lambda x: x[0])
        # Keep un-aggregated list for AI context so it can cite specific sections (chunks)
        has_chunks = any(n.get('chunk_index') is not None for _, n in scored_notes)
        if has_chunks:
            self._scored_notes_for_context = list(scored_notes)
        else:
            self._scored_notes_for_context = None
        # Aggregate chunks by note id: one entry per note (best score), display full content
        scored_notes = self._aggregate_scored_notes_by_note_id(scored_notes)
        
        # Effective method shown to user (may differ from config if embeddings unavailable)
        if original_search_method == "keyword_rerank":
            effective_method = "Keyword + Re-rank"
        elif search_method == "embedding" and embeddings_available:
            effective_method = "Embedding only"
        elif search_method == "hybrid" and embeddings_available:
            effective_method = "Hybrid"
        else:
            effective_method = "Keyword only"
        
        # Optional: Cross-encoder re-ranking in background (avoids UI freeze)
        if scored_notes and (search_config.get('enable_rerank', False) or original_search_method == 'keyword_rerank'):
            return ("PENDING_RERANK", scored_notes, effective_method + " + Re-ranked", 0, notes)
        
        # Stored superset for mode switching: keep notes above a low bar, cap size (filter_and_display_notes applies mode + sensitivity).
        MAX_STORED_FOR_MODES = 100
        min_relevance = max(15, min(75, search_config.get('min_relevance_percent', 55)))
        min_relevance_stored = min(20, min_relevance)
        all_above_stored = [x for x in scored_notes if x[0] >= min_relevance_stored]
        scored_notes = all_above_stored[:MAX_STORED_FOR_MODES] if all_above_stored else scored_notes[:5]
        total_above_threshold = sum(1 for x in scored_notes if x[0] >= min_relevance)
        return scored_notes, effective_method, total_above_threshold

    def keyword_filter_continue(self, state, embedding_results, progress_callback=None):
        """Continue keyword_filter after embedding search worker finishes. Uses state from keyword_filter.
        progress_callback(idx, total) is called every 500 notes when provided (e.g. from a worker thread).
        When notes count is large and we have embedding results, only score a subset (top by TF-IDF + embedding note ids) to avoid long freezes."""
        import re
        notes = state["notes"]
        query = state["query"]
        keywords = state["keywords"]
        stems = state["stems"]
        phrases = state["phrases"]
        tfidf_scores = state["tfidf_scores"]
        search_method = state["search_method"]
        original_search_method = state["original_search_method"]
        search_config = state["search_config"]
        use_context_boost = state["use_context_boost"]
        config = state["config"]
        embedding_scores = None
        embeddings_available = False
        if embedding_results:
            embedding_scores = {note['id']: score for score, note in embedding_results}
            embeddings_available = True
            # Limit to subset when very large so "Combining results" does not take minutes
            COMBINE_MAX_NOTES = 6000
            if len(notes) > COMBINE_MAX_NOTES:
                emb_ids = {note['id'] for _, note in embedding_results}
                top_by_tfidf = sorted(notes, key=lambda n: tfidf_scores.get(n['id'], 0), reverse=True)[:COMBINE_MAX_NOTES]
                top_ids = {n['id'] for n in top_by_tfidf}
                subset_ids = top_ids | emb_ids
                notes = [n for n in notes if n['id'] in subset_ids]
                log_debug(f"keyword_filter_continue: limited to {len(notes)} notes (top {COMBINE_MAX_NOTES} by TF-IDF + embedding results)")
        else:
            if search_method == 'embedding' and not hasattr(self, '_embedding_warning_shown'):
                tooltip(
                    "No embeddings found for this search. Using keyword search.\n\n"
                    "If you already ran Create/Update Embeddings: the selected decks/note types may not match.",
                    period=5000,
                )
                self._embedding_warning_shown = True
            elif search_method == 'hybrid' and not hasattr(self, '_hybrid_warning_shown'):
                tooltip(
                    "No embeddings for these notes. Using keyword-only search.\n\n"
                    "Run Create/Update Embeddings (Settings → Search & Embeddings) for the selected decks/note types.",
                    period=4000,
                )
                self._hybrid_warning_shown = True
        scored_notes = []
        keyword_scored_list = []
        max_score = 0
        max_emb_score = max(embedding_scores.values()) if embedding_scores else 0.0
        min_emb_frac = 0.25
        very_high_emb_frac = 0.9
        use_rrf = (search_method == 'hybrid' and embedding_scores and embeddings_available)
        total_notes = len(notes)
        high_freq_keywords = getattr(self, "_query_high_freq_keywords", set()) or set()
        if not isinstance(high_freq_keywords, set):
            try:
                high_freq_keywords = set(high_freq_keywords)
            except Exception:
                high_freq_keywords = set()
        for idx, note in enumerate(notes):
            if progress_callback and idx > 0 and idx % 500 == 0:
                try:
                    progress_callback(idx + 1, total_notes)
                except Exception:
                    pass
            content_lower = note['content'].lower()
            keyword_score = 0
            matched_keywords = 0
            for keyword in keywords:
                if keyword in high_freq_keywords:
                    continue
                whole_word = bool(re.search(r'\b' + re.escape(keyword) + r'\b', content_lower))
                if whole_word:
                    matched_keywords += 1
                    count = content_lower.count(keyword)
                    keyword_score += min(count * 12, 45) + 10
                elif keyword in content_lower:
                    matched_keywords += 1
                    keyword_score += 4
                stem = self._simple_stem(keyword)
                if stem != keyword and stem in content_lower:
                    keyword_score += 3
            for phrase in phrases:
                tokens = phrase.split()
                if tokens and all(t in high_freq_keywords for t in tokens):
                    continue
                if phrase in content_lower:
                    keyword_score += 18
            keyword_score += tfidf_scores.get(note['id'], 0) * 2
            if search_method == 'embedding':
                final_score = embedding_scores.get(note['id'], 0) if (embedding_scores and embeddings_available) else keyword_score
            else:
                final_score = keyword_score
            if use_context_boost:
                final_score = self._context_aware_boost(note, final_score)
            if use_rrf:
                emb_score = embedding_scores.get(note['id'], 0) if embedding_scores else 0
                passes_focused, passes_balanced, passes_broad = self._passes_focused_balanced_broad(
                    matched_keywords, final_score, emb_score, max_emb_score, keywords,
                    search_method, embeddings_available, min_emb_frac, very_high_emb_frac,
                )
                note['_passes_focused'] = passes_focused
                note['_passes_balanced'] = passes_balanced
                note['_passes_broad'] = passes_broad
                keyword_scored_list.append((keyword_score, final_score, note))
                continue
            # Inclusion: include if passes Broad; attach flags for UI mode filtering.
            emb_score = embedding_scores.get(note['id'], 0) if embedding_scores else 0
            passes_focused, passes_balanced, passes_broad = self._passes_focused_balanced_broad(
                matched_keywords, final_score, emb_score, max_emb_score, keywords,
                search_method, embeddings_available, min_emb_frac, very_high_emb_frac,
            )
            note['_passes_focused'] = passes_focused
            note['_passes_balanced'] = passes_balanced
            note['_passes_broad'] = passes_broad
            if passes_broad:
                scored_notes.append((final_score, note))
                max_score = max(max_score, final_score)
        if progress_callback and total_notes > 0:
            try:
                progress_callback(total_notes, total_notes)
            except Exception:
                pass
        # Optional verbose debug logging for search behavior tuning
        if search_config.get("verbose_search_debug", False):
            try:
                high_freq = getattr(self, "_query_high_freq_keywords", set()) or set()
                if not isinstance(high_freq, set):
                    high_freq = set(high_freq)
            except Exception:
                high_freq = set()
            top_notes = scored_notes[:5]
            debug_rows = []
            for score, note in top_notes:
                nid = note.get("id")
                emb_score = embedding_scores.get(nid, 0) if embedding_scores else 0
                tfidf = tfidf_scores.get(nid, 0)
                snippet = (note.get("content", "") or "")[:160].replace("\n", " ")
                debug_rows.append(
                    {
                        "note_id": nid,
                        "score": round(score, 2),
                        "embedding_score": round(emb_score, 4) if isinstance(emb_score, (int, float)) else emb_score,
                        "tfidf": round(tfidf, 4) if isinstance(tfidf, (int, float)) else tfidf,
                        "snippet": snippet,
                    }
                )
            log_debug(
                f"verbose_search_debug: query={query!r}, keywords={keywords}, phrases={phrases[:6]}, "
                f"high_freq_keywords={sorted(list(high_freq))[:12]}, top_notes={debug_rows}"
            )
        if use_rrf and embedding_results:
            k = RRF_K
            emb_weight = max(0, min(1, (search_config.get('hybrid_embedding_weight', 50) or 50) / 100.0))
            kw_weight = 1.0 - emb_weight
            keyword_ranked = sorted(keyword_scored_list, key=lambda x: (x[0], x[1]), reverse=True)
            kw_rank = {}
            for rank, (_, _, note) in enumerate(keyword_ranked, start=1):
                nid = note['id']
                kw_rank[nid] = min(kw_rank.get(nid, rank), rank)
            emb_rank = {}
            for rank, (_, note) in enumerate(embedding_results, start=1):
                nid = note['id']
                emb_rank[nid] = min(emb_rank.get(nid, rank), rank)
            all_ids = set(kw_rank) | set(emb_rank)
            rrf_scores = []
            for nid in all_ids:
                rrf = 0
                if nid in kw_rank:
                    rrf += kw_weight * (1.0 / (k + kw_rank[nid]))
                if nid in emb_rank:
                    rrf += emb_weight * (1.0 / (k + emb_rank[nid]))
                if rrf > 0:
                    note = next((n for _, _, n in keyword_ranked if n['id'] == nid), None) or next((n for _, n in embedding_results if n['id'] == nid), None)
                    if note:
                        rrf_scores.append((rrf, note))
            scored_notes = sorted(rrf_scores, reverse=True, key=lambda x: x[0])
            for _s, note in scored_notes:
                if '_passes_broad' not in note:
                    note['_passes_broad'] = True
                    note['_passes_balanced'] = True
                    note['_passes_focused'] = False
            max_score = scored_notes[0][0] if scored_notes else 1
        if max_score > 0 and max_score != 100:
            scored_notes = [(score / max_score * 100, note) for score, note in scored_notes]
        scored_notes.sort(reverse=True, key=lambda x: x[0])
        has_chunks_cont = any(n.get('chunk_index') is not None for _, n in scored_notes)
        if has_chunks_cont:
            self._scored_notes_for_context = list(scored_notes)
        else:
            self._scored_notes_for_context = None
        scored_notes = self._aggregate_scored_notes_by_note_id(scored_notes)
        if original_search_method == "keyword_rerank":
            effective_method = "Keyword + Re-rank"
        elif search_method == "embedding" and embeddings_available:
            effective_method = "Embedding only"
        elif search_method == "hybrid" and embeddings_available:
            effective_method = "Hybrid"
        else:
            effective_method = "Keyword only"
        if scored_notes and (search_config.get('enable_rerank', False) or original_search_method == 'keyword_rerank'):
            return ("PENDING_RERANK", notes, scored_notes, effective_method + " + Re-ranked", 0)
        MAX_STORED_FOR_MODES = 100
        min_relevance = max(15, min(75, search_config.get('min_relevance_percent', 55)))
        min_relevance_stored = min(20, min_relevance)
        all_above_stored = [x for x in scored_notes if x[0] >= min_relevance_stored]
        scored_notes = all_above_stored[:MAX_STORED_FOR_MODES] if all_above_stored else scored_notes[:5]
        total_above_threshold = sum(1 for x in scored_notes if x[0] >= min_relevance)
        return scored_notes, effective_method, total_above_threshold
    
    def get_best_model(self, provider):
        models = {
            'anthropic': 'claude-sonnet-4-20250514',
            'openai': 'gpt-4o-mini',
            'google': 'gemini-1.5-flash',
            'openrouter': 'google/gemini-flash-1.5',
            'ollama': 'llama3.2'
        }
        return models.get(provider, 'gpt-4o-mini')
    
    def ask_ai(self, query, notes, context, config):
        provider = config.get('provider', 'openai')
        api_key = config.get('api_key', '')
        model = self.get_best_model(provider)
        
        num_notes = len(notes)
        focus_line = _get_query_focus_instruction(query)
        focus_block = "\n" + focus_line if focus_line else ""
        prompt = f"""You are an assistant for question-answering over provided notes. Use ONLY the numbered notes below as your factual source (you may add brief connecting logic, but no outside facts or external guidelines). 
If the notes contain at least some relevant information, give the **best partial answer you can** based only on these notes and then briefly mention what is missing.
Only if the notes are essentially unrelated to the question, say exactly: "The provided notes do not contain enough information to answer this."

Context information is below. There are exactly {num_notes} notes: Note 1 = highest relevance, Note 2 = second, ... Note {num_notes} = last. Cite ONLY using numbers from 1 to {num_notes} (e.g. [1], [2], [1,3]). Do not use numbers outside 1–{num_notes}.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question.

Question: {query}{focus_block}

Rules:
- Base every claim strictly on these notes. Do **not** invent mechanisms, receptor types, dosages, diagnostic criteria, or risk factors that are not supported by the notes. One sentence or bullet per idea is fine.
- Write in a clear, exam-oriented style: use bullet points (•) for key points; use 2-space indented bullets for sub-points. Use **double asterisks** around important terms (diagnoses, drugs, criteria). Do not use ## for headings—use a single bold line with ● then bullets underneath.
- When the question asks about **receptors, mechanisms, pathways, or numbered lists (e.g. 1st–6th diseases, steps 1–6)**, present them in a clean ordered list and attach citations for each item.
- NUMBERED LISTS: When the question or notes refer to a numbered list (e.g. six childhood exanthems, 1st–6th disease, steps 1–6 of ventricular partitioning), **always** present in **strict numerical order**: 1st, then 2nd, then 3rd, then 4th, then 5th, then 6th. **Include every step or number that any note describes**—if Note X explicitly mentions "3rd step" or "third step", you MUST include it in your answer and cite it [X]. Do not skip steps; do not reorder by relevance.
- INLINE CITATIONS: Cite the supporting note(s) using [N] or [N,M] where N is between 1 and {num_notes} only. Example: "Hypertension increases stroke risk [1,3]." Do not use citation numbers outside 1–{num_notes}.
- At the end, on one line, list all note numbers you cited. Format: RELEVANT_NOTES: 1,3,5"""

        # Estimate input tokens
        input_tokens = estimate_tokens(prompt)
        
        if provider == "ollama":
            sc = config.get('search_config') or {}
            base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()
            model = (sc.get('ollama_chat_model') or 'llama3.2').strip()
            answer, relevant_indices = self.call_ollama(prompt, base_url, model, notes)
        elif provider == "anthropic":
            system_blocks, user_content = _build_anthropic_prompt_parts(query, context, _get_query_focus_instruction(query))
            answer, relevant_indices = self.call_anthropic(
                api_key=api_key, model=model, notes=notes,
                system_blocks=system_blocks, user_content=user_content
            )
        elif provider == "openai":
            answer, relevant_indices = self.call_openai(prompt, api_key, model, notes)
        elif provider == "google":
            answer, relevant_indices = self.call_google(prompt, api_key, model, notes)
        elif provider == "openrouter":
            answer, relevant_indices = self.call_openrouter(prompt, api_key, model, notes)
        else:
            api_url = config.get('api_url', '')
            answer, relevant_indices = self.call_custom(prompt, api_key, model, api_url, notes)
        
        log_debug(f"AI answer length: input ~{input_tokens} tokens, output ~{estimate_tokens(answer)} tokens")
        return answer, relevant_indices
    
    def call_ollama(self, prompt, base_url, model, notes):
        """Call Ollama /api/generate for AI answers (no API key)."""
        import json
        import urllib.request
        import urllib.error
        log_debug(f"Calling Ollama API: {base_url}, model={model}")
        url = base_url.rstrip("/") + "/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 4096}
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            # Reasoning models (e.g. deepseek-r1) can take several minutes; use 5 min timeout
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            # /api/generate returns "response"; /api/chat returns message.content; some models use "thinking"
            full_response = (
                result.get("response")
                or (result.get("message") or {}).get("content")
                or result.get("thinking")
                or ""
            )
            if isinstance(full_response, list):
                # Some models return content as list of parts
                full_response = "".join(
                    p.get("text", p) if isinstance(p, dict) else str(p)
                    for p in full_response
                )
            full_response = (full_response or "").strip()
            return self.parse_response(full_response, notes)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = str(e)
            log_debug(f"Ollama HTTP error: {e.code} {err_body}")
            raise Exception(f"Ollama error ({e.code}): {err_body[:200]}")
        except urllib.error.URLError as e:
            msg = str(getattr(e, "reason", e))
            if "timed out" in msg.lower():
                raise Exception("Ollama request timed out. Try a smaller model or more notes.")
            raise Exception(f"Cannot reach Ollama: {msg}. Is Ollama running (ollama serve)?")
        except Exception as e:
            log_debug(f"Ollama error: {e}")
            raise Exception(f"Ollama error: {e}")
    
    def call_anthropic(self, prompt=None, api_key=None, model=None, notes=None, system_blocks=None, user_content=None):
        """Call Anthropic API. Use system_blocks+user_content for prompt caching (recommended); else single prompt."""
        log_debug(f"Calling Anthropic API with model: {model}")
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        if system_blocks is not None and user_content is not None:
            data = {
                "model": model,
                "max_tokens": 4096,
                "system": system_blocks,
                "messages": [{"role": "user", "content": user_content}]
            }
        else:
            data = {
                "model": model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}]
            }
        response_text = self.make_request(url, headers, data)
        result = json.loads(response_text)
        full_response = result['content'][0]['text']
        return self.parse_response(full_response, notes)
    
    def call_openai(self, prompt, api_key, model, notes):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096
        }
        
        response_text = self.make_request(url, headers, data)
        result = json.loads(response_text)
        full_response = result['choices'][0]['message']['content']
        return self.parse_response(full_response, notes)
    
    def call_google(self, prompt, api_key, model, notes):
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 4096
            }
        }
        
        response_text = self.make_request(url, headers, data)
        result = json.loads(response_text)
        full_response = result['candidates'][0]['content']['parts'][0]['text']
        return self.parse_response(full_response, notes)
    
    def call_openrouter(self, prompt, api_key, model, notes):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096
        }
        
        response_text = self.make_request(url, headers, data)
        result = json.loads(response_text)
        full_response = result['choices'][0]['message']['content']
        return self.parse_response(full_response, notes)
    
    def call_custom(self, prompt, api_key, model, api_url, notes):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096
        }
        
        response_text = self.make_request(api_url, headers, data)
        result = json.loads(response_text)
        
        if 'choices' in result:
            full_response = result['choices'][0]['message']['content']
        elif 'content' in result:
            full_response = result['content'][0]['text']
        else:
            full_response = str(result)
        
        return self.parse_response(full_response, notes)
    
    def make_request(self, url, headers, data):
        """Make HTTP request with proper timeout and error handling"""
        log_debug(f"Making API request to: {url}")
        log_debug(f"Request data keys: {list(data.keys())}")
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        
        try:
            # Use 30 second timeout (60 was too long)
            log_debug("Opening URL connection (timeout: 30 seconds)...")
            with urllib.request.urlopen(req, timeout=30) as response:
                log_debug(f"Received response, status: {response.status}")
                response_data = response.read().decode('utf-8')
                log_debug(f"Response length: {len(response_data)} characters")
                return response_data
                
        except urllib.error.HTTPError as e:
            try:
                error_msg = e.read().decode('utf-8')
            except:
                error_msg = str(e)
            log_debug(f"HTTP Error {e.code}: {error_msg}")
            raise Exception(f"API Error ({e.code}): {error_msg}")
            
        except urllib.error.URLError as e:
            # Handle common "no internet / host unreachable" cases more clearly
            reason = getattr(e, "reason", e)
            msg = str(reason)
            log_debug(f"URL Error: {msg}")
            lower = msg.lower()
            
            # Windows / general "no internet or cannot resolve host" patterns
            if (
                "getaddrinfo failed" in lower
                or "name or service not known" in lower
                or "temporary failure in name resolution" in lower
                or "nodename nor servname provided" in lower
                or "winerror 11001" in lower  # host not found
                or "winerror 10051" in lower  # network unreachable
                or "winerror 10065" in lower  # no route to host
            ):
                raise Exception("No internet connection or the API host cannot be reached.")
            
            if "timed out" in lower:
                raise Exception("Request timed out after 30 seconds. The API may be slow or overloaded.")
            
            raise Exception(f"Network error: {msg}")
            
        except Exception as e:
            log_debug(f"Unexpected error: {type(e).__name__}: {str(e)}")
            raise Exception(f"Request error: {str(e)}")
    
    def parse_response(self, full_response, notes):
        import re
        answer_part = ""
        relevant_notes = []
        
        if "RELEVANT_NOTES:" in full_response:
            parts = full_response.split("RELEVANT_NOTES:")
            answer_part = parts[0].strip()
            
            if len(parts) > 1:
                notes_str = parts[1].strip()
                numbers = re.findall(r'\d+', notes_str)
                relevant_notes = [int(n) - 1 for n in numbers if n.isdigit() and 0 <= int(n) - 1 < len(notes)]
        else:
            answer_part = full_response
            relevant_notes = list(range(min(3, len(notes))))
        
        log_debug(f"Parsed {len(relevant_notes)} relevant notes from AI response")
        return answer_part, relevant_notes
    
    def _start_anthropic_stream(self, query, context_notes, context, config, relevant_notes, scored_notes, context_note_ids):
        """Start Anthropic streaming worker; UI updates in real-time via chunk_signal."""
        import anthropic  # raise if not installed
        api_key = config.get('api_key', '')
        model = self.get_best_model('anthropic')
        system_blocks, user_content = _build_anthropic_prompt_parts(query, context, _get_query_focus_instruction(query))
        self._streamed_answer = ""
        self._stream_context_notes = context_notes
        self._stream_relevant_notes = relevant_notes
        self._stream_scored_notes = scored_notes
        self._stream_context_note_ids = context_note_ids
        self._stream_config = config
        self._stream_query = query
        self.answer_box.setPlainText("Thinking…")
        worker = AnthropicStreamWorker(api_key, model, system_blocks, user_content, context_notes)
        worker.chunk_signal.connect(self._append_stream_chunk)
        worker.done_signal.connect(self._on_anthropic_stream_done)
        worker.error_signal.connect(self._on_anthropic_stream_error)
        self._anthropic_stream_worker = worker
        worker.start()
    
    def _append_stream_chunk(self, chunk):
        """Append a streamed text chunk to the answer box; first chunk replaces 'Thinking…' placeholder."""
        self._streamed_answer = getattr(self, '_streamed_answer', '') + chunk
        self.answer_box.setPlainText(self._streamed_answer)
    
    def _on_anthropic_stream_done(self, full_text):
        """Handle stream completion: parse response, update cited notes, format and display."""
        worker = getattr(self, '_anthropic_stream_worker', None)
        if worker:
            worker.chunk_signal.disconnect()
            worker.done_signal.disconnect()
            worker.error_signal.disconnect()
            self._anthropic_stream_worker = None
        context_notes = getattr(self, '_stream_context_notes', [])
        relevant_notes = getattr(self, '_stream_relevant_notes', [])
        scored_notes = getattr(self, '_stream_scored_notes', [])
        context_note_ids = getattr(self, '_stream_context_note_ids', [])
        config = getattr(self, '_stream_config', {})
        query = getattr(self, '_stream_query', '')
        answer, relevant_indices = self.parse_response(full_text, context_notes)
        relevant_note_ids = [relevant_notes[idx]['id'] for idx in relevant_indices if 0 <= idx < len(relevant_notes)]
        save_search_history(query, answer, relevant_note_ids, scored_notes, context_note_ids)
        self._context_note_ids = context_note_ids
        self._context_note_id_and_chunk = [(n['id'], n.get('chunk_index')) for n in context_notes]
        self.current_answer = answer
        ai_relevant_note_ids = set()
        for idx in relevant_indices:
            if 0 <= idx < len(relevant_notes):
                ai_relevant_note_ids.add(relevant_notes[idx]['id'])
        self._cited_note_ids = ai_relevant_note_ids
        improved_scored_notes = []
        for score, note in self.all_scored_notes:
            improved_score = score * 2 if note['id'] in ai_relevant_note_ids else score
            improved_scored_notes.append((improved_score, note))
        improved_scored_notes.sort(reverse=True, key=lambda x: x[0])
        if improved_scored_notes:
            max_boosted = improved_scored_notes[0][0]
            self.all_scored_notes = [(score / max_boosted * 100.0, note) for score, note in improved_scored_notes] if max_boosted > 0 else improved_scored_notes
        else:
            self.all_scored_notes = improved_scored_notes
        search_config = config.get('search_config') or {}
        if search_config.get('relevance_from_answer'):
            answer_text = (answer[:8000]).strip()
            note_texts = []
            for _, note in self.all_scored_notes:
                raw = note.get('display_content') or note.get('content', '')
                text = self.reveal_cloze_for_display(raw) if hasattr(self, 'reveal_cloze_for_display') else raw
                note_texts.append((text[:2000]) if text else "")
            if answer_text and note_texts:
                self.status_label.setText("Re-ranking by relevance to answer…")
                self._show_centile_progress("Re-ranking by relevance…", 0)
                self._relevance_rerank_worker = RelevanceRerankWorker(answer_text, note_texts, self.all_scored_notes, config)
                self._relevance_rerank_worker.progress_signal.connect(lambda p, m: self._show_centile_progress(m, p))
                self._relevance_rerank_worker.finished_signal.connect(
                    lambda res: self._on_relevance_rerank_done_stream(res, answer, config)
                )
                self._relevance_rerank_worker.start()
                return
            try:
                self._rerank_by_relevance_to_answer(answer, config)
            except Exception as e:
                log_debug(f"Relevance-from-answer rerank failed: {e}")
        self._finish_anthropic_stream_display(answer, config)

    def _on_relevance_rerank_done_stream(self, result, answer, config):
        """Called when RelevanceRerankWorker finishes (streaming path). Apply result and finish display."""
        self._hide_busy_progress()
        if getattr(self, '_relevance_rerank_worker', None):
            self._relevance_rerank_worker = None
        if result is not None:
            self.all_scored_notes = result
        self._finish_anthropic_stream_display(answer, config)

    def _finish_anthropic_stream_display(self, answer, config):
        """Format answer, update table, and set status (streaming path)."""
        formatted_answer = self.format_answer(answer)
        self._last_formatted_answer = formatted_answer
        self.answer_box.setHtml(formatted_answer)
        if hasattr(self, 'answer_source_label'):
            src = self._get_answer_source_text(config)
            self.answer_source_label.setText(f"Answer from: {src}" if src else "")
        for attr in ('copy_answer_btn',):
            if hasattr(self, attr):
                getattr(self, attr).setEnabled(True)
        self.filter_and_display_notes()
        if self.all_scored_notes:
            threshold = self.sensitivity_slider.value() if getattr(self, 'sensitivity_slider', None) else 0
            max_score = self.all_scored_notes[0][0]
            min_score = (threshold / 100.0) * max_score if max_score > 0 else 0
            effective_pct = round(100 * min_score / max_score) if (max_score > 0 and threshold > 0) else None
            sensitivity_text = f" (score ≥ {effective_pct}%)" if effective_pct is not None else " (sensitivity filter)"
            filtered_count = sum(1 for score, _ in self.all_scored_notes if score >= min_score)
            total_in_result = len(self.all_scored_notes)
            mode = getattr(self, "_effective_relevance_mode", getattr(self, "relevance_mode", "balanced")) or "balanced"
            mode = (mode or "").lower()
            mode_label = {"focused": "Focused", "balanced": "Balanced", "broad": "Broad"}.get(mode, "Balanced")
            mode_suffix = f" | Mode: {mode_label}"
            self.status_label.setText(
                f"Showing {filtered_count} of {total_in_result}{sensitivity_text}{mode_suffix} "
                f"| Answer from: {self._get_answer_source_text(config) or 'Anthropic'}"
            )
        else:
            self.status_label.setText("Answer from: Anthropic (streaming)")
    def _on_anthropic_stream_error(self, error_msg):
        """Show streaming error in answer box."""
        if getattr(self, '_anthropic_stream_worker', None):
            self._anthropic_stream_worker = None
        if hasattr(self, 'answer_source_label'):
            self.answer_source_label.setText("")
        self.answer_box.setText(f"❌ Error calling Anthropic API:\n{error_msg}\n\nCheck your API key and internet connection.")
        self.status_label.setText("Error occurred")
    
    def _rerank_by_relevance_to_answer(self, answer, config):
        """Re-rank all_scored_notes by similarity of each note to the AI answer text.
        Sets _display_relevance on each note and replaces all_scored_notes with (score, note) sorted by this.
        Uses the configured embedding engine (Voyage/Ollama)."""
        if not answer or not getattr(self, 'all_scored_notes', None):
            return
        import numpy as np
        sc = (config or load_config()).get('search_config') or {}
        try:
            answer_text = (answer[:8000]).strip()
            if not answer_text:
                return
            answer_emb = get_embedding_for_query(answer_text, config)
            if not answer_emb:
                return
            answer_vec = np.array(answer_emb, dtype=float)
            note_texts = []
            for _, note in self.all_scored_notes:
                raw = note.get('display_content') or note.get('content', '')
                text = self.reveal_cloze_for_display(raw) if hasattr(self, 'reveal_cloze_for_display') else raw
                note_texts.append((text[:2000]) if text else "")
            if not note_texts:
                return
            note_embs = get_embeddings_batch(note_texts, input_type="document", config=config)
            if not note_embs or len(note_embs) != len(self.all_scored_notes):
                return
            norm_a = max(np.linalg.norm(answer_vec), 1e-9)
            new_scores = []
            for i, (_, note) in enumerate(self.all_scored_notes):
                ne = np.array(note_embs[i], dtype=float)
                norm_n = max(np.linalg.norm(ne), 1e-9)
                sim = float(np.dot(answer_vec, ne) / (norm_a * norm_n))
                pct = max(0, min(100, round((sim + 1) * 50)))
                note['_display_relevance'] = pct
                new_scores.append((float(pct), note))
            new_scores.sort(reverse=True, key=lambda x: x[0])
            # Renormalize so top note(s) show 100% and rest spread below
            if new_scores:
                max_pct = new_scores[0][0]
                if max_pct > 0:
                    for score, note in new_scores:
                        note['_display_relevance'] = max(0, min(100, round(100 * (note['_display_relevance'] or 0) / max_pct)))
                    new_scores = [(100.0 if i == 0 else (note['_display_relevance'] or 0), note) for i, (_, note) in enumerate(new_scores)]
                    new_scores.sort(reverse=True, key=lambda x: x[0])
            self.all_scored_notes = new_scores
        except Exception as e:
            log_debug(f"Relevance-from-answer rerank failed: {e}")
    
    def _get_matching_terms_for_note(self, note, query):
        """Return list of query terms that appear in the note (for 'Why this result?' explainability)."""
        if not query or not hasattr(self, '_extract_keywords_improved'):
            return []
        try:
            keywords, stems, phrases = self._extract_keywords_improved(query)
            content_lower = (note.get('content') or note.get('display_content') or '').lower()
            if not content_lower:
                return []
            matched = set()
            # Phrase matches first – these are usually the most informative
            for p in phrases:
                if p and p.lower() in content_lower:
                    matched.add(p)
            # Exact keyword matches
            for w in keywords:
                wl = (w or '').lower()
                if wl and wl in content_lower:
                    matched.add(w)
            # Stem-based matches: map stem -> representative original query word
            stem_to_display = dict(stems or {})
            for w in keywords:
                stem = self._simple_stem(w)
                if stem:
                    stem_to_display.setdefault(stem, w)
            for stem, display in stem_to_display.items():
                # Skip extremely short stems to avoid over-matching
                if not stem or len(stem) <= 3:
                    continue
                if stem in content_lower:
                    matched.add(display)
            if not matched:
                return []
            # Filter out generic/low-information terms and very short tokens.
            # Also exclude per-query high-frequency terms (set during search) and AI-detected
            # generic terms so we don't show uninformative words in "Why this result?".
            stop_words = self._get_extended_stop_words()
            high_freq = getattr(self, '_query_high_freq_keywords', None) or set()
            if not isinstance(high_freq, set):
                try:
                    high_freq = set(high_freq)
                except Exception:
                    high_freq = set()
            ai_excluded = getattr(self, '_query_ai_excluded_terms', None) or set()
            if not isinstance(ai_excluded, set):
                try:
                    ai_excluded = set(ai_excluded)
                except Exception:
                    ai_excluded = set()

            def _is_meaningful(term: str) -> bool:
                t = (term or '').strip().lower()
                if not t:
                    return False
                if t in stop_words:
                    return False
                if t in high_freq:
                    return False
                if t in ai_excluded:
                    return False
                # Drop very short tokens by default (can whitelist later if needed)
                if len(t) <= 3:
                    return False
                return True
            filtered = [t for t in matched if _is_meaningful(t)]
            if not filtered:
                return []
            # Prefer more specific/phrase-like terms: phrases first, then by length
            filtered.sort(key=lambda t: (0 if ' ' in t else 1, -len(t), t.lower()))
            # region agent log
            try:
                if "trisom" in (query or "").lower():
                    _agent_debug_log(
                        run_id="pre-fix",
                        hypothesis_id="H2",
                        location="__init__._get_matching_terms_for_note",
                        message="matching_terms_for_note",
                        data={
                            "note_id": note.get("id"),
                            "query": query,
                            "matched_all": sorted(matched),
                            "filtered": filtered[:12],
                        },
                    )
            except Exception:
                pass
            # endregion
            return filtered[:12]
        except Exception as ex:
            return []

    def filter_and_display_notes(self):
        # Use chunk-level display list when AI received more items than aggregated display (fixes Ref 35 vs 32)
        display_source = getattr(self, '_display_scored_notes', None)
        ctx = getattr(self, '_context_note_ids', None) or []
        if display_source and ctx and len(ctx) > len(getattr(self, 'all_scored_notes', None) or []):
            notes_to_display = display_source
        elif hasattr(self, 'all_scored_notes') and self.all_scored_notes:
            notes_to_display = self.all_scored_notes
        else:
            return
        if not notes_to_display:
            return

        # Store current row count before clearing
        old_row_count = self.results_list.rowCount()
        
        # Clear table
        self.results_list.setRowCount(0)
        
        threshold = self.sensitivity_slider.value() if self.sensitivity_slider else 0
        sensitivity_threshold = threshold  # save before IDF block may overwrite threshold
        max_score = notes_to_display[0][0] if notes_to_display else 1
        min_score = (threshold / 100.0) * max_score if max_score > 0 else 0
        
        filtered_notes = [(score, note) for score, note in notes_to_display if score >= min_score]

        # Additional strict gating in Focused mode: IDF-based "specific keyword" filter.
        # Notes that match only generic words (appearing in >50% of results) are excluded.
        try:
            config = load_config()
            sc = config.get('search_config') or {}
        except Exception:
            sc = {}
        # Prefer per-search effective strictness when available
        strict = bool(getattr(self, '_effective_strict_relevance', sc.get('strict_relevance', False)))
        # Skip IDF filter when "Relevance from answer" is enabled: ranking is by similarity to answer,
        # not query keywords, so requiring query keywords in note text can empty the list incorrectly.
        relevance_from_answer_enabled = bool(sc.get('relevance_from_answer', False))
        if strict and filtered_notes and not relevance_from_answer_enabled:
            import re
            cq = getattr(self, 'current_query', '') or ''
            try:
                kw, _stems, _phrases = self._extract_keywords_improved(cq)
            except Exception:
                kw = []
            if kw:
                n_notes = len(filtered_notes)
                # Document frequency: how many notes contain each keyword
                doc_freq = {}
                for _score, note in filtered_notes:
                    content_lower = (note.get('content') or note.get('display_content') or '').lower()
                    for k in kw:
                        k_lower = (k or '').lower()
                        if not k_lower:
                            continue
                        if re.search(r'\b' + re.escape(k_lower) + r'\b', content_lower) or k_lower in content_lower:
                            doc_freq[k] = doc_freq.get(k, 0) + 1
                # Specific = appears in fewer than 50% of notes (discriminative)
                threshold = 0.5
                specific_kw = {k for k in kw if doc_freq.get(k, 0) / max(1, n_notes) < threshold}
                if specific_kw:
                    kept = []
                    for score, note in filtered_notes:
                        content_lower = (note.get('content') or note.get('display_content') or '').lower()
                        if any(
                            (k and (re.search(r'\b' + re.escape(k.lower()) + r'\b', content_lower) or k.lower() in content_lower))
                            for k in specific_kw
                        ):
                            kept.append((score, note))
                    filtered_notes = kept
                # else: all keywords generic, skip filter (keep all notes)
        
        # Optionally restrict to notes cited in the AI answer ([1], [2], …)
        if getattr(self, 'show_only_cited_cb', None) and self.show_only_cited_cb.isChecked():
            cited = getattr(self, '_cited_note_ids', None)
            if cited:
                filtered_notes = [(score, note) for score, note in filtered_notes if note['id'] in cited]
        
        # Filter by current relevance mode (Focused / Balanced / Broad) using precomputed flags; no extra search/API.
        mode = (getattr(self, 'relevance_mode', None) or 'balanced').lower()
        # When "Relevance from answer" is on, ranking is by similarity to answer; _passes_* often all True.
        # Differentiate modes by score percentile so Focused = fewer, Broad = all.
        if relevance_from_answer_enabled and filtered_notes:
            n_total = len(filtered_notes)
            if mode == 'focused':
                cap = max(1, int(n_total * 0.4))
                filtered_notes = filtered_notes[:cap]
            elif mode == 'balanced':
                cap = max(1, int(n_total * 0.7))
                filtered_notes = filtered_notes[:cap]
            # else broad: keep all
        else:
            if mode == 'focused':
                filtered_notes = [(s, n) for s, n in filtered_notes if n.get('_passes_focused')]
            elif mode == 'balanced':
                filtered_notes = [(s, n) for s, n in filtered_notes if n.get('_passes_balanced')]
            else:
                filtered_notes = [(s, n) for s, n in filtered_notes if n.get('_passes_broad', True)]

        # Set row count
        self.results_list.setRowCount(len(filtered_notes))
        # Disable sorting while populating so rows stay 0..N and every row gets the correct content (fixes empty rows when toggling "Show only cited notes")
        self.results_list.setSortingEnabled(False)
        
        # 1-based position in context (order sent to AI) so [1], [2], [19] in answer match this #
        cited_ids = getattr(self, '_cited_note_ids', set()) or set()
        # Build Ref = context position so citation [N] matches row labeled N (works after re-rank and with chunks)
        context_id_and_chunk = getattr(self, '_context_note_id_and_chunk', None)
        if context_id_and_chunk:
            ref_from_context = {(nid, cidx): i + 1 for i, (nid, cidx) in enumerate(context_id_and_chunk)}
            def get_ref(note, row):
                return ref_from_context.get((note['id'], note.get('chunk_index')), row + 1)
        else:
            order_for_note = {nid: i + 1 for i, nid in enumerate(ctx)} if ctx else {}
            def get_ref(note, row):
                return order_for_note.get(note['id'], row + 1)
        
        for row, (score, note) in enumerate(filtered_notes):
            # Content-based relevance when set (HyDE/query similarity); else rank-based
            percentage = note.get('_display_relevance')
            if percentage is None:
                percentage = int((score / max_score) * 100) if max_score > 0 else 0
            else:
                percentage = max(0, min(100, int(percentage)))
            # Steepen display so 100 stands out and lower scores spread (100→100, 99→97, 95→92, 80→72)
            display_pct = max(0, min(100, round(100 * (percentage / 100) ** 0.6))) if percentage else 0
            
            # Column 0 (Ref): Citation number from context (matches [1], [2] in AI answer). Cited notes: blue + bold.
            order_num = get_ref(note, row)
            order_item = QTableWidgetItem()
            order_item.setData(Qt.ItemDataRole.DisplayRole, order_num)
            order_item.setData(Qt.ItemDataRole.UserRole, note['id'])
            order_item.setFlags(order_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            order_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            if note['id'] in cited_ids:
                order_item.setForeground(QColor('#3498db'))
                font = order_item.font()
                font.setBold(True)
                order_item.setFont(font)
            why_ref = "Cited in answer: Yes" if note['id'] in cited_ids else "Cited in answer: No"
            order_item.setToolTip(f"Why this result?\n{why_ref}\nCitation [N] in answer matches Ref. Note ID: {note['id']}")
            self.results_list.setItem(row, 0, order_item)
            
            # Column 1: Content (with checkbox)
            preview_len = getattr(self, 'preview_length', 150)
            # Use full note content when available (chunk-level display); so preview shows start of full note, not chunk
            raw_for_display = note.get('_full_content') or note.get('display_content') or note['content']
            display_content = self.reveal_cloze_for_display(raw_for_display)
            content_preview = display_content[:preview_len] + "..." if len(display_content) > preview_len else display_content
            content_item = QTableWidgetItem(content_preview)
            content_item.setCheckState(Qt.CheckState.Unchecked)
            content_item.setData(Qt.ItemDataRole.UserRole, note['id'])
            content_item.setData(Qt.ItemDataRole.UserRole + 1, display_pct)
            content_item.setData(Qt.ItemDataRole.UserRole + 2, display_content)
            matching_terms = self._get_matching_terms_for_note(note, getattr(self, 'current_query', ''))
            why_line = f"Why this result?\nRelevance: {display_pct}%\n{why_ref}"
            if matching_terms:
                why_line += f"\nMatching terms: {', '.join(matching_terms[:8])}{'…' if len(matching_terms) > 8 else ''}"
            tooltip_text = f"{why_line}\n\nNote ID: {note['id']}\n\nFull Content:\n{display_content}"
            content_item.setToolTip(tooltip_text)
            content_item.setFlags(content_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self.results_list.setItem(row, 1, content_item)
            
            # Column 2: Note ID
            note_id_item = QTableWidgetItem(str(note['id']))
            note_id_item.setData(Qt.ItemDataRole.UserRole, note['id'])
            note_id_item.setFlags(note_id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            note_id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            note_id_item.setToolTip(f"Note ID: {note['id']}\nDouble-click to open in browser")
            self.results_list.setItem(row, 2, note_id_item)
            
            # Column 3: Relevance (steepened display %)
            percentage_item = QTableWidgetItem()
            percentage_item.setData(Qt.ItemDataRole.DisplayRole, display_pct)
            percentage_item.setData(Qt.ItemDataRole.UserRole, display_pct)
            percentage_item.setFlags(percentage_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            percentage_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if display_pct >= 80:
                percentage_item.setForeground(QColor("#27ae60"))
                relevance_desc = "High relevance"
            elif display_pct >= 50:
                percentage_item.setForeground(QColor("#f39c12"))
                relevance_desc = "Medium relevance"
            else:
                percentage_item.setForeground(QColor("#e74c3c"))
                relevance_desc = "Low relevance"
            why_pct = f"Why this result?\nRelevance: {display_pct}% ({relevance_desc})\n{why_ref}"
            if matching_terms:
                why_pct += f"\nMatching terms: {', '.join(matching_terms[:6])}{'…' if len(matching_terms) > 6 else ''}"
            # region agent log
            try:
                cq = getattr(self, "current_query", "") or ""
                if "trisom" in cq.lower():
                    _agent_debug_log(
                        run_id="pre-fix",
                        hypothesis_id="H3",
                        location="__init__.filter_and_display_notes",
                        message="note_row_display",
                        data={
                            "note_id": note.get("id"),
                            "row": row,
                            "raw_score": score,
                            "display_relevance": display_pct,
                            "matching_terms": matching_terms[:6] if matching_terms else [],
                        },
                    )
            except Exception:
                pass
            # endregion
            percentage_item.setToolTip(why_pct)
            self.results_list.setItem(row, 3, percentage_item)
        
        # Update status to match table: "Showing X of Y" so it always matches Matching notes row count
        filtered_count = len(filtered_notes)
        total_in_result = len(notes_to_display)
        # No slider: omit score % from status; with slider would show " (score ≥ X%)"
        sensitivity_text = ""
        if self.sensitivity_slider is not None:
            effective_pct = round(100 * min_score / max_score) if (max_score > 0 and sensitivity_threshold > 0) else None
            sensitivity_text = f" (score ≥ {effective_pct}%)" if effective_pct is not None else " (sensitivity filter)"
            if self.sensitivity_value_label is not None:
                if sensitivity_threshold == 0:
                    self.sensitivity_value_label.setText("0%")
                elif sensitivity_threshold > 0 and max_score > 0:
                    self.sensitivity_value_label.setText(f"≥{effective_pct}%")
        searched_suffix = ""
        if hasattr(self, 'total_notes_searched') and self.total_notes_searched is not None:
            searched_suffix = f" (searched {self.total_notes_searched} in {getattr(self, 'fields_description', 'Text & Extra')})"
        mode = getattr(self, "_effective_relevance_mode", getattr(self, "relevance_mode", "balanced")) or "balanced"
        mode = (mode or "").lower()
        mode_label = {"focused": "Focused", "balanced": "Balanced", "broad": "Broad"}.get(mode, "Balanced")
        # #region agent log
        _session_debug_log(
            "H1",
            "filter_and_display_notes.status_mode",
            "status bar mode",
            data={"relevance_mode": getattr(self, "relevance_mode", None), "_effective_relevance_mode": getattr(self, "_effective_relevance_mode", None), "mode_used": mode, "mode_label": mode_label},
        )
        # #endregion
        mode_suffix = f" | Mode: {mode_label}"
        self.status_label.setText(
            f"Showing {filtered_count} of {total_in_result}{sensitivity_text}{mode_suffix}{searched_suffix}"
        )
        
        # Enable/disable toggle button based on list content
        has_items = self.results_list.rowCount() > 0
        if hasattr(self, 'toggle_select_btn'):
            self.toggle_select_btn.setEnabled(has_items)
        
        # Restore selections from persistence
        if hasattr(self, 'selected_note_ids') and self.selected_note_ids:
            self.restore_selections()
        
        # Update selection count and button text
        self.update_selection_count()

        # Update View All button state and tooltip (enabled when list has rows)
        self._update_view_all_button_state()

        # Re-enable sorting and apply default sort (fixes empty rows when toggling "Show only cited notes")
        self.results_list.setSortingEnabled(True)
        if display_source is not None and display_source is notes_to_display:
            self.results_list.sortItems(0, Qt.SortOrder.AscendingOrder)
        else:
            self.results_list.sortItems(3, Qt.SortOrder.DescendingOrder)  # Sort by Relevance
    
    def update_selection_count(self):
        """Update the selected count display and toggle button text"""
        if not hasattr(self, 'results_list'):
            return
        
        checked_count = 0
        total_count = self.results_list.rowCount()
        
        # Initialize selected_note_ids if it doesn't exist
        if not hasattr(self, 'selected_note_ids'):
            self.selected_note_ids = set()
        
        # Update persistence set and count (check column 1 = Content which has the checkbox)
        for row in range(total_count):
            item = self.results_list.item(row, 1)  # Content column has checkbox
            if item:
                note_id = item.data(Qt.ItemDataRole.UserRole)
                if item.checkState() == Qt.CheckState.Checked:
                    checked_count += 1
                    if note_id:
                        self.selected_note_ids.add(note_id)
                else:
                    if note_id:
                        self.selected_note_ids.discard(note_id)
        
        # Update count label
        if hasattr(self, 'selected_count_label'):
            if total_count > 0:
                self.selected_count_label.setText(f"({checked_count} of {total_count} selected)")
            else:
                self.selected_count_label.setText("(0 selected)")
        
        # Update toggle button text
        if hasattr(self, 'toggle_select_btn'):
            if checked_count == total_count and total_count > 0:
                self.toggle_select_btn.setText("✗ Deselect All")
            else:
                self.toggle_select_btn.setText("✓ Select All")
    
    def on_preview_length_changed(self, value):
        """Update preview length and refresh display"""
        self.preview_length = value
        if hasattr(self, 'preview_length_label'):
            self.preview_length_label.setText(f"{value} chars")
        # Refresh the display if we have notes
        if hasattr(self, 'all_scored_notes') and self.all_scored_notes:
            self.filter_and_display_notes()
    
    def toggle_select_all(self):
        """Toggle between selecting all and deselecting all notes"""
        if not hasattr(self, 'results_list') or self.results_list.rowCount() == 0:
            return
        
        # Check if all are selected (check column 1 = Content which has the checkbox)
        all_selected = True
        for row in range(self.results_list.rowCount()):
            item = self.results_list.item(row, 1)
            if item and item.checkState() != Qt.CheckState.Checked:
                all_selected = False
                break
        
        # Toggle state
        if all_selected:
            self.deselect_all_notes()
        else:
            self.select_all_notes()
    
    def select_all_notes(self):
        """Select all notes in the results list"""
        if not hasattr(self, 'results_list'):
            return
        
        # Block signals to prevent multiple updates
        self.results_list.blockSignals(True)
        for row in range(self.results_list.rowCount()):
            item = self.results_list.item(row, 1)  # Content column has checkbox
            if item:
                item.setCheckState(Qt.CheckState.Checked)
                # Store in persistence
                note_id = item.data(Qt.ItemDataRole.UserRole)
                if note_id:
                    self.selected_note_ids.add(note_id)
        self.results_list.blockSignals(False)
        
        self.update_selection_count()
        tooltip(f"✓ Selected all {self.results_list.rowCount()} notes")
    
    def deselect_all_notes(self):
        """Deselect all notes in the results list"""
        if not hasattr(self, 'results_list'):
            return
        
        # Block signals to prevent multiple updates
        self.results_list.blockSignals(True)
        for row in range(self.results_list.rowCount()):
            item = self.results_list.item(row, 1)  # Content column has checkbox
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)
                # Remove from persistence
                note_id = item.data(Qt.ItemDataRole.UserRole)
                if note_id:
                    self.selected_note_ids.discard(note_id)
        self.results_list.blockSignals(False)
        
        self.update_selection_count()
        tooltip(f"✗ Deselected all notes")
    
    def restore_selections(self):
        """Restore selections from stored note IDs"""
        if not hasattr(self, 'selected_note_ids') or not self.selected_note_ids:
            return
        
        # Block signals to prevent multiple updates
        self.results_list.blockSignals(True)
        for row in range(self.results_list.rowCount()):
            item = self.results_list.item(row, 1)  # Content column has checkbox
            if item:
                note_id = item.data(Qt.ItemDataRole.UserRole)
                if note_id in self.selected_note_ids:
                    item.setCheckState(Qt.CheckState.Checked)
        self.results_list.blockSignals(False)
    
    def open_selected_in_browser(self):
        checked_ids = []
        for row in range(self.results_list.rowCount()):
            item = self.results_list.item(row, 1)  # Content column has checkbox
            if item and item.checkState() == Qt.CheckState.Checked:
                note_id = item.data(Qt.ItemDataRole.UserRole)
                checked_ids.append(str(note_id))
        
        if not checked_ids:
            tooltip("Please check at least one note to view")
            return
        
        browser = aqt.dialogs.open("Browser", mw)
        search_query = "nid:" + ",".join(checked_ids)
        browser.form.searchEdit.lineEdit().setText(search_query)
        browser.onSearchActivated()
        QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))
        tooltip(f"✓ Opened {len(checked_ids)} selected notes in browser")
    
    def open_all_in_browser(self):
        if self.results_list.rowCount() == 0:
            tooltip("No notes to view")
            return
        
        note_ids = []
        for row in range(self.results_list.rowCount()):
            item = self.results_list.item(row, 1)  # Content column has note ID in UserRole
            if item:
                note_id = item.data(Qt.ItemDataRole.UserRole)
                note_ids.append(str(note_id))
        
        browser = aqt.dialogs.open("Browser", mw)
        search_query = "nid:" + ",".join(note_ids)
        browser.form.searchEdit.lineEdit().setText(search_query)
        browser.onSearchActivated()
        QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))
        tooltip(f"✓ Opened {len(note_ids)} notes in browser")
    
    def open_in_browser(self, item):
        """Open note in browser when double-clicked"""
        # Get the row of the clicked item
        row = item.row()
        # Get note ID from content column (column 1 = Content)
        content_item = self.results_list.item(row, 1)
        if content_item:
            note_id = content_item.data(Qt.ItemDataRole.UserRole)
            browser = aqt.dialogs.open("Browser", mw)
            browser.form.searchEdit.lineEdit().setText(f"nid:{note_id}")
            browser.onSearchActivated()
            QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))
            tooltip("✓ Note opened in browser")


# Copy methods that were defined on EmbeddingSearchWorker or RerankWorker to AISearchDialog
# (they are indented under those worker classes but are used on AISearchDialog instances)
_aisearch_methods_from_worker = (
    '_get_answer_source_text', '_on_embedding_search_progress', '_show_busy_progress', '_show_centile_progress',
    '_start_estimated_progress_timer', '_stop_estimated_progress_timer', '_hide_busy_progress', '_on_embedding_search_finished',
    '_on_keyword_filter_continue_done', '_on_embedding_search_error', '_on_rerank_done', '_on_get_notes_done', '_on_keyword_filter_done',
    '_perform_search_continue', 'perform_search',
    '_on_relevance_rerank_done', '_display_answer_and_notes_after_rerank', '_on_relevance_rerank_done_stream', '_finish_anthropic_stream_display',
    '_on_ask_ai_success', '_on_ask_ai_error', '_on_ask_ai_worker_finished',
    '_rerank_with_cross_encoder', 'keyword_filter', 'keyword_filter_continue', 'get_best_model',
    'ask_ai', 'call_ollama', 'call_anthropic', 'call_openai', 'call_google', 'call_openrouter',
    'call_custom', 'make_request', 'parse_response', '_rerank_by_relevance_to_answer', 'filter_and_display_notes', '_get_matching_terms_for_note', 'update_selection_count',
    '_start_anthropic_stream', '_append_stream_chunk', '_on_anthropic_stream_done', '_on_anthropic_stream_error',
    'on_preview_length_changed', 'toggle_select_all', 'select_all_notes', 'deselect_all_notes',
    'restore_selections', '_bring_browser_to_front', 'open_selected_in_browser', 'open_all_in_browser', 'open_in_browser',
)
for _name in _aisearch_methods_from_worker:
    _method = (
        getattr(EmbeddingSearchWorker, _name, None)
        or getattr(RerankWorker, _name, None)
        or getattr(AnthropicStreamWorker, _name, None)
    )
    if _method is not None:
        setattr(AISearchDialog, _name, _method)


# Singleton: only one search dialog instance so data is not loaded multiple times
_ai_search_dialog_instance = None

def show_search_dialog():
    global _ai_search_dialog_instance
    log_debug("Opening search dialog")
    if _ai_search_dialog_instance is not None:
        try:
            if _ai_search_dialog_instance.isVisible():
                _ai_search_dialog_instance.raise_()
                _ai_search_dialog_instance.activateWindow()
                return
            _ai_search_dialog_instance.show()
            _ai_search_dialog_instance.raise_()
            _ai_search_dialog_instance.activateWindow()
            return
        except RuntimeError:
            _ai_search_dialog_instance = None
    dialog = AISearchDialog(mw)
    _ai_search_dialog_instance = dialog
    def _clear_search_dialog_ref():
        global _ai_search_dialog_instance
        if _ai_search_dialog_instance is dialog:
            _ai_search_dialog_instance = None
    try:
        dialog.destroyed.connect(_clear_search_dialog_ref)
    except Exception:
        pass
    dialog.setWindowModality(Qt.WindowModality.NonModal)
    dialog.show()

def show_settings_dialog(open_to_embeddings=False):
    log_debug("Opening settings dialog" + (" (Embeddings)" if open_to_embeddings else ""))
    dialog = SettingsDialog(mw, open_to_embeddings=open_to_embeddings)
    dialog.setWindowModality(Qt.WindowModality.NonModal)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()

def show_debug_log():
    try:
        addon_dir = os.path.dirname(__file__)
        log_file = os.path.join(addon_dir, "debug_log.txt")
        if os.path.exists(log_file):
            if os.name == 'nt':
                os.startfile(log_file)
            else:
                import subprocess
                subprocess.call(['open' if sys.platform == 'darwin' else 'xdg-open', log_file])
        else:
            showInfo("Debug log file not found. Try using the add-on first.")
    except Exception as e:
        showInfo(f"Error opening log file: {e}")

def check_vc_redistributables():
    """Check if Visual C++ Redistributables are installed"""
    import os
    import winreg
    
    if os.name != 'nt':  # Not Windows
        return True  # Assume OK on non-Windows
    
    try:
        # Check for Visual C++ 2015-2022 Redistributables (x64)
        # They're registered in the Windows registry
        vc_versions = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x86"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x86"),
        ]
        
        # Also check for newer versions (2017-2022)
        for major_version in range(14, 18):  # 14.0 to 17.0
            vc_versions.extend([
                (winreg.HKEY_LOCAL_MACHINE, f"SOFTWARE\\Microsoft\\VisualStudio\\{major_version}.0\\VC\\Runtimes\\x64"),
                (winreg.HKEY_LOCAL_MACHINE, f"SOFTWARE\\WOW6432Node\\Microsoft\\VisualStudio\\{major_version}.0\\VC\\Runtimes\\x64"),
            ])
        
        # Check if any version is installed - need to actually read a value to verify
        found_any = False
        for hkey, key_path in vc_versions:
            try:
                with winreg.OpenKey(hkey, key_path) as key:
                    # Try to read a value to ensure the key is valid
                    try:
                        version = winreg.QueryValueEx(key, "Version")[0]
                        if version:  # If we got a version, it's installed
                            found_any = True
                            log_debug(f"Found VC++ Redistributables: {key_path}, Version: {version}")
                            break
                    except (FileNotFoundError, OSError):
                        # Key exists but no Version value - still might be installed
                        # Check for other indicators
                        try:
                            # Try to enumerate values (with safety limit to prevent infinite loop)
                            i = 0
                            max_iterations = 1000  # Safety limit
                            while i < max_iterations:
                                try:
                                    name, value, _ = winreg.EnumValue(key, i)
                                    if name and value:
                                        found_any = True
                                        break
                                    i += 1
                                except OSError:
                                    break
                            if found_any:
                                break
                        except:
                            pass
            except (FileNotFoundError, OSError):
                continue
        
        if found_any:
            return True
        return False  # No VC++ redistributables found
    except Exception as e:
        log_debug(f"Error checking VC++ redistributables: {e}")
        # If we can't check, return None (unknown) instead of assuming True
        return None  # Unknown status

def install_vc_redistributables():
    """Download and install Visual C++ Redistributables"""
    import os
    import sys
    import urllib.request
    import subprocess
    import tempfile
    
    if os.name != 'nt':  # Not Windows
        showInfo("Visual C++ Redistributables are only needed on Windows.")
        return False
    
    vc_redist_url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    vc_redist_filename = "vc_redist.x64.exe"
    
    try:
        # Check if already installed - but be more thorough
        vc_status = check_vc_redistributables()
        if vc_status is True:
            # Double-check by trying to actually use a DLL that requires VC++
            # If PyTorch is failing, VC++ might not actually be working
            reply = QMessageBox.question(
                mw,
                "VC++ Redistributables Check",
                "VC++ Redistributables appear to be installed according to the registry.\n\n"
                "However, if you're still experiencing PyTorch DLL errors, they may not be working correctly.\n\n"
                "Options:\n"
                "1. Reinstall VC++ Redistributables anyway (recommended)\n"
                "2. Use keyword-only search (no PyTorch needed)\n"
                "Reinstall VC++ Redistributables?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False
            # Continue with installation anyway
        elif vc_status is None:
            # Unknown status - proceed with installation
            pass
        # If False, continue with installation
        
        # Create a progress dialog
        progress_dialog = QDialog(mw)
        progress_dialog.setWindowTitle("Installing Visual C++ Redistributables")
        progress_dialog.setMinimumWidth(500)
        progress_dialog.setMinimumHeight(200)
        progress_dialog.setModal(True)
        layout = QVBoxLayout(progress_dialog)
        
        status_label = QLabel("Downloading Visual C++ Redistributables...")
        status_label.setWordWrap(True)
        layout.addWidget(status_label)
        
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(progress_bar)
        
        close_btn = QPushButton("Close")
        close_btn.setEnabled(False)
        layout.addWidget(close_btn)
        
        progress_dialog.show()
        QApplication.processEvents()
        
        # Download the installer
        temp_dir = tempfile.gettempdir()
        installer_path = os.path.join(temp_dir, vc_redist_filename)
        
        def download_installer():
            try:
                status_label.setText("Downloading Visual C++ Redistributables installer...\nThis may take a minute.")
                QApplication.processEvents()
                
                urllib.request.urlretrieve(vc_redist_url, installer_path)
                
                status_label.setText("Download complete. Launching installer...\n\nYou may need to grant administrator privileges.")
                QApplication.processEvents()
                
                # Launch the installer
                # /quiet = silent install, /norestart = don't restart
                # /passive = show progress but no user interaction needed
                subprocess.Popen([installer_path, "/passive", "/norestart"], shell=True)
                
                status_label.setText("✅ Installer launched!\n\nPlease follow the installation wizard.\nAfter installation completes, restart Anki.")
                close_btn.setEnabled(True)
                progress_bar.setRange(0, 100)
                progress_bar.setValue(100)
                
                log_debug("VC++ Redistributables installer launched successfully")
                return True
            except Exception as e:
                status_label.setText(f"❌ Error: {str(e)}\n\nYou can manually download and install from:\n{vc_redist_url}")
                close_btn.setEnabled(True)
                log_debug(f"Error installing VC++ redistributables: {e}")
                return False
        
        # Run download in a thread to avoid blocking
        import threading
        thread = threading.Thread(target=download_installer, daemon=True)
        thread.start()
        
        # Don't wait for thread - let user close dialog when ready
        close_btn.clicked.connect(progress_dialog.close)
        
        return True
    except Exception as e:
        error_msg = (
            f"Error preparing VC++ Redistributables installation: {str(e)}\n\n"
            f"Please manually download and install from:\n{vc_redist_url}\n\n"
            "After installation, restart Anki."
        )
        showInfo(error_msg)
        log_debug(f"Error in install_vc_redistributables: {e}")
        return False

def get_pytorch_dll_error_guidance():
    """Get guidance message for PyTorch DLL loading errors"""
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    vc_status = check_vc_redistributables()
    vc_message = ""
    if vc_status is False:
        vc_message = "\n⚠️ Visual C++ Redistributables appear to be MISSING!\n   Click 'Install VC++ Redistributables' button below.\n\n"
    elif vc_status is None:
        vc_message = "\n⚠️ Could not verify Visual C++ Redistributables installation.\n   You may need to install them manually.\n\n"
    
    guidance = (
        "PyTorch DLL Loading Error Detected\n\n"
        f"Python version: {python_version}\n\n"
        f"{vc_message}"
        "Common causes and solutions:\n\n"
        "1. Missing Visual C++ Redistributables:\n"
        "   - Click 'Install VC++ Redistributables' button below\n"
        "   - Or download manually from:\n"
        "   - https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
        "2. Python 3.13 compatibility:\n"
        "   - Python 3.13 is very new and PyTorch may not have full support yet\n"
        "   - Try reinstalling PyTorch with CPU-only version:\n"
        "   - Use 'Fix PyTorch DLL Issue' button below\n\n"
        "3. Corrupted installation:\n"
        "   - Try: pip uninstall sentence-transformers torch\n"
        "   - Then reinstall using 'Install Dependencies' in settings\n\n"
        "4. Alternative: Use Anki with Python 3.11 or 3.12 for better compatibility"
    )
    return guidance

def _patch_colorama_for_transformers():
    """Patch colorama ErrorHandler to add flush attribute for transformers compatibility.
    This is a wrapper that ensures the patch is applied (the actual patch runs at module load)."""
    # The actual patching happens at module load time via _patch_colorama_early()
    # This function is kept for backward compatibility and to ensure patch is applied
    # if called before module initialization completes
    try:
        _patch_colorama_early()
    except:
        pass  # Silently fail

def check_dependency_installed(package_name):
    """Check if a Python package is installed"""
    try:
        # Patch colorama before importing sentence_transformers to avoid AttributeError
        if 'sentence_transformers' in package_name or 'transformers' in package_name:
            _patch_colorama_for_transformers()
            _ensure_stderr_patched()
        __import__(package_name.replace('-', '_'))
        return True
    except (ImportError, OSError, ModuleNotFoundError, AttributeError, Exception) as e:
        # OSError can occur when PyTorch DLLs fail to load (e.g., missing Visual C++ Redistributables)
        # ModuleNotFoundError is a subclass of ImportError but we catch it explicitly for clarity
        # AttributeError can occur due to library compatibility issues (e.g., colorama/transformers)
        if isinstance(e, OSError) and 'torch' in str(e).lower():
            log_debug(f"PyTorch DLL error detected: {e}")
        elif isinstance(e, AttributeError):
            log_debug(f"AttributeError during import (likely compatibility issue): {e}")
        return False

def _resolve_external_python_exe(python_path):
    """Resolve 'Python for Cross-Encoder' path to python executable. Returns None if invalid."""
    import os
    path = (python_path or "").strip()
    if not path:
        return None
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        exe = os.path.join(path, "python.exe")
        if os.path.isfile(exe):
            return exe
        exe = os.path.join(path, "python")
        if os.path.isfile(exe):
            return exe
    return None


def try_alternative_pytorch_install():
    """Try alternative PyTorch installation methods"""
    import sys
    import subprocess
    
    methods = [
        {
            "name": "Method 1: PyTorch 2.0.1 (Older, more stable)",
            "command": [sys.executable, "-m", "pip", "install", "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2", "--index-url", "https://download.pytorch.org/whl/cpu"]
        },
        {
            "name": "Method 2: PyTorch 2.1.0 (Mid-version)",
            "command": [sys.executable, "-m", "pip", "install", "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0", "--index-url", "https://download.pytorch.org/whl/cpu"]
        },
        {
            "name": "Method 3: Latest PyTorch (Current default)",
            "command": [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]
        },
        {
            "name": "Method 4: PyTorch without CUDA (pip default)",
            "command": [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        }
    ]
    
    dialog = QDialog(mw)
    dialog.setWindowTitle("Try Alternative PyTorch Installation")
    dialog.setMinimumWidth(500)
    layout = QVBoxLayout(dialog)
    
    info_label = QLabel(
        "If the standard installation failed, try these alternative methods:\n\n"
        "Select a method to try:"
    )
    info_label.setWordWrap(True)
    layout.addWidget(info_label)
    
    method_combo = QComboBox()
    for method in methods:
        method_combo.addItem(method["name"])
    layout.addWidget(method_combo)
    
    button_layout = QHBoxLayout()
    try_btn = QPushButton("Try This Method")
    cancel_btn = QPushButton("Cancel")
    button_layout.addWidget(try_btn)
    button_layout.addWidget(cancel_btn)
    layout.addLayout(button_layout)
    
    def try_method():
        selected_idx = method_combo.currentIndex()
        method = methods[selected_idx]
        dialog.close()
        
        # Show progress
        progress = QDialog(mw)
        progress.setWindowTitle("Installing PyTorch")
        progress_layout = QVBoxLayout(progress)
        status = QLabel(f"Trying: {method['name']}\n\nThis may take several minutes...")
        status.setWordWrap(True)
        progress_layout.addWidget(status)
        progress.show()
        QApplication.processEvents()
        
        try:
            result = subprocess.run(
                method["command"],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                status.setText("✅ Installation successful!\n\nTesting import...")
                QApplication.processEvents()
                
                # Test import
                try:
                    _patch_colorama_for_transformers()
                    _ensure_stderr_patched()
                    import torch
                    status.setText(f"✅ Success! PyTorch {torch.__version__} installed and working.\n\nNow install sentence-transformers.")
                    showInfo(f"PyTorch {torch.__version__} installed successfully!\n\nNow click 'Install Dependencies' to install sentence-transformers.")
                except Exception as e:
                    status.setText(f"⚠️ Installed but import failed: {e}\n\nTry installing VC++ Redistributables.")
                    showInfo(f"PyTorch installed but import failed: {e}\n\nTry installing VC++ Redistributables first.")
            else:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                status.setText(f"❌ Installation failed:\n{error_msg}")
                showInfo(f"Installation failed. Try another method or install VC++ Redistributables.")
        except Exception as e:
            status.setText(f"❌ Error: {e}")
            showInfo(f"Error: {e}")
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(progress.close)
        progress_layout.addWidget(close_btn)
    
    try_btn.clicked.connect(try_method)
    cancel_btn.clicked.connect(dialog.close)
    
    dialog.exec()

def fix_pytorch_dll_issue():
    """Fix PyTorch DLL issues by reinstalling with CPU-only version"""
    import sys
    import subprocess
    
    reply = QMessageBox.question(
        mw,
        "Fix PyTorch DLL Issue",
        "This will reinstall PyTorch with a CPU-only version that's more compatible.\n\n"
        "Steps:\n"
        "1. Uninstall existing PyTorch packages\n"
        "2. Install CPU-only PyTorch from official repository\n"
        "3. Reinstall sentence-transformers\n\n"
        "⚠️ IMPORTANT: If this fails, you may need to:\n"
        "- Install Visual C++ Redistributables first\n"
        "- Try alternative PyTorch versions\n"
        "- Use keyword-only search (no embeddings needed)\n\n"
        "This may take a few minutes. Continue?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes
    )
    
    if reply != QMessageBox.StandardButton.Yes:
        return
    
    # Create progress dialog
    progress_dialog = QDialog(mw)
    progress_dialog.setWindowTitle("Fixing PyTorch DLL Issue")
    progress_dialog.setMinimumWidth(600)
    progress_dialog.setMinimumHeight(500)
    progress_dialog.setModal(False)
    progress_layout = QVBoxLayout(progress_dialog)
    
    status_label = QLabel("Preparing...")
    status_label.setWordWrap(True)
    progress_layout.addWidget(status_label)
    
    log_text = QTextEdit()
    log_text.setReadOnly(True)
    log_text.setMaximumHeight(300)
    log_text.setFont(QFont("Courier", 9))
    log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
    progress_layout.addWidget(log_text)
    
    close_button = QPushButton("Close")
    close_button.setEnabled(False)
    close_button.clicked.connect(progress_dialog.close)
    progress_layout.addWidget(close_button)
    
    progress_dialog.show()
    QApplication.processEvents()
    
    def log(msg):
        log_text.append(msg)
        log_text.verticalScrollBar().setValue(log_text.verticalScrollBar().maximum())
        QApplication.processEvents()
        log_debug(msg)
    
    try:
        # Step 1: Uninstall PyTorch packages
        status_label.setText("Step 1/3: Uninstalling existing PyTorch packages...")
        log("Uninstalling torch, torchvision, torchaudio...")
        
        packages_to_uninstall = ['torch', 'torchvision', 'torchaudio', 'sentence-transformers']
        for pkg in packages_to_uninstall:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", pkg, "-y"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    log(f"✅ Uninstalled {pkg}")
                else:
                    log(f"⚠️ {pkg} may not have been installed")
            except Exception as e:
                log(f"⚠️ Error uninstalling {pkg}: {e}")
        
        # Step 2: Install CPU-only PyTorch
        status_label.setText("Step 2/3: Installing CPU-only PyTorch...")
        log("Installing PyTorch CPU-only version from official repository...")
        log("This may take several minutes...")
        
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            log("✅ PyTorch CPU-only installed successfully")
        else:
            log(f"❌ Error installing PyTorch:")
            for line in result.stderr.split('\n')[-10:]:
                if line.strip():
                    log(line)
            raise Exception("PyTorch installation failed")
        
        # Step 3: Reinstall sentence-transformers
        status_label.setText("Step 3/3: Reinstalling sentence-transformers...")
        log("Installing sentence-transformers...")
        
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "sentence-transformers"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            log("✅ sentence-transformers installed successfully")
        else:
            log(f"❌ Error installing sentence-transformers:")
            for line in result.stderr.split('\n')[-10:]:
                if line.strip():
                    log(line)
            raise Exception("sentence-transformers installation failed")
        
        # Verify installation
        status_label.setText("Verifying installation...")
        log("Testing import...")
        
        try:
            _patch_colorama_for_transformers()
            _ensure_stderr_patched()
            from sentence_transformers import SentenceTransformer
            log("✅ Import test successful!")
            status_label.setText("✅ Fix completed successfully!")
            status_label.setStyleSheet("color: green; font-weight: bold;")
            showInfo("PyTorch DLL issue fixed! You may need to restart Anki for changes to take effect.")
        except Exception as e:
            log(f"❌ Import test failed: {e}")
            status_label.setText("⚠️ Installation completed but import test failed")
            status_label.setStyleSheet("color: orange; font-weight: bold;")
            
            # Add helpful buttons
            button_layout = QHBoxLayout()
            
            try_alt_btn = QPushButton("Try Alternative PyTorch Version")
            try_alt_btn.clicked.connect(lambda: (progress_dialog.close(), try_alternative_pytorch_install()))
            button_layout.addWidget(try_alt_btn)
            
            vc_btn = QPushButton("Install VC++ Redistributables")
            vc_btn.clicked.connect(lambda: (progress_dialog.close(), install_vc_redistributables()))
            button_layout.addWidget(vc_btn)
            
            use_keyword_btn = QPushButton("Use Keyword-Only Search (No PyTorch)")
            use_keyword_btn.clicked.connect(lambda: (
                progress_dialog.close(),
                showInfo("You can use the addon in keyword-only mode!\n\n"
                        "1. Go to Settings\n"
                        "2. Change 'Search Method' to 'Keyword Only'\n"
                        "3. The addon will work without embeddings.")
            ))
            button_layout.addWidget(use_keyword_btn)
            
            progress_layout.addLayout(button_layout)
            
            error_msg = (
                f"Installation completed but verification failed: {e}\n\n"
                "Options:\n"
                "1. Try alternative PyTorch version (button above)\n"
                "2. Install VC++ Redistributables (button above)\n"
                "3. Use keyword-only search mode (no embeddings needed)\n"
                "4. Check the log for details"
            )
            showInfo(error_msg)
        
    except Exception as e:
        log(f"❌ Error: {e}")
        status_label.setText(f"❌ Error: {e}")
        status_label.setStyleSheet("color: red; font-weight: bold;")
        showInfo(f"Error fixing PyTorch: {e}")
    finally:
        close_button.setEnabled(True)

def _check_sentence_transformers_installed_subprocess():
    """Check if sentence-transformers is usable in Anki's Python via subprocess (avoids in-process import failures on Python 3.13)."""
    try:
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-c", "from sentence_transformers import CrossEncoder; print('ok')"],
            capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        return result.returncode == 0 and 'ok' in (result.stdout or '')
    except Exception:
        return False


def install_dependencies(python_exe=None):
    """Show manual install instructions for optional dependencies (no auto pip install).
    python_exe: None = Anki's Python; else path to external python.exe for Cross-Encoder."""
    import sys

    if python_exe:
        target_python = python_exe
        target_label = "Python for Cross-Encoder (from Settings)"
    else:
        target_python = sys.executable
        target_label = "Anki's Python"

    # Check if already installed
    try:
        import subprocess
        result = subprocess.run(
            [target_python, "-c", "from sentence_transformers import CrossEncoder; print('ok')"],
            capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        if result.returncode == 0 and 'ok' in (result.stdout or ''):
            showInfo("✅ sentence-transformers is already available.\n\nClick 'Check again' in Settings to enable Cross-Encoder.")
            return
    except Exception:
        pass

    pip_cmd = f'"{target_python}" -m pip install sentence-transformers'
    msg = (
        "Optional: Cross-Encoder re-ranking (better retrieval quality)\n\n"
        f"Python executable: {target_python}\n\n"
        "Copy this command and run it in a terminal:\n\n"
        f"  {pip_cmd}\n\n"
        f"Where to run: Use the Python above, or the one set as 'Python for Cross-Encoder' in Settings.\n\n"
        "See config.md in the add-on folder for troubleshooting."
    )
    dlg = QMessageBox(mw)
    dlg.setWindowTitle("Manual Install: sentence-transformers")
    dlg.setText(msg)
    dlg.setIcon(QMessageBox.Icon.Information)
    copy_btn = dlg.addButton("Copy command", QMessageBox.ButtonRole.ActionRole)
    dlg.addButton(QMessageBox.StandardButton.Ok)
    dlg.exec()
    if dlg.clickedButton() == copy_btn:
        QApplication.clipboard().setText(pip_cmd)
        tooltip("Command copied to clipboard")

log_debug("=== Anki Semantic Search Add-on Loaded ===")
log_debug(f"Addon directory: {os.path.dirname(__file__)}")
log_debug(f"Addon folder name: {ADDON_NAME}")

# Note: colorama is already patched at module load time via _patch_colorama_early()

# Add menu items
ai_search_menu = QMenu("🔍 Anki Semantic Search", mw)

search_action = QAction("Search Notes", mw)
search_action.triggered.connect(show_search_dialog)
search_action.setToolTip("Open Anki Semantic Search window")
ai_search_menu.addAction(search_action)

settings_action = QAction("Settings", mw)
settings_action.triggered.connect(lambda: show_settings_dialog(open_to_embeddings=False))
settings_action.setToolTip("Configure API, note types, search, and embeddings")
ai_search_menu.addAction(settings_action)

create_embeddings_action = QAction("Set up embeddings (one-time)", mw)
create_embeddings_action.triggered.connect(lambda: show_settings_dialog(open_to_embeddings=True))
create_embeddings_action.setToolTip(
    "Open Settings to the Embeddings panel to run the one-time embeddings setup so AI can search your notes. "
    "You can re-run this later if you change engines or add many new notes."
)
ai_search_menu.addAction(create_embeddings_action)

install_deps_action = QAction("Install extra model for better ranking", mw)
install_deps_action.triggered.connect(install_dependencies)
install_deps_action.setToolTip(
    "Install the optional sentence-transformers cross-encoder model for improved result ordering. "
    "Not required for basic Anki Semantic Search."
)
ai_search_menu.addAction(install_deps_action)

ai_search_menu.addSeparator()

debug_action = QAction("View Debug Log", mw)
debug_action.triggered.connect(show_debug_log)
ai_search_menu.addAction(debug_action)

mw.form.menuTools.addMenu(ai_search_menu)

mw.addonManager.setConfigAction(ADDON_NAME, show_settings_dialog)

# Background indexer: re-enabled using QueryOp — collection access only on main thread via QueryOp; worker thread does API + save only (no mw.col).
def _on_main_window_did_init():
    # Delay 15s after Anki start so the app is fully ready before any addon background work
    QTimer.singleShot(15000, _start_background_indexer)
try:
    gui_hooks.main_window_did_init.append(_on_main_window_did_init)
except Exception as e:
    log_debug(f"Could not register main_window_did_init hook: {e}")

log_debug("Menu items added successfully")

# END OF PART 3 - This is the complete file!