"""Search worker threads and standalone rerank helpers used by the search dialog."""

# ============================================================================
# Imports
# ============================================================================

import json
import os
import urllib.request

from aqt.qt import QThread, pyqtSignal

from ..core.compat import _ensure_stderr_patched, _patch_colorama_early
from ..core.engine import (
    get_embedding_for_query,
    get_embeddings_batch,
    load_embedding,
    load_embeddings_bulk,
)
from ..core.errors import _is_embedding_dimension_mismatch
from ..utils.config import (
    DEFAULT_RERANK_MODEL,
    RERANK_TOP_K_DEFAULT,
    get_rerank_config,
    get_retrieval_config,
)
from ..utils.log import log_debug
from ..utils.text import semantic_chunk_text
from .image_attachments import IMAGE_SUPPORT_ERROR, is_image_support_error


BULK_SEARCH_SOFT_MEMORY_BYTES = 256 * 1024 * 1024
BULK_SEARCH_LIMIT = 80
_LAST_EMBEDDING_MATCH_DIAGNOSTICS = {
    "rows": 0,
    "exact_matches": 0,
    "note_id_fallback_matches": 0,
}


# ============================================================================
# Search Worker Compatibility And Standalone Search Helpers
# ============================================================================


class AgenticPlanWorker(QThread):
    """Build the Agentic RAG plan away from the UI thread."""

    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, build_plan_callable, query, history, config):
        super().__init__()
        self._build_plan_callable = build_plan_callable
        self._query = query
        self._history = history
        self._config = config

    def run(self):
        try:
            plan = self._build_plan_callable(self._query, self._history, self._config)
            self.finished_signal.emit(plan)
        except Exception as exc:
            log_debug(f"AgenticPlanWorker error: {exc}")
            self.error_signal.emit(str(exc))

def _embedding_search_backend_sqlite_scan(embedding_query, notes, config, db_path=None):
    """Embedding-search boundary for future FAISS/sqlite-vec backends.

    Current default remains the existing SQLite blob scan plus Python cosine
    scoring. Future vector stores should implement this same return shape:
    list[(score, note)] sorted descending, or None on unavailable embeddings.
    """
    return _run_embedding_search_sync(embedding_query, notes, config, db_path=db_path)


def get_last_embedding_match_diagnostics():
    return dict(_LAST_EMBEDDING_MATCH_DIAGNOSTICS)


def _set_embedding_match_diagnostics(rows=0, exact_matches=0, note_id_fallback_matches=0):
    _LAST_EMBEDDING_MATCH_DIAGNOSTICS.update(
        {
            "rows": int(rows or 0),
            "exact_matches": int(exact_matches or 0),
            "note_id_fallback_matches": int(note_id_fallback_matches or 0),
        }
    )


def _note_content_hash(note):
    import hashlib

    content_hash = note.get("content_hash")
    if content_hash is None:
        content_hash = hashlib.md5(note.get("content", "").encode()).hexdigest()
    return content_hash


def _run_embedding_search_bulk_numpy(
    embedding_list,
    notes,
    db_path=None,
    progress_callback=None,
    should_cancel=None,
    result_limit=BULK_SEARCH_LIMIT,
    memory_cap_bytes=BULK_SEARCH_SOFT_MEMORY_BYTES,
):
    """Score embeddings with batched NumPy matrix multiplication."""
    try:
        import numpy as np
    except ImportError:
        return None

    query_vector = np.asarray(embedding_list, dtype=np.float32)
    query_dim = int(query_vector.shape[0]) if query_vector.ndim == 1 else 0
    query_norm = float(np.linalg.norm(query_vector)) if query_dim else 0.0
    if not query_dim or query_norm <= 0:
        return None
    query_vector = query_vector / query_norm

    note_lookup = {}
    note_lookup_by_id = {}
    note_order = {}
    for idx, note in enumerate(notes or []):
        key = (note.get("id"), _note_content_hash(note))
        note_lookup[key] = note
        note_lookup_by_id.setdefault(note.get("id"), note)
        note_order[key] = idx

    rows = load_embeddings_bulk(db_path=db_path)
    if not rows:
        _set_embedding_match_diagnostics(0, 0, 0)
        return None
    exact_available_note_ids = {
        row.get("note_id")
        for row in rows
        if (row.get("note_id"), row.get("content_hash")) in note_lookup
    }

    bytes_per_vector = max(1, query_dim * 4)
    batch_size = max(1, int(memory_cap_bytes // bytes_per_vector))
    scored_notes = []
    vectors = []
    vector_notes = []
    checked = 0
    total = len(rows)
    exact_matches = 0
    fallback_matches = 0

    def flush_batch():
        nonlocal vectors, vector_notes, scored_notes
        if not vectors:
            return
        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1)
        valid = norms > 0
        if valid.any():
            matrix = matrix[valid] / norms[valid, None]
            notes_valid = [note for note, keep in zip(vector_notes, valid.tolist()) if keep]
            similarities = matrix @ query_vector
            scored_notes.extend(((float(sim) + 1.0) * 50.0, note) for sim, note in zip(similarities, notes_valid))
        vectors = []
        vector_notes = []

    for row in rows:
        if should_cancel and should_cancel():
            return None
        checked += 1
        if progress_callback and (checked == 1 or checked == total or checked % max(50, total // 40 or 1) == 0):
            pct = int(100 * checked / total) if total else 0
            progress_callback(checked, total, f"Embedding search: {checked}/{total} ({pct}%)")

        key = (row.get("note_id"), row.get("content_hash"))
        note = note_lookup.get(key)
        if note is not None:
            exact_matches += 1
            note["_embedding_match_status"] = "exact"
        else:
            note = None
            if row.get("note_id") not in exact_available_note_ids:
                note = note_lookup_by_id.get(row.get("note_id"))
            if note is not None:
                fallback_matches += 1
                note["_embedding_match_status"] = "fallback"
        if note is None:
            continue
        try:
            emb = np.frombuffer(row.get("embedding_blob"), dtype=np.float32)
        except Exception:
            continue
        if emb.shape[0] != query_dim:
            log_debug(
                f"Embedding search: skipping note {row.get('note_id')} "
                f"(dimension mismatch query={query_dim} stored={emb.shape[0]})"
            )
            continue
        vectors.append(emb)
        vector_notes.append(note)
        if len(vectors) >= batch_size:
            flush_batch()

    flush_batch()
    _set_embedding_match_diagnostics(total, exact_matches, fallback_matches)
    log_debug(
        "Embedding search bulk match diagnostics: "
        f"rows={total}, exact_matches={exact_matches}, note_id_fallback_matches={fallback_matches}"
    )
    if not scored_notes:
        return []

    # If duplicate rows match a note, keep the strongest chunk score.
    best_by_note_key = {}
    for score, note in scored_notes:
        key = (note.get("id"), note.get("chunk_index"), _note_content_hash(note))
        if key not in best_by_note_key or score > best_by_note_key[key][0]:
            best_by_note_key[key] = (score, note)
    scored_notes = list(best_by_note_key.values())
    scored_notes.sort(key=lambda item: (-item[0], note_order.get((item[1].get("id"), _note_content_hash(item[1])), 10**12)))
    return scored_notes[:result_limit]


def _run_embedding_search_fallback_loop(embedding_list, notes, db_path=None):
    import array
    import math

    query_embedding = [float(x) for x in (embedding_list or [])]
    query_dim = len(query_embedding)
    query_norm = math.sqrt(sum(x * x for x in query_embedding))
    if not query_dim or query_norm <= 0:
        return None
    query_embedding = [x / query_norm for x in query_embedding]

    rows = load_embeddings_bulk(db_path=db_path)
    if rows:
        note_lookup = {
            (note.get("id"), _note_content_hash(note)): note
            for note in notes or []
        }
        note_lookup_by_id = {}
        for note in notes or []:
            note_lookup_by_id.setdefault(note.get("id"), note)
        exact_available_note_ids = {
            row.get("note_id")
            for row in rows
            if (row.get("note_id"), row.get("content_hash")) in note_lookup
        }
        iterable = []
        exact_matches = 0
        fallback_matches = 0
        for row in rows:
            note = note_lookup.get((row.get("note_id"), row.get("content_hash")))
            if note is not None:
                exact_matches += 1
                note["_embedding_match_status"] = "exact"
            else:
                note = None
                if row.get("note_id") not in exact_available_note_ids:
                    note = note_lookup_by_id.get(row.get("note_id"))
                if note is not None:
                    fallback_matches += 1
                    note["_embedding_match_status"] = "fallback"
            if note is not None:
                iterable.append((note, row.get("embedding_blob")))
        log_debug(
            "Embedding fallback-loop match diagnostics: "
            f"rows={len(rows)}, exact_matches={exact_matches}, note_id_fallback_matches={fallback_matches}"
        )
        _set_embedding_match_diagnostics(len(rows), exact_matches, fallback_matches)
    else:
        _set_embedding_match_diagnostics(0, 0, 0)
        iterable = []
        for note in notes:
            content_hash = _note_content_hash(note)
            note_embedding = load_embedding(note.get("id"), content_hash, db_path=db_path)
            if note_embedding is not None:
                iterable.append((note, note_embedding))

    scored_notes = []
    for note, raw_embedding in iterable:
        if raw_embedding is None:
            continue
        if isinstance(raw_embedding, (bytes, bytearray, memoryview)):
            try:
                emb = array.array("f")
                emb.frombytes(bytes(raw_embedding))
            except Exception:
                continue
        else:
            emb = [float(x) for x in raw_embedding]
        if len(emb) != query_dim:
            log_debug(
                f"Embedding search: skipping note {note.get('id')} "
                f"(dimension mismatch query={query_dim} stored={len(emb)})"
            )
            continue
        emb_norm = math.sqrt(sum(float(x) * float(x) for x in emb))
        if emb_norm <= 0:
            continue
        similarity = sum(q * (float(e) / emb_norm) for q, e in zip(query_embedding, emb))
        scored_notes.append(((similarity + 1.0) * 50.0, note))
    scored_notes.sort(reverse=True, key=lambda x: x[0])
    return scored_notes[:BULK_SEARCH_LIMIT]


def _run_embedding_search_sync(embedding_query, notes, config, db_path=None):



    """Run embedding search in a background thread (for taskman). Returns scored_notes or None.



    db_path: from main thread so profile-specific path is correct in background."""



    try:



        embedding_list = get_embedding_for_query(embedding_query, config)



        if not embedding_list:



            return None



        bulk_results = _run_embedding_search_bulk_numpy(embedding_list, notes, db_path=db_path)
        if bulk_results is not None:
            return bulk_results

        return _run_embedding_search_fallback_loop(embedding_list, notes, db_path=db_path)



    except Exception as e:



        log_debug(f"Embedding search error: {e}")



        if _is_embedding_dimension_mismatch(e):



            return {"embedding_results": None, "error": "dimension_mismatch"}



        return None











# --- Embedding Search Worker ---

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
        try:



            embedding_list = get_embedding_for_query(self.embedding_query, self.config)



            if not embedding_list:



                self.finished_signal.emit(None)



                return



            total = len(self.notes)
            def _progress(done, count, message):
                self.progress_signal.emit(done, count, message)

            bulk_results = _run_embedding_search_bulk_numpy(
                embedding_list,
                self.notes,
                db_path=getattr(self, "db_path", None),
                progress_callback=_progress,
                should_cancel=self.isInterruptionRequested,
            )
            if self.isInterruptionRequested():
                self.finished_signal.emit(None)
                return
            if bulk_results is not None:
                self.finished_signal.emit(bulk_results)
                return

            progress_interval = max(50, total // 40)
            fallback_results = []
            # Keep progress responsive while the fallback loop does per-note DB reads.
            for idx, _note in enumerate(self.notes):
                if idx % progress_interval == 0 or idx == total - 1:
                    pct = int(100 * (idx + 1) / total) if total else 0
                    self.progress_signal.emit(idx + 1, total, f"Embedding search: {idx + 1}/{total} ({pct}%)")
            fallback_results = _run_embedding_search_fallback_loop(
                embedding_list,
                self.notes,
                db_path=getattr(self, "db_path", None),
            )
            self.finished_signal.emit(fallback_results)



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











MAX_RERANK_COUNT = RERANK_TOP_K_DEFAULT



RRF_K = 60  # Reciprocal Rank Fusion constant (standard in retrieval literature; 1/(k+rank) per list)











# --- Chunking And Reranking Helpers ---

_semantic_chunk_text = semantic_chunk_text
_RERANK_MODEL_CACHE = {}


def _truncate_for_rerank(text):
    return (text or "")[:512]


def _get_cross_encoder(CrossEncoder, model_name):
    model_name = (model_name or "").strip() or DEFAULT_RERANK_MODEL
    if model_name not in _RERANK_MODEL_CACHE:
        _RERANK_MODEL_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANK_MODEL_CACHE[model_name]


def _clamp_display_relevance(value):
    try:
        return max(0, min(100, round(float(value))))
    except Exception:
        return 0


def _log_rerank_score_scale(source, min_s, max_s, span, normalized, scaled):
    try:
        normalized_values = [float(rn) for rn, _note in (normalized or [])]
        display_values = [
            note.get("_display_relevance")
            for _score, note in (scaled or [])[:8]
            if note.get("_display_relevance") is not None
        ]
        log_debug(
            "Rerank score scale: "
            f"source={source}, raw_min={float(min_s):.6f}, raw_max={float(max_s):.6f}, "
            f"raw_span={float(span):.6f}, "
            f"normalized_min={min(normalized_values) if normalized_values else None}, "
            f"normalized_max={max(normalized_values) if normalized_values else None}, "
            f"display_relevance_sample={display_values}"
        )
    except Exception as exc:
        log_debug(f"Rerank score scale logging failed: {exc}")











def _do_rerank(query, scored_notes, top_k, search_config):



    """



    Re-rank top results using a cross-encoder (gold standard for NotebookLM-style accuracy).



    Uses configured top_k by default to avoid CPU bottleneck. Blends cross-encoder scores with pre-rerank.



    Returns (scored_notes, success).



    """



    import json



    import os



    import subprocess



    rerank_config = get_rerank_config(search_config)
    rerank_model = rerank_config.get("rerank_model") or DEFAULT_RERANK_MODEL
    rerank_top_k = int(top_k or rerank_config.get("rerank_top_k") or RERANK_TOP_K_DEFAULT)
    rerank_timeout = int(rerank_config.get("rerank_timeout_seconds") or 90)
    top_k = max(1, min(100, rerank_top_k))
    log_debug(f"Rerank model: {rerank_model}, top_k: {top_k}, timeout: {rerank_timeout}s")



    top_notes = scored_notes[:top_k]



    if not top_notes:



        return scored_notes, False



    pre_scores = {note['id']: score for score, note in top_notes}



    contents = [_truncate_for_rerank(note.get('content', '')) for _, note in top_notes]



    rerank_python = (search_config.get('rerank_python_path') or '').strip()



    if rerank_python:



        python_exe = rerank_python



        if os.path.isdir(rerank_python):



            python_exe = os.path.join(rerank_python, "python.exe")



            if not os.path.isfile(python_exe):



                python_exe = os.path.join(rerank_python, "python")



        if os.path.isfile(python_exe):



            addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



            helper_path = os.path.join(addon_dir, "rerank_helper.py")



            if os.path.isfile(helper_path):



                try:



                    payload = json.dumps({"query": query, "contents": contents, "model": rerank_model})



                    creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0



                    env = os.environ.copy()
                    env["HF_HUB_OFFLINE"] = "1"
                    env["TRANSFORMERS_OFFLINE"] = "1"

                    proc = subprocess.Popen(



                        [python_exe, helper_path],



                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,



                        text=True, creationflags=creationflags, env=env



                    )



                    out, err = proc.communicate(input=payload, timeout=rerank_timeout)



                    if proc.returncode != 0:



                        log_debug(f"Rerank helper failed: {err or out}")



                        return scored_notes, False



                    data = json.loads(out)



                    if "error" in data:



                        log_debug(f"Rerank helper error for {data.get('model') or rerank_model}: {data['error']}")



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



                        note['_display_relevance'] = _clamp_display_relevance((rn - 50) * 2)



                    blend.sort(reverse=True, key=lambda x: x[0])



                    max_b = blend[0][0] if blend else 1



                    scaled = [(b / max_b * 100.0, note) for b, note in blend]



                    # Soft floor: notes that were top-ranked before rerank don't show below 55%



                    top_pre_ids = set(nid for _, nid in sorted(pre_scores.items(), key=lambda x: -x[1])[:top_k])



                    scaled = [(max(pct, 55.0) if note['id'] in top_pre_ids else pct, note) for pct, note in scaled]



                    scaled.sort(reverse=True, key=lambda x: x[0])



                    _log_rerank_score_scale("external_helper", min_s, max_s, span, normalized, scaled)



                    # Rest notes weren't reranked; no _display_relevance (UI will use score)



                    rest = [(0.0, note) for _, note in scored_notes[top_k:]]



                    return scaled + rest, True



                except subprocess.TimeoutExpired:



                    proc.kill()



                    log_debug(f"Rerank helper timed out after {rerank_timeout}s")



                    return scored_notes, False

                except Exception as e:



                    log_debug(f"Rerank subprocess failed: {e}")



                    return scored_notes, False

            else:



                log_debug(f"Rerank helper not found at expected path: {helper_path}")



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



                "'Install Dependencies' in the AI Search menu (or in Settings \u2192 Search & Embeddings)."



            )



        except Exception:



            pass



        return scored_notes, False



    except OSError as e:



        log_debug(f"Cross-encoder re-ranking skipped: library load failed ({e})")



        # Do not use showInfo in background thread. Return success=False and handle in UI.



        return scored_notes, "LIBRARY_LOAD_FAILED"



    except Exception as e:



        log_debug(f"Cross-encoder re-ranking failed with unexpected error: {e}")



        return scored_notes, False



    try:



        old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            model = _get_cross_encoder(CrossEncoder, rerank_model)
        finally:
            if old_hf_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_hf_offline
            if old_transformers_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_transformers_offline



        pairs = [(query, _truncate_for_rerank(note.get('content', ''))) for _, note in top_notes]



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



            note['_display_relevance'] = _clamp_display_relevance((rn - 50) * 2)



        blend.sort(reverse=True, key=lambda x: x[0])



        max_b = blend[0][0] if blend else 1



        scaled = [(b / max_b * 100.0, note) for b, note in blend]



        # Soft floor: notes that were top-ranked before rerank don't show below 55%



        top_pre_ids = set(nid for _, nid in sorted(pre_scores.items(), key=lambda x: -x[1])[:top_k])



        scaled = [(max(pct, 55.0) if note['id'] in top_pre_ids else pct, note) for pct, note in scaled]



        scaled.sort(reverse=True, key=lambda x: x[0])



        _log_rerank_score_scale("in_process", min_s, max_s, span, normalized, scaled)



        # Rest notes weren't reranked; no _display_relevance (UI will use score)



        rest = [(0.0, note) for _, note in scored_notes[top_k:]]



        return scaled + rest, True



    except Exception as e:



        log_debug(f"Cross-encoder re-ranking failed: {e}")



        return scored_notes, False











# --- Rerank And Keyword Workers ---

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











# --- Answer Generation Workers ---

class AskAIWorker(QThread):



    """Run ask_ai in a background thread so the main thread stays responsive (no 'Not Responding')."""



    success_signal = pyqtSignal(object, object)  # (answer, relevant_indices)



    error_signal = pyqtSignal(str)



    chunk_signal = pyqtSignal(str)







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



                self._query, self._context_notes, self._context, self._config, self.chunk_signal.emit



            )



            self.success_signal.emit(answer, relevant_indices)



        except Exception as e:



            log_debug(f"AskAIWorker error: {e}")



            self.error_signal.emit(str(e))


class DirectAIWorker(QThread):
    """Run a direct no-retrieval AI answer in a background thread."""

    success_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    chunk_signal = pyqtSignal(str)
    warning_signal = pyqtSignal(str)

    def __init__(self, dialog, query, chat_history, config, image_payloads=None, allow_image_fallback=True):
        super().__init__()
        self._dialog = dialog
        self._query = query
        self._chat_history = chat_history
        self._config = config
        self._image_payloads = list(image_payloads or [])
        self._allow_image_fallback = bool(allow_image_fallback)

    def _is_image_support_error(self, error):
        return is_image_support_error(error)

    def run(self):
        try:
            try:
                answer = self._dialog.ask_ai_direct(
                    self._query,
                    self._chat_history,
                    self._config,
                    self.chunk_signal.emit,
                    image_payloads=self._image_payloads,
                )
            except Exception as image_error:
                if not self._image_payloads or not self._is_image_support_error(image_error):
                    raise
                if not self._allow_image_fallback:
                    raise Exception(IMAGE_SUPPORT_ERROR)
                self.warning_signal.emit(
                    "The selected model does not support image input, so Ask AI retried using the current note text only."
                )
                answer = self._dialog.ask_ai_direct(
                    self._query,
                    self._chat_history,
                    self._config,
                    self.chunk_signal.emit,
                    image_payloads=[],
                )
            self.success_signal.emit(answer)
        except Exception as e:
            log_debug(f"DirectAIWorker error: {e}")
            self.error_signal.emit(str(e))
