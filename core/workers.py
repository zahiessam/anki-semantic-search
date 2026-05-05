# ============================================================================
# Imports
# ============================================================================

import time
import hashlib
import datetime
import numpy as np
import json
import urllib.request
import subprocess
import os
import sys
from aqt.qt import QThread, pyqtSignal, QApplication
from aqt import mw

from .engine import (
    get_embedding_for_query,
    get_embeddings_batch,
    save_embedding,
    load_embedding,
    load_embedding_exact,
    load_embedding_key_index,
    flush_embedding_batch,
    _build_deck_query,
    get_embedding_engine_id,
    make_embedding_scope_id,
    save_checkpoint,
    extract_keywords_improved,
    compute_tfidf_scores,
    aggregate_scored_notes_by_note_id
)
from .errors import _is_embedding_dimension_mismatch
from ..utils.config import load_config, get_config_value
from ..utils.paths import get_checkpoint_path, get_embeddings_db_path
from ..utils.log import log_debug
from ..utils.embeddings_status import format_partial_failure_progress

# ============================================================================
# Search And Embedding Workers
# ============================================================================

class EmbeddingSearchWorker(QThread):
    """Worker thread for embedding search (prevents UI freezing)."""
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

            query_embedding = np.array(embedding_list)
            total = len(self.notes)
            scored_notes = []
            progress_interval = max(50, total // 40)

            for idx, note in enumerate(self.notes):
                if self.isInterruptionRequested():
                    return

                if idx % progress_interval == 0 or idx == total - 1:
                    pct = int(100 * (idx + 1) / total) if total else 0
                    self.progress_signal.emit(idx + 1, total, f"Embedding search: {idx + 1}/{total} ({pct}%)")

                content_hash = note.get('content_hash')
                if content_hash is None:
                    content_hash = hashlib.md5(note['content'].encode()).hexdigest()

                note_embedding = load_embedding(note.get('id'), content_hash, db_path=self.db_path)
                if note_embedding is not None:
                    emb = np.array(note_embedding)
                    if len(emb) != len(query_embedding):
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
                self.error_signal.emit("dimension_mismatch")
            else:
                self.error_signal.emit(str(e))

# ============================================================================
# Rerank Availability And Embedding Index Workers
# ============================================================================

class RerankCheckWorker(QThread):
    """Worker thread for checking sentence-transformers availability."""
    finished_signal = pyqtSignal(bool)

    def __init__(self, dialog, python_path=None):
        super().__init__()
        self.python_path = python_path

    def run(self):
        try:
            if self.python_path:
                p = self.python_path.strip()
                if os.path.isdir(p):
                    exe = os.path.join(p, "python.exe") if os.name == 'nt' else os.path.join(p, "python")
                    p = exe if os.path.isfile(exe) else p
                result = subprocess.run([p, "-c", "from sentence_transformers import CrossEncoder; print('ok')"], capture_output=True, text=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                self.finished_signal.emit(result.returncode == 0 and 'ok' in result.stdout)
                return
            result = subprocess.run([sys.executable, "-c", "from sentence_transformers import CrossEncoder; print('ok')"], capture_output=True, text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
            self.finished_signal.emit(result.returncode == 0 and 'ok' in result.stdout)
        except:
            self.finished_signal.emit(False)

class EmbeddingWorker(QThread):
    """Worker thread for embedding generation (prevents UI blocking)"""
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    progress_detail = pyqtSignal(object)
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal(int, int, int, int, int)
    error_signal = pyqtSignal(str)

    def __init__(self, ntf, note_count, checkpoint, resume_available, config=None):
        super().__init__()
        self.ntf = ntf
        self.note_count = note_count
        self.checkpoint = checkpoint
        self.resume_available = resume_available
        self._is_paused = False
        # explanation: use dialog-provided embedding settings when available.
        self.config = config or load_config()
        config = self.config
        self.progress_update_interval = get_config_value(config, 'progress_update_interval', 10)
        self.last_progress_update = 0
        self._scan_ui_update_interval = max(100, int(self.progress_update_interval or 10))
        self._scan_ui_yield_seconds = 0.02
        self._last_scan_debug_bucket = -1
        self._skip_debug_counts = {}
        sc = config.get("search_config") or {}
        engine = sc.get("embedding_engine") or "voyage"
        try:
            self.effective_batch_size = int(sc.get("ollama_batch_size" if engine == "ollama" else "voyage_batch_size") or 64)
        except: self.effective_batch_size = 64
        self._dynamic_batch_size = self.effective_batch_size
        self._use_dynamic_batch = bool(sc.get("use_dynamic_batch_size", True))
        self._start_time = None
        self._last_checkpoint_count = 0
        self.scope_id = make_embedding_scope_id(self.ntf)

    def _format_eta(self, checked):
        if not self._start_time or checked <= 0:
            return "calculating..."
        elapsed = max(0.1, time.time() - self._start_time)
        remaining = max(0, self.note_count - checked)
        seconds = int(remaining * (elapsed / checked))
        if seconds < 60:
            return f"{seconds}s"
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {seconds}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m"

    def _emit_detail(self, checked, processed, refreshed, skipped, errors, pending_count=0):
        self.progress_detail.emit({
            "checked": checked,
            "total": self.note_count,
            "processed": processed,
            "refreshed": refreshed,
            "skipped": skipped,
            "errors": errors,
            "batch_size": self._dynamic_batch_size,
            "pending_count": pending_count,
            "eta": self._format_eta(checked),
        })

    def _emit_scan_progress(self, checked, processed, refreshed, skipped, errors, pending_count=0, force=False):
        if not force and checked - self.last_progress_update < self._scan_ui_update_interval:
            return
        self.progress_update.emit(checked)
        self.last_progress_update = checked
        self._emit_detail(checked, processed, refreshed, skipped, errors, pending_count=pending_count)
        self.status_update.emit(f"Checking notes for embedding updates... {checked:,}/{self.note_count:,}")
        if self.note_count:
            bucket = checked // 5000
            if bucket != self._last_scan_debug_bucket:
                self._last_scan_debug_bucket = bucket
                log_debug(
                    f"Embedding scan progress: checked={checked}/{self.note_count}, "
                    f"new={processed}, updated={refreshed}, current={skipped}, pending={pending_count}"
                )
        if not force:
            time.sleep(self._scan_ui_yield_seconds)

    def _debug_skip_reason(self, reason):
        count = self._skip_debug_counts.get(reason, 0) + 1
        self._skip_debug_counts[reason] = count
        if count in (1, 10, 100, 1000, 5000, 10000, 20000):
            log_debug(f"Embedding scan skip reason: {reason} count={count}")

    def _save_progress_checkpoint(self, processed_note_ids, errors, engine_id, checked, processed, skipped, force=False):
        if not processed_note_ids:
            return False
        if not force and len(processed_note_ids) - self._last_checkpoint_count < 500:
            return False
        if save_checkpoint(
            processed_note_ids,
            self.note_count,
            errors=errors,
            engine_id=engine_id,
            scope_id=self.scope_id,
        ):
            self._last_checkpoint_count = len(processed_note_ids)
            return True
        return False

    def run(self):
        try:
            if self.isInterruptionRequested(): return
            log_debug("EmbeddingWorker debug: run started")
            try:
                self.status_update.emit("Testing connection...")
                test_emb = get_embedding_for_query("Test connection", config=self.config)
                self.log_message.emit("\u2705 Connection verified")
                log_debug("EmbeddingWorker debug: connection test passed")
            except Exception as e: raise Exception(f"API test failed: {str(e)[:100]}")
            expected_dim = len(test_emb) if test_emb else None
            config = self.config
            engine_id = get_embedding_engine_id(config)
            self.log_message.emit(
                f"Embedding engine: {engine_id}"
            )
            self.status_update.emit("Loading existing embedding index...")
            index_started = time.time()
            existing_embedding_keys, existing_note_ids_any_engine, existing_note_timestamps = load_embedding_key_index(
                engine_id=engine_id,
                db_path=get_embeddings_db_path(),
                include_note_timestamps=True,
            )
            log_debug(
                f"EmbeddingWorker debug: index loaded in {time.time() - index_started:.2f}s; "
                f"keys={len(existing_embedding_keys)}, note_timestamps={len(existing_note_timestamps)}"
            )
            if existing_embedding_keys:
                self.log_message.emit(
                    f"Fast skip index loaded: {len(existing_embedding_keys):,} existing embeddings"
                )
            self._start_time = time.time()
            deck_q = _build_deck_query(self.ntf.get('enabled_decks'))
            note_lookup_started = time.time()
            note_ids = mw.col.find_notes(deck_q) if deck_q else mw.col.find_notes("")
            log_debug(
                f"EmbeddingWorker debug: find_notes returned {len(note_ids):,} ids in "
                f"{time.time() - note_lookup_started:.2f}s; deck_q={deck_q!r}"
            )
            enabled_raw = self.ntf.get('enabled_note_types')
            enabled_set = None if enabled_raw is None else set(enabled_raw)
            search_all = bool(self.ntf.get('search_all_fields', False))
            ntf_fields = self.ntf.get('note_type_fields') or {}
            use_first = bool(self.ntf.get('use_first_field_fallback', True))
            processed_note_ids = set(self.checkpoint.get('processed_note_ids', [])) if self.checkpoint and self.resume_available else set()
            processed = 0
            refreshed = 0
            self._last_checkpoint_count = len(processed_note_ids)
            skipped = 0
            errors = 0
            pending_notes = []
            failed_notes = {}
            checked = len(processed_note_ids)
            if checked:
                self.log_message.emit(f"Resuming from checkpoint: {checked:,}/{self.note_count:,} notes already processed")
            self.log_message.emit(
                f"Batch size: {self._dynamic_batch_size} notes/call"
                + (" (dynamic)" if self._use_dynamic_batch else "")
            )
            self.status_update.emit("Checking notes for embedding updates...")
            self.log_message.emit(f"Scanning {self.note_count:,} eligible notes for changes...")
            self._emit_scan_progress(checked, processed, refreshed, skipped, errors, force=True)
            model_map = {}
            for m in mw.col.models.all():
                if enabled_set is not None and m['name'] not in enabled_set: continue
                if search_all: indices = list(range(len(m['flds'])))
                else:
                    wanted = set(f.lower() for f in (ntf_fields.get(m['name']) or []))
                    if not wanted and use_first and m['flds']: wanted = {m['flds'][0]['name'].lower()}
                    indices = [i for i, f in enumerate(m['flds']) if f['name'].lower() in wanted]
                if indices: model_map[m['id']] = indices
            log_debug(
                f"EmbeddingWorker debug: model_map has {len(model_map)} models; "
                f"enabled_note_types={sorted(enabled_set) if enabled_set is not None else 'ALL'}"
            )
            note_id_list = list(note_ids)
            chunk_size = 500
            for start in range(0, len(note_id_list), chunk_size):
                if self.isInterruptionRequested(): break
                chunk = note_id_list[start:start + chunk_size]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                chunk_started = time.time()
                log_debug(f"EmbeddingWorker debug: loading note chunk {start + 1}-{start + len(chunk)}")
                rows = mw.col.db.execute(
                    f"select id, mid, mod, flds from notes where id in ({placeholders})",
                    *chunk,
                )
                log_debug(
                    f"EmbeddingWorker debug: chunk {start + 1}-{start + len(chunk)} returned "
                    f"{len(rows)} rows in {time.time() - chunk_started:.2f}s"
                )
                for nid, mid, note_mod, flds_str in rows:
                    if self.isInterruptionRequested(): break
                    while self._is_paused: time.sleep(0.1)
                    try:
                        if nid in processed_note_ids:
                            self._debug_skip_reason("checkpoint_processed")
                            continue
                        indices = model_map.get(mid)
                        if not indices:
                            skipped += 1
                            checked += 1
                            self._debug_skip_reason("no_selected_model_or_fields")
                            self._emit_scan_progress(checked, processed, refreshed, skipped, errors, pending_count=len(pending_notes))
                            continue
                        fields = flds_str.split("\x1f")
                        content = " ".join(fields[i].strip() for i in indices if i < len(fields) and fields[i].strip())
                        if not content:
                            skipped += 1
                            checked += 1
                            self._debug_skip_reason("empty_selected_content")
                            self._emit_scan_progress(checked, processed, refreshed, skipped, errors, pending_count=len(pending_notes))
                            continue
                        ch = hashlib.md5(content.encode()).hexdigest()
                        existing_key = (nid, ch) in existing_embedding_keys
                        existing = None
                        if not existing_key:
                            existing = load_embedding_exact(nid, ch, engine_id=engine_id)
                            existing_key = existing is not None and (expected_dim is None or len(existing) == expected_dim)
                        if not existing_key and nid in existing_note_timestamps:
                            try:
                                embedded_at = datetime.datetime.fromisoformat(existing_note_timestamps[nid])
                                note_modified_at = datetime.datetime.fromtimestamp(int(note_mod))
                                if embedded_at >= note_modified_at:
                                    existing_key = True
                            except Exception:
                                pass
                        if existing_key:
                            if nid not in processed_note_ids:
                                processed_note_ids.add(nid)
                            skipped += 1
                            checked += 1
                            self._emit_scan_progress(checked, processed, refreshed, skipped, errors, pending_count=len(pending_notes))
                            self._save_progress_checkpoint(processed_note_ids, errors, engine_id, checked, processed, skipped)
                            continue
                        pending_notes.append((nid, ch, content, nid in existing_note_timestamps))
                        checked += 1
                        self._emit_scan_progress(checked, processed, refreshed, skipped, errors, pending_count=len(pending_notes))
                        if len(pending_notes) >= self._dynamic_batch_size:
                            t0 = time.time()
                            batch_count = len(pending_notes)
                            self.status_update.emit(f"Embedding {batch_count} changed/missing notes...")
                            self.log_message.emit(f"Embedding batch: {batch_count:,} changed/missing notes")
                            embs = get_embeddings_batch([n[2] for n in pending_notes], config=config)
                            if embs and len(embs) == len(pending_notes):
                                for (rnid, rch, _, had_embedding), remb in zip(pending_notes, embs):
                                    save_embedding(rnid, rch, remb, batch_mode=True, engine_id=engine_id)
                                    existing_embedding_keys.add((rnid, rch))
                                    existing_note_ids_any_engine.add(rnid)
                                    existing_note_timestamps[rnid] = datetime.datetime.now().isoformat()
                                    processed_note_ids.add(rnid)
                                    if had_embedding:
                                        refreshed += 1
                                    else:
                                        processed += 1
                                flush_embedding_batch()
                                self._emit_scan_progress(checked, processed, refreshed, skipped, errors, force=True)
                                self._save_progress_checkpoint(processed_note_ids, errors, engine_id, checked, processed, skipped)
                                if self._use_dynamic_batch:
                                    dur = time.time() - t0
                                    old_batch_size = self._dynamic_batch_size
                                    if dur > 15: self._dynamic_batch_size = max(8, int(self._dynamic_batch_size * 0.6))
                                    elif dur < 6: self._dynamic_batch_size = min(256, int(self._dynamic_batch_size * 1.2))
                                    if self._dynamic_batch_size != old_batch_size:
                                        self.log_message.emit(f"Batch size adjusted: {old_batch_size} -> {self._dynamic_batch_size} notes/call")
                            pending_notes = []
                            if not self.isInterruptionRequested():
                                self.status_update.emit(f"Checking notes for embedding updates... {checked:,}/{self.note_count:,}")
                    except Exception as e:
                        errors += 1
                        self._debug_skip_reason(f"exception:{type(e).__name__}")
                        log_debug(f"EmbeddingWorker debug: note {nid} scan exception: {e}")
            if pending_notes and not self.isInterruptionRequested():
                self.status_update.emit(f"Embedding final batch of {len(pending_notes)} changed/missing notes...")
                self.log_message.emit(f"Embedding final batch: {len(pending_notes):,} changed/missing notes")
                embs = get_embeddings_batch([n[2] for n in pending_notes], config=config)
                if embs and len(embs) == len(pending_notes):
                    for (rnid, rch, _, had_embedding), remb in zip(pending_notes, embs):
                        save_embedding(rnid, rch, remb, batch_mode=True, engine_id=engine_id)
                        existing_embedding_keys.add((rnid, rch))
                        existing_note_ids_any_engine.add(rnid)
                        existing_note_timestamps[rnid] = datetime.datetime.now().isoformat()
                        processed_note_ids.add(rnid)
                        if had_embedding:
                            refreshed += 1
                        else:
                            processed += 1
                    flush_embedding_batch()
                    self._emit_scan_progress(checked, processed, refreshed, skipped, errors, force=True)
                    self._save_progress_checkpoint(processed_note_ids, errors, engine_id, checked, processed, skipped, force=True)
            if checked != self.last_progress_update:
                self.progress_update.emit(checked)
                self._emit_detail(checked, processed, refreshed, skipped, errors)
            self._save_progress_checkpoint(processed_note_ids, errors, engine_id, checked, processed, skipped, force=True)
            self.log_message.emit(
                f"Scan complete: {checked:,} checked, {processed:,} new, {refreshed:,} updated, {skipped:,} already current"
            )
            self.finished_signal.emit(processed, errors, skipped, refreshed, 0)
        except Exception as e: self.error_signal.emit(str(e))


# ============================================================================
# Keyword Filtering And Rerank Workers
# ============================================================================

class KeywordFilterWorker(QThread):
    """Worker thread for keyword_filter so search doesn't freeze the main thread."""

    finished_signal = pyqtSignal(object)

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
    """Worker thread for keyword_filter_continue so combining results doesn't freeze the UI."""

    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(object)

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
                self._state,
                self._embedding_results,
                progress_callback=progress_callback,
            )
            self.finished_signal.emit(result)
        except Exception as e:
            log_debug(f"KeywordFilterContinueWorker error: {e}")
            self.finished_signal.emit(None)


class RerankWorker(QThread):
    """Worker thread for cross-encoder reranking so the UI stays responsive."""

    finished_signal = pyqtSignal(object, bool)

    def __init__(self, query, scored_notes, top_k, search_config):
        super().__init__()
        self.query = query
        self.scored_notes = scored_notes
        self.top_k = top_k
        self.search_config = search_config

    def run(self):
        try:
            from ..ui.dialogs import _do_rerank

            scored_notes, success = _do_rerank(
                self.query,
                self.scored_notes,
                self.top_k,
                self.search_config,
            )
            self.finished_signal.emit(scored_notes, success)
        except Exception as e:
            log_debug(f"RerankWorker error: {e}")
            self.finished_signal.emit(self.scored_notes, False)


class RelevanceRerankWorker(QThread):
    """Worker for re-ranking notes by similarity to the AI answer."""

    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(object)

    def __init__(self, answer_text, note_texts, all_scored_notes, config):
        super().__init__()
        self.answer_text = answer_text
        self.note_texts = note_texts
        self.all_scored_notes = all_scored_notes
        self.config = config

    def run(self):
        try:
            self.progress_signal.emit(5, "Re-ranking by relevance... (embedding answer)")
            answer_emb = get_embedding_for_query(self.answer_text, self.config)
            if not answer_emb:
                self.finished_signal.emit(None)
                return

            self.progress_signal.emit(20, "Re-ranking by relevance... (embedding notes)")
            note_embs = get_embeddings_batch(self.note_texts, input_type="document", config=self.config)
            if not note_embs or len(note_embs) != len(self.all_scored_notes):
                self.finished_signal.emit(None)
                return

            self.progress_signal.emit(70, "Re-ranking by relevance... (scoring)")
            answer_vec = np.array(answer_emb, dtype=float)
            norm_a = max(np.linalg.norm(answer_vec), 1e-9)
            new_scores = []

            for i, (_, note) in enumerate(self.all_scored_notes):
                ne = np.array(note_embs[i], dtype=float)
                norm_n = max(np.linalg.norm(ne), 1e-9)
                sim = float(np.dot(answer_vec, ne) / (norm_a * norm_n))
                pct = max(0, min(100, round((sim + 1) * 50)))
                note["_display_relevance"] = pct
                new_scores.append((float(pct), note))

            new_scores.sort(reverse=True, key=lambda x: x[0])
            if new_scores:
                max_pct = new_scores[0][0]
                if max_pct > 0:
                    for score, note in new_scores:
                        note["_display_relevance"] = max(
                            0, min(100, round(100 * (note.get("_display_relevance") or 0) / max_pct))
                        )
                    new_scores = [
                        (100.0 if i == 0 else (note.get("_display_relevance") or 0), note)
                        for i, (_, note) in enumerate(new_scores)
                    ]
                    new_scores.sort(reverse=True, key=lambda x: x[0])

            self.progress_signal.emit(100, "Re-ranking by relevance... (done)")
            self.finished_signal.emit(new_scores)
        except Exception as e:
            log_debug(f"RelevanceRerankWorker error: {e}")
            self.finished_signal.emit(None)


# ============================================================================
# Answer Generation Workers
# ============================================================================

class AskAIWorker(QThread):
    """Run ask_ai in a background thread so the main thread stays responsive."""

    success_signal = pyqtSignal(object, object)
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
                self._query,
                self._context_notes,
                self._context,
                self._config,
            )
            self.success_signal.emit(answer, relevant_indices)
        except Exception as e:
            log_debug(f"AskAIWorker error: {e}")
            self.error_signal.emit(str(e))


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
