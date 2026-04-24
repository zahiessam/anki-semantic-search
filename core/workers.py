import time
import hashlib
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
    flush_embedding_batch,
    _build_deck_query,
    get_embedding_engine_id,
    _is_embedding_dimension_mismatch,
    save_checkpoint,
    extract_keywords_improved,
    compute_tfidf_scores,
    aggregate_scored_notes_by_note_id
)
from ..utils.config import load_config, get_config_value
from ..utils.paths import get_checkpoint_path, get_embeddings_db_path
from ..utils.log import log_debug
from ..utils.embeddings_status import format_partial_failure_progress

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
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal(int, int, int, int)
    error_signal = pyqtSignal(str)

    def __init__(self, ntf, note_count, checkpoint, resume_available):
        super().__init__()
        self.ntf = ntf
        self.note_count = note_count
        self.checkpoint = checkpoint
        self.resume_available = resume_available
        self._is_paused = False
        config = load_config()
        self.progress_update_interval = get_config_value(config, 'progress_update_interval', 10)
        self.last_progress_update = 0
        sc = config.get("search_config") or {}
        engine = sc.get("embedding_engine") or "voyage"
        try:
            self.effective_batch_size = int(sc.get("ollama_batch_size" if engine == "ollama" else "voyage_batch_size") or 64)
        except: self.effective_batch_size = 64
        self._dynamic_batch_size = self.effective_batch_size
        self._use_dynamic_batch = bool(sc.get("use_dynamic_batch_size", True))
        self._start_time = None

    def run(self):
        try:
            if self.isInterruptionRequested(): return
            try:
                self.status_update.emit("Testing connection...")
                test_emb = get_embedding_for_query("Test connection")
                self.log_message.emit("\u2705 Connection verified")
            except Exception as e: raise Exception(f"API test failed: {str(e)[:100]}")
            expected_dim = len(test_emb) if test_emb else None
            config = load_config()
            engine_id = get_embedding_engine_id(config)
            self._start_time = time.time()
            deck_q = _build_deck_query(self.ntf.get('enabled_decks'))
            note_ids = mw.col.find_notes(deck_q) if deck_q else mw.col.find_notes("")
            enabled_set = set(self.ntf.get('enabled_note_types') or []) if self.ntf.get('enabled_note_types') else None
            search_all = bool(self.ntf.get('search_all_fields', False))
            ntf_fields = self.ntf.get('note_type_fields') or {}
            use_first = bool(self.ntf.get('use_first_field_fallback', True))
            processed_note_ids = set(self.checkpoint.get('processed_note_ids', [])) if self.checkpoint and self.resume_available else set()
            processed = len(processed_note_ids)
            skipped = 0
            errors = 0
            pending_notes = []
            failed_notes = {}
            checked = processed
            model_map = {}
            for m in mw.col.models.all():
                if enabled_set and m['name'] not in enabled_set: continue
                if search_all: indices = list(range(len(m['flds'])))
                else:
                    wanted = set(f.lower() for f in (ntf_fields.get(m['name']) or []))
                    if not wanted and use_first and m['flds']: wanted = {m['flds'][0]['name'].lower()}
                    indices = [i for i, f in enumerate(m['flds']) if f['name'].lower() in wanted]
                if indices: model_map[m['id']] = indices
            id_list = ",".join(map(str, note_ids))
            for nid, mid, flds_str in mw.col.db.execute(f"select id, mid, flds from notes where id in ({id_list})"):
                if self.isInterruptionRequested(): break
                while self._is_paused: time.sleep(0.1)
                try:
                    indices = model_map.get(mid)
                    if not indices: continue
                    fields = flds_str.split("\x1f")
                    content = " ".join(fields[i].strip() for i in indices if i < len(fields) and fields[i].strip())
                    if not content: continue
                    ch = hashlib.md5(content.encode()).hexdigest()
                    existing = load_embedding(nid, ch, engine_id=engine_id)
                    if existing is not None and (expected_dim is None or len(existing) == expected_dim):
                        if nid not in processed_note_ids:
                            processed_note_ids.add(nid)
                        skipped += 1
                        checked += 1
                        if checked - self.last_progress_update >= self.progress_update_interval:
                            self.progress_update.emit(checked); self.last_progress_update = checked
                        continue
                    pending_notes.append((nid, ch, content))
                    if len(pending_notes) >= self._dynamic_batch_size:
                        t0 = time.time()
                        embs = get_embeddings_batch([n[2] for n in pending_notes], config=config)
                        if embs and len(embs) == len(pending_notes):
                            for (rnid, rch, _), remb in zip(pending_notes, embs):
                                save_embedding(rnid, rch, remb, batch_mode=True, engine_id=engine_id)
                                processed_note_ids.add(rnid)
                                processed += 1
                                checked += 1
                            flush_embedding_batch()
                            if checked - self.last_progress_update >= self.progress_update_interval:
                                self.progress_update.emit(checked); self.last_progress_update = checked
                            if self._use_dynamic_batch:
                                dur = time.time() - t0
                                if dur > 15: self._dynamic_batch_size = max(8, int(self._dynamic_batch_size * 0.6))
                                elif dur < 6: self._dynamic_batch_size = min(256, int(self._dynamic_batch_size * 1.2))
                        pending_notes = []
                except Exception as e: errors += 1
            if pending_notes and not self.isInterruptionRequested():
                embs = get_embeddings_batch([n[2] for n in pending_notes], config=config)
                if embs and len(embs) == len(pending_notes):
                    for (rnid, rch, _), remb in zip(pending_notes, embs):
                        save_embedding(rnid, rch, remb, batch_mode=True, engine_id=engine_id)
                        processed_note_ids.add(rnid)
                        processed += 1
                        checked += 1
                    flush_embedding_batch()
            if checked != self.last_progress_update:
                self.progress_update.emit(checked)
            self.finished_signal.emit(processed, errors, skipped, 0)
        except Exception as e: self.error_signal.emit(str(e))


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
