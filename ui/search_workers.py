"""Search worker threads and standalone rerank helpers used by the search dialog."""

# ============================================================================
# Imports
# ============================================================================

import json
import os
import urllib.request

from aqt.qt import QThread, pyqtSignal

from ..core.compat import _ensure_stderr_patched, _patch_colorama_early
from ..core.engine import get_embedding_for_query, get_embeddings_batch, load_embedding
from ..core.errors import _is_embedding_dimension_mismatch
from ..utils.log import log_debug
from ..utils.text import semantic_chunk_text


# ============================================================================
# Search Worker Compatibility And Standalone Search Helpers
# ============================================================================

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











# --- Chunking And Reranking Helpers ---

_semantic_chunk_text = semantic_chunk_text











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



            self.progress_signal.emit(100, "Re-ranking by relevance... (done)")



            self.finished_signal.emit(new_scores)



        except Exception as e:



            log_debug(f"RelevanceRerankWorker error: {e}")



            self.finished_signal.emit(None)











# --- Answer Generation Workers ---

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
