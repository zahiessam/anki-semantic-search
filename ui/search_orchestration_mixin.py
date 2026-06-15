"""Search orchestration callbacks for AISearchDialog."""

import datetime

from aqt import mw
from aqt.qt import QApplication

from .search_workers import (
    MAX_RERANK_COUNT,
    KeywordFilterContinueWorker,
    RerankWorker,
    _do_rerank,
    get_last_embedding_match_diagnostics,
)
from ..utils import log_debug

class SearchOrchestrationMixin:
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
                if hasattr(self, "_set_search_chat_busy"):
                    self._set_search_chat_busy(False)



                return



            combine_in_background = False



            # Handle dimension-mismatch dict from _run_embedding_search_sync



            if isinstance(embedding_results, dict) and "error" in embedding_results:



                setattr(self, "_last_embedding_error", embedding_results.get("error"))



                embedding_results = embedding_results.get("embedding_results")



            else:



                setattr(self, "_last_embedding_error", None)



            match_diag = get_last_embedding_match_diagnostics()
            self._last_embedding_rows_checked = int(match_diag.get("rows") or 0)
            self._last_embedding_exact_matches = int(match_diag.get("exact_matches") or 0)
            self._last_embedding_note_id_fallback_matches = int(match_diag.get("note_id_fallback_matches") or 0)



            # Run keyword_filter_continue in a worker so the UI shows progress and does not freeze.



            if embedding_results:



                if hasattr(self, 'status_label') and self.status_label:



                    self.status_label.setText("Combining keyword + semantic matches...")

                if hasattr(self, "_set_chat_transient"):
                    self._set_chat_transient("Combining keyword + semantic matches...", mode="info")



                if hasattr(self, 'search_progress_bar') and self.search_progress_bar:



                    self.search_progress_bar.setVisible(True)



                    self.search_progress_bar.setRange(0, 100)



                    self.search_progress_bar.setValue(0)



                if hasattr(self, 'search_progress_label') and self.search_progress_label:



                    self.search_progress_label.setVisible(True)



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



            self.answer_box.setText(f"Error after embedding search:\n{str(e)}")



        finally:



            if not combine_in_background and not getattr(self, '_pending_rerank', False):



                self.search_btn.setEnabled(True)
                if hasattr(self, "_set_search_chat_busy"):
                    self._set_search_chat_busy(False)



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
            if hasattr(self, "_set_search_chat_busy"):
                self._set_search_chat_busy(False)



            return



        if isinstance(result, tuple) and result[0] == "PENDING_RERANK":



            _, notes, scored_notes, effective_method, _ = result
            scored_notes, effective_method = self._maybe_apply_agentic_pass2(
                scored_notes,
                notes,
                state.get("search_config") or {},
                effective_method,
            )



            self._pending_rerank = True



            search_config = state.get("search_config") or {}
            pre_budget = self._dynamic_note_budget(
                state.get("query", ""),
                scored_notes,
                search_config,
                (state.get("config") or {}).get("provider", "openai"),
                phase="rerank",
                pinned_count=len((getattr(self, "selected_note_ids", set()) or set()) | (getattr(self, "_pinned_note_ids", set()) or set())),
                rerank_used=False,
            )
            rerank_limit = pre_budget.get("rerank_candidate_limit", MAX_RERANK_COUNT)
            self._last_dynamic_note_budget = pre_budget



            self._rerank_continue = (notes, effective_method, search_config)
            self._analytics_pre_rerank_results = self._analytics_results(scored_notes, limit=100)

            self._analytics_stage(
                "pre_cross_encoder_candidates",
                effective_method=effective_method,
                total_candidates=len(scored_notes or []),
                top_results=self._analytics_pre_rerank_results[:50],
            )



            if hasattr(self, 'status_label') and self.status_label:



                self.status_label.setText("Reranking strongest matches...")



            if hasattr(self, "_set_chat_transient"):
                self._set_chat_transient("Reranking strongest matches...", mode="info")

            self._show_centile_progress("Reranking strongest matches...", 0)



            self._start_estimated_progress_timer(25, 5, 95)



            query = state.get("query", "")



            try:



                from aqt.operations import QueryOp



                op = QueryOp(



                    parent=mw,



                    op=lambda col: _do_rerank(query, scored_notes, rerank_limit, search_config),



                    success=lambda pair: self._on_rerank_done(pair[0], pair[1]),



                )



                op.run_in_background()



            except Exception:



                self._rerank_worker = RerankWorker(query, scored_notes, rerank_limit, search_config)



                self._rerank_worker.finished_signal.connect(self._on_rerank_done)



                self._rerank_worker.start()



            return



        scored_notes, effective_method, total_above_threshold = result
        scored_notes, effective_method = self._maybe_apply_agentic_pass2(
            scored_notes,
            notes,
            state.get("search_config") or {},
            effective_method,
        )



        self._perform_search_continue(notes, scored_notes, effective_method, total_above_threshold)



        self.search_btn.setEnabled(True)
        if hasattr(self, "_set_search_chat_busy"):
            self._set_search_chat_busy(False)


    def _on_embedding_search_error(self, error_msg):



        """Handle embedding search worker error."""



        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:



            self.search_progress_bar.setVisible(False)



        if hasattr(self, 'search_progress_label') and self.search_progress_label:



            self.search_progress_label.setVisible(False)



        self.status_label.setText("Error occurred")

        self._analytics_stage("embedding_error", error=error_msg)

        self._write_search_analytics_report(
            completed_at=datetime.datetime.now().isoformat(),
            final_status="embedding_error",
            error=error_msg,
            final_results=self._analytics_results(getattr(self, "all_scored_notes", []), limit=100),
        )



        if hasattr(self, "_append_system_chat_message"):
            self._set_chat_transient(None)
            self._append_system_chat_message(f"Embedding search failed:\n{error_msg}", kind="error")
        else:
            self.answer_box.setText(f"Embedding search failed:\n{error_msg}")



        self.search_btn.setEnabled(True)
        if hasattr(self, "_set_search_chat_busy"):
            self._set_search_chat_busy(False)
        if hasattr(self, "_notify_long_job_done"):
            self._notify_long_job_done("Ask Notes failed", "Embedding search failed", kind="error")


