import datetime

from ..utils import log_debug
from ..utils.search_analytics_log import make_answer_payload, rank_deltas


class SearchAnswerDisplayMixin:
    def _display_answer_and_notes_after_rerank(self, answer, config, used_history, notes):



        """Display formatted answer, filter notes, and update status (shared after rerank or when no rerank)."""



        log_debug("Displaying answer and filtering notes...")

        self._stop_estimated_progress_timer()

        self._hide_busy_progress()



        formatted_answer = self.format_answer(answer)
        self._sources_rank_mode = False



        self._last_formatted_answer = formatted_answer



        if hasattr(self, "_set_chat_transient"):
            self._set_chat_transient(None)
        src = self._get_answer_source_text(config)
        if hasattr(self, "_append_assistant_chat_message"):
            self._append_assistant_chat_message(
                answer,
                mode="notes",
                context_note_ids=list(getattr(self, "_context_note_ids", []) or []),
                cited_note_ids=list(getattr(self, "_cited_note_ids", set()) or []),
                source_text=src,
            )
        else:
            self.answer_box.setHtml(formatted_answer)



        for attr in ('copy_answer_btn',):



            if hasattr(self, attr):



                getattr(self, attr).setEnabled(True)



        self.filter_and_display_notes()

        final_results = self._analytics_results(getattr(self, "all_scored_notes", []), limit=100)
        relevance_threshold = {}
        if hasattr(self, "_analytics_relevance_threshold_data"):
            relevance_threshold = self._analytics_relevance_threshold_data(getattr(self, "all_scored_notes", []))
        context_note_ids = list(getattr(self, "_context_note_ids", []) or [])
        answer_payload = make_answer_payload(
            answer,
            context_note_ids=context_note_ids,
            final_results=final_results,
            mode=self._analytics_mode(),
        )
        cited_refs = sorted(getattr(self, "_cited_refs", set()) or [])
        if cited_refs and not answer_payload.get("relevant_notes_refs"):
            answer_payload["relevant_notes_refs"] = cited_refs

        self._analytics_stage(
            "final_display",
            used_history=bool(used_history),
            displayed_rows=self.results_list.rowCount() if hasattr(self, "results_list") else None,
            threshold_pct=relevance_threshold.get("threshold_percent"),
            threshold_source=relevance_threshold.get("threshold_source"),
            show_all_override=relevance_threshold.get("show_all_override"),
            visible=relevance_threshold.get("final_visible_count"),
            hidden=relevance_threshold.get("hidden_count"),
            final_results=final_results,
            relevance_threshold=relevance_threshold,
            answer=answer_payload,
            cited_note_ids=list(getattr(self, "_cited_note_ids", set()) or []),
            context_note_ids=context_note_ids,
        )

        self._write_search_analytics_report(
            completed_at=datetime.datetime.now().isoformat(),
            final_status="answer_generated",
            used_history=bool(used_history),
            final_answer_length=len(answer or ""),
            answer=answer_payload,
            final_results=final_results,
            relevance_threshold=relevance_threshold,
            rank_deltas=rank_deltas(
                getattr(self, "_analytics_pre_rerank_results", []),
                self._analytics_results(getattr(self, "all_scored_notes", []), limit=100),
                final_results,
            ),
            cited_note_ids=list(getattr(self, "_cited_note_ids", set()) or []),
            context_note_ids=context_note_ids,
        )



        threshold = self.sensitivity_slider.value() if getattr(self, 'sensitivity_slider', None) else 0



        cache_indicator = " (\U0001F4DA from cache)" if used_history else ""



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
        if hasattr(self, "_set_search_chat_busy"):
            self._set_search_chat_busy(False)
        if hasattr(self, "_notify_long_job_done"):
            context_count = len(context_note_ids or [])
            cited_count = len(getattr(self, "_cited_note_ids", set()) or [])
            detail = f"Answer ready from {context_count} sources"
            if cited_count:
                detail += f" ({cited_count} cited)"
            if used_history:
                detail += " from cache"
            self._notify_long_job_done("Ask Notes complete", detail)



        log_debug("Search completed successfully")


    def _finish_anthropic_stream_display(self, answer, config):



        """Format answer, update table, and set status (streaming path)."""



        formatted_answer = self.format_answer(answer)
        self._sources_rank_mode = False



        self._last_formatted_answer = formatted_answer



        if hasattr(self, "_set_chat_transient"):
            self._set_chat_transient(None)
        src = self._get_answer_source_text(config)
        if hasattr(self, "_append_assistant_chat_message"):
            self._append_assistant_chat_message(
                answer,
                mode="notes",
                context_note_ids=list(getattr(self, "_context_note_ids", []) or []),
                cited_note_ids=list(getattr(self, "_cited_note_ids", set()) or []),
                source_text=src,
            )
        else:
            self.answer_box.setHtml(formatted_answer)



        for attr in ('copy_answer_btn',):



            if hasattr(self, attr):



                getattr(self, attr).setEnabled(True)



        self.filter_and_display_notes()



        if self.all_scored_notes:



            threshold = self.sensitivity_slider.value() if getattr(self, 'sensitivity_slider', None) else 0



            max_score = self.all_scored_notes[0][0]



            min_score = (threshold / 100.0) * max_score if max_score > 0 else 0



            effective_pct = round(100 * min_score / max_score) if (max_score > 0 and threshold > 0) else None



            sensitivity_text = f" (score \xe2\u2030\xa5 {effective_pct}%)" if effective_pct is not None else " (sensitivity filter)"



            filtered_count = getattr(self, "_last_visible_result_count", None)
            if filtered_count is None:
                filtered_count = self.results_list.rowCount() if hasattr(self, "results_list") else 0



            total_in_result = getattr(self, "_last_total_result_count", None)
            if total_in_result is None:
                total_in_result = len(self.all_scored_notes)



            threshold = getattr(self, "_effective_relevance_threshold_percent", 65)
            mode_suffix = f" | Threshold: {threshold}%"
            budget_message = self._dynamic_budget_message() if hasattr(self, "_dynamic_budget_message") else ""
            budget_suffix = f" | {budget_message}" if budget_message else ""



            self.status_label.setText(



                f"Showing {filtered_count} of {total_in_result}{sensitivity_text}{mode_suffix}{budget_suffix} "



                f"| Answer from: {self._get_answer_source_text(config) or 'Anthropic'}"



            )



        else:



            self.status_label.setText("Answer from: Anthropic (streaming)")

        if hasattr(self, "_notify_long_job_done"):
            context_count = len(getattr(self, "_context_note_ids", []) or [])
            self._notify_long_job_done("Ask Notes complete", f"Streaming answer ready from {context_count} sources")
