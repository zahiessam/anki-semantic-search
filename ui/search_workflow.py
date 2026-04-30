"""Search workflow methods and Anthropic streaming worker.

This module preserves the historical dynamic method-copy behavior by exposing
`install_search_workflow_methods()` for `ui.dialogs` to call after AISearchDialog
and worker classes are available.
"""

# ============================================================================
# Imports
# ============================================================================

import json
import re
import urllib.request

from aqt import dialogs, mw
from aqt.qt import QApplication, QColor, QTableWidgetItem, QThread, QTimer, Qt, pyqtSignal
from aqt.utils import tooltip

from .answer_prompts import _build_anthropic_prompt_parts
from .search_workers import (
    MAX_RERANK_COUNT,
    RRF_K,
    AskAIWorker,
    EmbeddingSearchWorker,
    KeywordFilterContinueWorker,
    KeywordFilterWorker,
    RelevanceRerankWorker,
    RerankWorker,
    _do_rerank,
)
from ..core.engine import estimate_tokens, get_embedding_for_query, get_embeddings_batch
from ..core.errors import _is_embedding_dimension_mismatch
from ..utils import (
    format_dimension_mismatch_hint,
    get_embeddings_db_path,
    load_config,
    load_search_history,
    log_debug,
    save_search_history,
)


# ============================================================================
# Search Workflow Methods Copied Onto AISearchDialog
# ============================================================================

class AnthropicStreamWorker(QThread):



    """Worker thread for Anthropic streaming. Emits text chunks for real-time UI updates."""



    chunk_signal = pyqtSignal(str)



    done_signal = pyqtSignal(str)



    error_signal = pyqtSignal(str)







    # --- Anthropic Streaming Thread Lifecycle ---

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







    # --- Copied Search Progress Handlers ---

    def _get_answer_source_text(self, config):



        """Return a short hint: where the answer came from (online API name or local model)."""



        if not config:



            return ""



        provider = config.get("provider", "openai")



        if provider == "ollama":



            sc = config.get("search_config") or {}



            model = (sc.get("ollama_chat_model") or "llama3.2").strip()



            return f"Ollama (local) \u2014 {model}"

        if provider in ("local_openai", "local_server"):
            sc = config.get("search_config") or {}
            model = (
                sc.get("local_llm_model")
                or config.get("local_llm_model")
                or "local-model"
            ).strip()
            base_url = (
                sc.get("local_llm_url")
                or config.get("local_llm_url")
                or config.get("api_url")
                or ""
            ).strip()
            label = "Local Server (Ollama, LM Studio, Jan)"
            if "11434" in base_url:
                label = "Ollama (local)"
            elif "1234" in base_url:
                label = "LM Studio/Jan (local)"
            return f"{label} \u2014 {model}"



        names = {



            "anthropic": "Anthropic (Claude)",



            "openai": "OpenAI (GPT)",



            "google": "Google (Gemini)",



            "openrouter": "OpenRouter",



            "custom": "Custom / OpenAI-compatible",



        }



        name = names.get(provider, "API")



        model = self.get_best_model(provider)



        return f"{name} \u2014 {model}"







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



        """Show 0\xe2\u20ac\u201c100% progress bar and label. Use for estimated or real progress during long operations."""



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



            self.answer_box.setText(f"Error after embedding search:\n{str(e)}")



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



                self.status_label.setText("Re-ranking results... (may take 10-30 s)")



            self._show_centile_progress("Re-ranking...", 0)



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



        self.answer_box.setText(f"Embedding search failed:\n{error_msg}")



        self.search_btn.setEnabled(True)







    # --- Copied Search Continuation Flow ---

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



            label_text = f"Results from: {effective_method} \u00b7 {mode_display} \u00b7 Embeddings: {engine_display}"



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



                    hint = "embedding unavailable \u2014 run Create/Update Embeddings in Settings \u2192 Search & Embeddings, or check API key"



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



            self.status_label.setText("\U0001F4DA Loading from cache... (saved AI API call)")



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



            relevant_note_ids = set(history_result.get('relevant_note_ids', []))



            self._cited_note_ids = relevant_note_ids



            self._cited_refs = {
                idx + 1
                for idx, note_id in enumerate(self._context_note_ids or [])
                if note_id in relevant_note_ids
            }



            if answer and str(answer).strip():



                log_debug(
                    "Displaying cached answer "
                    f"(answer length: {len(answer)}, cited notes: {len(relevant_note_ids)})"
                )



                self._display_answer_and_notes_after_rerank(answer, config, used_history, notes)



                return



            log_debug("Cached search result had an empty answer; showing matching notes only", is_error=True)



            self.answer_box.setPlainText("Cached search result had no saved answer. Run a fresh search after clearing history.")



            self.filter_and_display_notes()



            return



        else:



            # Cap notes/chunks sent to the AI (avoids rate limits and token overflow; chunked results can be huge)



            search_config = config.get('search_config') or {}



            provider_for_context = config.get('provider', 'openai')



            local_context_plan = self._local_context_usage_plan(
                query, len(relevant_notes), provider_for_context, search_config
            )



            if local_context_plan:



                max_context = max(3, min(local_context_plan['max_notes'], search_config.get('max_results', 12)))



            else:



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



            # Per-note limit: 0 = full content; >0 = truncate (Settings \u2192 Search & Embeddings \u2192 Max chars per note)



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



            context_lines = [_context_line(i, n) for i, n in enumerate(context_notes)]



            if local_context_plan:



                context_lines = self._fit_context_lines_to_token_budget(
                    context_lines, local_context_plan['context_token_budget']
                )



                if len(context_lines) < len(context_notes):



                    context_notes = context_notes[:len(context_lines)]



                    context_note_ids = context_note_ids[:len(context_lines)]



                    self._context_note_id_and_chunk = self._context_note_id_and_chunk[:len(context_lines)]



                    if self._display_scored_notes:



                        self._display_scored_notes = self._display_scored_notes[:len(context_lines)]



                log_debug(
                    "Local context plan: "
                    f"{local_context_plan['mode']}, notes={len(context_notes)}, "
                    f"context_tokens~{estimate_tokens(chr(10).join(context_lines))}, "
                    f"answer_tokens={local_context_plan['max_output_tokens']}"
                )



            context = "\n\n".join(context_lines)



            n_notes = len(context_notes)



            self.status_label.setText(f"Asking AI... (sending top {n_notes} notes, 10-30 s)")



            self._show_centile_progress(f"Asking AI... ({n_notes} notes)", 0)



            self._start_estimated_progress_timer(30, 5, 95)



            self.answer_box.setPlainText("Thinking...")



            QApplication.processEvents()



            provider = config.get('provider', 'openai')



            if provider == "anthropic":



                try:



                    self._start_anthropic_stream(query, context_notes, context, config, relevant_notes, scored_notes, context_note_ids)



                    return



                except Exception as e:



                    log_debug(f"Anthropic streaming not available, falling back to non-streaming: {e}")



            # Run ask_ai in background so UI stays responsive (avoids "Not Responding" during 10\xe2\u20ac\u201c300 s request)



            self._ask_ai_relevant_notes = context_notes



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



            provider = (config.get("provider") or "").lower()



            if provider in ("local_openai", "local_server", "ollama"):



                self.answer_box.setText(



                    "Request timed out. The local model is still too slow for this answer.\n"



                    "- Try a smaller or faster local model\n"



                    "- Reduce the result count or selected notes\n"



                    "- Check that LM Studio is loaded and actively generating"



                )



                self._stop_estimated_progress_timer()



                self._hide_busy_progress()



                self.status_label.setText("Error occurred")



                return



            self.answer_box.setText(



                "Request timed out. This could mean:\n"



                "\u2022 The API service is slow or overloaded\n"



                "\u2022 Your internet connection is unstable\n\n"



                "Try again or reduce the number of notes in your collection."



            )



        elif "401" in error_msg or "403" in error_msg or "authentication" in error_msg.lower():



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.answer_box.setText("\u26a0\ufe0f Authentication Error:\nYour API key appears to be invalid or expired.\n\nPlease check your API key in Settings.")



        elif "429" in error_msg or "rate limit" in error_msg.lower():



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.answer_box.setText("\xe2\u0161\xa0\ufe0f Rate Limit Exceeded:\nYou've made too many requests.\n\nPlease wait a few minutes and try again.")



        elif config.get('provider') == 'ollama' and any(x in error_msg.lower() for x in ('connection', 'refused', 'connect', 'unreachable')):



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.answer_box.setText(



                "Error: Cannot reach Ollama.\n\n"



                "Make sure Ollama is running (ollama serve) and the URL in Settings \u2192 Search & Embeddings (Ollama URL) is correct."



            )

        elif any(x in error_msg.lower() for x in ('winerror 10054', 'forcibly closed', 'connection reset', 'server closed the connection')):



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.answer_box.setText(



                "Local AI server closed the connection.\n\n"



                "This is usually a local server/provider issue, not an API key or internet problem.\n\n"



                "Check that the Server URL is an OpenAI-compatible base URL:\n"
                "- Ollama: http://localhost:11434/v1\n"
                "- LM Studio: http://localhost:1234/v1\n"
                "- Jan: http://localhost:1337/v1\n\n"
                "Also confirm the selected model is loaded/running and try a smaller or faster model if the provider keeps closing the request."



            )


        elif any(x in error_msg.lower() for x in ('n_ctx', 'n_keep', 'context length', 'context_length', 'maximum context')):



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.answer_box.setText(



                "Local LLM context is too small for this answer.\n\n"



                "LM Studio rejected the prompt because the retrieved note context plus the requested answer length exceeded the loaded Context Length.\n\n"



                "Fix: open LM Studio -> Developer -> Local Server -> loaded model -> Context and Offload, then increase Context Length. For Qwen3-VL-8B-Instruct, try 8192 first; use 12288-16384 if your GPU/RAM can handle it.\n\n"



                "You can also reduce the number of retrieved notes or set a smaller context-per-note limit in the add-on settings."



            )



        else:



            self.answer_box.setText(f"Error calling AI API:\n{error_msg}\n\nPlease check your API key and internet connection.")



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
        ai_relevant_refs = set()



        for idx in relevant_indices:



            if 0 <= idx < len(relevant_notes):



                ai_relevant_note_ids.add(relevant_notes[idx]['id'])
                ai_relevant_refs.add(idx + 1)



        self._cited_note_ids = ai_relevant_note_ids
        self._cited_refs = ai_relevant_refs



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



                self.status_label.setText("Re-ranking by relevance to answer...")



                self._show_centile_progress("Re-ranking by relevance...", 0)



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



        log_debug("Search completed successfully")







    def perform_search(self):



        """Perform search with proper error handling and UI updates"""



        log_debug("=== Perform Search Called ===")







        # Check for config



        config = self.get_config()



        log_debug(f"Retrieved config for search: {get_safe_config(config)}")







        if not config:



            log_debug("ERROR: No config found")



            self.answer_box.setText("Please configure your API key first. Click the \u2699 button.")



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

        if hasattr(self, 'view_btn'):



            self.view_btn.setEnabled(False)



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



            self._show_centile_progress("Loading notes...", 0)



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



                f"Unexpected Error:\n{str(e)}\n\n"



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



                self.status_label.setText("Re-ranking results... (may take 10-30 s)")



            self._show_centile_progress("Re-ranking...", 0)



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







        if success == "LIBRARY_LOAD_FAILED":



            from aqt.utils import showInfo



            showInfo(



                "Reranking was skipped: sentence-transformers/torch could not be loaded (e.g. DLL error on Windows).\n\n"



                "This usually means the Visual C++ Redistributable is missing or corrupted.\n\n"



                "Search results are still shown using the initial ranking. To fix reranking, try:\n"



                "1. Tools \u2192 Anki Semantic Search \u2192 Install extra model (reinstalls dependencies)\n"



                "2. Check 'Python for Cross-Encoder' in Settings if using an external Python."



            )



            # Continue with original notes as success=False but success=="LIBRARY_LOAD_FAILED"



            # we already have scored_notes as the original ones.



            success = False







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
            mode = 'balanced'



        self._effective_relevance_mode = mode



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



                    self.status_label.setText("Generating HyDE... (one short API call, usually 5-30 s)")



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



                    "Run Create/Update Embeddings (Settings \u2192 Search & Embeddings) for the selected decks/note types.",



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







    def _local_context_usage_plan(self, query, available_notes, provider, search_config):
        """Choose a local-model context budget based on question complexity."""
        if provider not in ("local_openai", "local_server", "ollama"):
            return None

        try:
            n_ctx = int(search_config.get('local_llm_context_tokens') or 12288)
        except Exception:
            n_ctx = 12288
        n_ctx = max(4096, min(n_ctx, 32768))

        query_text = query or ""
        query_tokens = estimate_tokens(query_text)
        lower_query = query_text.lower()
        complex_markers = (
            'compare', 'differentiate', 'difference', 'mechanism', 'pathway',
            'steps', 'sequence', 'algorithm', 'diagnosis', 'management',
            'treatment', 'why', 'explain', 'list', 'table', 'vs', 'versus',
            'all', 'causes', 'risk factors', 'complications'
        )

        complexity = 0
        if query_tokens > 18:
            complexity += 1
        if query_tokens > 35:
            complexity += 1
        if any(marker in lower_query for marker in complex_markers):
            complexity += 1
        if available_notes > 12:
            complexity += 1
        if search_config.get('enable_hyde') or search_config.get('enable_query_expansion'):
            complexity += 1

        if complexity <= 1:
            mode = "light"
            max_output_tokens = min(1024, max(512, n_ctx // 6))
            desired_context_tokens = min(2400, max(1200, n_ctx - max_output_tokens - 900))
            max_notes = 8
        elif complexity <= 3:
            mode = "balanced"
            max_output_tokens = min(2048, max(768, n_ctx // 5))
            desired_context_tokens = min(5600, max(2200, n_ctx - max_output_tokens - 900))
            max_notes = 16
        else:
            mode = "deep"
            max_output_tokens = min(4096, max(1024, n_ctx // 4))
            desired_context_tokens = min(10000, max(3500, n_ctx - max_output_tokens - 900))
            max_notes = 32

        return {
            "mode": mode,
            "model_context_tokens": n_ctx,
            "context_token_budget": desired_context_tokens,
            "max_output_tokens": max_output_tokens,
            "max_notes": max_notes,
        }



    def _fit_context_lines_to_token_budget(self, context_lines, token_budget):
        """Keep note blocks within a token budget while preserving note order."""
        fitted = []
        used = 0
        token_budget = max(600, int(token_budget or 0))

        for line in context_lines:
            line_tokens = estimate_tokens(line)
            remaining = token_budget - used
            if remaining <= 120:
                break
            if line_tokens <= remaining:
                fitted.append(line)
                used += line_tokens
                continue

            keep_chars = max(240, remaining * 4)
            fitted.append(line[:keep_chars].rstrip() + " ...")
            break

        return fitted or context_lines[:1]



    # --- Copied Answer Provider Calls ---

    def ask_ai(self, query, notes, context, config):



        provider = config.get('provider', 'openai')



        api_key = config.get('api_key', '')



        model = self.get_best_model(provider)







        num_notes = len(notes)



        focus_block = ""



        prompt = f"""You are an assistant for question-answering over provided notes. Use ONLY the numbered notes below as your factual source (you may add brief connecting logic, but no outside facts or external guidelines).



If the notes contain at least some relevant information, give the **best partial answer you can** based only on these notes and then briefly mention what is missing.



Only if the notes are essentially unrelated to the question, say exactly: "The provided notes do not contain enough information to answer this."







Context information is below. There are exactly {num_notes} notes: Note 1 = highest relevance, Note 2 = second, ... Note {num_notes} = last. Cite ONLY using numbers from 1 to {num_notes} (e.g. [1], [2], [1,3]). Do not use numbers outside 1\xe2\u20ac\u201c{num_notes}.



---------------------



{context}



---------------------



Given the context information and not prior knowledge, answer the question.







Question: {query}{focus_block}







Rules:



- Base every claim strictly on these notes. Do **not** invent mechanisms, receptor types, dosages, diagnostic criteria, or risk factors that are not supported by the notes. One sentence or bullet per idea is fine.



- Write in a clear, exam-oriented style: use bullet points (\u2022) for key points; use 2-space indented bullets for sub-points. Use **double asterisks** around important terms (diagnoses, drugs, criteria). Do not use ## for headings\u2014use a single bold line with \u25cf\x8f then bullets underneath.



- When the question asks about **receptors, mechanisms, pathways, or numbered lists (e.g. 1st\xe2\u20ac\u201c6th diseases, steps 1\xe2\u20ac\u201c6)**, present them in a clean ordered list and attach citations for each item.



- NUMBERED LISTS: When the question or notes refer to a numbered list (e.g. six childhood exanthems, 1st\xe2\u20ac\u201c6th disease, steps 1\xe2\u20ac\u201c6 of ventricular partitioning), **always** present in **strict numerical order**: 1st, then 2nd, then 3rd, then 4th, then 5th, then 6th. **Include every step or number that any note describes**\u2014if Note X explicitly mentions "3rd step" or "third step", you MUST include it in your answer and cite it [X]. Do not skip steps; do not reorder by relevance.



- INLINE CITATIONS: Cite the supporting note(s) using [N] or [N,M] where N is between 1 and {num_notes} only. Example: "Hypertension increases stroke risk [1,3]." Do not use citation numbers outside 1\xe2\u20ac\u201c{num_notes}.



- At the end, on one line, list all note numbers you cited. Format: RELEVANT_NOTES: 1,3,5"""







        # Estimate input tokens



        input_tokens = estimate_tokens(prompt)







        if provider == "ollama":



            sc = config.get('search_config') or {}



            base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()



            model = (sc.get('ollama_chat_model') or 'llama3.2').strip()



            answer, relevant_indices = self.call_ollama(prompt, base_url, model, notes)



        elif provider == "anthropic":



            system_blocks, user_content = _build_anthropic_prompt_parts(query, context)



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



        elif provider in ("local_openai", "local_server"):



            sc = config.get('search_config') or {}



            base_url = (
                sc.get('local_llm_url')
                or config.get('local_llm_url')
                or config.get('api_url')
                or 'http://localhost:1234/v1'
            ).strip()



            local_model = (
                sc.get('local_llm_model')
                or config.get('local_llm_model')
                or model
                or 'local-model'
            ).strip()



            api_url = self._openai_compatible_chat_url(base_url)



            local_context_plan = self._local_context_usage_plan(
                query, len(notes), provider, sc
            ) or {}



            answer, relevant_indices = self.call_custom(
                prompt,
                "",
                local_model,
                api_url,
                notes,
                timeout_seconds=300,
                max_tokens=local_context_plan.get('max_output_tokens', 4096),
            )



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



            "Content-Type": "application/json"



        }


        if api_key:



            headers["Authorization"] = f"Bearer {api_key}"



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



            "Content-Type": "application/json"



        }



        if api_key:



            headers["Authorization"] = f"Bearer {api_key}"



        data = {



            "model": model,



            "messages": [{"role": "user", "content": prompt}],



            "max_tokens": 4096



        }







        response_text = self.make_request(url, headers, data)



        result = json.loads(response_text)



        full_response = result['choices'][0]['message']['content']



        return self.parse_response(full_response, notes)







    def call_custom(self, prompt, api_key, model, api_url, notes, timeout_seconds=30, max_tokens=4096):



        headers = {



            "Content-Type": "application/json",



            "Authorization": f"Bearer {api_key}"



        }



        data = {



            "model": model,



            "messages": [{"role": "user", "content": prompt}],



            "max_tokens": max_tokens



        }







        response_text = self.make_request(api_url, headers, data, timeout_seconds=timeout_seconds)



        result = json.loads(response_text)







        if 'choices' in result:



            full_response = result['choices'][0]['message']['content']



        elif 'content' in result:



            full_response = result['content'][0]['text']



        else:



            full_response = str(result)







        return self.parse_response(full_response, notes)







    def _openai_compatible_chat_url(self, base_url):



        """Return a chat-completions endpoint for OpenAI-compatible servers."""



        url = (base_url or '').strip()



        if not url:



            raise ValueError("Local server URL is empty. Set it in Settings > API Settings.")



        if "://" not in url:



            url = "http://" + url



        url = url.rstrip("/")



        lower_url = url.lower()



        if lower_url.endswith("/chat/completions"):



            return url


        if lower_url.endswith("/v1"):



            return url + "/chat/completions"


        for suffix in ("/api/chat", "/api/generate", "/api/tags"):



            if lower_url.endswith(suffix):



                return url[: -len(suffix)] + "/v1/chat/completions"


        if lower_url.endswith("/v1/models"):



            return url[:-7] + "/chat/completions"


        if lower_url.endswith("/models"):



            base = url[:-7]



            if ":11434" in lower_url:



                return base + "/v1/chat/completions"



            return base + "/chat/completions"


        if ":11434" in lower_url:



            return url + "/v1/chat/completions"


        if lower_url.endswith("/api"):



            return url[:-4] + "/v1/chat/completions"


        return url + "/chat/completions"



    def make_request(self, url, headers, data, timeout_seconds=30):



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



            timeout_seconds = max(1, int(timeout_seconds or 30))



            log_debug(f"Opening URL connection (timeout: {timeout_seconds} seconds)...")



            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:



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



                raise Exception(f"Request timed out after {timeout_seconds} seconds. The API may be slow or overloaded.")







            raise Exception(f"Network error: {msg}")







        except Exception as e:



            log_debug(f"Unexpected error: {type(e).__name__}: {str(e)}")


            lower = str(e).lower()



            if "winerror 10054" in lower or "forcibly closed" in lower or "connection reset" in lower:



                raise Exception(
                    "Local AI server closed the connection. This usually means the server URL endpoint is wrong, "
                    "the selected model crashed/unloaded, or the request exceeded what the provider can handle. "
                    "For OpenAI-compatible local servers use a /v1 base URL such as http://localhost:11434/v1 "
                    "for Ollama, http://localhost:1234/v1 for LM Studio, or http://localhost:1337/v1 for Jan."
                )



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

                numbers = DIGIT_RE.findall(notes_str)

                relevant_notes = [int(n) - 1 for n in numbers if n.isdigit() and 0 <= int(n) - 1 < len(notes)]

        else:

            answer_part = full_response

            relevant_notes = list(range(min(3, len(notes))))







        log_debug(f"Parsed {len(relevant_notes)} relevant notes from AI response")



        return answer_part, relevant_notes







    # --- Copied Streaming Answer Display ---

    def _start_anthropic_stream(self, query, context_notes, context, config, relevant_notes, scored_notes, context_note_ids):



        """Start Anthropic streaming worker; UI updates in real-time via chunk_signal."""



        import anthropic  # raise if not installed



        api_key = config.get('api_key', '')



        model = self.get_best_model('anthropic')



        system_blocks, user_content = _build_anthropic_prompt_parts(query, context)



        self._streamed_answer = ""



        self._stream_context_notes = context_notes



        self._stream_relevant_notes = relevant_notes



        self._stream_scored_notes = scored_notes



        self._stream_context_note_ids = context_note_ids



        self._stream_config = config



        self._stream_query = query



        self.answer_box.setPlainText("Thinking...")



        worker = AnthropicStreamWorker(api_key, model, system_blocks, user_content, context_notes)



        worker.chunk_signal.connect(self._append_stream_chunk)



        worker.done_signal.connect(self._on_anthropic_stream_done)



        worker.error_signal.connect(self._on_anthropic_stream_error)



        self._anthropic_stream_worker = worker



        worker.start()







    def _append_stream_chunk(self, chunk):



        """Append a streamed text chunk to the answer box; first chunk replaces 'Thinking...' placeholder."""



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



        relevant_note_ids = [context_notes[idx]['id'] for idx in relevant_indices if 0 <= idx < len(context_notes)]



        save_search_history(query, answer, relevant_note_ids, scored_notes, context_note_ids)



        self._context_note_ids = context_note_ids



        self._context_note_id_and_chunk = [(n['id'], n.get('chunk_index')) for n in context_notes]



        self.current_answer = answer



        ai_relevant_note_ids = set()
        ai_relevant_refs = set()



        for idx in relevant_indices:



            if 0 <= idx < len(context_notes):



                ai_relevant_note_ids.add(context_notes[idx]['id'])
                ai_relevant_refs.add(idx + 1)



        self._cited_note_ids = ai_relevant_note_ids
        self._cited_refs = ai_relevant_refs



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



                self.status_label.setText("Re-ranking by relevance to answer...")



                self._show_centile_progress("Re-ranking by relevance...", 0)



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



            sensitivity_text = f" (score \xe2\u2030\xa5 {effective_pct}%)" if effective_pct is not None else " (sensitivity filter)"



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



        self.answer_box.setText(f"Error calling Anthropic API:\n{error_msg}\n\nCheck your API key and internet connection.")



        self.status_label.setText("Error occurred")







    # --- Copied Result Rerank And Display Helpers ---

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



            # Phrase matches first \xe2\u20ac\u201c these are usually the most informative



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



        checked_note_ids = set(getattr(self, 'selected_note_ids', set()) or [])



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







        # Clear and repopulate the table without letting transient unchecked items erase
        # the user's checked-note selections.



        self.results_list.blockSignals(True)



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



        mode = getattr(self, '_effective_relevance_mode', getattr(self, 'relevance_mode', None))
        if not mode:
            mode = (sc.get('relevance_mode') or 'balanced')
        strict = str(mode or 'balanced').lower() == 'focused'



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







        # Filter by current relevance mode (Focused / Balanced / Broad) using precomputed flags; no extra search/API.



        pre_mode_filtered_notes = list(filtered_notes)



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



        # Optionally restrict to the exact refs cited in the AI answer ([1], [2], ...).



        if getattr(self, 'show_only_cited_cb', None) and self.show_only_cited_cb.isChecked():



            cited = getattr(self, '_cited_note_ids', None) or set()



            cited_refs = getattr(self, '_cited_refs', None) or set()



            context_id_and_chunk = getattr(self, '_context_note_id_and_chunk', None) or []



            cited_pairs = {
                context_id_and_chunk[ref - 1]
                for ref in cited_refs
                if isinstance(ref, int) and 1 <= ref <= len(context_id_and_chunk)
            }



            if cited_pairs:



                filtered_notes = [
                    (score, note)
                    for score, note in pre_mode_filtered_notes
                    if (note['id'], note.get('chunk_index')) in cited_pairs
                ]



            elif cited:



                filtered_notes = [(score, note) for score, note in pre_mode_filtered_notes if note['id'] in cited]







        # Set row count



        self.results_list.setRowCount(len(filtered_notes))



        # Disable sorting while populating so rows stay 0..N and every row gets the correct content (fixes empty rows when toggling "Show only cited notes")



        self.results_list.setSortingEnabled(False)







        # 1-based position in context (order sent to AI) so [1], [2], [19] in answer match this #



        cited_ids = getattr(self, '_cited_note_ids', set()) or set()
        cited_refs = getattr(self, '_cited_refs', set()) or set()



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



            # Steepen display so 100 stands out and lower scores spread (100\u2192100, 99\u219297, 95\u219292, 80\u219272)



            display_pct = max(0, min(100, round(100 * (percentage / 100) ** 0.6))) if percentage else 0







            # Column 0 (Ref): Citation number from context (matches [1], [2] in AI answer). Cited notes: blue + bold.



            order_num = get_ref(note, row)



            order_item = QTableWidgetItem()



            order_item.setData(Qt.ItemDataRole.DisplayRole, order_num)



            order_item.setData(Qt.ItemDataRole.UserRole, note['id'])



            order_item.setFlags(order_item.flags() & ~Qt.ItemFlag.ItemIsEditable)



            order_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)



            is_cited_ref = order_num in cited_refs or note['id'] in cited_ids



            if is_cited_ref:



                order_item.setForeground(QColor('#3498db'))



                font = order_item.font()



                font.setBold(True)



                order_item.setFont(font)



            why_ref = "Cited in answer: Yes" if is_cited_ref else "Cited in answer: No"



            order_item.setToolTip(f"Why this result?\n{why_ref}\nCitation [N] in answer matches Ref. Note ID: {note['id']}")



            select_item = QTableWidgetItem()
            select_item.setCheckState(
                Qt.CheckState.Checked if note['id'] in checked_note_ids else Qt.CheckState.Unchecked
            )
            select_item.setData(Qt.ItemDataRole.UserRole, note['id'])
            select_item.setFlags(
                (select_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                & ~Qt.ItemFlag.ItemIsEditable
            )
            select_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            select_item.setToolTip("Select this note for View Selected")
            self.results_list.setItem(row, 0, select_item)

            self.results_list.setItem(row, 1, order_item)







            # Column 2: Content






            # Use full note content when available (chunk-level display); so preview shows start of full note, not chunk



            raw_for_display = (
                note.get('_full_display_content')
                or note.get('display_content')
                or note.get('_full_content')
                or note['content']
            )



            display_content = self.reveal_cloze_for_display(raw_for_display)



            content_item = QTableWidgetItem(display_content)



            content_item.setData(Qt.ItemDataRole.UserRole, note['id'])



            content_item.setData(Qt.ItemDataRole.UserRole + 1, display_pct)



            content_item.setData(Qt.ItemDataRole.UserRole + 2, display_content)



            matching_terms = self._get_matching_terms_for_note(note, getattr(self, 'current_query', ''))



            content_item.setToolTip("")
            content_item.setData(Qt.ItemDataRole.UserRole + 3, {
                "id": note['id'],
                "display_content": display_content,
                "relevance": display_pct,
                "why_ref": why_ref,
                "matching_terms": matching_terms,
            })



            content_item.setFlags(content_item.flags() & ~Qt.ItemFlag.ItemIsEditable)



            self.results_list.setItem(row, 2, content_item)







            # Column 3: Note ID



            note_id_item = QTableWidgetItem(str(note['id']))



            note_id_item.setData(Qt.ItemDataRole.UserRole, note['id'])



            note_id_item.setFlags(note_id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)



            note_id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)



            note_id_item.setToolTip(f"Note ID: {note['id']}\nDouble-click to open in browser")



            self.results_list.setItem(row, 3, note_id_item)







            # Column 4: Relevance (steepened display %)



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



                why_pct += f"\nMatching terms: {', '.join(matching_terms[:6])}{'...' if len(matching_terms) > 6 else ''}"



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



            self.results_list.setItem(row, 4, percentage_item)







        # Update status to match table: "Showing X of Y" so it always matches Matching notes row count



        filtered_count = len(filtered_notes)



        total_in_result = len(notes_to_display)



        # No slider: omit score % from status; with slider would show " (score \xe2\u2030\xa5 X%)"



        sensitivity_text = ""



        if self.sensitivity_slider is not None:



            effective_pct = round(100 * min_score / max_score) if (max_score > 0 and sensitivity_threshold > 0) else None



            sensitivity_text = f" (score \xe2\u2030\xa5 {effective_pct}%)" if effective_pct is not None else " (sensitivity filter)"



            if self.sensitivity_value_label is not None:



                if sensitivity_threshold == 0:



                    self.sensitivity_value_label.setText("0%")



                elif sensitivity_threshold > 0 and max_score > 0:



                    self.sensitivity_value_label.setText(f">={effective_pct}%")



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







        self.results_list.blockSignals(False)



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



            self.results_list.sortItems(1, Qt.SortOrder.AscendingOrder)



        else:



            self.results_list.sortItems(4, Qt.SortOrder.DescendingOrder)  # Sort by Relevance







    # --- Copied Result Selection And Browser Actions ---

    def update_selection_count(self):



        """Update the selected count display and toggle button text"""



        if not hasattr(self, 'results_list'):



            return







        checked_count = 0



        total_count = self.results_list.rowCount()







        # Initialize selected_note_ids if it doesn't exist



        if not hasattr(self, 'selected_note_ids'):



            self.selected_note_ids = set()







        # Update persistence set and count (column 0 is the selection checkbox)



        for row in range(total_count):



            item = self.results_list.item(row, 0)



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



                self.toggle_select_btn.setText("Deselect All")



            else:



                self.toggle_select_btn.setText("Select All")



        if hasattr(self, 'view_btn'):
            self.view_btn.setEnabled(checked_count > 0)



































    def toggle_select_all(self):



        """Toggle between selecting all and deselecting all notes"""



        if not hasattr(self, 'results_list') or self.results_list.rowCount() == 0:



            return







        # Check if all visible rows are selected.



        all_selected = True



        for row in range(self.results_list.rowCount()):



            item = self.results_list.item(row, 0)



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



            item = self.results_list.item(row, 0)



            if item:



                item.setCheckState(Qt.CheckState.Checked)



                # Store in persistence



                note_id = item.data(Qt.ItemDataRole.UserRole)



                if note_id:



                    self.selected_note_ids.add(note_id)



        self.results_list.blockSignals(False)







        self.update_selection_count()



        tooltip(f"\u2713 Selected all {self.results_list.rowCount()} notes")







    def deselect_all_notes(self):



        """Deselect all notes in the results list"""



        if not hasattr(self, 'results_list'):



            return







        # Block signals to prevent multiple updates



        self.results_list.blockSignals(True)



        for row in range(self.results_list.rowCount()):



            item = self.results_list.item(row, 0)



            if item:



                item.setCheckState(Qt.CheckState.Unchecked)



                # Remove from persistence



                note_id = item.data(Qt.ItemDataRole.UserRole)



                if note_id:



                    self.selected_note_ids.discard(note_id)



        self.results_list.blockSignals(False)







        self.update_selection_count()



        tooltip(f"\xe2\u0153\u2014 Deselected all notes")







    def restore_selections(self):



        """Restore selections from stored note IDs"""



        if not hasattr(self, 'selected_note_ids') or not self.selected_note_ids:



            return







        # Block signals to prevent multiple updates



        self.results_list.blockSignals(True)



        for row in range(self.results_list.rowCount()):



            item = self.results_list.item(row, 0)



            if item:



                note_id = item.data(Qt.ItemDataRole.UserRole)



                if note_id in self.selected_note_ids:



                    item.setCheckState(Qt.CheckState.Checked)



        self.results_list.blockSignals(False)







    def open_selected_in_browser(self):



        checked_ids = []



        for row in range(self.results_list.rowCount()):



            item = self.results_list.item(row, 0)



            if item and item.checkState() == Qt.CheckState.Checked:



                note_id = item.data(Qt.ItemDataRole.UserRole)



                checked_ids.append(str(note_id))







        if not checked_ids:



            tooltip("Please check at least one note to view")



            return







        browser = dialogs.open("Browser", mw)



        search_query = "nid:" + ",".join(checked_ids)



        browser.form.searchEdit.lineEdit().setText(search_query)



        browser.onSearchActivated()



        QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))



        tooltip(f"\u2713 Opened {len(checked_ids)} selected notes in browser")







    def open_all_in_browser(self):



        if self.results_list.rowCount() == 0:



            tooltip("No notes to view")



            return







        note_ids = []



        for row in range(self.results_list.rowCount()):



            item = self.results_list.item(row, 0)



            if item:



                note_id = item.data(Qt.ItemDataRole.UserRole)



                note_ids.append(str(note_id))







        browser = dialogs.open("Browser", mw)



        search_query = "nid:" + ",".join(note_ids)



        browser.form.searchEdit.lineEdit().setText(search_query)



        browser.onSearchActivated()



        QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))



        tooltip(f"\u2713 Opened {len(note_ids)} notes in browser")







    def open_in_browser(self, item):



        """Open note in browser when double-clicked"""



        # Get the row of the clicked item



        row = item.row()



        # Get note ID from the hidden selection-data column.



        content_item = self.results_list.item(row, 0)



        if content_item:



            note_id = content_item.data(Qt.ItemDataRole.UserRole)



            browser = dialogs.open("Browser", mw)



            browser.form.searchEdit.lineEdit().setText(f"nid:{note_id}")



            browser.onSearchActivated()



            QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))



            tooltip("\u2713 Note opened in browser")


    def _show_note_preview_for_cell(self, row, column):
        if column != 2 or not hasattr(self, '_note_preview_popup'):
            if hasattr(self, '_note_preview_popup'):
                self._note_preview_popup.hide()
            return

        item = self.results_list.item(row, column)
        if not item:
            self._note_preview_popup.hide()
            return

        note_info = item.data(Qt.ItemDataRole.UserRole + 3)
        if not isinstance(note_info, dict):
            self._note_preview_popup.hide()
            return

        self._note_preview_popup.show_note(note_info)


# ============================================================================
# Dynamic Method Compatibility Wiring
# ============================================================================

_AISEARCH_METHODS_FROM_WORKER = (
    '_get_answer_source_text', '_on_embedding_search_progress', '_show_busy_progress', '_show_centile_progress',
    '_start_estimated_progress_timer', '_stop_estimated_progress_timer', '_hide_busy_progress', '_on_embedding_search_finished',
    '_on_keyword_filter_continue_done', '_on_embedding_search_error', '_on_rerank_done', '_on_get_notes_done', '_on_keyword_filter_done',
    '_perform_search_continue', 'perform_search',
    '_on_relevance_rerank_done', '_display_answer_and_notes_after_rerank', '_on_relevance_rerank_done_stream', '_finish_anthropic_stream_display',
    '_on_ask_ai_success', '_on_ask_ai_error', '_on_ask_ai_worker_finished',
    '_rerank_with_cross_encoder', 'keyword_filter', 'keyword_filter_continue', 'get_best_model',
    '_local_context_usage_plan', '_fit_context_lines_to_token_budget',
    'ask_ai', 'call_ollama', 'call_anthropic', 'call_openai', 'call_google', 'call_openrouter',
    'call_custom', '_openai_compatible_chat_url', 'make_request', 'parse_response', '_rerank_by_relevance_to_answer', 'filter_and_display_notes', '_get_matching_terms_for_note', 'update_selection_count',
    '_start_anthropic_stream', '_append_stream_chunk', '_on_anthropic_stream_done', '_on_anthropic_stream_error',
    'toggle_select_all', 'select_all_notes', 'deselect_all_notes',
    'restore_selections', '_bring_browser_to_front', 'open_selected_in_browser', 'open_all_in_browser', 'open_in_browser',
    '_show_note_preview_for_cell',
)


def install_search_workflow_methods(search_dialog_cls):
    for name in _AISEARCH_METHODS_FROM_WORKER:
        method = (
            getattr(EmbeddingSearchWorker, name, None)
            or getattr(RerankWorker, name, None)
            or getattr(AnthropicStreamWorker, name, None)
        )
        if method is not None:
            setattr(search_dialog_cls, name, method)


def configure_search_workflow_globals(**values):
    globals().update(values)
