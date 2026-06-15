import json

from aqt.qt import QThread, pyqtSignal

from .answer_prompts import _build_anthropic_prompt_parts
from ..core.engine import estimate_tokens
from ..utils import log_debug, save_search_history
from ..utils.search_analytics_log import make_answer_payload, make_prompt_payload


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


class SearchAnthropicStreamingMixin:
    def _start_anthropic_stream(self, query, context_notes, context, config, relevant_notes, scored_notes, context_note_ids):



        """Start Anthropic streaming worker; UI updates in real-time via chunk_signal."""



        import anthropic  # raise if not installed



        api_key = config.get('api_key', '')



        model = self.get_best_model('anthropic')



        system_blocks, user_content = _build_anthropic_prompt_parts(query, context)
        self._analytics_stage(
            "answer_prompt_built",
            provider="anthropic",
            model=model,
            streaming=True,
            input_tokens=estimate_tokens(json.dumps(user_content, ensure_ascii=False)),
            **make_prompt_payload(
                prompt_parts={"system_blocks": system_blocks, "user_content": user_content},
                mode=self._analytics_mode(),
            ),
        )



        self._streamed_answer = ""



        self._stream_context_notes = context_notes



        self._stream_relevant_notes = relevant_notes



        self._stream_scored_notes = scored_notes



        self._stream_context_note_ids = context_note_ids



        self._stream_config = config



        self._stream_query = query



        if hasattr(self, "_set_chat_transient"):
            self._set_chat_transient("Answering from your notes...", mode="info")
        else:
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
        footer_marker = "RELEVANT_NOTES:"
        display_answer = self._streamed_answer.split(footer_marker, 1)[0].rstrip()
        for i in range(1, len(footer_marker)):
            if display_answer.endswith(footer_marker[:i]):
                display_answer = display_answer[:-i].rstrip()
                break



        if hasattr(self, "_set_chat_transient"):
            preview = display_answer[-500:] if display_answer else "Answering from your notes..."
            self._set_chat_transient(preview, mode="info")
        else:
            self.answer_box.setPlainText(display_answer)







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



        context_note_id_and_chunk = [(n['id'], n.get('chunk_index')) for n in context_notes]
        context_note_identity_keys = [
            (n['id'], n.get('chunk_index'), n.get('content_hash'))
            for n in context_notes
        ]
        save_search_history(
            query,
            answer,
            relevant_note_ids,
            scored_notes,
            context_note_ids,
            context_note_id_and_chunk=context_note_id_and_chunk,
            context_note_identity_keys=context_note_identity_keys,
        )

        answer_payload = make_answer_payload(
            answer,
            context_note_ids=context_note_ids,
            final_results=self._analytics_results(scored_notes, limit=100),
            mode=self._analytics_mode(),
        )
        if relevant_indices and not answer_payload.get("relevant_notes_refs"):
            answer_payload["relevant_notes_refs"] = [idx + 1 for idx in relevant_indices]

        self._analytics_stage(
            "answer_generated",
            answer_model=self._get_answer_source_text(config),
            answer_length=len(answer or ""),
            answer=answer_payload,
            context_note_ids=list(context_note_ids or []),
            cited_note_ids=list(relevant_note_ids or []),
            relevant_indices=list(relevant_indices or []),
            streaming=True,
        )



        self._context_note_ids = context_note_ids



        self._context_note_id_and_chunk = context_note_id_and_chunk
        self._context_note_identity_keys = context_note_identity_keys



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



        self._finish_anthropic_stream_display(answer, config)








    def _on_anthropic_stream_error(self, error_msg):



        """Show streaming error in answer box."""



        if getattr(self, '_anthropic_stream_worker', None):



            self._anthropic_stream_worker = None



        if hasattr(self, 'answer_source_label'):



            self.answer_source_label.setText("")



        if hasattr(self, "_append_system_chat_message"):
            self._set_chat_transient(None)
            self._append_system_chat_message(
                f"Error calling Anthropic API:\n{error_msg}\n\nCheck your API key and internet connection.",
                kind="error",
            )
        else:
            self.answer_box.setText(f"Error calling Anthropic API:\n{error_msg}\n\nCheck your API key and internet connection.")



        self.status_label.setText("Error occurred")







    # --- Copied Result Rerank And Display Helpers ---
