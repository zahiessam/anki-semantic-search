import datetime
import json

from .answer_prompts import _build_anthropic_prompt_parts
from .image_attachments import (
    IMAGE_SUPPORT_ERROR,
    clean_visual_terms,
    has_user_upload,
    is_image_support_error,
    normalize_review_image_payloads,
    snapshot_image_payloads,
)
from .search_workers import DirectAIWorker
from ..core.engine import estimate_tokens
from ..utils import log_debug, save_search_history
from ..utils.anthropic_response import extract_anthropic_text
from ..utils.search_analytics_log import make_answer_payload, make_prompt_payload


class SearchAnswerWorkflowMixin:
    def _direct_ai_review_image_payloads(self):
        context = getattr(self, "_review_context", None) or {}
        if not (getattr(self, "_review_context_text", "") or "").strip():
            return []
        return normalize_review_image_payloads(context.get("image_payloads") or [])

    def _direct_ai_composer_image_payloads(self):
        if hasattr(self, "_snapshot_composer_image_payloads"):
            return self._snapshot_composer_image_payloads()
        return snapshot_image_payloads(getattr(self, "_composer_image_payloads", []) or [])

    def _openai_direct_user_content(self, prompt, image_payloads=None):
        images = list(image_payloads or [])
        if not images:
            return prompt
        content = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image['mime_type']};base64,{image['base64']}",
                },
            })
        return content

    def _anthropic_direct_user_content(self, prompt, image_payloads=None):
        content = [{"type": "text", "text": prompt}]
        for image in image_payloads or []:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image["mime_type"],
                    "data": image["base64"],
                },
            })
        return content if image_payloads else prompt

    def _google_direct_parts(self, prompt, image_payloads=None):
        parts = [{"text": prompt}]
        for image in image_payloads or []:
            parts.append({
                "inline_data": {
                    "mime_type": image["mime_type"],
                    "data": image["base64"],
                },
            })
        return parts

    def _direct_ai_prompt(self, query, chat_history):
        history_lines = []
        for message in (chat_history or [])[-6:]:
            role = message.get("role", "user")
            content = (message.get("content") or "").strip()
            if content:
                history_lines.append(f"{role.upper()}: {content}")
        history_block = "\n\nRecent conversation:\n" + "\n".join(history_lines) if history_lines else ""
        review_context = (getattr(self, "_review_context_text", "") or "").strip()
        if review_context:
            grounding_rule = (
                "A current review note context is available below. Use it first when it helps answer the question, "
                "especially for phrases like \"this card\", \"this note\", \"this answer\", or \"the note I am reviewing\". "
                "If images are attached from the configured review fields, inspect them as part of the current note context. "
                "If the note context is incomplete or not relevant, answer directly using your general reasoning and say what is not shown by the current note."
            )
            review_block = (
                "\n\nCurrent review note context:\n"
                "---------------------\n"
                f"{review_context[:3000]}\n"
                "---------------------"
            )
            citation_rule = (
                "Do not claim that you searched the user's Anki collection. You may say you used the current review note context when relevant. "
                "Do not include citations or source-number brackets unless the user explicitly asks for references."
            )
        else:
            grounding_rule = (
                "Answer the user's question using your general reasoning and knowledge. Do not claim that you searched or used the user's Anki notes. "
                "If the user needs an answer grounded in their Anki collection, briefly say they can use Ask Notes."
            )
            review_block = ""
            citation_rule = "Do not include citations or source-number brackets unless the user explicitly asks for references. Ask AI is not grounded in the user's Anki notes."
        selected_block = self._selected_answer_context_block() if hasattr(self, "_selected_answer_context_block") else ""
        return f"""You are a helpful AI assistant inside Anki Semantic Search, speaking to a medical doctor.

{grounding_rule}

Respond in the same language as the user's question when possible. Format the answer for fast study review:
- Start with "Direct answer:" followed by 1-3 concise sentences.
- Default to a direct, concise clinical style. Avoid introductory filler, over-explaining basic medical concepts, and chatty commentary.
- Use short headings and concise bullets. Prefer 6-8 main bullets maximum and at most one sub-bullet level.
- Use a compact Markdown table when comparing criteria, lab findings, causes, diagnoses, classifications, treatments, or stepwise tests.
- Bold only key terms and abnormal/diagnostic findings; do not bold every medical noun.
- Do not use LaTeX/math markup. Write symbols plainly, for example beta-glucuronidase, ↓, ↑, and →.
- Keep paragraphs short and visually organized.
- {citation_rule}{review_block}{history_block}

Question: {query}{selected_block}"""

    def ask_ai_direct_from_composer(self):
        if (
            hasattr(self, "_collapse_sources_panel")
            and not getattr(self, "_sources_manually_expanded", False)
        ):
            self._collapse_sources_panel(manual=False)
        self._sources_rank_mode = False
        config = self.get_config()
        if not config:
            self._append_system_chat_message("Please configure your API key first. Click the Settings button.", kind="error")
            return
        query = self.search_input.toPlainText().strip() if hasattr(self, "search_input") else ""
        if hasattr(self, "_strip_selected_answer_snippet_from_query"):
            query = self._strip_selected_answer_snippet_from_query(query)
        selected_context = self._composer_selected_answer_context() if hasattr(self, "_composer_selected_answer_context") else ""
        composer_image_payloads = self._direct_ai_composer_image_payloads()
        if not query and selected_context:
            query = "Explain this selected text."
        if not query and composer_image_payloads:
            query = "What is shown in this image?"
        if not query:
            from aqt.utils import tooltip
            tooltip("Please enter a question")
            return
        if getattr(self, "_direct_ai_worker", None) is not None or getattr(self, "_ask_ai_worker", None) is not None:
            from aqt.utils import tooltip
            tooltip("Please wait for the current answer to finish")
            return

        history = list(getattr(self, "_chat_history", []) or [])
        self._pending_selected_answer_context = selected_context
        self._composer_selected_answer_context_text = ""
        if hasattr(self, "_update_selected_answer_context_chip"):
            self._update_selected_answer_context_chip()
        self._append_user_chat_message(query, mode="ai", image_payloads=composer_image_payloads)
        self.search_input.clear()
        self._set_search_chat_busy(True, "Asking AI...")
        self._set_chat_transient("Asking AI...", mode="info")
        self._show_centile_progress("Asking AI...", 0)
        self._start_estimated_progress_timer(60, 5, 95)
        self._ask_ai_config = config

        self._streamed_answer = ""
        image_payloads = self._direct_ai_review_image_payloads() + composer_image_payloads
        allow_image_fallback = not has_user_upload(image_payloads)
        self._direct_ai_worker = DirectAIWorker(
            self,
            query,
            history,
            config,
            image_payloads=image_payloads,
            allow_image_fallback=allow_image_fallback,
        )
        self._direct_ai_worker.chunk_signal.connect(self._append_stream_chunk)
        self._direct_ai_worker.success_signal.connect(self._on_direct_ai_success)
        self._direct_ai_worker.error_signal.connect(self._on_direct_ai_error)
        if hasattr(self._direct_ai_worker, "warning_signal"):
            self._direct_ai_worker.warning_signal.connect(self._on_direct_ai_warning)
        self._direct_ai_worker.finished.connect(self._on_direct_ai_finished)
        self._direct_ai_worker.start()
        if image_payloads and hasattr(self, "_clear_composer_image_attachment"):
            self._clear_composer_image_attachment()

    def ask_ai_direct(self, query, chat_history, config, chunk_callback=None, image_payloads=None):
        provider = config.get('provider', 'openai')
        api_key = config.get('api_key', '')
        model = self.get_best_model(provider)
        prompt = self._direct_ai_prompt(query, chat_history)
        image_payloads = list(image_payloads or [])

        if provider == "ollama":
            sc = config.get('search_config') or {}
            base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()
            model = (sc.get('ollama_chat_model') or 'llama3.2').strip()
            if chunk_callback and not image_payloads:
                answer = self._ollama_generate_stream_response(
                    prompt, base_url, model, timeout_seconds=75, chunk_callback=chunk_callback, num_predict=1024
                ).strip()
                if not answer:
                    raise Exception("Ollama returned an empty Ask AI response.")
                return answer
            url = base_url.rstrip("/") + "/api/chat"
            message = {"role": "user", "content": prompt}
            if image_payloads:
                message["images"] = [image["base64"] for image in image_payloads]
            data = {
                "model": model,
                "messages": [message],
                "stream": False,
                "options": {"num_predict": 1024},
            }
            result = json.loads(self.make_request(url, {"Content-Type": "application/json"}, data, timeout_seconds=75))
            answer = (
                (result.get("message") or {}).get("content")
                or result.get("response")
                or result.get("thinking")
                or ""
            )
            if isinstance(answer, list):
                answer = "".join(p.get("text", p) if isinstance(p, dict) else str(p) for p in answer)
            answer = (answer or "").strip()
            if not answer:
                raise Exception("Ollama returned an empty Ask AI response.")
            return answer

        if provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"}
            data = {
                "model": model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": self._anthropic_direct_user_content(prompt, image_payloads)}],
            }
            result = json.loads(self.make_request(url, headers, data, timeout_seconds=75))
            return extract_anthropic_text(result, source="Anthropic direct answer").strip()

        if provider == "google":
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
            data = {"contents": [{"parts": self._google_direct_parts(prompt, image_payloads)}], "generationConfig": {"maxOutputTokens": 1024}}
            result = json.loads(self.make_request(url, {"Content-Type": "application/json"}, data, timeout_seconds=75))
            return result['candidates'][0]['content']['parts'][0]['text'].strip()

        if provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            data = {"model": model, "messages": [{"role": "user", "content": self._openai_direct_user_content(prompt, image_payloads)}], "max_tokens": 1024}
            if chunk_callback:
                return self._openai_stream_response(
                    url, headers, data, timeout_seconds=75, chunk_callback=chunk_callback
                ).strip()
            result = json.loads(self.make_request(url, headers, data, timeout_seconds=75))
            return result['choices'][0]['message']['content'].strip()

        if provider in ("local_openai", "local_server"):
            sc = config.get('search_config') or {}
            base_url = (sc.get('local_llm_url') or config.get('local_llm_url') or config.get('api_url') or 'http://localhost:1234/v1').strip()
            local_model = (sc.get('answer_local_model') or sc.get('local_llm_model') or config.get('local_llm_model') or model or 'local-model').strip()
            url = self._openai_compatible_chat_url(base_url)
            data = {"model": local_model, "messages": [{"role": "user", "content": self._openai_direct_user_content(prompt, image_payloads)}], "max_tokens": 1024}
            if chunk_callback:
                return self._openai_stream_response(
                    url,
                    {"Content-Type": "application/json", "Authorization": "Bearer "},
                    data,
                    timeout_seconds=75,
                    chunk_callback=chunk_callback,
                ).strip()
            result = json.loads(self.make_request(url, {"Content-Type": "application/json", "Authorization": "Bearer "}, data, timeout_seconds=75))
            return result['choices'][0]['message']['content'].strip()

        url = "https://api.openai.com/v1/chat/completions" if provider == "openai" else config.get('api_url', '')
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = {"model": model, "messages": [{"role": "user", "content": self._openai_direct_user_content(prompt, image_payloads)}], "max_tokens": 1024}
        if provider == "openai" and chunk_callback:
            return self._openai_stream_response(
                url, headers, data, timeout_seconds=75, chunk_callback=chunk_callback
            ).strip()
        result = json.loads(self.make_request(url, headers, data, timeout_seconds=75))
        if 'choices' in result:
            return result['choices'][0]['message']['content'].strip()
        if 'content' in result:
            return extract_anthropic_text(result, source="direct answer").strip()
        return str(result).strip()

    def _notes_search_rewrite_prompt(self, query, chat_history):
        history_lines = []
        for message in (chat_history or [])[-6:]:
            role = message.get("role", "user")
            mode = message.get("mode", "")
            content = (message.get("content") or "").strip()
            if content:
                history_lines.append(f"{role.upper()}[{mode}]: {content}")
        history_block = "\n".join(history_lines) if history_lines else "(none)"
        review_context = (getattr(self, "_review_context_text", "") or "").strip()
        review_block = (
            "\n\nCurrent review note context:\n"
            f"{review_context[:1800]}"
            if review_context else ""
        )
        selected_block = self._selected_answer_context_block() if hasattr(self, "_selected_answer_context_block") else ""
        return f"""Rewrite the user's Ask Notes request into one standalone Anki note-search query.

Use the recent conversation only to resolve follow-up references like "that", "it", "the previous answer", or "search notes for this".
Use the current review note context to resolve phrases like "this card", "this note", or "this answer".
Use selected answer text context to resolve phrases like "this", "that line", or "explain selected text".
Keep important medical terms, diagnoses, drugs, mechanisms, lists, and constraints.
Do not answer the question.
Return only the standalone search query as plain text. No bullets, quotes, labels, or explanation.

Recent conversation:
{history_block}{review_block}{selected_block}

Ask Notes request:
{query}

Standalone search query:"""

    def _clean_rewritten_notes_query(self, rewritten, fallback):
        text = (rewritten or "").strip()
        if not text:
            return fallback
        if text.startswith("```"):
            text = text.strip("`").strip()
        for prefix in (
            "Standalone search query:",
            "Search query:",
            "Query:",
            "Rewritten query:",
        ):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        text = text.strip().strip('"').strip("'").strip()
        lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
        if lines:
            text = " ".join(lines[:3]).strip()
        return text or fallback

    def _image_search_terms_prompt(self, query):
        context = (query or "").strip()
        context_line = f"\nUser/search context: {context}" if context else ""
        return (
            "Extract concise visual findings and likely medical/search terms for finding related Anki notes. "
            "Do not answer the question. Return only a short comma-separated phrase list, no labels, no bullets."
            f"{context_line}"
        )

    def _extract_text_from_openai_result(self, result):
        content = ((result.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text") or "")
                else:
                    parts.append(str(part))
            content = " ".join(parts)
        return str(content or "").strip()

    def generate_image_search_terms(self, query, config, image_payloads):
        image_payloads = snapshot_image_payloads(image_payloads)
        if not image_payloads:
            return ""

        provider = (config.get('provider', 'openai') or "openai").strip().lower()
        api_key = config.get('api_key', '')
        model = self.get_best_model(provider)
        prompt = self._image_search_terms_prompt(query)

        try:
            if provider == "ollama":
                sc = config.get('search_config') or {}
                base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()
                model = (sc.get('ollama_chat_model') or 'llama3.2').strip()
                url = base_url.rstrip("/") + "/api/chat"
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt, "images": [image["base64"] for image in image_payloads]}],
                    "stream": False,
                    "options": {"num_predict": 120, "temperature": 0.1},
                }
                result = json.loads(self.make_request(url, {"Content-Type": "application/json"}, data, timeout_seconds=75))
                raw = (
                    (result.get("message") or {}).get("content")
                    or result.get("response")
                    or result.get("thinking")
                    or ""
                )
                if isinstance(raw, list):
                    raw = " ".join(p.get("text", p) if isinstance(p, dict) else str(p) for p in raw)
            elif provider == "anthropic":
                url = "https://api.anthropic.com/v1/messages"
                headers = {"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"}
                data = {
                    "model": model,
                    "max_tokens": 120,
                    "messages": [{"role": "user", "content": self._anthropic_direct_user_content(prompt, image_payloads)}],
                }
                result = json.loads(self.make_request(url, headers, data, timeout_seconds=75))
                raw = extract_anthropic_text(result, source="Anthropic image search terms")
            elif provider == "google":
                url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
                data = {
                    "contents": [{"parts": self._google_direct_parts(prompt, image_payloads)}],
                    "generationConfig": {"maxOutputTokens": 120, "temperature": 0.1},
                }
                result = json.loads(self.make_request(url, {"Content-Type": "application/json"}, data, timeout_seconds=75))
                raw = ((result.get("candidates") or [{}])[0].get("content") or {}).get("parts", [{}])[0].get("text", "")
            elif provider == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": self._openai_direct_user_content(prompt, image_payloads)}],
                    "max_tokens": 120,
                    "temperature": 0.1,
                }
                result = json.loads(self.make_request(url, headers, data, timeout_seconds=75))
                raw = self._extract_text_from_openai_result(result)
            elif provider in ("local_openai", "local_server"):
                sc = config.get('search_config') or {}
                base_url = (sc.get('local_llm_url') or config.get('local_llm_url') or config.get('api_url') or 'http://localhost:1234/v1').strip()
                local_model = (sc.get('answer_local_model') or sc.get('local_llm_model') or config.get('local_llm_model') or model or 'local-model').strip()
                url = self._openai_compatible_chat_url(base_url)
                data = {
                    "model": local_model,
                    "messages": [{"role": "user", "content": self._openai_direct_user_content(prompt, image_payloads)}],
                    "max_tokens": 120,
                    "temperature": 0.1,
                }
                result = json.loads(self.make_request(url, {"Content-Type": "application/json", "Authorization": "Bearer "}, data, timeout_seconds=75))
                raw = self._extract_text_from_openai_result(result)
            else:
                url = "https://api.openai.com/v1/chat/completions" if provider == "openai" else config.get('api_url', '')
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": self._openai_direct_user_content(prompt, image_payloads)}],
                    "max_tokens": 120,
                    "temperature": 0.1,
                }
                result = json.loads(self.make_request(url, headers, data, timeout_seconds=75))
                raw = self._extract_text_from_openai_result(result) if 'choices' in result else str(result)
        except Exception as exc:
            if is_image_support_error(exc):
                raise Exception(IMAGE_SUPPORT_ERROR)
            raise

        terms = clean_visual_terms(raw)
        if not terms:
            log_debug(f"Image visual terms unusable, falling back to text-only query. Raw response: {str(raw or '')[:500]}")
        return terms

    def rewrite_notes_search_query(self, query, chat_history, config):
        """Use the configured chat model to turn a follow-up into a retrieval query."""
        fallback = (query or "").strip()
        if not fallback:
            return ""
        if not chat_history:
            return fallback

        provider = config.get('provider', 'openai')
        api_key = config.get('api_key', '')
        model = self.get_best_model(provider)
        prompt = self._notes_search_rewrite_prompt(fallback, chat_history)

        if provider == "ollama":
            sc = config.get('search_config') or {}
            base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()
            model = (sc.get('ollama_chat_model') or 'llama3.2').strip()
            url = base_url.rstrip("/") + "/api/chat"
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": 80, "temperature": 0.1},
            }
            result = json.loads(self.make_request(url, {"Content-Type": "application/json"}, data, timeout_seconds=45))
            rewritten = (
                (result.get("message") or {}).get("content")
                or result.get("response")
                or result.get("thinking")
                or ""
            )
            if isinstance(rewritten, list):
                rewritten = "".join(p.get("text", p) if isinstance(p, dict) else str(p) for p in rewritten)
            return self._clean_rewritten_notes_query(rewritten, fallback)

        if provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"}
            data = {"model": model, "max_tokens": 80, "messages": [{"role": "user", "content": prompt}]}
            result = json.loads(self.make_request(url, headers, data, timeout_seconds=45))
            return self._clean_rewritten_notes_query(extract_anthropic_text(result, source="Anthropic notes query rewrite"), fallback)

        if provider == "google":
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
            data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 80, "temperature": 0.1}}
            result = json.loads(self.make_request(url, {"Content-Type": "application/json"}, data, timeout_seconds=45))
            return self._clean_rewritten_notes_query(result['candidates'][0]['content']['parts'][0]['text'], fallback)

        if provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 80, "temperature": 0.1}
            result = json.loads(self.make_request(url, headers, data, timeout_seconds=45))
            return self._clean_rewritten_notes_query(result['choices'][0]['message']['content'], fallback)

        if provider in ("local_openai", "local_server"):
            sc = config.get('search_config') or {}
            base_url = (sc.get('local_llm_url') or config.get('local_llm_url') or config.get('api_url') or 'http://localhost:1234/v1').strip()
            local_model = (sc.get('answer_local_model') or sc.get('local_llm_model') or config.get('local_llm_model') or model or 'local-model').strip()
            url = self._openai_compatible_chat_url(base_url)
            data = {"model": local_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 80, "temperature": 0.1}
            result = json.loads(self.make_request(url, {"Content-Type": "application/json", "Authorization": "Bearer "}, data, timeout_seconds=45))
            return self._clean_rewritten_notes_query(result['choices'][0]['message']['content'], fallback)

        url = "https://api.openai.com/v1/chat/completions" if provider == "openai" else config.get('api_url', '')
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 80, "temperature": 0.1}
        result = json.loads(self.make_request(url, headers, data, timeout_seconds=45))
        if 'choices' in result:
            return self._clean_rewritten_notes_query(result['choices'][0]['message']['content'], fallback)
        if 'content' in result:
            return self._clean_rewritten_notes_query(extract_anthropic_text(result, source="notes query rewrite"), fallback)
        return fallback

    def _on_direct_ai_success(self, answer):
        self._stop_estimated_progress_timer()
        self._hide_busy_progress()
        self._set_chat_transient(None)
        self._pending_selected_answer_context = ""
        self.current_answer = answer or ""
        self._last_formatted_answer = ""
        src = self._get_answer_source_text(getattr(self, '_ask_ai_config', None) or self.get_config())
        self._append_assistant_chat_message(
            answer or "The provider returned an empty answer.",
            mode="ai",
            source_text=src,
        )
        if hasattr(self, "status_label"):
            self.status_label.setText("Ready")
        if hasattr(self, "_notify_long_job_done"):
            self._notify_long_job_done("Ask AI complete", "Answer ready")

    def _on_direct_ai_warning(self, warning_msg):
        if warning_msg and hasattr(self, "_append_system_chat_message"):
            self._append_system_chat_message(warning_msg, kind="warning")

    def _on_direct_ai_error(self, error_msg):
        self._stop_estimated_progress_timer()
        self._hide_busy_progress()
        self._set_chat_transient(None)
        self._pending_selected_answer_context = ""
        lower = (error_msg or "").lower()
        if "timed out" in lower or "timeout" in lower:
            error_msg = (
                f"{error_msg}\n\n"
                "Ask AI is using the configured answer model directly. Try a smaller/faster Ollama chat model "
                "or confirm the selected model responds in Ollama."
            )
        self._append_system_chat_message(f"Could not get an AI answer:\n{error_msg}", kind="error")
        if hasattr(self, "status_label"):
            self.status_label.setText("Error occurred")
        if hasattr(self, "_notify_long_job_done"):
            self._notify_long_job_done("Ask AI failed", "Check the conversation for details", kind="error")

    def _on_direct_ai_finished(self):
        self._direct_ai_worker = None
        self._set_search_chat_busy(False)

    def _answer_error_context_payload(self):
        context_note_ids = list(
            getattr(self, "_ask_ai_context_note_ids", None)
            or getattr(self, "_context_note_ids", None)
            or []
        )
        context_notes = list(getattr(self, "_ask_ai_relevant_notes", None) or [])
        context_text = getattr(self, "_ask_ai_context", "") or ""
        return {
            "context_note_ids": context_note_ids,
            "context_count": len(context_notes) if context_notes else len(context_note_ids),
            "prompt_context_non_empty": bool(str(context_text).strip()),
        }

    def _on_ask_ai_worker_finished(self):



        """Clear worker reference after thread finishes."""



        self._ask_ai_worker = None
        self._pending_selected_answer_context = ""
        if hasattr(self, "_set_search_chat_busy"):
            self._set_search_chat_busy(False)


    def _on_ask_ai_error(self, error_msg):



        """Handle AI request failure (runs on main thread)."""



        log_debug(f"Error calling AI API: {error_msg}")
        if hasattr(self, "_append_system_chat_message"):
            self._stop_estimated_progress_timer()
            self._hide_busy_progress()
            self._set_chat_transient(None)
            self._append_system_chat_message(f"Could not get a notes answer:\n{error_msg}", kind="error")
            if hasattr(self, "status_label"):
                self.status_label.setText("Error occurred")
            error_context = self._answer_error_context_payload()
            self._analytics_stage("answer_error", error=error_msg, **error_context)
            self._write_search_analytics_report(
                completed_at=datetime.datetime.now().isoformat(),
                final_status="answer_error",
                error=error_msg,
                **error_context,
                final_results=self._analytics_results(getattr(self, "all_scored_notes", []), limit=100),
            )
            if hasattr(self, "_notify_long_job_done"):
                self._notify_long_job_done("Ask Notes failed", "Check the conversation for details", kind="error")
            return



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

                error_context = self._answer_error_context_payload()
                self._analytics_stage("answer_error", error=error_msg, **error_context)

                self._write_search_analytics_report(
                    completed_at=datetime.datetime.now().isoformat(),
                    final_status="answer_error",
                    error=error_msg,
                    **error_context,
                    final_results=self._analytics_results(getattr(self, "all_scored_notes", []), limit=100),
                )
                if hasattr(self, "_notify_long_job_done"):
                    self._notify_long_job_done("Ask Notes failed", "Request timed out", kind="error")



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

        error_context = self._answer_error_context_payload()
        self._analytics_stage("answer_error", error=error_msg, **error_context)

        self._write_search_analytics_report(
            completed_at=datetime.datetime.now().isoformat(),
            final_status="answer_error",
            error=error_msg,
            **error_context,
            final_results=self._analytics_results(getattr(self, "all_scored_notes", []), limit=100),
        )
        if hasattr(self, "_notify_long_job_done"):
            self._notify_long_job_done("Ask Notes failed", "Check the conversation for details", kind="error")


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



        save_search_history(
            getattr(self, '_notes_answer_query', None) or getattr(self, 'current_query', ''),
            answer,
            relevant_note_ids,
            scored_notes,
            context_note_ids,
            context_note_id_and_chunk=getattr(self, '_context_note_id_and_chunk', None),
            context_note_identity_keys=getattr(self, '_context_note_identity_keys', None),
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
        )



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



        self._display_answer_and_notes_after_rerank(answer, config, used_history, notes)


    def ask_ai(self, query, notes, context, config, chunk_callback=None):



        provider = config.get('provider', 'openai')



        api_key = config.get('api_key', '')



        model = self.get_best_model(provider)







        num_notes = len(notes)



        focus_block = ""
        review_context = (getattr(self, "_review_context_text", "") or "").strip()
        if review_context:
            focus_block = (
                "\n\nCurrent review note guidance: The first numbered note may be the active review note. "
                "When the question says this card, this note, or this answer, interpret it using that current review note first."
            )
        recent_chat = []
        for message in (getattr(self, "_chat_history", []) or [])[-6:]:
            role = message.get("role")
            content = (message.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                recent_chat.append(f"{role.upper()}: {content}")
        conversation_block = (
            "\n\nRecent conversation context for interpreting follow-up wording only:\n"
            + "\n".join(recent_chat)
            if recent_chat else ""
        )
        selected_block = self._selected_answer_context_block() if hasattr(self, "_selected_answer_context_block") else ""



        prompt = f"""You are an assistant for question-answering over provided notes, speaking to a medical doctor. Use the numbered notes as the primary factual source, and keep the answer grounded in them. You may add brief outside context only when it helps make a complete model answer.



If the notes contain at least some relevant information, start with a short "Direct answer:" paragraph, then give the best answer you can in one flowing response. Add a short Side note about important information that is missing or only supplied by outside context.



Only if the notes are essentially unrelated to the question, say exactly: "The provided notes do not contain enough information to answer this."







Context information is below. There are exactly {num_notes} notes: Note 1 = highest relevance, Note 2 = second, ... Note {num_notes} = last. Cite ONLY using numbers from 1 to {num_notes} (e.g. [1], [2], [1,3]). Do not use numbers outside 1\xe2\u20ac\u201c{num_notes}.



---------------------



{context}



---------------------



Given the context information, answer the question. Use citations only where the answer is directly supported by the numbered notes.







Question: {query}{focus_block}{conversation_block}{selected_block}







Rules:



- Start with "Direct answer:" followed by 1-3 concise sentences. If those sentences use note-supported facts, keep inline citations in that paragraph.

- Keep the main answer as one integrated response. Do not split it into separate "from notes" and "outside knowledge" sections.

- For claims explicitly supported by the notes, place the citation immediately after the supported sentence or clause.

- You may include limited outside context without a citation when needed for clarity, but do not make it look note-supported. Mention any important outside-added or missing information briefly in the Side note.

- Do **not** invent mechanisms, receptor types, dosages, diagnostic criteria, or risk factors that are neither supported by the notes nor clearly needed as minimal outside context. Every note-supported factual claim must keep an inline citation immediately after the supported sentence, clause, bullet, or table cell. One sentence or bullet per idea is fine.



- Default to a direct, concise clinical style for a physician reader. Avoid introductory filler, over-explaining basic medical concepts, and chatty commentary.

- Write in a clear, exam-oriented style: use short Markdown headings such as "Key points", "Details", or "Table" only when helpful. Prefer 6-8 main bullets maximum and at most one sub-bullet level. Prefer short labeled bullets over deep nesting. For classification, comparison, staging, criteria, treatment, or differential questions, prefer a compact Markdown table instead of deeply nested bullets. Reserve **double asterisks** for final answers, diagnoses, drugs, criteria, and section labels; do not bold every medical noun.

- Use a compact Markdown table when comparing diagnoses, criteria, lab findings, causes, treatments, or stepwise tests. Keep citations inside the table cells when the table contains factual claims.

- Do not use LaTeX/math markup. Write symbols plainly, for example beta-glucuronidase, ↓, ↑, and →.



- When the question asks about **receptors, mechanisms, pathways, or numbered lists (e.g. 1st\xe2\u20ac\u201c6th diseases, steps 1\xe2\u20ac\u201c6)**, present them in a clean ordered list and attach citations for each item.



- NUMBERED LISTS: When the question or notes refer to a numbered list (e.g. six childhood exanthems, 1st\xe2\u20ac\u201c6th disease, steps 1\xe2\u20ac\u201c6 of ventricular partitioning), **always** present in **strict numerical order**: 1st, then 2nd, then 3rd, then 4th, then 5th, then 6th. **Include every step or number that any note describes**\u2014if Note X explicitly mentions "3rd step" or "third step", you MUST include it in your answer and cite it [X]. Do not skip steps; do not reorder by relevance.



- INLINE CITATIONS: Cite only claims supported by the notes. Use inline citations [N] or [N,M] where N is between 1 and {num_notes} only. Example: "Hypertension increases stroke risk [1,3]." Do not cite outside context. Do not rely only on the final RELEVANT_NOTES line. Do not use citation numbers outside 1\xe2\u20ac\u201c{num_notes}. Never remove citations to make the answer look cleaner.

- Before the final RELEVANT_NOTES line, include a short "Side note:" paragraph. Briefly name what the notes do not explicitly include, or say "No major missing information from the provided notes." if the notes are sufficient.



- Respond in the same language as the query.

- Write the answer in normal Markdown prose. Do not output JSON or any structured format.

- End with exactly one plain-text line: RELEVANT_NOTES: 1,3,5"""







        # Estimate input tokens



        input_tokens = estimate_tokens(prompt)
        if provider != "anthropic":
            self._analytics_stage(
                "answer_prompt_built",
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                **make_prompt_payload(prompt=prompt, mode=self._analytics_mode()),
            )







        if provider == "ollama":



            sc = config.get('search_config') or {}



            base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()



            model = (sc.get('ollama_chat_model') or 'llama3.2').strip()



            if chunk_callback:
                answer, relevant_indices = self.call_ollama_stream(prompt, base_url, model, notes, chunk_callback)
            else:
                answer, relevant_indices = self.call_ollama(prompt, base_url, model, notes)



        elif provider == "anthropic":



            system_blocks, user_content = _build_anthropic_prompt_parts(query, context)
            self._analytics_stage(
                "answer_prompt_built",
                provider=provider,
                model=model,
                input_tokens=estimate_tokens(json.dumps(user_content, ensure_ascii=False)),
                **make_prompt_payload(
                    prompt_parts={"system_blocks": system_blocks, "user_content": user_content},
                    mode=self._analytics_mode(),
                ),
            )



            answer, relevant_indices = self.call_anthropic(



                api_key=api_key, model=model, notes=notes,



                system_blocks=system_blocks, user_content=user_content



            )



        elif provider == "openai":



            if chunk_callback:
                answer, relevant_indices = self.call_openai_stream(prompt, api_key, model, notes, chunk_callback)
            else:
                answer, relevant_indices = self.call_openai(prompt, api_key, model, notes)



        elif provider == "google":



            answer, relevant_indices = self.call_google(prompt, api_key, model, notes)



        elif provider == "openrouter":



            if chunk_callback:
                answer, relevant_indices = self.call_openrouter_stream(prompt, api_key, model, notes, chunk_callback)
            else:
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
                sc.get('answer_local_model')
                or sc.get('local_llm_model')
                or config.get('local_llm_model')
                or model
                or 'local-model'
            ).strip()



            api_url = self._openai_compatible_chat_url(base_url)



            local_context_plan = self._local_context_usage_plan(
                query, len(notes), provider, sc
            ) or {}



            if chunk_callback:
                answer, relevant_indices = self.call_custom_stream(
                    prompt,
                    "",
                    local_model,
                    api_url,
                    notes,
                    timeout_seconds=300,
                    max_tokens=local_context_plan.get('max_output_tokens', 4096),
                    chunk_callback=chunk_callback,
                )
            else:
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
