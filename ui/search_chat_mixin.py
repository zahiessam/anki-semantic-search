"""Chat transcript helpers for the search dialog."""

import html
import re

from aqt.qt import QApplication, QMimeData, QTextCursor, QTimer
from aqt.utils import tooltip

from .answer_formatting import format_direct_ai_answer_html
from .branding import CHATBOT_NAME
from .theme import get_addon_theme


class SearchChatMixin:
    """Owns session chat state, transcript rendering, and chat actions."""

    def _init_search_chat_state(self):
        self._chat_messages = []
        self._chat_history = []
        self._chat_transient_html = ""
        self._pending_chat_question = ""
        self._active_chat_mode = None
        self._chat_message_seq = 0
        try:
            from ..core.memory import SessionMemory

            self._agentic_session_memory = SessionMemory()
        except Exception:
            self._agentic_session_memory = None

    def _next_chat_message_id(self):
        if not hasattr(self, "_chat_message_seq"):
            self._chat_message_seq = 0
        self._chat_message_seq += 1
        return f"m{self._chat_message_seq}"

    def _notes_source_snapshot(self):
        return {
            "all_scored_notes": list(getattr(self, "all_scored_notes", []) or []),
            "display_scored_notes": list(getattr(self, "_display_scored_notes", []) or []),
            "context_note_ids": list(getattr(self, "_context_note_ids", []) or []),
            "context_note_id_and_chunk": list(getattr(self, "_context_note_id_and_chunk", []) or []),
            "context_note_identity_keys": list(getattr(self, "_context_note_identity_keys", []) or []),
            "cited_note_ids": list(getattr(self, "_cited_note_ids", set()) or []),
            "cited_refs": list(getattr(self, "_cited_refs", set()) or []),
        }

    def _chat_message_by_id(self, message_id):
        for message in getattr(self, "_chat_messages", []) or []:
            if str(message.get("message_id") or "") == str(message_id or ""):
                return message
        return None

    def _restore_source_snapshot_for_message(self, message):
        snapshot = (message or {}).get("source_snapshot") or {}
        scored = snapshot.get("all_scored_notes")
        if not scored:
            tooltip("Source notes for this older answer are not available.")
            return False

        self.all_scored_notes = list(scored)
        display_scored = snapshot.get("display_scored_notes") or []
        self._display_scored_notes = list(display_scored) if display_scored else None
        self._context_note_ids = list(snapshot.get("context_note_ids") or [])
        self._context_note_id_and_chunk = list(snapshot.get("context_note_id_and_chunk") or [])
        self._context_note_identity_keys = list(snapshot.get("context_note_identity_keys") or [])
        self._cited_note_ids = set(snapshot.get("cited_note_ids") or [])
        self._cited_refs = set(snapshot.get("cited_refs") or [])
        self._sources_rank_mode = False
        self._show_all_dynamic_results = False
        if hasattr(self, "filter_and_display_notes"):
            self.filter_and_display_notes()
        return True

    def _append_chat_message(self, role, content, mode="system", **metadata):
        if not hasattr(self, "_chat_messages"):
            self._init_search_chat_state()
        message = {"role": role, "mode": mode, "content": content or "", "message_id": self._next_chat_message_id()}
        message.update(metadata)
        if role == "assistant" and mode == "notes" and "source_snapshot" not in message:
            message["source_snapshot"] = self._notes_source_snapshot()
        self._chat_messages.append(message)
        if role in ("user", "assistant"):
            self._chat_history.append({"role": role, "content": content or "", "mode": mode})
            self._chat_history = self._chat_history[-6:]
        self._render_chat_transcript()

    def _append_user_chat_message(self, query, mode, image_payloads=None):
        self._pending_chat_question = query or ""
        self._active_chat_mode = mode
        reply_context = (getattr(self, "_pending_selected_answer_context", "") or "").strip()
        metadata = {"reply_context": reply_context} if reply_context else {}
        if image_payloads:
            metadata["image_payloads"] = [dict(payload) for payload in image_payloads if payload]
        self._append_chat_message("user", query, mode=mode, **metadata)

    def _append_assistant_chat_message(self, answer, mode, **metadata):
        self._active_chat_mode = None
        self._append_chat_message("assistant", answer, mode=mode, **metadata)
        if hasattr(self, "copy_answer_btn"):
            self.copy_answer_btn.setEnabled(True)

    def _append_system_chat_message(self, text, kind="info"):
        self._active_chat_mode = None
        self._append_chat_message("system", text, mode="system", kind=kind)

    def _set_chat_transient(self, text=None, mode="system"):
        if text:
            self._chat_transient_html = self._render_system_row(text, kind=mode)
        else:
            self._chat_transient_html = ""
        self._render_chat_transcript()

    def _clear_search_chat(self):
        self._init_search_chat_state()
        if hasattr(self, "_clear_composer_image_attachment"):
            self._clear_composer_image_attachment()
        self._last_formatted_answer = None
        self.current_answer = ""
        if hasattr(self, "copy_answer_btn"):
            self.copy_answer_btn.setEnabled(False)
        if hasattr(self, "status_label"):
            self.status_label.setText("Ready")
        self._render_chat_transcript()

    def _render_chat_transcript(self):
        if not hasattr(self, "answer_box"):
            return
        if not hasattr(self, "_chat_messages"):
            self._init_search_chat_state()
        theme = getattr(self, "_theme", get_addon_theme())
        font_size = getattr(self, "_answer_font_size", 13)
        parts = [f'<div style="font-size: {font_size}px; line-height: 1.36; padding: 4px 4px 10px;">']
        if not self._chat_messages and not self._chat_transient_html:
            parts.append(
                f'<div style="color: {theme["quiet_text"]}; padding: 34px 12px; text-align: center;">'
                "Ask AI for direct reasoning, or Ask Notes to search your Anki collection with citations."
                "</div>"
            )
        for message in self._chat_messages:
            role = message.get("role")
            if role == "user":
                parts.append(self._render_user_chat_bubble(message))
            elif role == "assistant":
                parts.append(self._render_assistant_chat_bubble(message))
            else:
                parts.append(self._render_system_row(message.get("content") or "", message.get("kind") or "info"))
        if self._chat_transient_html:
            parts.append(self._chat_transient_html)
        parts.append("</div>")
        self.answer_box.setHtml("".join(parts))
        try:
            cursor = self.answer_box.textCursor()
            cursor.clearSelection()
            self.answer_box.setTextCursor(cursor)
        except Exception:
            pass
        QTimer.singleShot(0, self._scroll_search_chat_to_bottom)

    def _render_user_chat_bubble(self, message):
        theme = getattr(self, "_theme", get_addon_theme())
        content = html.escape(message.get("content") or "").replace("\n", "<br>")
        image_html = self._render_user_image_preview(message.get("image_payloads") or [])
        reply_context = self._reply_context_preview(message.get("reply_context") or "")
        safe_reply_context = html.escape(reply_context).replace("\n", "<br>")
        reply_html = ""
        if reply_context:
            reply_html = (
                f'<div style="color: {theme["accent_text"]}; opacity: 0.68; '
                'font-size: 11px; line-height: 1.28; margin-bottom: 6px; '
                'padding-left: 8px; border-left: 2px solid rgba(255,255,255,0.45);">'
                '<span style="font-weight: 700;">Replying to</span><br>'
                f'{safe_reply_context}</div>'
            )
        return (
            '<table width="100%" cellspacing="0" cellpadding="0" style="margin: 7px 0;">'
            '<tr><td align="right">'
            '<table width="88%" cellspacing="0" cellpadding="0" align="right">'
            '<tr><td style="text-align: left; '
            f'background-color: {theme["accent"]}; color: {theme["accent_text"]}; '
            'border-radius: 16px 16px 4px 16px; padding: 9px 13px;">'
            f"{reply_html}{image_html}{content}</td></tr></table>"
            "</td></tr></table>"
        )

    def _render_user_image_preview(self, image_payloads):
        if not image_payloads:
            return ""
        image = image_payloads[0] or {}
        mime_type = html.escape(image.get("mime_type") or "image/png")
        data = html.escape(image.get("base64") or "")
        filename = html.escape(image.get("filename") or "Attached image")
        if not data:
            return ""
        return (
            '<div style="margin-bottom: 7px;">'
            f'<img src="data:{mime_type};base64,{data}" '
            'width="120" style="max-width: 120px; max-height: 120px; '
            'border-radius: 6px; border: 1px solid rgba(255,255,255,0.45); '
            'display: block; object-fit: contain;" />'
            f'<div style="font-size: 10px; opacity: 0.72; padding-top: 2px;">{filename}</div>'
            '</div>'
        )

    def _reply_context_preview(self, text, max_chars=280):
        preview = re.sub(r"\s+", " ", text or "").strip()
        if len(preview) > max_chars:
            preview = preview[:max_chars].rstrip() + "..."
        return preview

    def _render_assistant_chat_bubble(self, message):
        theme = getattr(self, "_theme", get_addon_theme())
        mode = message.get("mode") or "ai"
        is_notes = mode == "notes"
        accent = theme["teal"] if is_notes else theme["accent"]
        label = "Ask Notes answer" if is_notes else "Ask AI answer"
        subtitle = "Searched Anki notes" if is_notes else ""
        content = message.get("content") or ""
        if is_notes:
            body = self.format_answer(
                content,
                context_note_ids=message.get("context_note_ids") or [],
                citation_scope=message.get("message_id"),
            )
        else:
            body = self._format_direct_ai_chat_html(content)
        source_text = html.escape(message.get("source_text") or "")
        source_footer = (
            f'<div style="color: {theme["quiet_text"]}; font-size: 10px; '
            'font-style: italic; padding-top: 6px; margin-top: 6px; '
            'background-color: transparent; opacity: 0.86; '
            f'border-top: 1px solid {theme["subtle_border"]};">'
            f'Answer from: {source_text}</div>'
            if source_text else ""
        )
        return (
            '<table width="98%" cellspacing="0" cellpadding="0" style="margin: 8px 0 11px;">'
            '<tr><td valign="top">'
            '<table width="100%" cellspacing="0" cellpadding="0">'
            f'<tr><td style="color: {accent}; font-size: 11px; font-weight: 700; padding: 0 0 4px 2px;">'
            f'{html.escape(label)}'
            + (
                f' <span style="color: {theme["quiet_text"]}; font-weight: 500;">'
                f'{html.escape(subtitle)}</span>'
                if subtitle else ""
            )
            + '</td></tr>'
            '<tr><td style="text-align: left; '
            f'background-color: {theme["control_bg"]}; color: {theme["input_text"]}; '
            f'border: 1px solid {theme["subtle_border"]}; border-left: 3px solid {accent}; '
            'border-radius: 8px; padding: 10px 12px;">'
            f"{body}{source_footer}</td></tr></table>"
            '</td></tr></table>'
        )

    def _render_system_row(self, text, kind="info"):
        theme = getattr(self, "_theme", get_addon_theme())
        color = theme["danger"] if kind == "error" else theme["warning"] if kind == "warning" else theme["quiet_text"]
        safe = html.escape(text or "").replace("\n", "<br>")
        return (
            f'<div style="color: {color}; text-align: center; font-size: 12px; '
            f'padding: 7px 10px; margin: 6px 0;">{safe}</div>'
        )

    def _format_direct_ai_chat_html_legacy(self, content):
        safe = html.escape(content or "")
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        lines = safe.splitlines()
        parts = []
        in_list = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_list:
                    parts.append("</ul>")
                    in_list = False
                continue
            if stripped.startswith(("- ", "* ", "• ")):
                if not in_list:
                    parts.append('<ul style="margin: 0.35em 0 0.5em 1.1em; padding-left: 1.1em;">')
                    in_list = True
                parts.append(f'<li style="margin: 0.2em 0;">{stripped[2:].strip()}</li>')
            else:
                if in_list:
                    parts.append("</ul>")
                    in_list = False
                parts.append(f'<p style="margin: 0.35em 0;">{stripped}</p>')
        if in_list:
            parts.append("</ul>")
        return "".join(parts) or "&nbsp;"

    def _format_direct_ai_chat_html(self, content):
        spacing = getattr(self, "styling_config", {}).get("answer_spacing", "normal")
        return format_direct_ai_answer_html(content, spacing) or "&nbsp;"

    def _scroll_search_chat_to_bottom(self):
        try:
            bar = self.answer_box.verticalScrollBar()
            if bar:
                bar.setValue(bar.maximum())
        except Exception:
            pass

    def _latest_assistant_message(self):
        for message in reversed(getattr(self, "_chat_messages", []) or []):
            if message.get("role") == "assistant":
                return message
        return None

    def _latest_user_question(self):
        for message in reversed(getattr(self, "_chat_history", []) or []):
            if message.get("role") == "user":
                content = (message.get("content") or "").strip()
                if content:
                    return content
        return ""

    def _selected_answer_text_context(self, max_chars=1800):
        try:
            cursor = self.answer_box.textCursor() if hasattr(self, "answer_box") else None
            if not cursor or not cursor.hasSelection():
                return ""
            text = (cursor.selectedText() or "").replace("\u2029", "\n").replace("\u2028", "\n")
            return self._normalize_selected_answer_context(text, max_chars=max_chars)
        except Exception:
            return ""

    def _normalize_selected_answer_context(self, text, max_chars=1800):
        text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n..."
        return text

    def _composer_selected_answer_context(self):
        text = (getattr(self, "_composer_selected_answer_context_text", "") or "").strip()
        if text:
            return text
        return self._selected_answer_text_context() if hasattr(self, "_selected_answer_text_context") else ""

    def _strip_selected_answer_snippet_from_query(self, query):
        return (query or "").strip()

    def _selection_chip_preview(self, selected_text, max_chars=180):
        preview = re.sub(r"\s+", " ", selected_text or "").strip()
        if len(preview) > max_chars:
            preview = preview[:max_chars].rstrip() + "..."
        return preview

    def _update_selected_answer_context_chip(self):
        chip = getattr(self, "selected_answer_context_chip", None)
        label = getattr(self, "selected_answer_context_preview_label", None)
        text = (getattr(self, "_composer_selected_answer_context_text", "") or "").strip()
        if not chip or not label:
            return
        if not text:
            chip.hide()
            label.setText("")
            label.setToolTip("")
            return
        label.setText(self._selection_chip_preview(text))
        label.setToolTip(text)
        chip.show()

    def _clear_composer_selected_answer_context(self):
        self._composer_selected_answer_context_text = ""
        self._update_selected_answer_context_chip()
        if hasattr(self, "search_input"):
            self.search_input.setFocus()
        tooltip("Removed selection")

    def _reply_about_selected_answer_text(self):
        selected_text = self._selected_answer_text_context() if hasattr(self, "_selected_answer_text_context") else ""
        if not selected_text:
            tooltip("Select answer text first")
            return
        self._composer_selected_answer_context_text = selected_text
        self._update_selected_answer_context_chip()
        if hasattr(self, "search_input"):
            cursor = self.search_input.textCursor()
            cursor.movePosition(
                QTextCursor.MoveOperation.End
                if hasattr(QTextCursor, "MoveOperation")
                else QTextCursor.End
            )
            self.search_input.setTextCursor(cursor)
            self.search_input.setFocus()
        tooltip("Selection attached")

    def _selected_answer_context_block(self):
        text = (getattr(self, "_pending_selected_answer_context", "") or "").strip()
        if not text:
            return ""
        return (
            "\n\nSelected answer text context:\n"
            f"{text}\n\n"
            "Use this selected text to resolve references like this, that, it, or explain this."
        )

    def _show_answer_context_menu(self, pos):
        menu = self.answer_box.createStandardContextMenu(pos)
        selected_text = self._selected_answer_text_context() if hasattr(self, "_selected_answer_text_context") else ""
        if selected_text:
            menu.addSeparator()
            action = menu.addAction("Reply about selection")
            action.triggered.connect(self._reply_about_selected_answer_text)
        menu.exec(self.answer_box.mapToGlobal(pos))

    def _ask_ai_about_selected_answer_text(self):
        self._reply_about_selected_answer_text()

    def copy_answer_to_clipboard(self):
        message = self._latest_assistant_message()
        plain = (message or {}).get("content", "").strip()
        if not plain:
            plain = self.answer_box.toPlainText().strip() if hasattr(self, "answer_box") else ""
        if not plain:
            tooltip("No answer to copy")
            return
        cb = QApplication.clipboard()
        if cb:
            mime = QMimeData()
            mime.setText(plain)
            cb.setMimeData(mime)
        tooltip("Copied latest answer")

    def _ask_notes_from_composer(self):
        if hasattr(self, "set_search_mode"):
            self.set_search_mode("ask_notes")
        self.perform_search()

    def _set_search_chat_busy(self, busy, message=None):
        for attr in ("search_btn", "ask_notes_btn", "ask_ai_btn", "find_related_btn", "attach_image_btn"):
            button = getattr(self, attr, None)
            if button is not None:
                button.setEnabled(not busy)
        if message and hasattr(self, "status_label"):
            self.status_label.setText(message)
