"""Review Ask AI context helpers and legacy popup implementation."""

import base64
import html
import json
import mimetypes
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from aqt import dialogs, mw
from aqt.qt import (
    QCursor,
    QDialog,
    QEvent,
    QFrame,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QThread,
    QTimer,
    QVBoxLayout,
    Qt,
    pyqtSignal,
)
from aqt.utils import tooltip

from .answer_formatting import format_answer_html
from .branding import CHATBOT_ICON, CHATBOT_NAME, CHATBOT_REFERENCE
from .note_preview_popup import NotePreviewPopup
from .theme import get_addon_theme
from ..utils.config import load_config
from ..utils.log import log_debug
from ..utils.text import clean_html_for_display, reveal_cloze


IMG_RE = re.compile(r"<\s*img\b[^>]*\bsrc\s*=\s*['\"]?([^'\"\s>]+)", re.IGNORECASE)
_CITATION_RE = re.compile(r"\[((?:\d+[A-Za-z]?\s*,\s*)*\d+[A-Za-z]?)\]")
_CITATION_N_RE = re.compile(r"\[N((?:\d+[A-Za-z]?\s*,\s*N?)*\d+[A-Za-z]?)\]")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_BOLD_ALT_RE = re.compile(r"__(.+?)__")
_MD_HIGHLIGHT_RE = re.compile(r"~~(.+?)~~")
_MD_HEADER_RE = re.compile(r"^(.{1,50}):(\s*)$", re.MULTILINE)
_MD_UNTERMINATED_BOLD_RE = re.compile(r"\*\*([^*]+)$")
NOTE_PREVIEW_SHOW_DELAY_MS = 1500


def _openai_compatible_chat_url(base_url):
    url = (base_url or "").strip()
    if not url:
        raise ValueError("Local server URL is empty. Set it in Settings.")
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")
    lower = url.lower()
    if lower.endswith("/chat/completions"):
        return url
    if lower.endswith("/v1"):
        return url + "/chat/completions"
    if lower.endswith("/api/chat") or lower.endswith("/api/generate") or lower.endswith("/api/tags"):
        return re.sub(r"/api/(chat|generate|tags)$", "/v1/chat/completions", url, flags=re.IGNORECASE)
    if ":11434" in lower:
        return url + "/v1/chat/completions"
    return url + "/chat/completions"


def _request_json(url, headers, data, timeout_seconds=60):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = str(exc)
        raise Exception(f"Provider returned HTTP {exc.code}: {body[:300]}")
    except urllib.error.URLError as exc:
        raise Exception(f"Provider connection failed: {getattr(exc, 'reason', exc)}")


def _extract_text_from_anthropic(result):
    parts = []
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "\n".join(part for part in parts if part).strip()


def _strip_relevant_notes(answer):
    if "RELEVANT_NOTES:" in (answer or ""):
        return answer.split("RELEVANT_NOTES:", 1)[0].strip()
    return (answer or "").strip()


def _format_review_answer(answer, context_note_ids=None):
    chunks = []
    text_buffer = []
    lines = (answer or "").splitlines()
    index = 0
    while index < len(lines):
        if (
            index + 1 < len(lines)
            and "|" in lines[index]
            and "|" in lines[index + 1]
            and _is_table_separator(lines[index + 1])
        ):
            if text_buffer:
                chunks.append(_format_review_text("\n".join(text_buffer), context_note_ids))
                text_buffer = []
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            chunks.append(_table_html(table_lines))
            continue
        text_buffer.append(lines[index])
        index += 1
    if text_buffer:
        chunks.append(_format_review_text("\n".join(text_buffer), context_note_ids))
    return "".join(chunks)


def _format_review_text(answer, context_note_ids=None):
    formatted = format_answer_html(
        answer,
        context_note_ids or [],
        "normal",
        {
            "citation_n": _CITATION_N_RE,
            "citation": _CITATION_RE,
            "md_bold": _MD_BOLD_RE,
            "md_unterminated_bold": _MD_UNTERMINATED_BOLD_RE,
            "md_bold_alt": _MD_BOLD_ALT_RE,
            "md_header": _MD_HEADER_RE,
            "md_highlight": _MD_HIGHLIGHT_RE,
        },
    )
    formatted = _format_ai_generated_citations(formatted)
    return _format_review_note_citations(formatted)


def _format_ai_generated_citations(formatted_html):
    return re.sub(
        r"\[(?:AI\s+Chatbot|ai\s+chatbot|AI\s+generated|ai\s+generated|Ai\s+Generated|AI|ai)\]",
        (
            '<span style="display: inline-block; color: #bfdbfe; '
            'background: rgba(37, 99, 235, 0.18); border: 1px solid rgba(96, 165, 250, 0.45); '
            f'border-radius: 6px; padding: 1px 5px; font-size: 10px; font-weight: 600;" '
            f'title="{html.escape(CHATBOT_REFERENCE)}">{html.escape(CHATBOT_ICON)}</span>'
        ),
        formatted_html or "",
    )


def _format_review_note_citations(formatted_html):
    return re.sub(
        r'<a href="cite:(\d+)"[^>]*>\[[^\]]+\]</a>',
        (
            r'<a href="cite:\1" style="display: inline-block; color: #bfdbfe !important; '
            r'background: rgba(37, 99, 235, 0.18); border: 1px solid rgba(96, 165, 250, 0.45); '
            r'border-radius: 6px; padding: 1px 5px; font-size: 10px; font-weight: 700; '
            r'text-decoration: none !important;" title="Open source note in Anki Browser">Note</a>'
        ),
        formatted_html or "",
    )


def _is_table_separator(line):
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def _render_pipe_tables(answer):
    lines = (answer or "").splitlines()
    out = []
    index = 0
    while index < len(lines):
        if (
            index + 1 < len(lines)
            and "|" in lines[index]
            and "|" in lines[index + 1]
            and _is_table_separator(lines[index + 1])
        ):
            header = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
            rows = []
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                rows.append(cells)
                index += 1
            table = [
                '<table style="border-collapse: collapse; margin: 0.45em 0; width: 100%;">',
                "<thead><tr>",
            ]
            for cell in header:
                table.append(
                    '<th style="border: 1px solid #6b7280; padding: 4px 6px; text-align: left;">'
                    + html.escape(cell)
                    + "</th>"
                )
            table.append("</tr></thead><tbody>")
            for row in rows:
                table.append("<tr>")
                for idx in range(len(header)):
                    cell = row[idx] if idx < len(row) else ""
                    table.append(
                        '<td style="border: 1px solid #6b7280; padding: 4px 6px;">'
                        + html.escape(cell)
                        + "</td>"
                    )
                table.append("</tr>")
            table.append("</tbody></table>")
            out.append("".join(table))
            continue
        out.append(lines[index])
        index += 1
    return "\n".join(out)


def _table_html(table_lines):
    if len(table_lines) < 2:
        return html.escape("\n".join(table_lines))
    header = [cell.strip() for cell in table_lines[0].strip().strip("|").split("|")]
    body_lines = table_lines[2:]
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in body_lines
    ]
    table = [
        '<table style="border-collapse: collapse; margin: 0.5em 0; width: 100%; font-size: 13px;">',
        "<thead><tr>",
    ]
    for cell in header:
        table.append(
            '<th style="border: 1px solid #6b7280; padding: 5px 7px; text-align: left; font-weight: 600;">'
            + html.escape(cell)
            + "</th>"
        )
    table.append("</tr></thead><tbody>")
    for row in rows:
        table.append("<tr>")
        for idx in range(len(header)):
            cell = row[idx] if idx < len(row) else ""
            table.append(
                '<td style="border: 1px solid #6b7280; padding: 5px 7px;">'
                + html.escape(cell)
                + "</td>"
            )
        table.append("</tr>")
    table.append("</tbody></table>")
    return "".join(table)


def _field_names_for_note(note):
    try:
        model = note.note_type()
        return [field.get("name", "") for field in (model.get("flds") or [])]
    except Exception:
        return []


def _note_type_name_for_note(note):
    try:
        model = note.note_type()
        return model.get("name", "") if model else ""
    except Exception:
        try:
            model = note.model()
            return model.get("name", "") if model else ""
        except Exception:
            return ""


def _review_context_field_names(note, config=None):
    field_names = _field_names_for_note(note)
    if not field_names:
        return []
    config = (config or load_config()) or {}
    review_cfg = config.get("review_ask_ai") or {}

    if review_cfg.get("context_source", "embedding_fields") == "custom_fields":
        custom_cfg = dict(review_cfg)
        custom_cfg["use_first_field_fallback"] = False
        fields = _field_names_from_filter_config(note, field_names, custom_cfg)
        if fields:
            return fields
        fields = _field_names_from_filter_config(
            note,
            field_names,
            config.get("note_type_filter") or {},
        )
        if fields:
            return fields
        if review_cfg.get("use_first_field_fallback", True):
            return field_names[:2]
        return []

    return _field_names_from_filter_config(
        note,
        field_names,
        config.get("note_type_filter") or {},
    )


def _field_names_from_filter_config(note, field_names, filter_cfg):
    if filter_cfg.get("search_all_fields"):
        return field_names
    note_type_name = _note_type_name_for_note(note)
    selected = (filter_cfg.get("note_type_fields") or {}).get(note_type_name) or []
    selected_lookup = {str(name).strip().lower() for name in selected if str(name or "").strip()}
    fields = [name for name in field_names if name.strip().lower() in selected_lookup]
    if not fields and filter_cfg.get("use_first_field_fallback", True):
        fields = field_names[:2]
    return fields


def _extract_image_refs(raw_html):
    refs = []
    for src in IMG_RE.findall(raw_html or ""):
        src = html.unescape(urllib.parse.unquote(src)).strip()
        if not src:
            continue
        refs.append(os.path.basename(src))
    return refs


def _detect_image_mime(data, path=None):
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    guessed = mimetypes.guess_type(path or "")[0] or "image/png"
    return guessed if guessed.startswith("image/") else "image/png"


def _image_payloads_for_refs(image_refs):
    payloads = []
    if not image_refs or not mw or not getattr(mw, "col", None):
        return payloads
    try:
        media_dir = mw.col.media.dir()
    except Exception:
        return payloads
    for filename in sorted(set(image_refs)):
        if not filename:
            continue
        path = os.path.abspath(os.path.join(media_dir, filename))
        try:
            if not path.startswith(os.path.abspath(media_dir) + os.sep):
                continue
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as image_file:
                raw_data = image_file.read()
            mime_type = _detect_image_mime(raw_data, path)
            if not mime_type.startswith("image/"):
                continue
            data = base64.b64encode(raw_data).decode("ascii")
            payloads.append({
                "filename": filename,
                "mime_type": mime_type,
                "base64": data,
            })
        except Exception as exc:
            log_debug(f"Could not attach review image {filename}: {exc}")
    return payloads


def extract_review_note_context(note, config=None):
    """Return text/image context from the configured Review Ask AI fields."""
    field_names = _review_context_field_names(note, config)
    fields = []
    image_refs = []
    for index, name in enumerate(field_names):
        try:
            raw_value = note[name]
        except Exception:
            raw_value = ""
        field_image_refs = _extract_image_refs(raw_value)
        image_refs.extend(field_image_refs)
        display_text = clean_html_for_display(reveal_cloze(raw_value)).strip()
        fields.append(
            {
                "index": index,
                "name": name or f"Field {index + 1}",
                "text": display_text,
                "images": sorted(set(field_image_refs)),
            }
        )

    context_lines = []
    for field in fields:
        if field["text"]:
            context_lines.append(f"{field['name']}: {field['text']}")
        elif field.get("images"):
            context_lines.append(f"{field['name']}: [image attached: {', '.join(field['images'])}]")
    if image_refs:
        unique_images = sorted(set(image_refs))
        context_lines.append("Images present: " + ", ".join(unique_images))

    return {
        "fields": fields,
        "images": sorted(set(image_refs)),
        "image_payloads": _image_payloads_for_refs(image_refs),
        "text": "\n".join(context_lines).strip(),
    }


class ReviewAskAIWorker(QThread):
    success_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    warning_signal = pyqtSignal(str)

    def __init__(self, question, context_text, config, chat_history=None, image_payloads=None):
        super().__init__()
        self.question = question
        self.context_text = context_text
        self.config = config or {}
        self.chat_history = list(chat_history or [])
        self.image_payloads = list(image_payloads or [])
        self._retried_without_images = False

    def run(self):
        try:
            answer = self._ask_provider()
            self.success_signal.emit(_strip_relevant_notes(answer))
        except Exception as exc:
            log_debug(f"Review {CHATBOT_NAME} worker error: {exc}", is_error=True)
            self.error_signal.emit(str(exc))

    def _build_prompt(self):
        history_text = self._format_chat_history()
        history_block = (
            "\nRecent conversation for resolving follow-up references:\n"
            "---------------------\n"
            f"{history_text}\n"
            "---------------------\n"
            if history_text
            else ""
        )
        source_rule = """Use the current note context first when it helps answer the question.
If the note context is incomplete or not relevant, answer directly using your general reasoning and knowledge.
Use [1] only for facts directly supported by the current note context.
Do not cite or label general reasoning unless the user explicitly asks for a source breakdown."""

        return f"""You are an assistant helping with the current Anki review note.
{source_rule}
Use the recent conversation only to understand follow-up references such as "that", "why not", or "the other diagnosis".
If the recent conversation conflicts with the current note context, trust the current note context.

Current note context:
---------------------
{self.context_text or f"(No readable text found in the configured {CHATBOT_NAME} fields.)"}
---------------------
{history_block}
Question: {self.question}

Answer clearly and concisely. If images are attached, inspect them directly and incorporate relevant visual details. Source visual observations as [1]."""

    def _format_chat_history(self):
        lines = []
        for turn in self.chat_history[-4:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = str(turn.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _openai_user_content(self, prompt):
        if not self.image_payloads:
            return prompt
        content = [{"type": "text", "text": prompt}]
        for image in self.image_payloads:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image['mime_type']};base64,{image['base64']}",
                },
            })
        return content

    def _anthropic_user_content(self, prompt):
        content = [{"type": "text", "text": prompt}]
        for image in self.image_payloads:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image["mime_type"],
                    "data": image["base64"],
                },
            })
        return content

    def _google_parts(self, prompt):
        parts = [{"text": prompt}]
        for image in self.image_payloads:
            parts.append({
                "inline_data": {
                    "mime_type": image["mime_type"],
                    "data": image["base64"],
                },
            })
        return parts

    def _provider(self):
        return (self.config.get("provider") or "openai").strip().lower()

    def _is_image_support_error(self, error_message):
        """Check if an error is related to image input not being supported."""
        error_str = str(error_message).lower()
        image_error_patterns = [
            "does not support image input",
            "image input not supported",
            "does not support images",
            "images not supported",
            "multimodal",
            "vision",
        ]
        return any(pattern in error_str for pattern in image_error_patterns)

    def _ask_provider(self):
        provider = self._provider()
        prompt = self._build_prompt()
        sc = self.config.get("search_config") or {}
        api_key = self.config.get("api_key") or ""

        try:
            if provider == "ollama":
                base_url = (sc.get("ollama_base_url") or "http://localhost:11434").rstrip("/")
                model = (sc.get("ollama_chat_model") or "llama3.2").strip()
                result = _request_json(
                    base_url + "/api/generate",
                    {"Content-Type": "application/json"},
                    {
                        "model": model,
                        "prompt": prompt,
                        "images": [image["base64"] for image in self.image_payloads],
                        "stream": False,
                        "options": {"num_predict": 2048},
                    },
                    timeout_seconds=120,
                )
                return result.get("response") or (result.get("message") or {}).get("content") or ""

            if provider == "anthropic":
                model = self.config.get("anthropic_model") or "claude-sonnet-4-6"
                result = _request_json(
                    "https://api.anthropic.com/v1/messages",
                    {
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    {
                        "model": model,
                        "max_tokens": 2048,
                        "system": "Answer questions using only the supplied Anki note context.",
                        "messages": [{"role": "user", "content": self._anthropic_user_content(prompt)}],
                    },
                )
                return _extract_text_from_anthropic(result)

            if provider == "google":
                model = self.config.get("google_model") or "gemini-1.5-flash"
                result = _request_json(
                    f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}",
                    {"Content-Type": "application/json"},
                    {"contents": [{"parts": self._google_parts(prompt)}], "generationConfig": {"maxOutputTokens": 2048}},
                )
                return ((result.get("candidates") or [{}])[0].get("content") or {}).get("parts", [{}])[0].get("text", "")

            if provider == "openrouter":
                model = self.config.get("openrouter_model") or "google/gemini-flash-1.5"
                url = "https://openrouter.ai/api/v1/chat/completions"
                key = api_key
            elif provider in ("local_openai", "local_server"):
                base_url = sc.get("local_llm_url") or self.config.get("local_llm_url") or "http://localhost:1234/v1"
                url = _openai_compatible_chat_url(base_url)
                model = sc.get("answer_local_model") or sc.get("local_llm_model") or "local-model"
                key = ""
            else:
                url = self.config.get("api_url") or "https://api.openai.com/v1/chat/completions"
                model = self.config.get("openai_chat_model") or "gpt-4o-mini"
                key = api_key

            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            result = _request_json(
                url,
                headers,
                {"model": model, "messages": [{"role": "user", "content": self._openai_user_content(prompt)}], "max_tokens": 2048},
                timeout_seconds=120 if provider in ("local_openai", "local_server") else 60,
            )
            return ((result.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        except Exception as exc:
            # If this is an image support error and we haven't retried yet, retry without images
            if self._is_image_support_error(exc) and not self._retried_without_images and self.image_payloads:
                self._retried_without_images = True
                self.image_payloads = []
                self.warning_signal.emit(
                    f"Note: The selected model does not support images. "
                    f"Answering based on text content only."
                )
                return self._ask_provider()
            # Otherwise, re-raise the exception
            raise


class AskAIReviewDialog(QDialog):
    def __init__(self, card, parent=None):
        super().__init__(parent or mw)
        self.card = card
        self.note = card.note()
        self.note_id = self.note.id
        self.config = load_config()
        self.context = extract_review_note_context(self.note, self.config)
        self._ask_worker = None
        self._request_id = 0
        self._chat_history = []
        self._chat_messages = []
        self._pending_question = ""
        self._transient_chat_html = ""
        self._setup_ui()
        self._review_card_timer = QTimer(self)
        self._review_card_timer.timeout.connect(self._sync_with_current_reviewer_card)
        self._review_card_timer.start(700)

    def _setup_ui(self):
        theme = get_addon_theme()
        self._theme = theme
        styling = (self.config or {}).get("styling") or {}
        self._question_font_size = max(10, min(24, int(styling.get("question_font_size", 13) or 13)))
        self._answer_font_size = max(10, min(24, int(styling.get("answer_font_size", 14) or 14)))
        self.setWindowTitle(CHATBOT_NAME)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(540, 580)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {theme['bg']}; }}
            QLabel {{ color: {theme['text']}; }}
            QPlainTextEdit#reviewQuestionInput {{
                background-color: {theme['control_bg']};
                color: {theme['input_text']};
                border: 1px solid {theme['control_border']};
                border-radius: 6px;
                padding: 6px;
                font-size: {self._question_font_size}px;
            }}
            QTextBrowser#reviewContextBox {{
                background-color: {theme['section_bg']};
                color: {theme['input_text']};
                border: 1px solid {theme['subtle_border']};
                border-radius: 6px;
                padding: 7px;
                font-size: {max(11, self._answer_font_size - 1)}px;
            }}
            QTextBrowser#reviewAnswerBox {{
                background-color: #000000;
                color: {theme['input_text']};
                border: 1px solid {theme['subtle_border']};
                border-radius: 6px;
                padding: 9px;
                font-size: {self._answer_font_size}px;
            }}
            QPushButton {{
                background-color: {theme['muted_btn']};
                color: {theme['muted_btn_text']};
                border: 1px solid {theme['control_border']};
                border-radius: 6px;
                padding: 7px 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {theme['muted_btn_hover']}; }}
            QPushButton#primaryAsk {{ background-color: {theme['accent']}; color: {theme['accent_text']}; border: none; min-height: 36px; padding: 8px 18px; }}
            QPushButton#primaryAsk:disabled {{ background-color: {theme['control_bg']}; color: {theme['quiet_text']}; }}
            QPushButton#secondaryAction {{ background-color: transparent; color: {theme['text']}; border: 1px solid {theme['control_border']}; min-height: 30px; padding: 5px 10px; font-weight: 600; }}
            QPushButton#secondaryAction:disabled {{ color: {theme['quiet_text']}; border-color: {theme['subtle_border']}; }}
            QPushButton#clearChatAction {{ background-color: transparent; color: {theme['quiet_text']}; border: none; min-height: 28px; padding: 4px 6px; font-weight: 500; text-decoration: underline; }}
            QPushButton#clearChatAction:disabled {{ color: {theme['quiet_text']}; text-decoration: none; }}
            QPushButton#contextToggle {{ background-color: transparent; color: {theme['text']}; border: none; padding: 4px 0; text-align: left; font-weight: bold; }}
            QFrame#inputBar {{
                background-color: {theme['section_bg']};
                border: 1px solid {theme['subtle_border']};
                border-radius: 8px;
            }}
            QFrame#questionInputContainer {{
                background-color: {theme['control_bg']};
                border: 1px solid {theme['control_border']};
                border-radius: 18px;
            }}
            QFrame#questionInputContainer QPlainTextEdit#reviewQuestionInput {{
                border: none;
                background-color: transparent;
                padding: 6px 8px;
            }}
        """)

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        layout = QVBoxLayout()
        layout.setSpacing(8)
        root_layout.addLayout(layout, 3)

        self._note_preview_popup = NotePreviewPopup(self)
        self._build_header_bar(layout, theme)
        self._build_message_thread(layout, theme)
        self._build_input_bar(layout, theme)
        self._render_chat_transcript()

    def set_review_card(self, card, clear_chat=None):
        if card is None:
            return False
        try:
            note = card.note()
            note_id = note.id
        except Exception:
            return False

        same_note = str(note_id) == str(getattr(self, "note_id", ""))
        self.card = card
        if same_note:
            self.note = note
            return False

        if clear_chat is None:
            clear_chat = True
        self._cancel_active_work()
        self.note = note
        self.note_id = note_id
        self.config = load_config()
        self.context = extract_review_note_context(self.note, self.config)
        if hasattr(self, "context_box"):
            self.context_box.setHtml(self._context_preview_html())
            self.context_box.setVisible(False)
        if hasattr(self, "context_toggle_btn"):
            self.context_toggle_btn.setText("Show note context \u25be")
        if clear_chat:
            self._reset_chat_state("New note loaded. Chat cleared.")
        else:
            self._render_chat_transcript()
        return True

    def _sync_with_current_reviewer_card(self):
        try:
            reviewer = getattr(mw, "reviewer", None)
            card = getattr(reviewer, "card", None)
        except Exception:
            card = None
        if card is not None:
            self.set_review_card(card)

    def _cancel_active_work(self):
        self._request_id += 1
        self._stop_loading()
        if self._ask_worker is not None and self._ask_worker.isRunning():
            self._ask_worker.requestInterruption()
        self.ask_btn.setEnabled(True)

    def _reset_chat_state(self, system_message=None):
        self._chat_history = []
        self._chat_messages = []
        self._pending_question = ""
        self._transient_chat_html = ""
        self.clear_chat_btn.setEnabled(False)
        if system_message:
            self._append_system_message(system_message)
        else:
            self._render_chat_transcript()

    def _build_header_bar(self, layout, theme):
        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        self.context_toggle_btn = QPushButton("Show note context \u25be")
        self.context_toggle_btn.setObjectName("contextToggle")
        self.context_toggle_btn.clicked.connect(self._toggle_context_preview)
        header_row.addWidget(self.context_toggle_btn)
        header_row.addStretch()
        layout.addLayout(header_row)

        self.context_box = QTextBrowser()
        self.context_box.setObjectName("reviewContextBox")
        self.context_box.setMaximumHeight(130)
        self.context_box.setHtml(self._context_preview_html())
        self.context_box.setVisible(False)
        layout.addWidget(self.context_box)

    def _build_message_thread(self, layout, theme):
        self.answer_box = QTextBrowser()
        self.answer_box.setObjectName("reviewAnswerBox")
        self.answer_box.setMinimumHeight(260)
        self.answer_box.setOpenLinks(False)
        self.answer_box.setOpenExternalLinks(False)
        self.answer_box.setMouseTracking(True)
        self.answer_box.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu if hasattr(Qt, "ContextMenuPolicy") else Qt.CustomContextMenu
        )
        self.answer_box.customContextMenuRequested.connect(self._show_answer_context_menu)
        self.answer_box.anchorClicked.connect(self._on_answer_link_clicked)
        self.answer_box.viewport().installEventFilter(self)
        self.answer_box.viewport().installEventFilter(self._note_preview_popup)
        layout.addWidget(self.answer_box, 1)

    def _build_input_bar(self, layout, theme):
        input_bar = QFrame()
        input_bar.setObjectName("inputBar")
        input_layout = QVBoxLayout(input_bar)
        input_layout.setContentsMargins(8, 8, 8, 7)
        input_layout.setSpacing(6)

        compose_row = QHBoxLayout()
        compose_row.setSpacing(7)
        input_container = QFrame()
        input_container.setObjectName("questionInputContainer")
        input_container_layout = QHBoxLayout(input_container)
        input_container_layout.setContentsMargins(6, 2, 6, 2)

        self.question_input = QPlainTextEdit()
        self.question_input.setObjectName("reviewQuestionInput")
        self.question_input.setPlaceholderText("Ask a follow-up about this note...")
        self.question_input.setMinimumHeight(56)
        self.question_input.setMaximumHeight(82)
        self.question_input.installEventFilter(self)
        input_container_layout.addWidget(self.question_input)
        compose_row.addWidget(input_container, 1)

        self.ask_btn = QPushButton("\u21b5")
        self.ask_btn.setObjectName("primaryAsk")
        self.ask_btn.setToolTip(f"Send to {CHATBOT_NAME}")
        self.ask_btn.setFixedWidth(44)
        self.ask_btn.clicked.connect(self._ask_ai)
        compose_row.addWidget(self.ask_btn)
        input_layout.addLayout(compose_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.search_collection_btn = QPushButton("Search collection")
        self.search_collection_btn.setObjectName("secondaryAction")
        self.search_collection_btn.setToolTip("Open AI Search with this question and search your notes.")
        self.search_collection_btn.clicked.connect(self._search_collection)
        action_row.addWidget(self.search_collection_btn)
        action_row.addStretch()

        self.clear_chat_btn = QPushButton("Clear chat")
        self.clear_chat_btn.setObjectName("clearChatAction")
        self.clear_chat_btn.setEnabled(False)
        self.clear_chat_btn.clicked.connect(self._clear_chat)
        action_row.addWidget(self.clear_chat_btn)
        input_layout.addLayout(action_row)

        layout.addWidget(input_bar)

    def _context_preview_html(self):
        parts = []
        for field in self.context["fields"]:
            if field["text"]:
                value = html.escape(field["text"])
            elif field.get("images"):
                image_count = len(field["images"])
                suffix = "s" if image_count != 1 else ""
                value = (
                    f'<span style="color: #bfe7d2;">{image_count} image{suffix} attached and visible to '
                    f'{html.escape(CHATBOT_NAME)}.</span>'
                )
            else:
                value = '<span style="color: #8b949e;">No readable text.</span>'
            parts.append(
                f'<p style="margin: 0 0 8px;"><span style="color: #8b949e; font-weight: 600;">'
                f'{html.escape(field["name"])}:</span> {value}</p>'
            )
        if self.context["images"]:
            images = html.escape(", ".join(self.context["images"]))
            parts.append(
                '<p style="margin: 0 0 8px;"><span style="color: #8b949e; font-weight: 600;">'
                f'Images:</span> {images}</p>'
            )
        if not parts:
            parts.append(f'<p style="margin: 0;">No readable text found in the configured {html.escape(CHATBOT_NAME)} fields.</p>')
        return f'<div style="font-size: 13px; line-height: 1.35;">{"".join(parts)}</div>'

    def _toggle_context_preview(self):
        visible = not self.context_box.isVisible()
        self.context_box.setVisible(visible)
        self.context_toggle_btn.setText("Hide note context \u25b4" if visible else "Show note context \u25be")

    def _render_chat_transcript(self, transient_html=""):
        if transient_html:
            self._transient_chat_html = transient_html
        transient_html = transient_html or self._transient_chat_html
        theme = getattr(self, "_theme", get_addon_theme())
        answer_font_size = getattr(self, "_answer_font_size", 14)
        parts = [
            f'<div style="font-size: {answer_font_size}px; line-height: 1.34; padding: 2px 2px 6px;">'
        ]
        if not self._chat_messages and not transient_html:
            parts.append(
                f'<div style="color: {theme["quiet_text"]}; padding: 30px 12px; text-align: center;">'
                f"Ask {html.escape(CHATBOT_NAME)} a question about this card. Follow-ups will keep the recent chat in mind."
                "</div>"
            )
        for message in self._chat_messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "user":
                parts.append(self._render_user_bubble(content))
            else:
                parts.append(self._render_assistant_bubble(content))
        if transient_html:
            parts.append(transient_html)
        parts.append("</div>")
        self.answer_box.setHtml("".join(parts))
        QTimer.singleShot(0, self._scroll_chat_to_bottom)

    def _render_user_bubble(self, content):
        theme = getattr(self, "_theme", get_addon_theme())
        safe_content = html.escape(content).replace("\n", "<br>")
        return (
            '<table width="100%" cellspacing="0" cellpadding="0" style="margin: 6px 0 7px;">'
            '<tr><td align="right">'
            '<table width="72%" cellspacing="0" cellpadding="0" align="right">'
            '<tr><td style="text-align: left; '
            f'background-color: {theme["accent"]}; color: {theme["accent_text"]}; '
            'border-radius: 18px 18px 4px 18px; padding: 9px 13px;">'
            f"{safe_content}</td></tr></table>"
            "</td></tr></table>"
        )

    def _render_assistant_bubble(self, content):
        theme = getattr(self, "_theme", get_addon_theme())
        return (
            '<table cellspacing="0" cellpadding="0" style="margin: 6px 0 9px;">'
            '<tr>'
            '<td valign="top" style="padding: 0 9px 0 0;">'
            f'<div style="width: 30px; height: 30px; line-height: 30px; text-align: center; '
            f'background: {theme["section_header_bg"]}; color: {theme["accent"]}; '
            f'border: 1px solid {theme["subtle_border"]}; border-radius: 15px; font-size: 18px;" '
            f'title="{html.escape(CHATBOT_REFERENCE)}">{html.escape(CHATBOT_ICON)}</div>'
            '</td>'
            '<td valign="top" width="82%">'
            '<table width="100%" cellspacing="0" cellpadding="0">'
            '<tr><td style="text-align: left; '
            f'background-color: {theme["control_bg"]}; color: {theme["input_text"]}; '
            'border-radius: 18px 18px 18px 4px; padding: 9px 12px;">'
            f"{self._format_assistant_chat_html(content)}</td></tr></table>"
            '</td></tr></table>'
        )

    def _format_assistant_chat_html(self, content):
        theme = getattr(self, "_theme", get_addon_theme())
        formatted = _format_review_answer(content, [self.note_id])
        formatted = re.sub(
            r'<strong style="font-weight: bold;">(.*?)</strong>',
            (
                '<strong style="font-weight: 700; color: {text}; background: rgba(245, 158, 11, 0.18); '
                'border: 1px solid rgba(245, 158, 11, 0.30); border-radius: 5px; padding: 1px 4px;">\\1</strong>'
            ).format(text=theme["input_text"]),
            formatted,
        )
        formatted = re.sub(
            r'<p style="margin: 0\.75em 0 0\.4em 0; font-weight: bold; font-size: 1\.22em;">',
            (
                '<p style="margin: 0.7em 0 0.35em 0; font-weight: bold; font-size: 1.08em; '
                f'color: {theme["accent"]};">'
            ),
            formatted,
        )
        formatted = re.sub(
            r'<ul style="([^"]*)">',
            '<ul style="margin: 0.35em 0 0.5em 0.2em; padding-left: 1.1em; list-style-type: disc;">',
            formatted,
        )
        formatted = re.sub(
            r'<li style="[^"]*">',
            (
                '<li style="margin: 0.2em 0; padding: 2px 0;">'
            ),
            formatted,
        )
        formatted = re.sub(
            r'<p style="margin: ([^"]*);">',
            r'<p style="margin: \1; padding: 2px 0;">',
            formatted,
        )
        return formatted

    def _scroll_chat_to_bottom(self):
        bar = self.answer_box.verticalScrollBar()
        if bar:
            bar.setValue(bar.maximum())

    def _on_answer_link_clicked(self, url):
        if str(url.toString()).startswith("cite:"):
            self._open_note_id_in_browser(self.note_id)
            self._render_chat_transcript()

    def _selected_answer_text_context(self, max_chars=1800):
        try:
            cursor = self.answer_box.textCursor() if hasattr(self, "answer_box") else None
            if not cursor or not cursor.hasSelection():
                return ""
            text = (cursor.selectedText() or "").replace("\u2029", "\n").replace("\u2028", "\n")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "\n..."
            return text
        except Exception:
            return ""

    def _show_answer_context_menu(self, pos):
        menu = self.answer_box.createStandardContextMenu(pos)
        if self._selected_answer_text_context():
            menu.addSeparator()
            action = menu.addAction(f"Ask {CHATBOT_NAME} about selection")
            action.triggered.connect(self._ask_ai_about_selected_answer_text)
        menu.exec(self.answer_box.mapToGlobal(pos))

    def _ask_ai_about_selected_answer_text(self):
        selected_text = self._selected_answer_text_context()
        if not selected_text:
            tooltip("Select answer text first")
            return
        self._pending_selected_answer_context = selected_text
        self.question_input.setPlainText("Explain this selected text.")
        self._ask_ai()

    def eventFilter(self, obj, event):
        if obj is getattr(self, "question_input", None) and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                modifiers = event.modifiers()
                if modifiers & Qt.KeyboardModifier.ShiftModifier:
                    return False
                if modifiers == Qt.KeyboardModifier.NoModifier or modifiers & Qt.KeyboardModifier.ControlModifier:
                    if self.ask_btn.isEnabled():
                        self._ask_ai()
                    return True
        if obj is getattr(getattr(self, "answer_box", None), "viewport", lambda: None)():
            if event.type() == QEvent.Type.MouseMove:
                anchor = self.answer_box.anchorAt(event.pos())
                if str(anchor or "").startswith("cite:"):
                    self._schedule_answer_note_preview(str(anchor))
                else:
                    self._cancel_answer_note_preview()
            elif event.type() in (QEvent.Type.Leave, QEvent.Type.Hide):
                self._cancel_answer_note_preview()
        return super().eventFilter(obj, event)

    def _answer_note_preview_info(self):
        display_content = " | ".join(
            field.get("text") or ""
            for field in self.context.get("fields", [])
            if field.get("text")
        )
        return {
            "id": self.note_id,
            "display_content": display_content or self.context.get("text") or "",
            "relevance": "source",
            "why_ref": "Source note for this answer citation",
            "matching_terms": [],
        }

    def _schedule_answer_note_preview(self, anchor):
        if getattr(self, "_pending_answer_note_preview_anchor", None) == anchor:
            timer = getattr(self, "_answer_note_preview_timer", None)
            if timer is not None and timer.isActive():
                return
        self._pending_answer_note_preview_anchor = anchor
        timer = getattr(self, "_answer_note_preview_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._show_pending_answer_note_preview)
            self._answer_note_preview_timer = timer
        timer.start(NOTE_PREVIEW_SHOW_DELAY_MS)

    def _cancel_answer_note_preview(self):
        timer = getattr(self, "_answer_note_preview_timer", None)
        if timer is not None:
            timer.stop()
        self._pending_answer_note_preview_anchor = None
        if hasattr(self, "_note_preview_popup"):
            self._note_preview_popup.schedule_hide_if_cursor_outside()

    def _show_pending_answer_note_preview(self):
        anchor = getattr(self, "_pending_answer_note_preview_anchor", None)
        self._pending_answer_note_preview_anchor = None
        if not anchor:
            return
        try:
            current_anchor = self.answer_box.anchorAt(self.answer_box.viewport().mapFromGlobal(QCursor.pos()))
            if current_anchor != anchor:
                return
        except Exception:
            pass
        self._note_preview_popup.show_note(self._answer_note_preview_info())

    def _ask_ai(self):
        question = self.question_input.toPlainText().strip()
        if not question:
            tooltip("Please enter a question")
            return
        selected_text = (getattr(self, "_pending_selected_answer_context", "") or "").strip()
        worker_question = question
        if selected_text:
            worker_question = (
                f"{question}\n\nSelected answer text context:\n"
                f"{selected_text}\n\n"
                "Use this selected text to resolve references like this, that, it, or explain this."
            )
        self._request_id += 1
        request_id = self._request_id
        self._pending_question = question
        self._chat_messages.append({"role": "user", "content": question})
        self._transient_chat_html = ""
        self.question_input.clear()
        self._pending_selected_answer_context = ""
        self._render_chat_transcript()
        self.ask_btn.setEnabled(False)
        self._start_loading()
        QTimer.singleShot(10000, lambda rid=request_id: self._show_still_waiting(rid))

        self._ask_worker = ReviewAskAIWorker(
            worker_question,
            self.context["text"],
            self.config,
            chat_history=self._chat_history[-4:],
            image_payloads=self.context.get("image_payloads") or [],
        )
        self._ask_worker.success_signal.connect(lambda answer, rid=request_id: self._on_ask_success(rid, answer))
        self._ask_worker.error_signal.connect(lambda error, rid=request_id: self._on_ask_error(rid, error))
        self._ask_worker.warning_signal.connect(lambda warning, rid=request_id: self._on_ask_warning(rid, warning))
        self._ask_worker.finished.connect(self._on_ask_finished)
        self._ask_worker.start()

    def _show_still_waiting(self, request_id):
        if request_id == self._request_id and self._ask_worker is not None:
            self._transient_chat_html = self._system_message_html(f"Still waiting for the {CHATBOT_NAME} provider...")
            self._render_chat_transcript()

    def _on_ask_success(self, request_id, answer):
        if request_id != self._request_id:
            return
        self._stop_loading()
        if answer:
            self._chat_messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )
            self._render_chat_transcript()
            self._chat_history.extend(
                [
                    {"role": "user", "content": self._pending_question},
                    {"role": "assistant", "content": answer},
                ]
            )
            self._chat_history = self._chat_history[-4:]
            self.clear_chat_btn.setEnabled(True)
        else:
            self._append_system_message("The provider returned an empty answer.")

    def _on_ask_error(self, request_id, error):
        if request_id != self._request_id:
            return
        self._stop_loading()
        self._append_system_message(f"Could not get a {CHATBOT_NAME} answer:\n{error}", kind="error")

    def _on_ask_warning(self, request_id, warning):
        if request_id != self._request_id:
            return
        self._append_system_message(warning, kind="warning")

    def _on_ask_finished(self):
        self.ask_btn.setEnabled(True)
        self._ask_worker = None

    def _start_loading(self):
        self._transient_chat_html = self._typing_indicator_html()
        self._render_chat_transcript()

    def _stop_loading(self):
        self._transient_chat_html = ""
        self._render_chat_transcript()

    def _set_message_html(self, message, kind="info"):
        self._transient_chat_html = self._system_message_html(message, kind=kind)
        self._render_chat_transcript()

    def _append_system_message(self, message, kind="info"):
        self._transient_chat_html = self._system_message_html(message, kind=kind)
        self._render_chat_transcript()

    def _typing_indicator_html(self):
        theme = getattr(self, "_theme", get_addon_theme())
        return (
            '<div style="text-align: left; margin: 6px 0 9px;">'
            '<div style="display: inline-block; max-width: 82%; text-align: left; '
            f'background: {theme["control_bg"]}; color: {theme["quiet_text"]}; '
            'border-radius: 18px 18px 18px 4px; padding: 9px 13px; font-size: 18px; '
            'letter-spacing: 2px;">...</div></div>'
        )

    def _system_message_html(self, message, kind="info"):
        theme = getattr(self, "_theme", get_addon_theme())
        if kind == "error":
            color = theme["danger"]
        elif kind == "warning":
            color = "#f59e0b"
        else:
            color = theme["quiet_text"]
        return (
            '<div style="text-align: center; margin: 8px 0;">'
            f'<div style="display: inline-block; max-width: 86%; color: {color}; '
            'background: rgba(127,127,127,0.10); border-radius: 10px; padding: 7px 10px; '
            'white-space: pre-wrap; font-size: 12px;">'
            f"{html.escape(message or '')}</div></div>"
        )

    def _clear_chat(self):
        self._reset_chat_state("Chat cleared.")

    def _search_collection(self):
        query = self.question_input.toPlainText().strip() or self._last_user_question()
        if not query:
            tooltip("Type a question first, then search the collection.")
            return
        try:
            from . import dialogs as dialogs_module
            dialogs_module.show_search_dialog(initial_query=query, auto_search=True)
        except Exception as exc:
            log_debug(f"Could not open AI Search from review chatbot: {exc}", is_error=True)
            tooltip("Could not open AI Search. Check the add-on log.")

    def _last_user_question(self):
        for message in reversed(self._chat_messages):
            if message.get("role") == "user":
                content = str(message.get("content") or "").strip()
                if content:
                    return content
        return ""

    def _open_note_id_in_browser(self, note_id):
        self._open_note_ids_in_browser([str(note_id)], "Opened note in Browser")

    def _open_note_ids_in_browser(self, note_ids, message):
        browser = dialogs.open("Browser", mw)
        browser.form.searchEdit.lineEdit().setText("nid:" + ",".join(str(note_id) for note_id in note_ids))
        browser.onSearchActivated()
        tooltip(message)

    def closeEvent(self, event):
        self._cancel_active_work()
        timer = getattr(self, "_review_card_timer", None)
        if timer is not None:
            timer.stop()
        super().closeEvent(event)
