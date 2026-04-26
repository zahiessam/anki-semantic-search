"""Scrollable note preview popup for search results."""

import html
import re

from aqt import mw
from aqt.qt import (
    QApplication,
    QCursor,
    QEvent,
    QFrame,
    QLabel,
    QPoint,
    QSize,
    QTextBrowser,
    QTimer,
    QVBoxLayout,
    Qt,
)

from ..utils import log_debug
from ..utils.text import reveal_cloze


class NotePreviewPopup(QFrame):
    """A browser-like hover preview for matching notes."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.ToolTip)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setObjectName("semanticNotePreviewPopup")
        self._current_note_id = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        self.header = QLabel(self)
        self.header.setObjectName("semanticNotePreviewHeader")
        self.header.setTextFormat(Qt.TextFormat.RichText)
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(True)
        self.browser.setFixedSize(QSize(620, 420))
        self.browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        if mw and mw.col:
            try:
                self.browser.document().setSearchPaths([mw.col.media.dir()])
            except Exception:
                pass
        layout.addWidget(self.browser)

        self.setStyleSheet(
            """
            QFrame#semanticNotePreviewPopup {
                background: #2f2f2f;
                border: 1px solid #5a5a5a;
                border-radius: 6px;
            }
            QLabel#semanticNotePreviewHeader {
                color: #f2f2f2;
                font-size: 12px;
            }
            QTextBrowser {
                background: #303030;
                color: #f6f6f6;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 0;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #2b2b2b;
                border: none;
                margin: 0;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #707070;
                border-radius: 4px;
                min-height: 24px;
                min-width: 24px;
            }
            """
        )

    def show_note(self, note_info, global_pos=None):
        note_id = note_info.get("id")
        if not note_id:
            self.hide()
            return

        if note_id != self._current_note_id:
            self._current_note_id = note_id
            self.header.setText(self._build_header(note_info))
            self.browser.setHtml(self._build_note_html(note_info))
            self.browser.verticalScrollBar().setValue(0)

        self.adjustSize()
        self.move(self._bounded_position(global_pos or QCursor.pos()))
        self.show()
        self.raise_()

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.Leave, QEvent.Type.Hide):
            QTimer.singleShot(180, self._hide_if_cursor_outside)
        return False

    def leaveEvent(self, event):
        QTimer.singleShot(180, self._hide_if_cursor_outside)
        super().leaveEvent(event)

    def _build_header(self, note_info):
        terms = note_info.get("matching_terms") or []
        terms_text = ", ".join(html.escape(str(term)) for term in terms[:8])
        if len(terms) > 8:
            terms_text += "..."
        chunks = [
            "<b>Why this result?</b>",
            f"Relevance: {html.escape(str(note_info.get('relevance', '')))}%",
            html.escape(note_info.get("why_ref", "")),
            f"Note ID: {html.escape(str(note_info.get('id', '')))}",
        ]
        if terms_text:
            chunks.append(f"Matching terms: {terms_text}")
        return " &nbsp; | &nbsp; ".join(chunk for chunk in chunks if chunk)

    def _build_note_html(self, note_info):
        try:
            note = mw.col.get_note(note_info["id"])
            model = note.note_type() if hasattr(note, "note_type") else note.model()
            field_names = [field.get("name", "") for field in (model or {}).get("flds", [])]
            preview_fields = [
                name for name in field_names
                if name.lower() in ("text", "extra")
            ]
            if not preview_fields:
                preview_fields = field_names[:2]
            field_sections = []

            for name in preview_fields:
                try:
                    value = note[name]
                except Exception:
                    value = ""
                field_sections.append(self._field_section_html(name, value))

            model_name = html.escape(model.get("name", "Note") if model else "Note")
            content = "\n".join(field_sections) or self._fallback_content(note_info)
            return self._wrap_html(model_name, content)
        except Exception as exc:
            log_debug(f"Could not build note preview popup: {exc}")
            return self._wrap_html("Note", self._fallback_content(note_info))

    def _field_section_html(self, name, value):
        safe_name = html.escape(name or "Field")
        value = self._normalize_field_html(reveal_cloze(value or ""))
        return f"""
        <section class="semantic-preview-field" style="margin: 0 0 10px;">
            <div class="semantic-preview-field-title" style="font-size: 13px; line-height: 1.25; font-weight: 600; margin: 0 0 4px;">{safe_name}</div>
            <div class="semantic-preview-field-body" style="font-size: 14px; line-height: 1.45; padding: 8px;">{value}</div>
        </section>
        """

    def _fallback_content(self, note_info):
        content = html.escape(str(note_info.get("display_content") or ""))
        content = re.sub(r"\s*\|\s*", "<br>", content)
        return f"<section class='semantic-preview-field' style='margin: 0 0 10px;'><div class='semantic-preview-field-body' style='font-size: 14px; line-height: 1.45; padding: 8px;'>{content}</div></section>"

    def _wrap_html(self, model_name, content):
        return f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: #303030;
                    color: #f6f6f6;
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                    line-height: 1.45;
                }}
                body {{ padding: 8px; }}
                .semantic-preview-root,
                .semantic-preview-root * {{
                    font-size: 14px !important;
                    line-height: 1.45 !important;
                    letter-spacing: 0 !important;
                }}
                .semantic-preview-root b,
                .semantic-preview-root strong {{
                    font-weight: 700 !important;
                    font-size: 14px !important;
                }}
                .semantic-preview-model {{
                    color: #a8c7ff;
                    font-size: 12px !important;
                    line-height: 1.25 !important;
                    margin: 0 0 8px;
                }}
                .semantic-preview-field {{ margin: 0 0 10px; }}
                .semantic-preview-field-title {{
                    color: #f4f4f4;
                    font-size: 13px !important;
                    line-height: 1.25 !important;
                    font-weight: 600;
                    margin: 0 0 4px;
                }}
                .semantic-preview-field-body {{
                    background: #353535;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    color: #f6f6f6;
                    font-size: 14px !important;
                    line-height: 1.45 !important;
                    min-height: 20px;
                    padding: 8px;
                    overflow-wrap: anywhere;
                }}
                .semantic-preview-field-body * {{
                    font-size: 14px !important;
                    line-height: 1.45 !important;
                    max-width: 100% !important;
                }}
                .semantic-preview-field-body:empty::after {{
                    content: " ";
                    white-space: pre;
                }}
                img, svg, video, audio, object, embed {{ display: none !important; }}
                hr {{
                    border: 0;
                    border-top: 1px solid #555;
                    margin: 10px 0;
                }}
                a {{ color: #8ab4f8; }}
            </style>
        </head>
        <body>
            <div class="semantic-preview-root" style="font-size: 14px; line-height: 1.45;">
                <div class="semantic-preview-model" style="font-size: 12px; line-height: 1.25; margin: 0 0 8px;">{html.escape(model_name)}</div>
                {content}
            </div>
        </body>
        </html>
        """

    def _bounded_position(self, global_pos):
        screen = QApplication.screenAt(global_pos)
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is None:
            return global_pos + QPoint(16, 16)

        rect = screen.availableGeometry()
        size = self.sizeHint()
        x = global_pos.x() + 16
        y = global_pos.y() + 16
        if x + size.width() > rect.right():
            x = global_pos.x() - size.width() - 16
        if y + size.height() > rect.bottom():
            y = global_pos.y() - size.height() - 16
        return QPoint(max(rect.left(), x), max(rect.top(), y))

    def _hide_if_cursor_outside(self):
        if self.isVisible() and not self.geometry().contains(QCursor.pos()):
            self.hide()

    def _normalize_field_html(self, value):
        value = re.sub(r"<\s*(img|svg|video|audio|object|embed)\b[^>]*>.*?<\s*/\s*\1\s*>", "", value, flags=re.IGNORECASE | re.DOTALL)
        value = re.sub(r"<\s*(img|source|track|embed|object)\b[^>]*?/?>", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s(?:class|style|width|height)\s*=\s*(['\"]).*?\1", "", value, flags=re.IGNORECASE | re.DOTALL)
        value = re.sub(r"font-size\s*:\s*[^;\"']+;?", "", value, flags=re.IGNORECASE)
        value = re.sub(r"line-height\s*:\s*[^;\"']+;?", "", value, flags=re.IGNORECASE)
        return value
