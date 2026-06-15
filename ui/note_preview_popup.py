"""Scrollable note preview popup for search results."""

import html
import math
import re

from aqt import mw
from aqt.qt import (
    QApplication,
    QCursor,
    QEvent,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPoint,
    QRect,
    QSize,
    QPushButton,
    QTextBrowser,
    QTimer,
    QVBoxLayout,
    Qt,
)

from ..utils import log_debug
from ..utils.text import reveal_cloze


PREVIEW_BROWSER_WIDTH = 460
PREVIEW_BROWSER_MAX_HEIGHT = 420
PREVIEW_BROWSER_MIN_HEIGHT = 84
PREVIEW_HIDE_GRACE_MS = 1500
PREVIEW_EDGE_GAP = 10

PHOTO_CREDIT_LINE_RE = re.compile(
    r"^\s*(?:"
    r"photo credit:.*"
    r"|.*(?:cc by(?:-sa)?|via flickr|all rights reserved|used with permission).*"
    r"|(?:<a\b[^>]*>\s*)?(?:https?://|www\.)[^\s<]+(?:\s*</a>)?"
    r")\s*$",
    re.IGNORECASE,
)


class NotePreviewPopup(QFrame):
    """A browser-like hover preview for matching notes."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.ToolTip)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setObjectName("semanticNotePreviewPopup")
        self._current_note_id = None
        self._pinned = False
        self._app_filter_installed = False
        self._container_rect = None
        self._watched_objects = set()
        self._browser_max_height = PREVIEW_BROWSER_MAX_HEIGHT

        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 8, 9, 9)
        layout.setSpacing(7)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        self.header = QLabel(self)
        self.header.setObjectName("semanticNotePreviewHeader")
        self.header.setTextFormat(Qt.TextFormat.RichText)
        self.header.setWordWrap(True)
        header_row.addWidget(self.header, 1)

        self.pin_btn = QPushButton("Pin", self)
        self.pin_btn.setObjectName("semanticNotePreviewPinButton")
        self.pin_btn.setCheckable(True)
        self.pin_btn.setFixedHeight(24)
        self.pin_btn.clicked.connect(self._toggle_pinned)
        header_row.addWidget(self.pin_btn)

        self.close_btn = QPushButton("x", self)
        self.close_btn.setObjectName("semanticNotePreviewCloseButton")
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.clicked.connect(self.close_preview)
        self.close_btn.hide()
        header_row.addWidget(self.close_btn)

        layout.addLayout(header_row)

        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(True)
        self.browser.setFixedWidth(PREVIEW_BROWSER_WIDTH)
        self.browser.setMinimumHeight(PREVIEW_BROWSER_MIN_HEIGHT)
        self.browser.setMaximumHeight(PREVIEW_BROWSER_MAX_HEIGHT)
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
                background: #2b2b2b;
                border: 1px solid #404040;
                border-radius: 7px;
            }
            QLabel#semanticNotePreviewHeader {
                color: #d8dddd;
                font-size: 11px;
            }
            QPushButton#semanticNotePreviewPinButton,
            QPushButton#semanticNotePreviewCloseButton {
                background: #343434;
                color: #e8e8e8;
                border: 1px solid #4b4b4b;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton#semanticNotePreviewPinButton:hover,
            QPushButton#semanticNotePreviewCloseButton:hover {
                background: #3d3d3d;
                border-color: #646464;
            }
            QPushButton#semanticNotePreviewPinButton:checked {
                background: #2e4e4a;
                border-color: #6aa9a0;
                color: #ffffff;
            }
            QTextBrowser {
                background: #2f2f2f;
                color: #f6f6f6;
                border: 1px solid #444444;
                border-radius: 5px;
                padding: 0;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #2b2b2b;
                border: none;
                margin: 0;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #606060;
                border-radius: 4px;
                min-height: 24px;
                min-width: 24px;
            }
            """
        )

    def show_note(self, note_info, global_pos=None, anchor_rect=None, container_rect=None):
        note_id = note_info.get("id")
        if not note_id:
            self.hide()
            return

        self._container_rect = container_rect
        self._apply_responsive_size(container_rect)
        if note_id != self._current_note_id:
            self._current_note_id = note_id
            self.header.setText(self._build_header(note_info))
            self.browser.setHtml(self._build_note_html(note_info))
            self._resize_browser_to_content()
            self.browser.verticalScrollBar().setValue(0)

        self.adjustSize()
        self.move(self._bounded_position(global_pos or QCursor.pos(), anchor_rect, container_rect))
        self.show()
        self._install_app_filter()
        self.raise_()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and self.isVisible():
            try:
                if event.key() == Qt.Key.Key_Escape:
                    self.close_preview()
                    return True
            except Exception:
                pass
        if event.type() in (QEvent.Type.Leave, QEvent.Type.Hide):
            if obj in self._watched_objects or obj is self:
                self.schedule_hide_if_cursor_outside()
        return False

    def leaveEvent(self, event):
        self.schedule_hide_if_cursor_outside()
        super().leaveEvent(event)

    def hideEvent(self, event):
        self._remove_app_filter()
        super().hideEvent(event)

    def mousePressEvent(self, event):
        if not self._pinned:
            self._set_pinned(True)
        super().mousePressEvent(event)

    def schedule_hide_if_cursor_outside(self):
        if self._pinned:
            return
        QTimer.singleShot(PREVIEW_HIDE_GRACE_MS, self._hide_if_cursor_outside)

    def reset_pinned(self):
        self._set_pinned(False)
        self.hide()

    def is_pinned(self):
        return self._pinned

    def close_preview(self):
        self._set_pinned(False)
        self.hide()

    def _build_header(self, note_info):
        terms = note_info.get("matching_terms") or []
        terms_text = ", ".join(html.escape(str(term)) for term in terms[:4])
        if len(terms) > 4:
            terms_text += "..."
        why_ref = str(note_info.get("why_ref", "") or "")
        why_ref = why_ref.replace("Cited in answer: ", "Cited: ").replace("Related-note rank", "Ranked note")
        chips = [
            f"{html.escape(str(note_info.get('relevance', '')))}% relevance",
            html.escape(why_ref),
            f"ID: {html.escape(str(note_info.get('id', '')))}",
        ]
        chip_html = "".join(
            f"<span style='display:inline-block; margin: 3px 5px 0 0; padding: 2px 6px; "
            f"border: 1px solid #434d4f; border-radius: 4px; color: #d5dfdd; "
            f"background: #303535; font-size: 11px; line-height: 1.25;'>{chunk}</span>"
            for chunk in chips if chunk
        )
        terms_html = ""
        if terms_text:
            terms_html = (
                "<div style='color: #aeb8ba; font-size: 11px; line-height: 1.25; margin-top: 4px;'>"
                f"Terms: {terms_text}</div>"
            )
        return (
            "<div style='color:#d8dddd; font-weight: 600; font-size: 11px; line-height: 1.25; margin-bottom: 2px;'>"
            "Note preview</div>"
            f"<div>{chip_html}</div>{terms_html}"
        )

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
                section = self._field_section_html(name, value)
                if section:
                    field_sections.append(section)

            model_name = html.escape(model.get("name", "Note") if model else "Note")
            content = "\n".join(field_sections) or self._fallback_content(note_info)
            return self._wrap_html(model_name, content)
        except Exception as exc:
            log_debug(f"Could not build note preview popup: {exc}")
            return self._wrap_html("Note", self._fallback_content(note_info))

    def _field_section_html(self, name, value):
        safe_name = html.escape(name or "Field")
        value = self._normalize_field_html(reveal_cloze(value or ""), name)
        if (name or "").strip().lower() == "extra" and not self._visible_text(value):
            return ""
        return f"""
        <section class="semantic-preview-field" style="margin: 0 0 10px;">
            <div class="semantic-preview-field-title" style="font-size: 11px; line-height: 1.25; font-weight: 700; padding: 5px 9px;">{safe_name}</div>
            <div class="semantic-preview-field-body" style="font-size: 13px; line-height: 1.42; padding: 9px 10px;">{value}</div>
        </section>
        """

    def _fallback_content(self, note_info):
        content = html.escape(str(note_info.get("display_content") or ""))
        content = re.sub(r"\s*\|\s*", "<br>", content)
        return f"<section class='semantic-preview-field' style='margin: 0 0 10px;'><div class='semantic-preview-field-body' style='font-size: 13px; line-height: 1.42; padding: 9px 10px;'>{content}</div></section>"

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
                    background: #2f2f2f;
                    color: #f6f6f6;
                    font-family: Arial, sans-serif;
                    font-size: 13px;
                    line-height: 1.42;
                }}
                body {{ padding: 9px; }}
                .semantic-preview-root,
                .semantic-preview-root * {{
                    font-size: 13px !important;
                    line-height: 1.42 !important;
                    letter-spacing: 0 !important;
                }}
                .semantic-preview-root b,
                .semantic-preview-root strong {{
                    font-weight: 700 !important;
                    font-size: 13px !important;
                }}
                .semantic-preview-model {{
                    color: #a7b4b8;
                    font-size: 11px !important;
                    line-height: 1.25 !important;
                    margin: 0 0 8px;
                }}
                .semantic-preview-field {{
                    background: #323232;
                    border: 1px solid #424242;
                    border-left: 2px solid #6f9691;
                    border-radius: 5px;
                    margin: 0 0 10px;
                    overflow: hidden;
                }}
                .semantic-preview-field:last-child {{
                    margin-bottom: 0;
                }}
                .semantic-preview-field-title {{
                    background: #363636;
                    border-bottom: 1px solid #424242;
                    color: #d5dfdd;
                    font-size: 11px !important;
                    line-height: 1.25 !important;
                    font-weight: 700;
                    margin: 0;
                    padding: 5px 9px;
                }}
                .semantic-preview-field-body {{
                    background: #2f2f2f;
                    border: none;
                    border-radius: 0;
                    color: #f6f6f6;
                    font-size: 13px !important;
                    line-height: 1.42 !important;
                    min-height: 20px;
                    padding: 9px 10px;
                    overflow-wrap: anywhere;
                }}
                .semantic-preview-field-body * {{
                    font-size: 13px !important;
                    line-height: 1.42 !important;
                    max-width: 100% !important;
                }}
                .semantic-preview-field-body:empty::after {{
                    content: " ";
                    white-space: pre;
                }}
                img, svg, video, audio, object, embed {{ display: none !important; }}
                hr {{
                    border: 0;
                    border-top: 1px solid #505050;
                    margin: 10px 0;
                }}
                a {{ color: #9bbff4; }}
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

    def _bounded_position(self, global_pos, anchor_rect=None, container_rect=None):
        screen = QApplication.screenAt(global_pos)
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is None:
            return global_pos + QPoint(16, 16)

        rect = container_rect if isinstance(container_rect, QRect) and container_rect.isValid() else screen.availableGeometry()
        size = self.sizeHint()
        if isinstance(anchor_rect, QRect) and anchor_rect.isValid():
            right_x = anchor_rect.right() + PREVIEW_EDGE_GAP
            left_x = anchor_rect.left() - size.width() - PREVIEW_EDGE_GAP
            if right_x + size.width() <= rect.right():
                x = right_x
            elif left_x >= rect.left():
                x = left_x
            else:
                x = min(max(rect.left() + PREVIEW_EDGE_GAP, right_x), rect.right() - size.width() - PREVIEW_EDGE_GAP)

            y = anchor_rect.top()
            if y + size.height() > rect.bottom() - PREVIEW_EDGE_GAP:
                y = anchor_rect.bottom() - size.height()
        else:
            x = global_pos.x() + PREVIEW_EDGE_GAP
            y = global_pos.y() + PREVIEW_EDGE_GAP
            if x + size.width() > rect.right():
                x = global_pos.x() - size.width() - PREVIEW_EDGE_GAP
            if y + size.height() > rect.bottom():
                y = global_pos.y() - size.height() - PREVIEW_EDGE_GAP
        max_x = max(rect.left() + PREVIEW_EDGE_GAP, rect.right() - size.width() - PREVIEW_EDGE_GAP)
        max_y = max(rect.top() + PREVIEW_EDGE_GAP, rect.bottom() - size.height() - PREVIEW_EDGE_GAP)
        return QPoint(
            min(max(rect.left() + PREVIEW_EDGE_GAP, x), max_x),
            min(max(rect.top() + PREVIEW_EDGE_GAP, y), max_y),
        )

    def _hide_if_cursor_outside(self):
        if self._pinned:
            return
        if self.isVisible() and not self.geometry().contains(QCursor.pos()):
            self.hide()

    def _resize_browser_to_content(self):
        try:
            document = self.browser.document()
            document.setTextWidth(self.browser.viewport().width())
            height = int(math.ceil(document.size().height())) + 18
            height = max(PREVIEW_BROWSER_MIN_HEIGHT, min(self._browser_max_height, height))
            self.browser.setFixedHeight(height)
        except Exception:
            self.browser.setFixedHeight(self._browser_max_height)

    def _apply_responsive_size(self, container_rect=None):
        if isinstance(container_rect, QRect) and container_rect.isValid():
            available_width = max(300, container_rect.width() - (PREVIEW_EDGE_GAP * 4))
            width = min(PREVIEW_BROWSER_WIDTH, available_width)
            max_height = min(PREVIEW_BROWSER_MAX_HEIGHT, max(PREVIEW_BROWSER_MIN_HEIGHT, int(container_rect.height() * 0.55) - 80))
        else:
            width = PREVIEW_BROWSER_WIDTH
            max_height = PREVIEW_BROWSER_MAX_HEIGHT
        self.browser.setFixedWidth(width)
        self._browser_max_height = max(PREVIEW_BROWSER_MIN_HEIGHT, max_height)
        self.browser.setMaximumHeight(self._browser_max_height)

    def watch_hover_object(self, obj):
        if obj is not None:
            self._watched_objects.add(obj)

    def _toggle_pinned(self, checked):
        self._set_pinned(bool(checked))

    def _set_pinned(self, pinned):
        self._pinned = bool(pinned)
        self.pin_btn.setChecked(self._pinned)
        self.pin_btn.setText("Pinned" if self._pinned else "Pin")
        self.close_btn.setVisible(self._pinned)

    def _install_app_filter(self):
        app = QApplication.instance()
        if app is not None and not self._app_filter_installed:
            app.installEventFilter(self)
            self._app_filter_installed = True

    def _remove_app_filter(self):
        app = QApplication.instance()
        if app is not None and self._app_filter_installed:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass
            self._app_filter_installed = False

    def _normalize_field_html(self, value, field_name=None):
        value = re.sub(r"<\s*(img|svg|video|audio|object|embed)\b[^>]*>.*?<\s*/\s*\1\s*>", "", value, flags=re.IGNORECASE | re.DOTALL)
        value = re.sub(r"<\s*(img|source|track|embed|object)\b[^>]*?/?>", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\[sound:[^\]]+\]", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s(?:class|style|width|height)\s*=\s*(['\"]).*?\1", "", value, flags=re.IGNORECASE | re.DOTALL)
        value = re.sub(r"font-size\s*:\s*[^;\"']+;?", "", value, flags=re.IGNORECASE)
        value = re.sub(r"line-height\s*:\s*[^;\"']+;?", "", value, flags=re.IGNORECASE)
        if (field_name or "").strip().lower() == "extra":
            value = self._strip_photo_credit_lines(value)
        value = self._collapse_empty_html(value)
        return value

    def _strip_photo_credit_lines(self, value):
        value = re.sub(
            r"<(?P<tag>div|p|li)\b[^>]*>\s*(?P<body>.*?)\s*</(?P=tag)>",
            lambda match: "" if self._is_photo_credit_line(match.group("body")) else match.group(0),
            value,
            flags=re.IGNORECASE | re.DOTALL,
        )

        lines = re.split(r"\r?\n|<br\s*/?>", value, flags=re.IGNORECASE)
        return "<br>".join(line for line in lines if not self._is_photo_credit_line(line))

    def _is_photo_credit_line(self, value):
        return bool(PHOTO_CREDIT_LINE_RE.match(self._visible_text(value)))

    def _collapse_empty_html(self, value):
        previous = None
        while previous != value:
            previous = value
            value = re.sub(r"<(div|p)\b[^>]*>\s*(?:&nbsp;|\s|<br\s*/?>)*</\1>", "", value, flags=re.IGNORECASE)
            value = re.sub(r"(?:\s|&nbsp;|<br\s*/?>)+$", "", value, flags=re.IGNORECASE)
        return value.strip()

    def _visible_text(self, value):
        value = re.sub(r"<[^>]+>", " ", value or "")
        value = html.unescape(value).replace("\xa0", " ")
        return re.sub(r"\s+", " ", value).strip()
