import os
import sys
import re
from aqt.qt import *
from .theme import (
    collapsible_section_button_stylesheet,
    collapsible_section_content_stylesheet,
    get_addon_theme,
    settings_field_row_stylesheet,
)
from ..utils import log_debug, load_config

def _get_spell_checker():
    """Return SpellChecker instance if pyspellchecker is available, else None."""
    try:
        from spellchecker import SpellChecker
        return SpellChecker()
    except ImportError:
        pass

    try:
        config = load_config()
        sc = config.get('search_config') or {}
        ext_path = (sc.get('rerank_python_path') or '').strip()
        if ext_path:
            py_dir = os.path.dirname(ext_path) if os.path.isfile(ext_path) else ext_path
            if py_dir:
                site_packages = os.path.join(py_dir, "Lib", "site-packages")
                if os.path.isdir(site_packages) and site_packages not in sys.path:
                    sys.path.insert(0, site_packages)
                    try:
                        from spellchecker import SpellChecker
                        return SpellChecker()
                    finally:
                        if site_packages in sys.path:
                            sys.path.remove(site_packages)
    except Exception:
        pass
    return None

class SpellCheckHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._spell = _get_spell_checker()
        self._format = QTextCharFormat()
        self._format.setUnderlineColor(QColor(get_addon_theme()["danger"]))
        ustyle = getattr(QTextCharFormat.UnderlineStyle, 'SpellCheckUnderline', None) or getattr(QTextCharFormat, 'SpellCheckUnderline', None)
        self._format.setUnderlineStyle(ustyle if ustyle is not None else QTextCharFormat.UnderlineStyle.SingleUnderline)
        self._custom_words = set()

    def highlightBlock(self, text):
        if not self._spell or not text:
            return
        txt = text if isinstance(text, str) else str(text)
        for m in re.finditer(r'\b[a-zA-Z]{2,}\b', txt):
            word = m.group()
            if word.lower() not in self._custom_words and word.lower() in self._spell.unknown([word.lower()]):
                self.setFormat(m.start(), m.end() - m.start(), self._format)

    def add_custom_word(self, word):
        if word:
            self._custom_words.add(word.lower())
            self.rehighlight()

class SpellCheckPlainTextEdit(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._spell_highlighter = SpellCheckHighlighter(self.document())
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def _on_context_menu(self, pos):
        menu = self.createStandardContextMenu()
        spell = _get_spell_checker()
        if spell:
            cursor = self.textCursor()
            if not cursor.hasSelection():
                cursor.select(QTextCursor.SelectionType.WordUnderCursor)
            word = cursor.selectedText().strip()
            if word and len(word) >= 2 and word.isalpha():
                word_lower = word.lower()
                if word_lower in spell.unknown([word_lower]):
                    candidates = list(spell.candidates(word_lower))[:5] if spell.candidates(word_lower) else []
                    if candidates:
                        if menu.actions():
                            menu.insertSeparator(menu.actions()[0])
                        for corr in candidates:
                            action = QAction(f"Replace with '{corr}'", menu)
                            action.triggered.connect(lambda checked, c=corr, cur=cursor: self._replace_word(cur, c))
                            menu.insertAction(menu.actions()[0], action)
        menu.exec(self.mapToGlobal(pos))

    def _replace_word(self, cursor, new_word):
        cursor.beginEditBlock()
        cursor.insertText(new_word)
        cursor.endEditBlock()


def _first_tooltip_from_widget(widget):
    if widget is None:
        return ""
    tooltip = widget.toolTip()
    if tooltip:
        return tooltip
    for child in widget.findChildren(QWidget):
        tooltip = child.toolTip()
        if tooltip:
            return tooltip
    return ""


def _first_tooltip_from_layout(layout):
    for i in range(layout.count()):
        item = layout.itemAt(i)
        widget = item.widget() if item is not None else None
        if widget is not None:
            tooltip = _first_tooltip_from_widget(widget)
            if tooltip:
                return tooltip
        child_layout = item.layout() if item is not None else None
        if child_layout is not None:
            tooltip = _first_tooltip_from_layout(child_layout)
            if tooltip:
                return tooltip
    return ""


def apply_setting_row_tooltip(row, tooltip=None):
    """Apply one setting tooltip across the whole row and its plain labels."""
    if row is None:
        return

    resolved = tooltip or row.toolTip() or _first_tooltip_from_widget(row)
    if not resolved:
        return

    row.setToolTip(resolved)
    for label in row.findChildren(QLabel):
        if not label.toolTip():
            label.setToolTip(resolved)


def sync_setting_row_tooltips(root):
    """Normalize tooltips for all settings rows under root."""
    if root is None:
        return
    rows = []
    if getattr(root, "objectName", lambda: "")() == "settingsFieldRow":
        rows.append(root)
    rows.extend(root.findChildren(QFrame, "settingsFieldRow"))
    for row in rows:
        apply_setting_row_tooltip(row)


def settings_field_row(theme, content=None, label=None, layout=None, vertical=False, tooltip=None):
    row = QFrame()
    row.setObjectName("settingsFieldRow")
    row.setStyleSheet(settings_field_row_stylesheet(theme))

    row_layout = QVBoxLayout(row) if vertical else QHBoxLayout(row)
    row_layout.setContentsMargins(8, 6, 8, 6)
    row_layout.setSpacing(8)

    resolved_tooltip = tooltip or (
        content.toolTip() if content is not None and hasattr(content, "toolTip") else ""
    )
    if not resolved_tooltip and layout is not None:
        resolved_tooltip = _first_tooltip_from_layout(layout)
    if resolved_tooltip:
        row.setToolTip(resolved_tooltip)

    if label:
        label_widget = label if isinstance(label, QLabel) else QLabel(str(label))
        if resolved_tooltip and not label_widget.toolTip():
            label_widget.setToolTip(resolved_tooltip)
        if not vertical:
            label_widget.setMinimumWidth(170)
        row_layout.addWidget(label_widget)

    if layout is not None:
        row_layout.addLayout(layout)
    elif content is not None:
        row_layout.addWidget(content)

    apply_setting_row_tooltip(row, resolved_tooltip)

    return row


class CollapsibleSection(QWidget):
    def __init__(self, title="", parent=None, is_expanded=False):
        super().__init__(parent)
        self.is_expanded = is_expanded
        theme = get_addon_theme()

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(self.is_expanded)
        self.toggle_button.setStyleSheet(collapsible_section_button_stylesheet(theme))

        self._update_arrow()
        self.toggle_button.clicked.connect(self.toggle)
        self.main_layout.addWidget(self.toggle_button)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(16, 14, 16, 16)
        self.content_layout.setSpacing(8)
        self.content_area.setStyleSheet(collapsible_section_content_stylesheet(theme))
        self.content_area.setVisible(self.is_expanded)
        self.main_layout.addWidget(self.content_area)

    def _update_arrow(self):
        arrow = "v " if self.is_expanded else "> "
        title = self.toggle_button.text()
        if title.startswith("v ") or title.startswith("> "):
            title = title[2:]
        self.toggle_button.setText(arrow + title)

    def toggle(self):
        self.is_expanded = not self.is_expanded
        self.content_area.setVisible(self.is_expanded)
        self.toggle_button.setChecked(self.is_expanded)
        self._update_arrow()

    def setExpanded(self, expanded):
        self.is_expanded = bool(expanded)
        self.content_area.setVisible(self.is_expanded)
        self.toggle_button.setChecked(self.is_expanded)
        self._update_arrow()

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)

class SidebarButton(QWidget):
    def __init__(self, icon, text, theme, callback, parent=None):
        super().__init__(parent)
        self.callback = callback
        self.theme = theme

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.icon_frame = QFrame()
        self.icon_frame.setFixedSize(40, 40)
        icon_layout = QVBoxLayout(self.icon_frame)
        icon_layout.setContentsMargins(0, 0, 0, 0)

        self.icon_label = QLabel(icon)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet(f"font-size: 20px; color: {theme['accent']};")
        icon_layout.addWidget(self.icon_label)

        self.drawer = QFrame()
        self.drawer.setObjectName("drawer")
        self.drawer.setMaximumWidth(0)
        self.drawer.setStyleSheet(f"""
            QFrame#drawer {{
                background-color: {theme['bg']};
                border: 1px solid {theme['panel_border']};
                border-left: none;
                border-top-right-radius: 15px;
                border-bottom-right-radius: 15px;
            }}
        """)

        drawer_layout = QHBoxLayout(self.drawer)
        drawer_layout.setContentsMargins(8, 0, 12, 0)
        self.text_label = QLabel(text)
        self.text_label.setStyleSheet(f"font-size: 13px; color: {theme['text']}; font-weight: bold;")
        drawer_layout.addWidget(self.text_label)

        self.layout.addWidget(self.icon_frame)
        self.layout.addWidget(self.drawer)
        self.layout.addStretch()

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.animation = QPropertyAnimation(self.drawer, b"maximumWidth")
        self.animation.setDuration(250)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def enterEvent(self, event):
        self.animation.stop()
        self.animation.setEndValue(100)
        self.animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.animation.stop()
        self.animation.setEndValue(0)
        self.animation.start()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.pos()):
            if self.callback:
                self.callback()
        super().mouseReleaseEvent(event)

class SemanticSearchSideBar(QWidget):
    def __init__(self, parent=None, callbacks=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(2, 4, 0, 0)
        self.root_layout.setSpacing(8)
        self.root_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        theme = get_addon_theme()
        callbacks = callbacks or {}

        self.btn_search = SidebarButton("\U0001F50D", "Search", theme, callbacks.get('search'), self)
        self.btn_embed = SidebarButton("\U0001F4DA", "Embed", theme, callbacks.get('embed'), self)
        self.btn_settings = SidebarButton("\u2699", "Settings", theme, callbacks.get('settings'), self)

        self.root_layout.addWidget(self.btn_search)
        self.root_layout.addWidget(self.btn_embed)
        self.root_layout.addWidget(self.btn_settings)

        self.setFixedWidth(160)
        self.setFixedHeight(160)
        self.setMouseTracking(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        if hasattr(QPainter, 'RenderHint'):
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        else:
            try: painter.setRenderHint(QPainter.Antialiasing)
            except: pass

        theme = get_addon_theme()
        painter.setBrush(QColor(theme['bg']))
        painter.setPen(QColor(theme['panel_border']))
        painter.drawRoundedRect(QRect(0, 0, 44, 144), 22, 22)
