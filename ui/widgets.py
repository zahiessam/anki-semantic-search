import os
import sys
import re
from aqt.qt import *
from .theme import (
    collapsible_section_button_stylesheet,
    collapsible_section_content_stylesheet,
    get_addon_theme,
    settings_field_row_stylesheet,
    settings_status_label_style,
    settings_text_style,
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
        self._image_paste_callback = None
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def set_image_paste_callback(self, callback):
        self._image_paste_callback = callback

    def _try_attach_image_from_mime_data(self, mime_data):
        if mime_data is None or not mime_data.hasImage() or not self._image_paste_callback:
            return False
        image_data = mime_data.imageData()
        image = None
        if isinstance(image_data, QImage):
            image = image_data
        elif isinstance(image_data, QPixmap):
            image = image_data.toImage()
        elif hasattr(image_data, "toImage"):
            try:
                image = image_data.toImage()
            except Exception:
                image = None
        if image is None or image.isNull():
            return False
        self._image_paste_callback(image)
        return True

    def insertFromMimeData(self, source):
        try:
            if self._try_attach_image_from_mime_data(source):
                return
        except Exception as exc:
            log_debug(f"Image paste insert handling skipped: {exc}")
        super().insertFromMimeData(source)

    def keyPressEvent(self, event):
        try:
            paste_match = False
            try:
                paste_match = event.matches(QKeySequence.StandardKey.Paste)
            except Exception:
                key_enum = getattr(Qt, "Key", Qt)
                modifier_enum = getattr(Qt, "KeyboardModifier", Qt)
                key = event.key()
                modifiers = event.modifiers()
                paste_match = bool(
                    key == getattr(key_enum, "Key_V", getattr(Qt, "Key_V", None))
                    and (
                        modifiers & getattr(modifier_enum, "ControlModifier", getattr(Qt, "ControlModifier", 0))
                        or modifiers & getattr(modifier_enum, "MetaModifier", getattr(Qt, "MetaModifier", 0))
                    )
                )
            if paste_match:
                clipboard = QApplication.clipboard()
                mime_data = clipboard.mimeData() if clipboard is not None else None
                if self._try_attach_image_from_mime_data(mime_data):
                    event.accept()
                    return
        except Exception as exc:
            log_debug(f"Image paste handling skipped: {exc}")
        super().keyPressEvent(event)

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


def _theme_int(theme, key, default):
    try:
        return int(theme.get(key, default))
    except (TypeError, ValueError, AttributeError):
        return default


def settings_page(theme, title, subtitle=None, min_width=None, max_width=None):
    """Create the standard centered settings tab page and return (page, body_layout)."""
    page = QWidget()
    outer_layout = QHBoxLayout(page)
    outer_layout.setContentsMargins(0, 0, 0, 0)
    outer_layout.setSpacing(0)

    body = QWidget()
    body.setMinimumWidth(min_width or _theme_int(theme, "settings_page_min_width", 760))
    body.setMaximumWidth(max_width or _theme_int(theme, "settings_page_max_width", 1100))
    body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    outer_layout.addStretch(1)
    outer_layout.addWidget(body)
    outer_layout.addStretch(1)

    body_layout = QVBoxLayout(body)
    margin = _theme_int(theme, "settings_page_margin", 20)
    body_layout.setContentsMargins(margin, margin, margin, margin)
    body_layout.setSpacing(_theme_int(theme, "settings_page_spacing", 14))

    heading = QLabel(str(title))
    heading.setStyleSheet(settings_text_style(theme, "heading"))
    body_layout.addWidget(heading)
    page.heading_label = heading

    if subtitle:
        subtitle_label = QLabel(str(subtitle))
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet(settings_text_style(theme, "subtitle"))
        body_layout.addWidget(subtitle_label)
        page.subtitle_label = subtitle_label

    return page, body_layout


def settings_field_row(theme, content=None, label=None, layout=None, vertical=False, tooltip=None):
    row = QFrame()
    row.setObjectName("settingsFieldRow")
    row.setStyleSheet(settings_field_row_stylesheet(theme))

    row_layout = QVBoxLayout(row) if vertical else QHBoxLayout(row)
    row_layout.setContentsMargins(6, 4, 6, 4)
    row_layout.setSpacing(7)

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


def settings_status(theme, text="", state="info"):
    """Create a compact themed status label."""
    label = QLabel(str(text))
    label.setWordWrap(True)
    label.setStyleSheet(settings_status_label_style(theme, state))
    label._settings_status_state = state
    return label


def _settings_indent(level):
    return max(0, int(level)) * 12


def settings_subsection_header(theme, title, level=1):
    """Create a lightweight in-section label for related child settings."""
    label = QLabel(str(title))
    indent = _settings_indent(level)
    label.setContentsMargins(indent, 6, 0, 2)
    label.setStyleSheet(f"color: {theme['subtext']}; font-weight: bold; background: transparent;")
    return label


def settings_child_row(theme, content=None, label=None, layout=None, vertical=False, tooltip=None, level=1):
    """Wrap a settings row as a child/sub-item with consistent indentation."""
    wrapper = QFrame()
    wrapper.setObjectName("settingsChildRow")
    wrapper.setStyleSheet("QFrame#settingsChildRow { background: transparent; border: none; }")
    outer = QHBoxLayout(wrapper)
    outer.setContentsMargins(_settings_indent(level), 0, 0, 0)
    outer.setSpacing(0)

    row = settings_field_row(theme, content=content, label=label, layout=layout, vertical=vertical, tooltip=tooltip)
    wrapper._settings_content_row = row
    outer.addWidget(row, 1)
    apply_setting_row_tooltip(wrapper, tooltip or row.toolTip())
    return wrapper


def settings_child_group(theme, level=1):
    """Create an indented container for multiple related child widgets."""
    group = QFrame()
    group.setObjectName("settingsChildGroup")
    group.setStyleSheet("QFrame#settingsChildGroup { background: transparent; border: none; }")
    outer = QHBoxLayout(group)
    outer.setContentsMargins(_settings_indent(level), 0, 0, 0)
    outer.setSpacing(0)

    content = QWidget()
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(7)
    outer.addWidget(content, 1)
    group.content_layout = content_layout
    return group


def settings_compact_group(theme, title=None):
    """Create a compact boxed subsection for related settings."""
    group = QFrame()
    group.setObjectName("settingsCompactGroup")
    group.setStyleSheet(
        f"""
        QFrame#settingsCompactGroup {{
            background-color: {theme['panel_bg']};
            border: 1px solid {theme['subtle_border']};
            border-radius: 6px;
        }}
        """
    )
    layout = QVBoxLayout(group)
    layout.setContentsMargins(10, 9, 10, 9)
    layout.setSpacing(8)
    group.content_layout = layout
    if title:
        label = QLabel(str(title))
        label.setStyleSheet(
            f"color: {theme['subtext']}; font-weight: bold; background: transparent; border: none;"
        )
        layout.addWidget(label)
        group.title_label = label
    return group


def settings_section(theme, title, collapsible=False, expanded=True):
    """Create a shared settings section; returns a widget with .content_layout."""
    if collapsible:
        return CollapsibleSection(str(title), is_expanded=expanded)
    radius = _theme_int(theme, "settings_radius", 6)
    section = QFrame()
    section.setObjectName("settingsSection")
    section.setStyleSheet(
        f"""
        QFrame#settingsSection {{
            background-color: {theme['section_bg']};
            border: 1px solid {theme['subtle_border']};
            border-radius: {radius}px;
        }}
        QFrame#settingsSectionHeader {{
            background-color: {theme['section_header_bg']};
            border: none;
            border-top-left-radius: {radius}px;
            border-top-right-radius: {radius}px;
            border-bottom: 1px solid {theme['subtle_border']};
        }}
        QLabel#settingsSectionTitle {{
            color: {theme['text']};
            font-weight: bold;
            background: transparent;
            border: none;
        }}
        QWidget#settingsSectionContent {{
            background: transparent;
            border: none;
        }}
        """
    )

    outer = QVBoxLayout(section)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    header = QFrame()
    header.setObjectName("settingsSectionHeader")
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(12, 8, 12, 8)
    header_layout.setSpacing(8)

    title_label = QLabel(str(title))
    title_label.setObjectName("settingsSectionTitle")
    header_layout.addWidget(title_label)
    header_layout.addStretch(1)
    outer.addWidget(header)

    content = QWidget()
    content.setObjectName("settingsSectionContent")
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(12, 10, 12, 12)
    content_layout.setSpacing(_theme_int(theme, "settings_section_spacing", 8))
    outer.addWidget(content)

    section.header = header
    section.title_label = title_label
    section.content_widget = content
    section.content_layout = content_layout
    return section


def settings_inline_row(theme, label, control, control_width=200, tooltip=None):
    """Create a compact form row with the control close to its label."""
    row = QFrame()
    row.setObjectName("settingsInlineRow")
    row.setToolTip(tooltip or getattr(control, "toolTip", lambda: "")())
    row.setStyleSheet("QFrame#settingsInlineRow { background: transparent; border: none; }")
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)

    label_widget = label if isinstance(label, QLabel) else QLabel(str(label))
    label_widget.setStyleSheet(
        f"color: {theme['subtext']}; background: transparent; border: none;"
    )
    label_widget.setMinimumWidth(_theme_int(theme, "settings_row_label_width", 150))
    label_widget.setMaximumWidth(_theme_int(theme, "settings_row_label_max_width", 210))
    label_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
    label_widget.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
    layout.addWidget(label_widget)

    if control_width:
        control.setMinimumWidth(control_width)
        control.setMaximumWidth(control_width)
        control.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    layout.addWidget(control, 0)
    layout.addStretch(1)
    return row


def settings_labeled_action_row(theme, label, field, action_button=None, control_width=620, tooltip=None):
    """Create a form row with one expanding field and an optional compact action."""
    control = QWidget()
    control.setObjectName("settingsLabeledActionControl")
    control.setStyleSheet("QWidget#settingsLabeledActionControl { background: transparent; border: none; }")
    control_layout = QHBoxLayout(control)
    control_layout.setContentsMargins(0, 0, 0, 0)
    control_layout.setSpacing(8)
    field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    control_layout.addWidget(field, 1)
    if action_button is not None:
        action_button.setMaximumWidth(_theme_int(theme, "settings_inline_action_max_width", 120))
        control_layout.addWidget(action_button, 0)
    if control_width:
        control.setMinimumWidth(min(320, control_width))
        control.setMaximumWidth(control_width)
    return settings_inline_row(theme, label, control, control_width=0, tooltip=tooltip)


def settings_form_row(theme, label, control, action=None, control_width=620, tooltip=None):
    """Canonical settings form row."""
    return settings_labeled_action_row(theme, label, control, action, control_width, tooltip)


def settings_button_strip(theme, *buttons, align_left=True):
    """Create a transparent row for secondary actions without making them full-width."""
    row = QFrame()
    row.setObjectName("settingsButtonStrip")
    row.setStyleSheet("QFrame#settingsButtonStrip { background: transparent; border: none; }")
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    if not align_left:
        layout.addStretch(1)
    for button in buttons:
        if button is None:
            continue
        if button.maximumWidth() >= 16777215:
            button.setMaximumWidth(_theme_int(theme, "settings_action_max_width", 260))
        layout.addWidget(button, 0)
    if align_left:
        layout.addStretch(1)
    return row


def settings_toolbar(theme, *buttons, align_left=True):
    """Canonical settings action toolbar."""
    return settings_button_strip(theme, *buttons, align_left=align_left)


def settings_compact_checkbox_row(theme, checkbox):
    """Create a low-padding row for a checkbox setting."""
    row = QFrame()
    row.setObjectName("settingsCompactCheckboxRow")
    row.setStyleSheet("QFrame#settingsCompactCheckboxRow { background: transparent; border: none; }")
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    checkbox.setStyleSheet("QCheckBox { background: transparent; border: none; }")
    layout.addWidget(checkbox)
    layout.addStretch(1)
    return row


def settings_checkbox_row(theme, checkbox, hint=None):
    """Canonical checkbox row with optional subtle helper text."""
    if not hint:
        return settings_compact_checkbox_row(theme, checkbox)
    wrapper = QFrame()
    wrapper.setObjectName("settingsCheckboxWithHint")
    wrapper.setStyleSheet("QFrame#settingsCheckboxWithHint { background: transparent; border: none; }")
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(settings_compact_checkbox_row(theme, checkbox))
    hint_label = QLabel(str(hint))
    hint_label.setWordWrap(True)
    hint_label.setStyleSheet(settings_text_style(theme, "subtle"))
    layout.addWidget(hint_label)
    return wrapper


def settings_hint_box(theme, text_or_widget):
    """Create a compact muted hint panel."""
    box = QFrame()
    box.setObjectName("settingsHintBox")
    box.setStyleSheet(
        f"""
        QFrame#settingsHintBox {{
            background-color: {theme['panel_bg']};
            border: none;
            border-radius: 5px;
        }}
        """
    )
    layout = QVBoxLayout(box)
    layout.setContentsMargins(8, 4, 8, 4)
    if isinstance(text_or_widget, QWidget):
        widget = text_or_widget
        widget.setStyleSheet(f"color: {theme['subtext']}; background: transparent; border: none;")
    else:
        widget = QLabel(str(text_or_widget))
        widget.setWordWrap(True)
        widget.setStyleSheet(f"color: {theme['subtext']}; background: transparent; border: none;")
    layout.addWidget(widget)
    return box


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
        self.content_area.setObjectName("collapsibleSectionContent")
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(12, 10, 12, 12)
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

    def setTitle(self, title):
        self.toggle_button.setText(str(title))
        self._update_arrow()

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
        self.animation.setEndValue(130)
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

        from .branding import CHATBOT_ICON, CHATBOT_NAME

        self.btn_ask_ai = SidebarButton(CHATBOT_ICON, CHATBOT_NAME, theme, callbacks.get('ask_ai'), self)
        self.btn_embed = SidebarButton("\U0001F4DA", "Embed", theme, callbacks.get('embed'), self)
        self.btn_settings = SidebarButton("\u2699", "Settings", theme, callbacks.get('settings'), self)

        self.root_layout.addWidget(self.btn_ask_ai)
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
