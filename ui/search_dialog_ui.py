"""Search dialog defaults and UI construction."""

# ============================================================================
# Imports
# ============================================================================

from aqt.qt import *
from aqt.utils import askUser, showInfo, tooltip

from .search_dialog import ContentDelegate, RelevanceBarDelegate
from .branding import CHATBOT_ICON, CHATBOT_NAME
from .note_preview_popup import NotePreviewPopup
from .theme import get_addon_theme
from .image_attachments import ATTACHMENT_THUMBNAIL_SIZE
from .widgets import SpellCheckPlainTextEdit, _get_spell_checker
from ..utils import clamp_relevance_threshold_percent, get_search_history_queries, load_config


# ============================================================================
# Search Dialog UI Construction
# ============================================================================

_addon_theme = get_addon_theme


def _panel_style(theme):
    return (
        f"background-color: {theme['section_bg']}; "
        f"border: 1px solid {theme['subtle_border']}; "
        "border-radius: 6px;"
    )


def _filter_strip_style(theme):
    return (
        f"background-color: {theme['panel_bg']}; "
        f"border: 1px solid {theme['subtle_border']}; "
        "border-radius: 5px;"
    )


def reset_to_medical_defaults(self):

    """Compatibility wrapper; the active preset lives in SettingsRerankUiMixin."""
    showInfo("Open Settings to apply the Clinical High-Yield preset.")


def _set_sources_toggle_text(self):
    button = getattr(self, "sources_toggle_btn", None)
    if button is None:
        return
    collapsed = bool(getattr(self, "_sources_collapsed", False))
    button.setText("\u2039" if collapsed else "\u203a")
    button.setToolTip("Show Sources" if collapsed else "Hide Sources")


def _collapse_sources_panel(self, manual=False):
    if not getattr(self, "use_side_by_side", False):
        return
    content = getattr(self, "results_container", None)
    shell = getattr(self, "sources_shell", None)
    splitter = getattr(self, "main_splitter", None)
    if content is None or shell is None or splitter is None:
        return

    if manual:
        self._sources_manually_expanded = False
    self._sources_collapsed = True
    content.setVisible(False)
    shell.setMinimumWidth(18)
    shell.setMaximumWidth(18)
    splitter.setSizes([max(800, self.width() - 18), 18])
    _set_sources_toggle_text(self)


def _expand_sources_panel(self, manual=False):
    if not getattr(self, "use_side_by_side", False):
        return
    content = getattr(self, "results_container", None)
    shell = getattr(self, "sources_shell", None)
    splitter = getattr(self, "main_splitter", None)
    if content is None or shell is None or splitter is None:
        return

    if manual:
        self._sources_manually_expanded = True
    self._sources_collapsed = False
    shell.setMaximumWidth(16777215)
    shell.setMinimumWidth(260)
    content.setVisible(True)
    total = max(900, self.width())
    sources_width = max(360, int(total * 0.42))
    splitter.setSizes([max(360, total - sources_width), sources_width])
    _set_sources_toggle_text(self)


def _toggle_sources_panel(self):
    if getattr(self, "_sources_collapsed", False):
        _expand_sources_panel(self, manual=True)
    else:
        _collapse_sources_panel(self, manual=True)



def setup_ui(self):



    root_layout = QHBoxLayout()



    root_layout.setSpacing(8)



    root_layout.setContentsMargins(6, 4, 6, 6)







    main_container = QWidget()



    layout = QVBoxLayout(main_container)



    section_spacing = self.styling_config.get('section_spacing', 8)



    layout.setSpacing(section_spacing)



    layout.setContentsMargins(8, 4, 8, 6)







    palette = QApplication.palette()



    is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128







    theme = _addon_theme(is_dark)
    self._theme = theme
    self._answer_font_size = self.styling_config.get('answer_font_size', 13)
    if hasattr(self, "_init_search_chat_state"):
        self._init_search_chat_state()







    # Compact header: keep metadata and settings out of the conversation lane.



    header_layout = QHBoxLayout()



    header_layout.addStretch()



    self.scope_banner = QLabel("")



    self.scope_banner.setStyleSheet(f"font-size: 10px; color: {'#95a5a6' if is_dark else '#7f8c8d'}; padding: 2px 8px;")



    self.scope_banner.setToolTip("Search scope. Use the Settings button to change note types, fields, decks.")



    header_layout.addWidget(self.scope_banner)

    self.review_context_banner = QLabel("")
    self.review_context_banner.setStyleSheet(
        f"font-size: 10px; color: {theme['teal']}; padding: 2px 8px; font-weight: 700;"
    )
    self.review_context_banner.setToolTip("Current review note context is available for Ask Notes and Find Related Notes.")
    self.review_context_banner.setVisible(False)
    header_layout.addWidget(self.review_context_banner)



    settings_btn = QPushButton("\u2699 Settings")



    settings_btn.setObjectName("settingsBtn")



    settings_btn.setToolTip("Configure API key, note types, decks, and search behavior")



    settings_btn.clicked.connect(self.open_settings)



    header_layout.addWidget(settings_btn)



    layout.addLayout(header_layout)







    # Chat composer. It is built early so existing helpers can keep
    # using self.search_input, then added directly under the answer box.



    search_container = QWidget()
    search_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)



    search_container.setStyleSheet("background-color: transparent; border-radius: 0; padding: 0;")



    search_layout = QVBoxLayout(search_container)



    search_layout.setSpacing(4)



    search_layout.setContentsMargins(2, 4, 2, 2)







    label_font_size = max(11, self.styling_config.get('label_font_size', 13) - 2)



    search_label = QLabel("Ask:")



    search_label.setToolTip("Type a question. Ask AI answers directly; Ask Notes searches your Anki notes with citations.")



    search_label.setStyleSheet(f"font-weight: bold; font-size: {label_font_size}px; color: {'#ffffff' if is_dark else '#2c3e50'};")



    search_layout.addWidget(search_label)







    search_input_layout = QVBoxLayout()
    search_input_layout.setSpacing(4)
    search_input_layout.setContentsMargins(0, 0, 0, 0)



    try:



        self.search_input = SpellCheckPlainTextEdit()
        if hasattr(self.search_input, "set_image_paste_callback"):
            self.search_input.set_image_paste_callback(self._attach_composer_image_from_qimage)



    except Exception:



        self.search_input = QPlainTextEdit()



    self.search_input.setPlaceholderText("Ask AI, or search your notes with Ask Notes...")



    self.search_input.setMinimumHeight(44)



    self.search_input.setMaximumHeight(100)



    spell_hint = " Right-click misspelled words for corrections." if _get_spell_checker() else ""



    self.search_input.setToolTip(f"Type your question.{spell_hint}")



    question_font_size = max(11, self.styling_config.get('question_font_size', 13) - 1)



    self.search_input.setStyleSheet(
        f"QPlainTextEdit {{ font-size: {question_font_size}px; border: 1px solid {theme['subtle_border']}; }}"
        f"QPlainTextEdit:focus {{ border: 1px solid {theme['focus_border']}; }}"
    )

    # History dropdown for recent searches, with per-item delete buttons.

    self._search_history_model = QStringListModel(get_search_history_queries())

    self.search_history_btn = QToolButton()

    self.search_history_btn.setObjectName("searchHistoryBtn")
    self.search_history_btn.setText("Recent")

    try:

        self.search_history_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

    except AttributeError:

        self.search_history_btn.setPopupMode(QToolButton.InstantPopup)

    self.search_history_btn.setMinimumHeight(30)
    self.search_history_btn.setMinimumWidth(72)

    self.search_history_btn.setMaximumWidth(110)

    self._rebuild_search_history_menu(self._search_history_model.stringList())



    clear_history_btn = QPushButton("Clear")
    clear_history_btn.setObjectName("clearHistoryBtn")

    clear_history_btn.setToolTip("Delete all search history")

    clear_history_btn.setMinimumHeight(30)
    clear_history_btn.setMaximumWidth(68)

    clear_history_btn.clicked.connect(self._on_clear_search_history)


    self.selected_answer_context_chip = QFrame()
    self.selected_answer_context_chip.setObjectName("selectedAnswerContextChip")
    self.selected_answer_context_chip.setStyleSheet(
        "QFrame#selectedAnswerContextChip {"
        f"  background-color: {theme['control_bg']};"
        f"  border: 1px solid {theme['subtle_border']};"
        "  border-radius: 8px;"
        "}"
        "QLabel { background: transparent; border: none; }"
        "QPushButton {"
        "  background: transparent;"
        "  border: none;"
        "  border-radius: 7px;"
        "  padding: 1px 5px;"
        "  font-weight: bold;"
        f"  color: {theme['quiet_text']};"
        "}"
        "QPushButton:hover {"
        f"  background-color: {theme['panel_bg']};"
        f"  color: {theme['text']};"
        "}"
    )
    selected_context_layout = QHBoxLayout(self.selected_answer_context_chip)
    selected_context_layout.setContentsMargins(8, 5, 6, 5)
    selected_context_layout.setSpacing(6)

    self.selected_answer_context_count_label = QLabel("1 selection")
    self.selected_answer_context_count_label.setStyleSheet(
        f"color: {theme['text']}; font-size: 11px; font-weight: 700;"
    )
    selected_context_layout.addWidget(self.selected_answer_context_count_label, 0)

    self.selected_answer_context_preview_label = QLabel("")
    self.selected_answer_context_preview_label.setStyleSheet(
        f"color: {theme['quiet_text']}; font-size: 11px;"
    )
    self.selected_answer_context_preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    selected_context_layout.addWidget(self.selected_answer_context_preview_label, 1)

    self.selected_answer_context_clear_btn = QPushButton("\u00d7")
    self.selected_answer_context_clear_btn.setFixedSize(22, 22)
    self.selected_answer_context_clear_btn.setToolTip("Remove selected text context")
    self.selected_answer_context_clear_btn.clicked.connect(self._clear_composer_selected_answer_context)
    selected_context_layout.addWidget(self.selected_answer_context_clear_btn, 0)

    self.selected_answer_context_chip.hide()

    self.composer_image_chip = QFrame()
    self.composer_image_chip.setObjectName("composerImageChip")
    self.composer_image_chip.setStyleSheet(
        "QFrame#composerImageChip {"
        f"  background-color: {theme['control_bg']};"
        f"  border: 1px solid {theme['subtle_border']};"
        "  border-radius: 8px;"
        "}"
        "QLabel { background: transparent; border: none; }"
        "QPushButton {"
        "  background: transparent;"
        "  border: none;"
        "  border-radius: 7px;"
        "  padding: 1px 5px;"
        "  font-weight: bold;"
        f"  color: {theme['quiet_text']};"
        "}"
        "QPushButton:hover {"
        f"  background-color: {theme['panel_bg']};"
        f"  color: {theme['text']};"
        "}"
    )
    image_chip_layout = QHBoxLayout(self.composer_image_chip)
    image_chip_layout.setContentsMargins(8, 5, 6, 5)
    image_chip_layout.setSpacing(7)
    self.composer_image_thumb_label = QLabel()
    self.composer_image_thumb_label.setFixedSize(ATTACHMENT_THUMBNAIL_SIZE, ATTACHMENT_THUMBNAIL_SIZE)
    self.composer_image_thumb_label.setScaledContents(False)
    image_chip_layout.addWidget(self.composer_image_thumb_label, 0)
    self.composer_image_name_label = QLabel("")
    self.composer_image_name_label.setWordWrap(False)
    self.composer_image_name_label.setStyleSheet(f"color: {theme['text']}; font-size: 11px;")
    self.composer_image_name_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    image_chip_layout.addWidget(self.composer_image_name_label, 1)
    self.composer_image_clear_btn = QPushButton("\u00d7")
    self.composer_image_clear_btn.setFixedSize(22, 22)
    self.composer_image_clear_btn.setToolTip("Remove attached image")
    self.composer_image_clear_btn.clicked.connect(self._clear_composer_image_attachment)
    image_chip_layout.addWidget(self.composer_image_clear_btn, 0)
    self.composer_image_chip.hide()


    self.ask_ai_btn = QPushButton(f"{CHATBOT_ICON} Ask AI")
    self.ask_ai_btn.setObjectName("askAiBtn")
    self.ask_ai_btn.setMinimumHeight(32)
    self.ask_ai_btn.setMinimumWidth(82)
    self.ask_ai_btn.setToolTip("Answer using AI reasoning without searching notes or adding citations.")
    self.ask_ai_btn.clicked.connect(self.ask_ai_direct_from_composer)

    self.search_btn = QPushButton("\U0001F4DA Ask Notes")
    self.ask_notes_btn = self.search_btn

    self.search_btn.setObjectName("searchBtn")

    self.search_btn.setMinimumHeight(32)

    self.search_btn.setMinimumWidth(92)
    self.search_btn.setToolTip("Search your Anki notes and answer with citations.")

    self.search_btn.clicked.connect(self._ask_notes_from_composer if hasattr(self, "_ask_notes_from_composer") else self.perform_search)

    self.clear_chat_btn = QPushButton("Clear Chat")
    self.clear_chat_btn.setObjectName("clearChatBtn")
    self.clear_chat_btn.setMinimumHeight(32)
    self.clear_chat_btn.setMinimumWidth(82)
    self.clear_chat_btn.setToolTip("Clear this chat session. Saved search history is unchanged.")
    self.clear_chat_btn.clicked.connect(self._clear_search_chat)

    self.find_related_btn = QPushButton("\U0001F50E Find Related Notes")
    self.find_related_btn.setObjectName("findRelatedBtn")
    self.find_related_btn.setMinimumHeight(32)
    self.find_related_btn.setToolTip("Search for notes related to the current review note.")
    self.find_related_btn.clicked.connect(self.find_related_notes_from_review)
    self.find_related_btn.setVisible(False)



    search_input_layout.addWidget(self.selected_answer_context_chip)
    search_input_layout.addWidget(self.composer_image_chip)
    search_input_layout.addWidget(self.search_input)

    composer_actions_layout = QHBoxLayout()
    composer_actions_layout.setSpacing(6)
    composer_actions_layout.setContentsMargins(0, 0, 0, 0)

    composer_utility_strip = QFrame()
    composer_utility_strip.setObjectName("composerUtilityStrip")
    composer_utility_strip.setStyleSheet(
        "QFrame#composerUtilityStrip {"
        f"  background-color: {theme['section_bg']};"
        f"  border: 1px solid {theme['subtle_border']};"
        "  border-radius: 7px;"
        "}"
        "QFrame#composerUtilityStrip QToolButton,"
        "QFrame#composerUtilityStrip QPushButton {"
        "  background-color: transparent;"
        f"  border: 1px solid transparent;"
        f"  color: {theme['subtext']};"
        "  border-radius: 5px;"
        "  padding: 4px 8px;"
        "  font-size: 12px;"
        "  font-weight: 600;"
        "}"
        "QFrame#composerUtilityStrip QToolButton:hover,"
        "QFrame#composerUtilityStrip QPushButton:hover {"
        f"  background-color: {theme['panel_bg']};"
        f"  border-color: {theme['control_hover_border']};"
        f"  color: {theme['text']};"
        "}"
        "QFrame#composerUtilityStrip QToolButton:disabled,"
        "QFrame#composerUtilityStrip QPushButton:disabled {"
        "  background-color: transparent;"
        f"  border-color: transparent;"
        f"  color: {theme['quiet_text']};"
        "}"
        "QPushButton#attachImageBtn {"
        "  padding: 0;"
        f"  color: {theme['text']};"
        "  font-size: 15px;"
        "  font-weight: 700;"
        "}"
    )
    utility_layout = QHBoxLayout(composer_utility_strip)
    utility_layout.setContentsMargins(4, 4, 4, 4)
    utility_layout.setSpacing(2)
    utility_layout.addWidget(self.search_history_btn)
    utility_layout.addWidget(clear_history_btn)
    self.attach_image_btn = QPushButton("\U0001F4CE")
    self.attach_image_btn.setObjectName("attachImageBtn")
    self.attach_image_btn.setFixedSize(32, 30)
    self.attach_image_btn.setToolTip("Attach image (PNG, JPG, WebP)")
    self.attach_image_btn.clicked.connect(self._choose_composer_image_attachment)
    utility_layout.addWidget(self.attach_image_btn)
    composer_actions_layout.addWidget(composer_utility_strip)
    composer_actions_layout.addStretch()
    composer_actions_layout.addWidget(self.find_related_btn)
    composer_actions_layout.addWidget(self.ask_ai_btn)
    composer_actions_layout.addWidget(self.search_btn)
    composer_actions_layout.addWidget(self.clear_chat_btn)
    search_input_layout.addLayout(composer_actions_layout)

    search_layout.addLayout(search_input_layout)













    # Ctrl+Enter to search



    search_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)



    search_shortcut.activated.connect(self._ask_notes_from_composer if hasattr(self, "_ask_notes_from_composer") else self.perform_search)







    # Keyboard shortcuts for select/deselect



    select_all_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)



    select_all_shortcut.activated.connect(self.select_all_notes)



    deselect_all_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)



    deselect_all_shortcut.activated.connect(self.deselect_all_notes)














    # Splitter for resizable sections (side-by-side or stacked)



    layout_mode = self.styling_config.get('layout_mode', 'side_by_side')



    self.use_side_by_side = layout_mode == 'side_by_side'



    split_orientation = Qt.Orientation.Horizontal if self.use_side_by_side else Qt.Orientation.Vertical



    main_splitter = QSplitter(split_orientation)
    self.main_splitter = main_splitter
    self._sources_collapsed = False
    self._sources_manually_expanded = False







    # Chat transcript section



    answer_container = QWidget()
    self.answer_container = answer_container



    answer_container.setStyleSheet(_panel_style(theme))



    answer_layout = QVBoxLayout(answer_container)
    answer_layout.setSpacing(7)
    answer_layout.setContentsMargins(8, 7, 8, 8)







    answer_header = QHBoxLayout()



    answer_label = QLabel(f"{CHATBOT_ICON} Conversation")



    answer_label.setStyleSheet(
        f"font-weight: bold; font-size: 13px; color: {theme['text']}; "
        "background: transparent; border: none; padding: 0;"
    )



    answer_header.addWidget(answer_label)



    answer_header.addStretch()



    answer_layout.addLayout(answer_header)







    try:



        self.answer_box = QTextBrowser()



    except NameError:



        self.answer_box = QTextEdit()



    self.answer_box.setObjectName("answerBox")
    self.answer_box.setReadOnly(True)

    if hasattr(Qt, 'ScrollBarPolicy'):
        self.answer_box.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.answer_box.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    else:
        self.answer_box.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.answer_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)



    self.answer_box.setOpenExternalLinks(False)



    # Ensure link clicks are emitted (not opened) so citation links work (Ollama and all providers)



    if hasattr(self.answer_box, 'setOpenLinks'):



        self.answer_box.setOpenLinks(False)



    if hasattr(self.answer_box, 'setOpenExternalLinks'):



        self.answer_box.setOpenExternalLinks(False)



    # Explicitly enable link interaction so [1], [2] citation links are clickable



    if hasattr(Qt, 'TextInteractionFlag') and hasattr(Qt.TextInteractionFlag, 'TextBrowserInteraction'):



        self.answer_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)



    elif hasattr(Qt, 'TextBrowserInteraction'):



        self.answer_box.setTextInteractionFlags(Qt.TextBrowserInteraction)



    if hasattr(self.answer_box, 'setPlaceholderText'):



        self.answer_box.setPlaceholderText("Ask AI for direct reasoning, or Ask Notes for cited answers from your Anki notes.")



    self.answer_box.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu if hasattr(Qt, "ContextMenuPolicy") else Qt.CustomContextMenu)
    self.answer_box.customContextMenuRequested.connect(self._show_answer_context_menu)

    self.answer_box.setMinimumHeight(100)



    self.answer_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



    answer_font_size = self.styling_config.get('answer_font_size', 13)
    answer_bg = '#2d2d2d' if is_dark else '#ffffff'
    answer_text = '#ffffff' if is_dark else '#1a1a1a'
    answer_selection_bg = "rgba(47, 129, 247, 0.24)" if is_dark else "rgba(47, 129, 247, 0.16)"
    answer_selection_text = "#ffffff" if is_dark else "#111827"
    self.answer_box.setStyleSheet(



        f"QTextBrowser#answerBox, QTextEdit#answerBox {{ background-color: {answer_bg}; "



        f"border: 1px solid {theme['subtle_border']}; color: {answer_text}; "



        f"font-size: {answer_font_size}px; padding: 9px; }} "

        f"QTextBrowser#answerBox::selection, QTextEdit#answerBox::selection {{ background-color: {answer_selection_bg}; color: {answer_selection_text}; }} "



        f"QTextBrowser#answerBox a, QTextEdit#answerBox a {{ color: #3498db; text-decoration: underline; }} "



        f"QTextBrowser#answerBox a:hover, QTextEdit#answerBox a:hover {{ color: #5dade2; }} "



    )



    # Connect link click: anchorClicked (Qt6) or linkActivated (PyQt5) so citation links work with Ollama and all providers



    if hasattr(self.answer_box, 'anchorClicked'):



        self.answer_box.anchorClicked.connect(self._on_answer_link_clicked)



    elif hasattr(self.answer_box, 'linkActivated'):



        self.answer_box.linkActivated.connect(self._on_answer_link_clicked)



    answer_layout.addWidget(self.answer_box, 1)

    answer_layout.addWidget(search_container, 0)







    main_splitter.addWidget(answer_container)







    if hasattr(self, "_render_chat_transcript"):
        self._render_chat_transcript()

    # Sources section



    results_container = QWidget()
    self.results_container = results_container
    results_container.setStyleSheet(_panel_style(theme))



    results_layout = QVBoxLayout(results_container)
    results_layout.setContentsMargins(8, 7, 8, 8)
    results_layout.setSpacing(7)







    results_header = QHBoxLayout()



    results_label = QLabel("\U0001F4CB Sources:")



    results_label.setToolTip("Notes found by Ask Notes or Find Related Notes. Citations in Ask Notes answers link back to these sources.")



    results_label.setStyleSheet(
        f"font-weight: bold; font-size: 13px; color: {theme['text']}; "
        "background: transparent; border: none; padding: 0;"
    )



    results_header.addWidget(results_label)



    self.selected_count_label = QLabel("(0 selected)")



    self.selected_count_label.setStyleSheet(f"font-size: {label_font_size - 2}px; color: {'#95a5a6' if is_dark else '#7f8c8d'}; font-style: italic;")



    self.selected_count_label.setToolTip(f"Notes you checked (for \"View Selected\"). Not the same as \"cited\" in the {CHATBOT_NAME} answer\u2014use \"Show only cited notes\" for that.")



    results_header.addWidget(self.selected_count_label)



    results_header.addStretch()

    results_layout.addLayout(results_header)







    # Create table: Select | Ref (citation [1],[2]...) | Content | Note ID | Relevance



    self.results_list = QTableWidget()



    self.results_list.setColumnCount(5)



    self.results_list.setHorizontalHeaderLabels(["✓", "Ref", "Content", "Note ID", "Relevance"])



    self.results_list.setMinimumHeight(120)



    self.results_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



    notes_font_size = self.styling_config.get('notes_font_size', 12)



    self.results_list.setStyleSheet(
        f"QTableWidget {{ font-size: {notes_font_size}px; border: 1px solid {theme['subtle_border']}; border-radius: 5px; }}"
        f"QTableWidget:focus {{ border: 1px solid {theme['focus_border']}; }}"
        "QTableWidget::item { padding: 5px 6px; }"
    )



    self.results_list.setWordWrap(True)
    self.results_list.verticalHeader().setDefaultSectionSize(max(32, notes_font_size + 18))







    # Configure columns



    self.results_list.setColumnWidth(0, 38)   # Selection checkbox



    self.results_list.setColumnWidth(1, 42)   # Ref (citation number matching [1], [2] in answer)



    self.results_list.setColumnWidth(2, 400)  # Content



    self.results_list.setColumnWidth(3, 80)   # Note ID (hidden by default)



    self.results_list.setColumnWidth(4, 100)  # Relevance (bar + %)

    self.results_list.setColumnHidden(3, True)  # Hide Note ID column (right-click header to show)



    self.results_list.horizontalHeader().setStretchLastSection(False)



    self.results_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)



    self.results_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)



    self.results_list.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)



    self.results_list.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)

    self.results_list.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)



    # Make column headers readable on dark and light themes (was hard to see on dark)



    header_color = "#ecf0f1" if is_dark else "#2c3e50"



    self.results_list.horizontalHeader().setStyleSheet(



        f"color: {header_color}; font-weight: bold; font-size: {max(11, notes_font_size - 1)}px;"



    )







    self.results_list.setSortingEnabled(True)
    self.results_list.setMouseTracking(True)



    self.results_list.setItemDelegateForColumn(2, ContentDelegate(self.results_list)) # First field normally, both on hover

    self.results_list.setItemDelegateForColumn(4, RelevanceBarDelegate(self.results_list))  # Relevance bar + %



    self.results_list.sortItems(4, Qt.SortOrder.DescendingOrder)  # Sort by Relevance







    # Enable double-click on rows



    self.results_list.itemDoubleClicked.connect(self.open_in_browser)
    self._note_preview_popup = NotePreviewPopup(self)
    self.results_list.cellEntered.connect(self._show_note_preview_for_cell)
    self.results_list.viewport().installEventFilter(self._note_preview_popup)







    # Hide vertical header (row numbers)



    self.results_list.verticalHeader().setVisible(False)







    # Set selection behavior to select entire rows



    self.results_list.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)



    self.results_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)







    # Enable alternating row colors (zebra striping)



    self.results_list.setAlternatingRowColors(True)







    results_layout.addWidget(self.results_list, 1)







    # Label at bottom of notes area: which search mechanism yielded these results



    self.search_method_result_label = QLabel("")



    self.search_method_result_label.setStyleSheet(



        "font-size: 11px; color: #7f8c8d; padding: 4px 0; margin-top: 4px;"



    )



    self.search_method_result_label.setWordWrap(True)



    results_layout.addWidget(self.search_method_result_label)







    sources_shell = QWidget()
    self.sources_shell = sources_shell
    sources_shell_layout = QHBoxLayout(sources_shell)
    sources_shell_layout.setContentsMargins(0, 0, 0, 0)
    sources_shell_layout.setSpacing(4)

    sources_handle_rail = QWidget()
    self.sources_handle_rail = sources_handle_rail
    sources_handle_rail.setFixedWidth(18)
    sources_handle_layout = QVBoxLayout(sources_handle_rail)
    sources_handle_layout.setContentsMargins(0, 0, 0, 0)
    sources_handle_layout.setSpacing(0)

    self.sources_toggle_btn = QPushButton("\u2039")
    self.sources_toggle_btn.setObjectName("sourcesToggleBtn")
    self.sources_toggle_btn.setFixedSize(18, 42)
    self.sources_toggle_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    self.sources_toggle_btn.setToolTip("Show Sources")
    self.sources_toggle_btn.setStyleSheet(
        "QPushButton#sourcesToggleBtn {"
        "  padding: 0;"
        "  margin: 0;"
        "  border-radius: 4px;"
        "  border: 1px solid rgba(149, 165, 166, 0.28);"
        "  background-color: rgba(255, 255, 255, 0.015);"
        "  color: rgba(236, 240, 241, 0.70);"
        "  font-size: 18px;"
        "  font-weight: bold;"
        "}"
        "QPushButton#sourcesToggleBtn:hover {"
        "  border: 1px solid #3498db;"
        "  background-color: rgba(52, 152, 219, 0.12);"
        "  color: #d6ecff;"
        "}"
        "QPushButton#sourcesToggleBtn:focus {"
        "  border: 1px solid #3498db;"
        "  background-color: rgba(52, 152, 219, 0.16);"
        "  color: #ffffff;"
        "}"
    )
    self.sources_toggle_btn.clicked.connect(self._toggle_sources_panel)
    self.sources_toggle_btn.setVisible(self.use_side_by_side)
    sources_handle_layout.addStretch()
    sources_handle_layout.addWidget(self.sources_toggle_btn)
    sources_handle_layout.addStretch()
    sources_shell_layout.addWidget(sources_handle_rail)
    sources_shell_layout.addWidget(results_container, 1)

    main_splitter.addWidget(sources_shell)



    main_splitter.setSizes([450, 550] if self.use_side_by_side else [350, 450])



    main_splitter.setChildrenCollapsible(False)



    main_splitter.setHandleWidth(8)







    layout.addWidget(main_splitter, 1)







    # Result tuning: single relevance threshold slider



    sensitivity_container = QWidget()



    sensitivity_container.setObjectName("sourceFilterStrip")
    sensitivity_container.setStyleSheet(_filter_strip_style(theme))



    sensitivity_layout = QVBoxLayout(sensitivity_container)



    sensitivity_layout.setSpacing(5)



    sensitivity_layout.setContentsMargins(8, 6, 8, 6)



    self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
    self.sensitivity_slider.setRange(20, 80)
    self.sensitivity_slider.setSingleStep(1)
    self.sensitivity_slider.setPageStep(5)
    self.sensitivity_slider.setMinimumWidth(120)
    self.sensitivity_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    self.sensitivity_slider.setToolTip("Controls visible results and AI context eligibility. Use Show All Source Notes to inspect lower-relevance retained notes.")
    try:
        threshold_value = clamp_relevance_threshold_percent(
            (load_config().get("search_config") or {}).get("relevance_threshold_percent", 65)
        )
    except Exception:
        threshold_value = 65
    self._effective_relevance_threshold_percent = threshold_value
    self._relevance_threshold_source = "config"
    self.sensitivity_slider.setValue(threshold_value)



    self.sensitivity_value_label = QLabel(f"{threshold_value}%")
    self.sensitivity_value_label.setMinimumWidth(42)
    self.sensitivity_value_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)







    # Relevance threshold. The cited-notes filter is placed on a second row
    # so narrow windows do not force the slider and checkbox to overlap.



    mode_and_filter_row = QHBoxLayout()



    mode_and_filter_row.setSpacing(8)



    mode_and_filter_row.setContentsMargins(0, 0, 0, 0)

    threshold_layout = QHBoxLayout()
    threshold_layout.setSpacing(6)
    threshold_layout.setContentsMargins(0, 0, 0, 0)
    threshold_name = QLabel("Threshold")
    threshold_name.setToolTip("Higher values show fewer, stricter matches and use the same cutoff for AI context.")
    low_label = QLabel("More")
    high_label = QLabel("Stricter")
    for threshold_text_label in (threshold_name, low_label, high_label):
        threshold_text_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        threshold_text_label.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {theme['subtext']};")
    threshold_layout.addWidget(threshold_name)
    threshold_layout.addWidget(low_label)
    threshold_layout.addWidget(self.sensitivity_slider, 20)
    threshold_layout.addWidget(high_label)
    threshold_layout.addWidget(self.sensitivity_value_label)
    mode_and_filter_row.addLayout(threshold_layout, 20)







    # Relevance mode selector (Focused / Balanced / Broad)



    mode_layout = QHBoxLayout()



    mode_layout.setSpacing(0)



    mode_layout.setContentsMargins(0, 0, 0, 0)



    self.relevance_mode_group = QButtonGroup(self)



    self.relevance_mode_group.setExclusive(True)







    # Determine initial mode from config.



    try:



        sc_mode = load_config().get("search_config", {}).get("relevance_mode", "")



    except Exception:



        sc_mode = ""



    if not sc_mode:
        sc_mode = "balanced"



    sc_mode = (sc_mode or "balanced").lower()



    if sc_mode not in ("focused", "balanced", "broad"):



        sc_mode = "balanced"



    self.relevance_mode = sc_mode







    # Shared segmented-control style for relevance mode buttons



    mode_border_color = theme["subtle_border"]



    mode_text_color = theme["subtext"]



    mode_checked_bg = theme["section_header_checked"]



    mode_checked_text = theme["text"]



    mode_hover_bg = theme["panel_bg"]



    mode_btn_style = (



        "QRadioButton {"



        "  font-size: 11px;"



        "  padding: 4px 10px;"



        "  border: 1px solid " + mode_border_color + ";"



        "  color: " + mode_text_color + ";"



        "  background-color: transparent;"



        "  margin: 0;"



        "}"



        "QRadioButton::indicator {"



        "  width: 0px;"



        "  height: 0px;"



        "}"



        "QRadioButton:hover {"



        "  background-color: " + mode_hover_bg + ";"



        "}"



        "QRadioButton:checked {"



        "  background-color: " + mode_checked_bg + ";"



        "  color: " + mode_checked_text + ";"



        "}"



    )







    def _add_mode_button(label, mode_key, tooltip):



        btn = QRadioButton(label)



        btn.setToolTip(tooltip)



        btn.setProperty("mode_key", mode_key)



        btn.setMinimumWidth(72)



        btn.setStyleSheet(mode_btn_style)



        self.relevance_mode_group.addButton(btn)



        mode_layout.addWidget(btn)



        if mode_key == self.relevance_mode:



            btn.setChecked(True)







    _add_mode_button(



        "Focused",



        "focused",



        "Fewer notes, most on-topic only. Same search results; only which notes are shown changes.",



    )



    _add_mode_button(



        "Balanced",



        "balanced",



        "Moderate set. Same search results; only which notes are shown changes.",



    )



    _add_mode_button(



        "Broad",



        "broad",



        "More notes, including tangential. Same search results; only which notes are shown changes.",



    )







    self.relevance_mode_group.buttonToggled.connect(self._on_relevance_mode_changed)



    for btn in self.relevance_mode_group.buttons():
        btn.setVisible(False)

    self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_changed)
    try:
        self.sensitivity_slider.sliderReleased.connect(self.on_relevance_threshold_released)
    except Exception:
        pass







    self.show_only_cited_cb = QCheckBox("Show only cited notes")
    self.show_only_cited_cb.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    self.show_only_cited_cb.setStyleSheet(f"font-size: 11px; color: {theme['subtext']};")



    self.show_only_cited_cb.setToolTip(



        "Show only notes that the AI explicitly cited in its answer."



    )



    try:



        sc = load_config().get('search_config', {})



        self.show_only_cited_cb.setChecked(bool(sc.get('show_only_cited', False)))



    except Exception:



        self.show_only_cited_cb.setChecked(False)



    self.show_only_cited_cb.stateChanged.connect(self._on_show_only_cited_changed)



    sensitivity_layout.addLayout(mode_and_filter_row)

    cited_filter_row = QHBoxLayout()
    cited_filter_row.setSpacing(4)
    cited_filter_row.setContentsMargins(0, 0, 0, 0)
    cited_filter_row.addStretch()
    cited_filter_row.addWidget(self.show_only_cited_cb)
    sensitivity_layout.addLayout(cited_filter_row)







    results_layout.addWidget(sensitivity_container)







    # Source action buttons



    btn_container = QWidget()



    btn_container.setObjectName("actionBar")



    btn_container.setStyleSheet("QWidget#actionBar { background-color: transparent; padding: 0; }")



    btn_layout = QHBoxLayout(btn_container)
    btn_layout.setContentsMargins(0, 0, 0, 0)
    btn_layout.setSpacing(7)







    self.view_btn = QPushButton("\U0001F441 View Selected")



    self.view_btn.setObjectName("viewBtn")



    self.view_btn.setToolTip("Open selected notes in the Anki browser")



    self.view_btn.setMinimumHeight(32)



    self.view_btn.clicked.connect(self.open_selected_in_browser)



    self.view_btn.setEnabled(False)







    self.view_all_btn = QPushButton("\U0001F4DA View All")



    self.view_all_btn.setObjectName("viewAllBtn")



    self.view_all_btn.setToolTip(



        "Open all visible notes in the Anki browser. "



        "No notes in the list \u2014 run a search first."



    )



    self.view_all_btn.setMinimumHeight(32)



    self.view_all_btn.clicked.connect(self.open_all_in_browser)



    self.view_all_btn.setEnabled(False)

    self.show_all_dynamic_results_btn = QPushButton("\U0001F4DA Show All Source Notes")
    self.show_all_dynamic_results_btn.setObjectName("showAllDynamicResultsBtn")
    self.show_all_dynamic_results_btn.setToolTip(
        "Show lower-relevance source notes hidden by the threshold or display limit. "
        "AI answer context stays unchanged."
    )
    self.show_all_dynamic_results_btn.setMinimumHeight(30)
    self.show_all_dynamic_results_btn.clicked.connect(self._on_show_all_dynamic_results)
    self.show_all_dynamic_results_btn.setVisible(False)







    btn_layout.addWidget(self.view_btn)



    btn_layout.addWidget(self.view_all_btn)



    btn_layout.addWidget(self.show_all_dynamic_results_btn)



    btn_layout.addStretch()







    results_layout.addWidget(btn_container)







    # Status bar



    status_container = QWidget()



    status_container.setStyleSheet(
        f"background-color: {theme['panel_bg']}; "
        f"border: 1px solid {theme['subtle_border']}; "
        "border-radius: 5px;"
    )



    status_layout = QHBoxLayout(status_container)



    status_layout.setContentsMargins(8, 4, 8, 4)







    status_icon = QLabel("\u2139")
    status_icon.setStyleSheet(f"color: {theme['quiet_text']}; font-size: 11px;")



    self.status_label = QLabel("Ready")



    self.status_label.setStyleSheet(f"color: {theme['subtext']}; font-size: 11px; font-weight: 500;")



    self.status_label.setToolTip(



        "Showing X of Y: X = notes passing the Relevance Threshold, Y = notes in this result set. "
        "Move the threshold slider to show more results or stricter matches."



    )







    status_layout.addWidget(status_icon)



    status_layout.addWidget(self.status_label)



    status_layout.addStretch()



    # Progress bar and % for embedding search (hidden when idle)



    self.search_progress_bar = QProgressBar()



    self.search_progress_bar.setMinimum(0)



    self.search_progress_bar.setMaximum(100)



    self.search_progress_bar.setValue(0)



    self.search_progress_bar.setMaximumWidth(120)



    self.search_progress_bar.setMinimumWidth(80)



    self.search_progress_bar.setTextVisible(True)



    self.search_progress_bar.setFormat("%p%")



    self.search_progress_bar.setStyleSheet("QProgressBar { text-align: center; color: palette(window-text); }")



    self.search_progress_bar.setVisible(False)



    status_layout.addWidget(self.search_progress_bar)



    self.search_progress_label = QLabel("")



    self.search_progress_label.setStyleSheet(f"font-size: 11px; color: {'#95a5a6' if is_dark else '#7f8c8d'};")



    self.search_progress_label.setVisible(False)



    status_layout.addWidget(self.search_progress_label)







    layout.addWidget(status_container)







    # Main content added directly to root layout



    root_layout.addWidget(main_container, 1)







    self.setLayout(root_layout)







    self.results_list.itemSelectionChanged.connect(self._update_view_all_button_state)







    # Track checkbox state changes to update count and button text



    # Only track changes in column 0 (dedicated checkbox column)



    self.results_list.itemChanged.connect(self.on_item_changed)







    # Store selected note IDs for persistence



    self.selected_note_ids = set()



    self._pinned_note_ids = set()  # note IDs from clicked [N] refs in AI answer



    self._cited_note_ids = set()   # note IDs cited in AI answer ([1], [2], ...) for "Show only cited" filter
    self._cited_refs = set()       # 1-based refs cited in AI answer; preserves chunk-level citations







    self._refresh_search_history()



    if self.use_side_by_side:
        QTimer.singleShot(0, self._collapse_sources_panel)

    QTimer.singleShot(100, self._refresh_scope_banner)


class SearchDialogUiMixin:
    """Owns search dialog defaults and widget construction."""

    def reset_to_medical_defaults(self):
        return reset_to_medical_defaults(self)

    def _collapse_sources_panel(self, manual=False):
        return _collapse_sources_panel(self, manual=manual)

    def _expand_sources_panel(self, manual=False):
        return _expand_sources_panel(self, manual=manual)

    def _toggle_sources_panel(self):
        return _toggle_sources_panel(self)

    def setup_ui(self):
        return setup_ui(self)
