"""Search dialog defaults and UI construction."""

# ============================================================================
# Imports
# ============================================================================

from aqt.qt import *
from aqt.utils import askUser, showInfo, tooltip

from .search_dialog import ContentDelegate, RelevanceBarDelegate
from .note_preview_popup import NotePreviewPopup
from .theme import get_addon_theme
from .widgets import SpellCheckPlainTextEdit, _get_spell_checker
from ..utils import get_search_history_queries, load_config


# ============================================================================
# Search Dialog UI Construction
# ============================================================================

_addon_theme = get_addon_theme

def reset_to_medical_defaults(self):

    """Resets all settings to high-yield medical defaults as suggested by Glutamine."""

    from aqt.utils import askUser

    if not askUser("Reset all settings to high-yield clinical defaults (Resident-approved)?"):

        return



    # 1. API & Provider Defaults

    self.answer_provider_combo.setCurrentIndex(0) # Default to Ollama (Local)

    self.local_llm_url.setText("http://localhost:1234/v1")

    self.local_llm_model.setText("llama3.2")



    # 2. Search & Embedding Defaults (Medical High-Yield)

    self.search_method_combo.setCurrentIndex(2) # Hybrid (RRF) - Best for Med

    self.min_relevance_spin.setValue(55) # Filter out noise

    self.max_results_spin.setValue(50) # Comprehensive for medical use

    self.hybrid_weight_spin.setValue(40) # Slightly favor keywords for drug names/genes

    self.enable_query_expansion_cb.setChecked(True) # AI synonyms are great for medicine

    self.use_ai_generic_term_detection_cb.setChecked(True)



    # 3. Styling Defaults (Dark mode friendly)

    self.question_font_spin.setValue(14)

    self.answer_font_spin.setValue(13)

    self.notes_font_spin.setValue(12)

    self.layout_combo.setCurrentIndex(0) # Side-by-side



    # 4. Note Types & Fields

    self.include_all_note_types_cb.setChecked(True)

    self.include_all_decks_cb.setChecked(True)

    self.use_first_field_cb.setChecked(True)



    from aqt.utils import showInfo

    showInfo("Settings reset to Clinical High-Yield defaults. Click 'Save' to apply.")



def setup_ui(self):



    root_layout = QHBoxLayout()



    root_layout.setSpacing(8)



    root_layout.setContentsMargins(8, 8, 8, 8)







    main_container = QWidget()



    layout = QVBoxLayout(main_container)



    section_spacing = self.styling_config.get('section_spacing', 8)



    layout.setSpacing(section_spacing)



    layout.setContentsMargins(10, 8, 10, 8)







    palette = QApplication.palette()



    is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128







    theme = _addon_theme(is_dark)







    # Compact header: title + hint + scope banner + settings (shrunk)



    header_layout = QHBoxLayout()



    title_label = QLabel("\U0001F50D Anki Semantic Search")



    title_label.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {'#ffffff' if is_dark else '#2c3e50'};")



    header_layout.addWidget(title_label)



    hint_label = QLabel("Ask a question \u2192 AI answer from your notes")



    hint_label.setStyleSheet(f"font-size: 10px; color: {'#95a5a6' if is_dark else '#7f8c8d'};")



    header_layout.addWidget(hint_label)



    self.scope_banner = QLabel("")



    self.scope_banner.setStyleSheet(f"font-size: 10px; color: {'#95a5a6' if is_dark else '#7f8c8d'}; padding: 2px 8px;")



    self.scope_banner.setToolTip("Search scope. Use the Settings button to change note types, fields, decks.")



    header_layout.addWidget(self.scope_banner)



    header_layout.addStretch()



    settings_btn = QPushButton("\u2699 Settings")



    settings_btn.setObjectName("settingsBtn")



    settings_btn.setToolTip("Configure API key, note types, decks, and search behavior")



    settings_btn.clicked.connect(self.open_settings)



    header_layout.addWidget(settings_btn)



    layout.addLayout(header_layout)







    # Search input (shrunk)



    search_container = QWidget()



    search_container.setStyleSheet("background-color: rgba(52, 152, 219, 0.12); border-radius: 4px; padding: 4px 6px;")



    search_layout = QVBoxLayout(search_container)



    search_layout.setSpacing(2)



    search_layout.setContentsMargins(4, 4, 4, 4)







    label_font_size = max(11, self.styling_config.get('label_font_size', 13) - 2)



    search_label = QLabel("Search:")



    search_label.setToolTip("Type a question; matching notes will be found and the AI will answer using them. Ctrl+Enter to search.")



    search_label.setStyleSheet(f"font-weight: bold; font-size: {label_font_size}px; color: {'#ffffff' if is_dark else '#2c3e50'};")



    search_layout.addWidget(search_label)







    search_input_layout = QHBoxLayout()



    try:



        self.search_input = SpellCheckPlainTextEdit()



    except Exception:



        self.search_input = QPlainTextEdit()



    self.search_input.setPlaceholderText("e.g., hypertension  or  causes of heart failure \u2014 Ctrl+Enter to search")



    self.search_input.setMinimumHeight(44)



    self.search_input.setMaximumHeight(100)



    spell_hint = " Right-click misspelled words for corrections." if _get_spell_checker() else ""



    self.search_input.setToolTip(f"Type your question. Ctrl+Enter to search.{spell_hint}")



    question_font_size = max(11, self.styling_config.get('question_font_size', 13) - 1)



    self.search_input.setStyleSheet(f"font-size: {question_font_size}px;")

    # History dropdown for recent searches

    self._search_history_model = QStringListModel(get_search_history_queries())

    self.search_history_combo = QComboBox()

    self.search_history_combo.setModel(self._search_history_model)

    self.search_history_combo.setEditable(False)

    self.search_history_combo.setMinimumWidth(80)

    self.search_history_combo.setMaximumWidth(150)

    self.search_history_combo.setToolTip("Recent searches \u2014 select to fill the search box, then click Search.")

    self.search_history_combo.activated.connect(self._on_search_history_selected)



    clear_history_btn = QPushButton("Clear history")

    clear_history_btn.setToolTip("Delete all search history")

    clear_history_btn.setMaximumWidth(90)

    clear_history_btn.clicked.connect(self._on_clear_search_history)



    self.search_btn = QPushButton("\U0001F50D Search")

    self.search_btn.setObjectName("searchBtn")

    self.search_btn.setMinimumHeight(42)

    self.search_btn.setMinimumWidth(100)

    self.search_btn.clicked.connect(self.perform_search)



    search_input_layout.addWidget(self.search_input, 4) # Give much more stretch to input

    search_input_layout.addWidget(self.search_history_combo, 1) # Keep history small

    search_input_layout.addWidget(clear_history_btn)

    search_input_layout.addWidget(self.search_btn)

    search_layout.addLayout(search_input_layout)







    # Shortcuts and History moved from sidebar to below search input



    extra_info_layout = QHBoxLayout()



    shortcuts_hint = QLabel("<b>Shortcuts:</b> Ctrl+Enter search | Ctrl+A select | Ctrl+D deselect")



    shortcuts_hint.setStyleSheet(f"font-size: 10px; color: {theme['subtext']};")



    extra_info_layout.addWidget(shortcuts_hint)



    extra_info_layout.addStretch()



    search_layout.addLayout(extra_info_layout)







    layout.addWidget(search_container)







    # Ctrl+Enter to search



    search_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)



    search_shortcut.activated.connect(self.perform_search)







    # Keyboard shortcuts for select/deselect



    select_all_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)



    select_all_shortcut.activated.connect(self.select_all_notes)



    deselect_all_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)



    deselect_all_shortcut.activated.connect(self.deselect_all_notes)







    layout.addWidget(search_container)







    # Splitter for resizable sections (side-by-side or stacked)



    layout_mode = self.styling_config.get('layout_mode', 'side_by_side')



    self.use_side_by_side = layout_mode == 'side_by_side'



    split_orientation = Qt.Orientation.Horizontal if self.use_side_by_side else Qt.Orientation.Vertical



    main_splitter = QSplitter(split_orientation)







    # AI Answer section



    answer_container = QWidget()



    answer_container.setStyleSheet("background-color: rgba(46, 204, 113, 0.12); border-radius: 6px; padding: 8px;")



    answer_layout = QVBoxLayout(answer_container)







    answer_header = QHBoxLayout()



    answer_label = QLabel("\U0001F4A1 AI Answer:")



    answer_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #27ae60;")



    answer_header.addWidget(answer_label)



    answer_header.addStretch()



    self.copy_answer_btn = QPushButton("\U0001F4CB Copy")



    self.copy_answer_btn.setMaximumWidth(80)



    self.copy_answer_btn.setToolTip("Copy AI answer (paste into Word for bullets and formatting)")



    self.copy_answer_btn.clicked.connect(self.copy_answer_to_clipboard)



    self.copy_answer_btn.setEnabled(False)



    answer_header.addWidget(self.copy_answer_btn)



    answer_layout.addLayout(answer_header)







    try:



        self.answer_box = QTextBrowser()



    except NameError:



        self.answer_box = QTextEdit()



    self.answer_box.setObjectName("answerBox")
    self.answer_box.setReadOnly(True)

    if hasattr(Qt, 'ScrollBarPolicy'):
        self.answer_box.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.answer_box.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    else:
        self.answer_box.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
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



        self.answer_box.setPlaceholderText("Enter a question above and click Search to see an AI answer based on your notes.")



    self.answer_box.setMinimumHeight(100)



    self.answer_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



    answer_font_size = self.styling_config.get('answer_font_size', 13)
    answer_bg = '#2d2d2d' if is_dark else '#ffffff'
    answer_text = '#ffffff' if is_dark else '#1a1a1a'
    self.answer_box.setStyleSheet(



        f"QTextBrowser#answerBox, QTextEdit#answerBox {{ background-color: {answer_bg}; "



        f"border: 2px solid #27ae60; color: {answer_text}; "



        f"font-size: {answer_font_size}px; padding: 10px; }} "



        f"QTextBrowser#answerBox a, QTextEdit#answerBox a {{ color: #3498db; text-decoration: underline; }} "



        f"QTextBrowser#answerBox a:hover, QTextEdit#answerBox a:hover {{ color: #5dade2; }} "



    )



    # Connect link click: anchorClicked (Qt6) or linkActivated (PyQt5) so citation links work with Ollama and all providers



    if hasattr(self.answer_box, 'anchorClicked'):



        self.answer_box.anchorClicked.connect(self._on_answer_link_clicked)



    elif hasattr(self.answer_box, 'linkActivated'):



        self.answer_box.linkActivated.connect(self._on_answer_link_clicked)



    answer_layout.addWidget(self.answer_box)







    # Hint: where the answer came from (API name or local model)



    self.answer_source_label = QLabel("")



    self.answer_source_label.setStyleSheet(
        f"font-size: 11px; color: {'#c2d2ca' if is_dark else '#586a61'}; "
        "font-style: italic; margin-top: 4px; padding: 2px 4px;"
    )



    self.answer_source_label.setWordWrap(True)



    answer_layout.addWidget(self.answer_source_label)







    main_splitter.addWidget(answer_container)







    # Results section



    results_container = QWidget()



    results_layout = QVBoxLayout(results_container)







    results_header = QHBoxLayout()



    results_label = QLabel("\U0001F4CB Matching notes:")



    results_label.setToolTip("Notes that match your question. Check the ones to send to the AI for the answer.")



    results_label.setStyleSheet(f"font-weight: bold; font-size: {label_font_size}px; color: {'#ffffff' if is_dark else '#2c3e50'};")



    results_header.addWidget(results_label)



    self.selected_count_label = QLabel("(0 selected)")



    self.selected_count_label.setStyleSheet(f"font-size: {label_font_size - 2}px; color: {'#95a5a6' if is_dark else '#7f8c8d'}; font-style: italic;")



    self.selected_count_label.setToolTip("Notes you checked (for \"View Selected\"). Not the same as \"cited\" in the AI answer\u2014use \"Show only cited notes\" for that.")



    results_header.addWidget(self.selected_count_label)



    results_header.addStretch()



    self.toggle_select_btn = QPushButton("Select All")



    self.toggle_select_btn.setObjectName("toggleSelectBtn")



    self.toggle_select_btn.setMinimumWidth(110)



    self.toggle_select_btn.setToolTip("Toggle select/deselect all (Ctrl+A / Ctrl+D)")



    self.toggle_select_btn.clicked.connect(self.toggle_select_all)



    self.toggle_select_btn.setEnabled(False)



    results_header.addWidget(self.toggle_select_btn)



    results_layout.addLayout(results_header)







    # Create table: Select | Ref (citation [1],[2]...) | Content | Note ID | Relevance



    self.results_list = QTableWidget()



    self.results_list.setColumnCount(5)



    self.results_list.setHorizontalHeaderLabels(["✓", "Ref", "Content", "Note ID", "Relevance"])



    self.results_list.setMinimumHeight(120)



    self.results_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



    notes_font_size = self.styling_config.get('notes_font_size', 12)



    self.results_list.setStyleSheet(f"font-size: {notes_font_size}px;")



    self.results_list.setWordWrap(True)







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







    main_splitter.addWidget(results_container)



    main_splitter.setSizes([450, 550] if self.use_side_by_side else [350, 450])



    main_splitter.setChildrenCollapsible(False)



    main_splitter.setHandleWidth(8)







    layout.addWidget(main_splitter, 1)







    # Result tuning: relevance mode controls (Focused / Balanced / Broad); no slider



    sensitivity_container = QWidget()



    sensitivity_container.setStyleSheet("background-color: rgba(241, 196, 15, 0.1); border-radius: 6px; padding: 6px;")



    sensitivity_layout = QVBoxLayout(sensitivity_container)



    sensitivity_layout.setSpacing(4)



    sensitivity_layout.setContentsMargins(4, 2, 4, 2)



    self.sensitivity_slider = None  # Slider removed; mode alone controls how many notes are shown



    self.sensitivity_value_label = None







    # Relevance modes + "Show only cited notes" aligned on one line



    mode_and_filter_row = QHBoxLayout()



    mode_and_filter_row.setSpacing(8)



    mode_and_filter_row.setContentsMargins(0, 0, 0, 0)







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



    mode_border_color = "#f1c40f"



    mode_text_color = "#f1c40f" if is_dark else "#7d6608"



    mode_checked_bg = "#f1c40f" if is_dark else "#f9e79f"



    mode_checked_text = "#1e1e1e" if is_dark else "#7d6608"



    mode_hover_bg = "rgba(241, 196, 15, 0.18)"



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



    mode_and_filter_row.addLayout(mode_layout)







    self.show_only_cited_cb = QCheckBox("Show only cited notes")



    self.show_only_cited_cb.setToolTip(



        "Show only notes that the AI explicitly cited in its answer."



    )



    try:



        sc = load_config().get('search_config', {})



        self.show_only_cited_cb.setChecked(bool(sc.get('show_only_cited', False)))



    except Exception:



        self.show_only_cited_cb.setChecked(False)



    self.show_only_cited_cb.stateChanged.connect(self._on_show_only_cited_changed)



    mode_and_filter_row.addWidget(self.show_only_cited_cb)



    mode_and_filter_row.addStretch()







    sensitivity_layout.addLayout(mode_and_filter_row)







    layout.addWidget(sensitivity_container)







    # Action buttons



    btn_container = QWidget()



    btn_container.setObjectName("actionBar")



    btn_bar_bg = "#2d2d2d" if is_dark else "#d5d8dc"



    btn_bar_border = "#555555" if is_dark else "#95a5a6"



    btn_container.setStyleSheet(f"""



        QWidget#actionBar {{



            background-color: {btn_bar_bg};



            border: 1px solid {btn_bar_border};



            border-radius: 6px;



            padding: 6px;



        }}



    """)



    btn_layout = QHBoxLayout(btn_container)







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







    btn_layout.addWidget(self.view_btn)



    btn_layout.addWidget(self.view_all_btn)



    btn_layout.addStretch()







    layout.addWidget(btn_container)







    # Status bar



    status_container = QWidget()



    status_container.setStyleSheet("background-color: rgba(52, 152, 219, 0.15); border-radius: 4px; padding: 4px;")



    status_layout = QHBoxLayout(status_container)



    status_layout.setContentsMargins(8, 4, 8, 4)







    status_icon = QLabel("\u2139\ufe0f")



    self.status_label = QLabel("Ready")



    self.status_label.setStyleSheet(f"color: {'#ffffff' if is_dark else '#2c3e50'}; font-size: 12px; font-weight: 600;")



    self.status_label.setToolTip(



        "Showing X of Y: X = notes passing the Sensitivity filter, Y = notes in this result set (from Min relevance % and Max results in Settings). "



        "Move the Sensitivity slider to change X. Raise Min relevance % in Settings to get a smaller result set (smaller Y)."



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



    QTimer.singleShot(100, self._refresh_scope_banner)
