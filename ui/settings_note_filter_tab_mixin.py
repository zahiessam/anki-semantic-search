"""Focused Settings dialog layout builder mixin."""

from aqt.qt import *

from .theme import settings_button_style, settings_text_style


class SettingsNoteFilterTabMixin:
    """Builds one focused section of the Settings dialog UI."""

    def _build_note_types_tab(self, theme):
        # --- Note Types & Fields Tab ---



        nt_tab = QWidget()



        nt_main = QVBoxLayout(nt_tab)



        nt_main.setSpacing(10)



        nt_main.setContentsMargins(20, 20, 20, 20)







        nt_info = QLabel("\U0001F4CB Note types, decks, and fields")
        nt_info.setStyleSheet(settings_text_style(theme, "section_heading"))



        nt_main.addWidget(nt_info)

        nt_hint = QLabel(
            "Choose what search can see. Leave everything selected to search the full collection."
        )
        nt_hint.setWordWrap(True)
        nt_hint.setStyleSheet(settings_text_style(theme, "subtle"))
        nt_main.addWidget(nt_hint)

        self.note_type_loading_status = QLabel("")
        self.note_type_loading_status.setWordWrap(True)
        self.note_type_loading_status.setStyleSheet(settings_text_style(theme, "subtle"))
        self.note_type_loading_status.hide()
        nt_main.addWidget(self.note_type_loading_status)

        self.note_type_loading_bar = QProgressBar()
        self.note_type_loading_bar.setRange(0, 0)
        self.note_type_loading_bar.setTextVisible(False)
        self.note_type_loading_bar.setMaximumHeight(8)
        self.note_type_loading_bar.hide()
        nt_main.addWidget(self.note_type_loading_bar)







        # Side-by-side: left = Note types + Decks (stacked), right = Fields by note type



        main_h_split = QSplitter(Qt.Orientation.Horizontal)



        left_v_split = QSplitter(Qt.Orientation.Vertical)







        # ---- Left column: Note types (top) ----



        nt_group = QGroupBox("Note types to include")



        nt_gl = QVBoxLayout(nt_group)



        nt_btn_row = QHBoxLayout()



        nt_select_btn = QPushButton("Select all note types")
        nt_select_btn.setToolTip("Select every note type currently listed.")



        nt_select_btn.clicked.connect(lambda: self._set_note_types_checked(True, manual=True))



        nt_deselect_btn = QPushButton("Deselect all note types")
        nt_deselect_btn.setToolTip("Clear every note type currently listed, then choose only the ones you want.")



        nt_deselect_btn.clicked.connect(lambda: self._set_note_types_checked(False, manual=True))



        nt_btn_row.addWidget(nt_select_btn)



        nt_btn_row.addWidget(nt_deselect_btn)



        nt_btn_row.addStretch()



        # Sort options



        sort_label = QLabel("Sort by:")



        self.sort_combo = QComboBox()



        self.sort_combo.addItem("Note Count (Desc)", "count_desc")



        self.sort_combo.addItem("Note Count (Asc)", "count_asc")



        self.sort_combo.addItem("Name (A-Z)", "name_asc")



        self.sort_combo.addItem("Name (Z-A)", "name_desc")



        self.sort_combo.currentIndexChanged.connect(self._on_sort_note_types_changed)



        nt_btn_row.addWidget(sort_label)



        nt_btn_row.addWidget(self.sort_combo)



        nt_gl.addLayout(nt_btn_row)



        self.note_types_table = QTableWidget()



        self.note_types_table.setColumnCount(2)



        self.note_types_table.setHorizontalHeaderLabels(["Note Type", "Note Count"])



        self.note_types_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)



        self.note_types_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)



        self.note_types_table.setColumnWidth(1, 120)  # Set minimum width for count column



        self.note_types_table.setMinimumHeight(80)



        self.note_types_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



        self.note_types_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)



        self.note_types_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)



        self.note_types_table.itemChanged.connect(self._on_note_type_item_changed)



        self.note_types_table.setSortingEnabled(True)



        self.note_types_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)  # Sort by count descending by default



        header = self.note_types_table.horizontalHeader()
        header.sortIndicatorChanged.connect(
            lambda _column, _order: QTimer.singleShot(0, self._sync_field_groups_to_note_type_order)
        )



        # Add spacing between columns



        self.note_types_table.setColumnWidth(0, 200)



        nt_gl.addWidget(self.note_types_table)



        left_v_split.addWidget(nt_group)







        # ---- Left column: Decks (bottom) ----



        deck_group = QGroupBox("Decks to search")



        deck_gl = QVBoxLayout(deck_group)



        self.include_all_decks_cb = QCheckBox("Include all decks")



        self.include_all_decks_cb.setChecked(True)



        self.include_all_decks_cb.stateChanged.connect(self._on_include_all_decks_toggled)



        deck_gl.addWidget(self.include_all_decks_cb)



        deck_btn_row = QHBoxLayout()



        deck_select_btn = QPushButton("Select All")



        deck_select_btn.clicked.connect(lambda: self._set_decks_checked(True))



        deck_deselect_btn = QPushButton("Deselect All")



        deck_deselect_btn.clicked.connect(lambda: self._set_decks_checked(False))



        deck_btn_row.addWidget(deck_select_btn)



        deck_btn_row.addWidget(deck_deselect_btn)



        deck_btn_row.addStretch()



        deck_gl.addLayout(deck_btn_row)



        # Use QTreeWidget for hierarchical deck display (like main Anki interface)



        self.decks_list = QTreeWidget()



        self.decks_list.setMinimumHeight(80)



        self.decks_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



        # Simplified header: only show deck name and total notes



        self.decks_list.setHeaderLabels(["Deck", "Notes"])



        self.decks_list.setRootIsDecorated(True)  # Show expand/collapse arrows



        self.decks_list.setAlternatingRowColors(True)



        # Set column widths: deck name and notes



        self.decks_list.setColumnWidth(0, 260)  # Deck name



        self.decks_list.setColumnWidth(1, 70)   # Notes



        self.decks_list.itemChanged.connect(self._on_deck_item_changed)



        deck_gl.addWidget(self.decks_list)



        left_v_split.addWidget(deck_group)







        left_v_split.setSizes([220, 180])



        left_v_split.setChildrenCollapsible(False)



        left_v_split.setHandleWidth(6)



        main_h_split.addWidget(left_v_split)







        # ---- Right column: Fields by note type ----



        fld_outer = QGroupBox("Fields to search per note type (greyed if note type unchecked)")



        fld_outer_l = QVBoxLayout(fld_outer)



        self.search_all_fields_cb = QCheckBox("Search in all fields (ignore selections below)")



        self.search_all_fields_cb.setChecked(False)



        self.search_all_fields_cb.stateChanged.connect(self._on_search_all_fields_toggled)



        fld_outer_l.addWidget(self.search_all_fields_cb)



        self.use_first_field_cb = QCheckBox("Use first two fields when no fields selected for a note type")



        self.use_first_field_cb.setChecked(True)



        self.use_first_field_cb.setToolTip("If a note type has no checked fields, use its first two fields instead of skipping.")
        self.use_first_field_cb.stateChanged.connect(lambda *_: self._persist_note_type_filter())



        fld_outer_l.addWidget(self.use_first_field_cb)


        ask_ai_context_hint = QLabel(
            "Review Ask AI uses these same note type and field selections for text and image context."
        )
        ask_ai_context_hint.setWordWrap(True)
        ask_ai_context_hint.setStyleSheet(settings_text_style(theme, "subtle"))
        fld_outer_l.addWidget(ask_ai_context_hint)



        self.fields_by_note_type_scroll = QScrollArea()
        self.fields_by_note_type_scroll.setObjectName("settingsFieldScroll")



        self.fields_by_note_type_scroll.setMinimumHeight(120)



        self.fields_by_note_type_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



        self.fields_by_note_type_scroll.setWidgetResizable(True)



        self.fields_by_note_type_inner = QWidget()



        self.fields_by_note_type_layout = QVBoxLayout(self.fields_by_note_type_inner)



        self.fields_by_note_type_layout.setAlignment(Qt.AlignmentFlag.AlignTop)



        self.fields_by_note_type_scroll.setWidget(self.fields_by_note_type_inner)



        fld_outer_l.addWidget(self.fields_by_note_type_scroll)



        main_h_split.addWidget(fld_outer)



        self._field_cbs = {}  # model_name -> { field_name: QCheckBox }



        self._field_groupboxes = {}  # model_name -> QGroupBox (for greying when note type unchecked)







        main_h_split.setSizes([380, 420])



        main_h_split.setChildrenCollapsible(False)



        main_h_split.setHandleWidth(6)



        nt_main.addWidget(main_h_split)







        # ---- Count notes, Save/Load/Delete, Refresh ----



        action_row = QHBoxLayout()



        self.count_notes_btn = QPushButton("\U0001F4CA Preview coverage")



        self.count_notes_btn.clicked.connect(self._on_count_notes)



        action_row.addWidget(self.count_notes_btn)



        action_row.addStretch()



        nt_main.addLayout(action_row)







        save_pl = QHBoxLayout()



        self.preset_name_edit = QLineEdit()



        self.preset_name_edit.setPlaceholderText("e.g. Default, Shared deck")



        self.preset_name_edit.setMaximumWidth(200)



        self.preset_name_edit.setToolTip("Name for this set of note types and decks (for quick switching)")



        save_pl.addWidget(QLabel("Save as preset:"))



        save_pl.addWidget(self.preset_name_edit)



        save_preset_btn = QPushButton("\U0001F4BE Save preset")



        save_preset_btn.clicked.connect(self._on_save_preset)



        save_pl.addWidget(save_preset_btn)



        save_pl.addStretch()



        save_pl.addWidget(QLabel("Load preset:"))



        self.load_preset_combo = QComboBox()



        self.load_preset_combo.setMaximumWidth(180)



        self.load_preset_combo.setEditable(False)



        save_pl.addWidget(self.load_preset_combo)



        load_preset_btn = QPushButton("Load")



        load_preset_btn.clicked.connect(self._on_load_preset)



        save_pl.addWidget(load_preset_btn)



        self.delete_preset_combo = QComboBox()



        self.delete_preset_combo.setMaximumWidth(160)



        save_pl.addWidget(QLabel("Delete:"))



        save_pl.addWidget(self.delete_preset_combo)



        delete_preset_btn = QPushButton("Delete preset")



        delete_preset_btn.clicked.connect(self._on_delete_preset)



        save_pl.addWidget(delete_preset_btn)



        nt_main.addLayout(save_pl)







        refresh_btn = QPushButton("\U0001F504 Refresh lists (detect new note types/fields/decks)")



        refresh_btn.clicked.connect(self._refresh_note_type_lists)



        nt_main.addWidget(refresh_btn)







        self._note_types_tab = nt_tab
        self._note_type_lists_loaded = False







        nt_main.addStretch()








        return nt_tab
