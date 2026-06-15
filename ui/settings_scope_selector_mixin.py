"""Scope selector UI mixin for AI Search/Embedding scope picker."""

from aqt import mw
from aqt.qt import *
from aqt.utils import showInfo

from ..core.engine import (
    analyze_note_eligibility,
    get_deck_names,
    get_models_with_fields,
    get_models_with_fields_for_deck,
    get_notes_count_per_deck,
)
from ..utils import load_config, log_debug, save_config
from .theme import get_addon_theme, settings_text_style
from .widgets import (
    settings_checkbox_row,
    settings_child_group,
    settings_child_row,
    settings_form_row,
    settings_page,
    settings_status,
    settings_toolbar,
)


class SettingsScopeSelectorMixin:
    """Builds and manages the new scope selector tree widget."""

    def _build_scope_selector_tab(self, theme):
        """Build the new scope selector tab replacing the old note types/decks/fields layout."""
        scope_tab, scope_main = settings_page(
            theme,
            "\U0001F4CB AI search scope",
            "Choose what AI Search can see. Select decks, then expand a deck to choose note types and fields.",
        )

        # Loading status
        self.scope_loading_status = settings_status(theme, "", "info")
        self.scope_loading_status.hide()
        scope_main.addWidget(self.scope_loading_status)

        self.scope_loading_bar = QProgressBar()
        self.scope_loading_bar.setRange(0, 0)
        self.scope_loading_bar.setTextVisible(False)
        self.scope_loading_bar.setMaximumHeight(8)
        self.scope_loading_bar.hide()
        scope_main.addWidget(self.scope_loading_bar)

        # Search/filter box
        self.scope_filter_edit = QLineEdit()
        self.scope_filter_edit.setPlaceholderText("Filter decks...")
        self.scope_filter_edit.textChanged.connect(self._on_scope_filter_changed)
        scope_main.addWidget(settings_form_row(theme, "Filter", self.scope_filter_edit))

        # Quick actions row
        select_all_btn = QPushButton("Select all visible")
        select_all_btn.clicked.connect(lambda: self._set_scope_tree_checked(True, visible_only=True))
        deselect_all_btn = QPushButton("Deselect all visible")
        deselect_all_btn.clicked.connect(lambda: self._set_scope_tree_checked(False, visible_only=True))
        expand_selected_btn = QPushButton("Expand selected")
        expand_selected_btn.clicked.connect(self._expand_selected_scope_items)
        collapse_all_btn = QPushButton("Collapse all")
        collapse_all_btn.clicked.connect(self._collapse_all_scope_items)
        scope_main.addWidget(settings_toolbar(
            theme,
            select_all_btn,
            deselect_all_btn,
            expand_selected_btn,
            collapse_all_btn,
        ))

        # Scope tree widget
        self.scope_tree = QTreeWidget()
        self.scope_tree.setHeaderLabels(["Scope / Note Type / Field", "Notes"])
        self.scope_tree.setRootIsDecorated(True)
        self.scope_tree.setAlternatingRowColors(True)
        self.scope_tree.setColumnWidth(0, 350)
        self.scope_tree.setColumnWidth(1, 80)
        self.scope_tree.itemChanged.connect(self._on_scope_tree_item_changed)
        self.scope_tree.itemExpanded.connect(self._on_scope_tree_item_expanded)
        scope_tree_group = settings_child_group(theme)
        scope_tree_group.content_layout.addWidget(self.scope_tree)
        scope_main.addWidget(scope_tree_group)

        # Existing controls retained
        self.search_all_fields_cb = QCheckBox("Search in all fields (ignore field selections below)")
        self.search_all_fields_cb.setChecked(False)
        self.search_all_fields_cb.stateChanged.connect(self._on_search_all_fields_toggled)
        scope_main.addWidget(settings_checkbox_row(theme, self.search_all_fields_cb))

        self.use_first_field_cb = QCheckBox("Use first two fields when no fields selected for a note type")
        self.use_first_field_cb.setChecked(True)
        self.use_first_field_cb.setToolTip("If a note type has no checked fields, use its first two fields instead of skipping.")
        self.use_first_field_cb.stateChanged.connect(lambda *_: self._persist_scope_config())
        scope_main.addWidget(settings_checkbox_row(theme, self.use_first_field_cb))

        ask_ai_hint = QLabel(
            "AI Search scope controls embeddings and search context. Review Ask AI field context is configured below."
        )
        ask_ai_hint.setWordWrap(True)
        ask_ai_hint.setStyleSheet(settings_text_style(theme, "subtle"))
        scope_main.addWidget(ask_ai_hint)

        # Preview coverage button
        self.scope_count_notes_btn = QPushButton("\U0001F4CA Preview coverage")
        self.scope_count_notes_btn.clicked.connect(self._on_scope_count_notes)
        scope_main.addWidget(settings_toolbar(theme, self.scope_count_notes_btn))

        # Refresh button
        refresh_btn = QPushButton("\U0001F504 Refresh lists")
        refresh_btn.clicked.connect(self._refresh_scope_lists)
        scope_main.addWidget(settings_toolbar(theme, refresh_btn))

        self._build_chatbot_fields_section(scope_main, theme)

        scope_main.addStretch()

        self._scope_tab = scope_tab
        self._scope_lists_loaded = False
        self._scope_tree_items = {}  # For tracking items by key

        return scope_tab

    def _build_chatbot_fields_section(self, scope_main, theme):
        """Build separate current-card field controls for Review Ask AI."""
        section = QFrame()
        self.chatbot_fields_section = section
        section.setObjectName("chatbotFieldsSection")
        section.setStyleSheet(
            f"QFrame#chatbotFieldsSection {{ background-color: transparent; border-top: 1px solid {theme['subtle_border']}; }}"
        )
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 12, 0, 0)
        section_layout.setSpacing(8)

        heading = QLabel("\U0001F50E Review Ask AI Current-Note Fields")
        heading.setStyleSheet(settings_text_style(theme, "section_heading"))
        section_layout.addWidget(heading)

        hint = QLabel(
            "Review Ask AI follows the AI Search decks and note types above. Use this only to override which fields from the current note are sent."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(settings_text_style(theme, "subtle"))
        section_layout.addWidget(hint)

        self.chatbot_same_fields_cb = QCheckBox("Use AI Search field selections")
        self.chatbot_same_fields_cb.setToolTip(
            "When enabled, Review Ask AI uses the AI Search field selections above. Disable to choose separate current-note fields."
        )
        self.chatbot_same_fields_cb.stateChanged.connect(self._on_chatbot_field_mode_changed)
        section_layout.addWidget(settings_checkbox_row(theme, self.chatbot_same_fields_cb))

        self.chatbot_same_fields_note = settings_status(
            theme,
            "Using AI Search field selections for current-note context.",
            "info",
        )
        self.chatbot_same_fields_note_row = self.chatbot_same_fields_note
        section_layout.addWidget(self.chatbot_same_fields_note_row)

        self.chatbot_fields_tree = QTreeWidget()
        self.chatbot_fields_tree.setHeaderLabels(["Note Type / Field", ""])
        self.chatbot_fields_tree.setRootIsDecorated(True)
        self.chatbot_fields_tree.setAlternatingRowColors(True)
        self.chatbot_fields_tree.setColumnWidth(0, 420)
        self.chatbot_fields_tree.setMinimumHeight(150)
        self.chatbot_fields_tree.setMaximumHeight(260)
        self.chatbot_fields_tree.itemChanged.connect(self._on_chatbot_field_tree_item_changed)
        self.chatbot_fields_by_note_type = {}
        self._theme = theme
        self.chatbot_fields_tree_group = settings_child_group(theme)
        self.chatbot_fields_tree_group.content_layout.addWidget(self.chatbot_fields_tree)
        section_layout.addWidget(self.chatbot_fields_tree_group)

        scope_main.addWidget(section)
        self._apply_chatbot_field_config((load_config() or {}).get("review_ask_ai") or {})

    def _on_scope_filter_changed(self, text):
        """Filter scope tree based on search text."""
        filter_text = text.lower().strip()
        
        for i in range(self.scope_tree.topLevelItemCount()):
            top_item = self.scope_tree.topLevelItem(i)
            self._filter_tree_item_recursive(top_item, filter_text)

    def _filter_tree_item_recursive(self, item, filter_text):
        """Recursively filter tree items and their children."""
        if not filter_text:
            item.setHidden(False)
            for i in range(item.childCount()):
                self._filter_tree_item_recursive(item.child(i), filter_text)
            return

        item_text = item.text(0).lower()
        matches = filter_text in item_text
        
        # Check children
        any_child_visible = False
        for i in range(item.childCount()):
            child = item.child(i)
            self._filter_tree_item_recursive(child, filter_text)
            if not child.isHidden():
                any_child_visible = True
        
        # Show if matches or has visible children
        item.setHidden(not (matches or any_child_visible))

    def _set_scope_tree_checked(self, checked, visible_only=False):
        """Set check state for all or only visible scope tree items."""
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.scope_tree.blockSignals(True)
        
        try:
            for i in range(self.scope_tree.topLevelItemCount()):
                top_item = self.scope_tree.topLevelItem(i)
                if visible_only and top_item.isHidden():
                    continue
                self._set_tree_item_checked_recursive(top_item, state, visible_only)
        finally:
            self.scope_tree.blockSignals(False)
            self._on_ai_search_scope_changed()

    def _set_tree_item_checked_recursive(self, item, state, visible_only=False):
        """Recursively set check state for tree item and children."""
        if visible_only and item.isHidden():
            return
        
        if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            item.setCheckState(0, state)
        
        for i in range(item.childCount()):
            self._set_tree_item_checked_recursive(item.child(i), state, visible_only)

    def _expand_selected_scope_items(self):
        """Expand all items that are checked."""
        for i in range(self.scope_tree.topLevelItemCount()):
            self._expand_selected_recursive(self.scope_tree.topLevelItem(i))

    def _expand_selected_recursive(self, item):
        """Recursively expand checked items."""
        if item.checkState(0) == Qt.CheckState.Checked:
            item.setExpanded(True)
        
        for i in range(item.childCount()):
            self._expand_selected_recursive(item.child(i))

    def _collapse_all_scope_items(self):
        """Collapse all items in the scope tree."""
        self.scope_tree.collapseAll()

    def _on_scope_tree_item_changed(self, item, column):
        """Handle item check state changes."""
        if column == 0 and item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            if getattr(self, '_applying_scope_config', False):
                return
            item_data = item.data(0, Qt.ItemDataRole.UserRole)
            if item_data and item_data[0] == "deck" and item.checkState(0) == Qt.CheckState.Checked:
                self._ensure_deck_tree_children(item)
            # Auto-expand children when parent is checked
            if item.checkState(0) == Qt.CheckState.Checked:
                item.setExpanded(True)
            self._persist_scope_config()
            self._on_ai_search_scope_changed()

    def _on_scope_tree_item_expanded(self, item):
        """Lazy-load note types and fields when the user opens a deck."""
        self._ensure_deck_tree_children(item)
        self._on_ai_search_scope_changed()

    def _on_search_all_fields_toggled(self):
        """Handle search all fields toggle."""
        # Disable field checkboxes when search all fields is enabled
        enabled = not self.search_all_fields_cb.isChecked()
        self._set_field_checkboxes_enabled(enabled)
        self._persist_scope_config()

    def _on_chatbot_field_mode_changed(self):
        if getattr(self, "_applying_chatbot_field_config", False):
            return
        if not self.chatbot_same_fields_cb.isChecked():
            self._refresh_chatbot_field_rows(preserve=True)
            self._seed_chatbot_fields_from_search_if_empty()
        self._set_chatbot_fields_enabled()
        self._persist_chatbot_field_config()

    def _on_chatbot_field_tree_item_changed(self, item, column):
        if getattr(self, "_applying_chatbot_field_config", False):
            return
        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        if item_data and item_data[0] == "chatbot_field":
            self._persist_chatbot_field_config()

    def _set_chatbot_fields_enabled(self):
        custom_enabled = not self.chatbot_same_fields_cb.isChecked()
        self.chatbot_same_fields_note.setVisible(not custom_enabled)
        if hasattr(self, "chatbot_same_fields_note_row"):
            self.chatbot_same_fields_note_row.setVisible(not custom_enabled)
        self.chatbot_fields_tree.setVisible(custom_enabled)
        if hasattr(self, "chatbot_fields_tree_group"):
            self.chatbot_fields_tree_group.setVisible(custom_enabled)
        self.chatbot_fields_tree.setEnabled(custom_enabled)

    def _seed_chatbot_fields_from_search_if_empty(self):
        if any(item.checkState(0) == Qt.CheckState.Checked for items in self.chatbot_fields_by_note_type.values() for item in items):
            return
        selected_by_note_type = self._build_scope_config_from_ui().get("note_type_fields") or {}
        if not selected_by_note_type:
            return
        previous_block_state = getattr(self, "_applying_chatbot_field_config", False)
        self._applying_chatbot_field_config = True
        try:
            for model_name, items in self.chatbot_fields_by_note_type.items():
                selected = {str(name).strip().lower() for name in selected_by_note_type.get(model_name, [])}
                for item in items:
                    item_data = item.data(0, Qt.ItemDataRole.UserRole)
                    field_name = item_data[2] if item_data and len(item_data) > 2 else item.text(0)
                    item.setCheckState(
                        0,
                        Qt.CheckState.Checked if field_name.strip().lower() in selected else Qt.CheckState.Unchecked,
                    )
        finally:
            self._applying_chatbot_field_config = previous_block_state

    def _selected_ai_search_note_type_rows(self):
        rows_by_model = {}
        field_fallbacks = {model_name: field_names for model_name, _count, field_names in get_models_with_fields()}

        for i in range(self.scope_tree.topLevelItemCount()):
            deck_item = self.scope_tree.topLevelItem(i)
            deck_data = deck_item.data(0, Qt.ItemDataRole.UserRole)
            if not deck_data or deck_data[0] != "deck" or deck_item.checkState(0) != Qt.CheckState.Checked:
                continue

            self._ensure_deck_tree_children(deck_item)
            for j in range(deck_item.childCount()):
                note_type_item = deck_item.child(j)
                note_type_data = note_type_item.data(0, Qt.ItemDataRole.UserRole)
                if (
                    not note_type_data
                    or note_type_data[0] != "note_type"
                    or note_type_item.checkState(0) != Qt.CheckState.Checked
                ):
                    continue

                model_name = note_type_data[2]
                try:
                    count = int(str(note_type_item.text(1)).replace(",", ""))
                except ValueError:
                    count = 0

                field_names = []
                for k in range(note_type_item.childCount()):
                    field_item = note_type_item.child(k)
                    field_data = field_item.data(0, Qt.ItemDataRole.UserRole)
                    if field_data and field_data[0] == "field":
                        field_names.append(field_data[3])
                if not field_names:
                    field_names = field_fallbacks.get(model_name, [])

                existing_count, existing_fields = rows_by_model.get(model_name, (0, []))
                merged_fields = existing_fields + [field for field in field_names if field not in existing_fields]
                rows_by_model[model_name] = (existing_count + count, merged_fields)

        rows = [(model_name, count, field_names) for model_name, (count, field_names) in rows_by_model.items()]
        return sorted(rows, key=lambda row: row[1], reverse=True)

    def _clear_chatbot_field_rows(self):
        self.chatbot_fields_tree.clear()
        self.chatbot_fields_by_note_type = {}

    def _refresh_chatbot_field_rows(self, preserve=True, selected_by_note_type=None):
        if not hasattr(self, "chatbot_fields_tree"):
            return
        if selected_by_note_type is not None:
            existing = selected_by_note_type
        elif preserve:
            existing = self._build_chatbot_field_config_from_ui().get("note_type_fields", {})
        else:
            existing = {}
        previous_block_state = getattr(self, "_applying_chatbot_field_config", False)
        self._applying_chatbot_field_config = True
        try:
            self._clear_chatbot_field_rows()
            rows = self._selected_ai_search_note_type_rows()
            if not rows:
                item = QTreeWidgetItem(["Select note types in AI Search above to choose Review Ask AI current-note fields.", ""])
                item.setData(0, Qt.ItemDataRole.UserRole, ("chatbot_empty",))
                item.setDisabled(True)
                self.chatbot_fields_tree.addTopLevelItem(item)
            for model_name, count, field_names in rows:
                selected = {str(name).strip().lower() for name in existing.get(model_name, [])}
                note_type_item = QTreeWidgetItem([f"{model_name} ({count:,})", ""])
                note_type_item.setData(0, Qt.ItemDataRole.UserRole, ("chatbot_note_type", model_name))
                self.chatbot_fields_tree.addTopLevelItem(note_type_item)

                field_items = []
                for field_name in field_names:
                    field_item = QTreeWidgetItem([field_name, ""])
                    field_item.setData(0, Qt.ItemDataRole.UserRole, ("chatbot_field", model_name, field_name))
                    field_item.setFlags(field_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    field_item.setCheckState(
                        0,
                        Qt.CheckState.Checked if field_name.strip().lower() in selected else Qt.CheckState.Unchecked,
                    )
                    note_type_item.addChild(field_item)
                    field_items.append(field_item)
                self.chatbot_fields_by_note_type[model_name] = field_items
            self.chatbot_fields_tree.collapseAll()
        finally:
            self._applying_chatbot_field_config = previous_block_state

    def _on_ai_search_scope_changed(self):
        if getattr(self, "_applying_scope_config", False):
            return
        if hasattr(self, "chatbot_fields_tree") and not self.chatbot_same_fields_cb.isChecked():
            self._refresh_chatbot_field_rows(preserve=True)
            self._persist_chatbot_field_config()

    def _refresh_chatbot_fields_after_scope_apply(self, config=None):
        """Refresh chatbot fields after saved AI Search scope has been restored."""
        if not hasattr(self, "chatbot_fields_tree"):
            return
        config = config or load_config()
        review_cfg = (config or {}).get("review_ask_ai") or {}
        if review_cfg.get("context_source", "embedding_fields") != "custom_fields":
            self._set_chatbot_fields_enabled()
            return

        selected_by_note_type = review_cfg.get("note_type_fields") or {}
        if not selected_by_note_type:
            selected_by_note_type = ((config or {}).get("note_type_filter") or {}).get("note_type_fields") or {}
        self._refresh_chatbot_field_rows(preserve=False, selected_by_note_type=selected_by_note_type)
        self._set_chatbot_fields_enabled()

    def _build_chatbot_field_config_from_ui(self):
        current = (load_config() or {}).get("review_ask_ai") or {}
        if self.chatbot_same_fields_cb.isChecked():
            return {
                "context_source": "embedding_fields",
                "note_type_fields": current.get("note_type_fields") or {},
                "search_all_fields": False,
                "use_first_field_fallback": True,
            }

        note_type_fields = {}
        for model_name, items in self.chatbot_fields_by_note_type.items():
            selected = []
            for item in items:
                if item.checkState(0) != Qt.CheckState.Checked:
                    continue
                item_data = item.data(0, Qt.ItemDataRole.UserRole)
                selected.append(item_data[2] if item_data and len(item_data) > 2 else item.text(0))
            if selected:
                note_type_fields[model_name] = selected
        if not self.chatbot_fields_by_note_type:
            note_type_fields = current.get("note_type_fields") or {}

        return {
            "context_source": "custom_fields",
            "note_type_fields": note_type_fields,
            "search_all_fields": False,
            "use_first_field_fallback": True,
        }

    def _persist_chatbot_field_config(self):
        if getattr(self, "_applying_chatbot_field_config", False):
            return
        try:
            config = load_config()
            config["review_ask_ai"] = self._build_chatbot_field_config_from_ui()
            save_config(config)
        except Exception as e:
            log_debug(f"Error persisting Review Ask AI field config: {e}")

    def _apply_chatbot_field_config(self, review_cfg):
        self._applying_chatbot_field_config = True
        try:
            review_cfg = review_cfg or {}
            self.chatbot_same_fields_cb.setChecked(
                review_cfg.get("context_source", "embedding_fields") != "custom_fields"
            )
        finally:
            self._applying_chatbot_field_config = False
            self._refresh_chatbot_field_rows(preserve=False, selected_by_note_type=review_cfg.get("note_type_fields") or {})
            self._set_chatbot_fields_enabled()

    def _set_field_checkboxes_enabled(self, enabled):
        """Enable or disable all field checkboxes in the tree."""
        for i in range(self.scope_tree.topLevelItemCount()):
            self._set_field_checkboxes_enabled_recursive(self.scope_tree.topLevelItem(i), enabled)

    def _set_field_checkboxes_enabled_recursive(self, item, enabled):
        """Recursively enable/disable field checkboxes."""
        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        item_type = item_data[0] if item_data else None
        if item_type == "field":
            item.setDisabled(not enabled)
        
        for i in range(item.childCount()):
            self._set_field_checkboxes_enabled_recursive(item.child(i), enabled)

    def _populate_scope_tree(self):
        """Populate the deck scope tree."""
        if not hasattr(self, 'scope_tree') or self.scope_tree is None:
            return

        # Guard against population during initialization (should only populate when tab is opened)
        # Only allow population if explicitly forced or if lists are being loaded via the proper lifecycle
        if not getattr(self, '_allow_scope_population', False):
            log_debug(f"  [Debug] _populate_scope_tree: Skipping (not allowed during init)")
            return

        log_debug(f"  [Debug] _populate_scope_tree: Starting population")

        try:
            self.scope_tree.clear()
            self._scope_tree_items.clear()

            self._populate_deck_scope_tree()

            # Collapse all items by default
            self.scope_tree.collapseAll()
            log_debug(f"  [Debug] _populate_scope_tree: Completed")
        except Exception as e:
            log_debug(f"Error populating scope tree: {e}")

    def _populate_deck_scope_tree(self):
        """Populate scope tree with main decks only."""
        try:
            deck_names = [name for name in get_deck_names() if "::" not in name]
            deck_counts = get_notes_count_per_deck()

            for deck_name in sorted(deck_names):
                self._create_deck_tree_item(deck_name, deck_counts)
            
        except Exception as e:
            log_debug(f"Error populating deck scope tree: {e}")

    def _create_deck_tree_item(self, deck_name, deck_counts):
        """Create a deck tree item. Note types are loaded lazily on expand."""
        note_count = deck_counts.get(deck_name, 0)
        
        # Hide empty Default deck
        if deck_name == "Default" and note_count == 0:
            return None
        
        item = QTreeWidgetItem([deck_name, str(note_count)])
        item.setData(0, Qt.ItemDataRole.UserRole, ("deck", deck_name))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, Qt.CheckState.Unchecked)

        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        self.scope_tree.addTopLevelItem(item)
        
        if note_count > 0:
            placeholder = QTreeWidgetItem(["Loading note types...", ""])
            placeholder.setData(0, Qt.ItemDataRole.UserRole, ("placeholder", deck_name))
            item.addChild(placeholder)
        
        return item

    def _ensure_deck_tree_children(self, deck_item):
        """Populate note-type/field children for one deck item if needed."""
        try:
            if deck_item is None:
                return
            item_data = deck_item.data(0, Qt.ItemDataRole.UserRole)
            if not item_data or item_data[0] != "deck":
                return
            has_placeholder = False
            for index in range(deck_item.childCount()):
                child_data = deck_item.child(index).data(0, Qt.ItemDataRole.UserRole)
                if child_data and child_data[0] == "placeholder":
                    has_placeholder = True
                    break
                if child_data and child_data[0] == "note_type":
                    return
            if deck_item.childCount() > 0 and not has_placeholder:
                return

            deck_name = item_data[1]
            model_data = get_models_with_fields_for_deck(deck_name)

            previous_block_state = self.scope_tree.blockSignals(True)
            try:
                for index in reversed(range(deck_item.childCount())):
                    child = deck_item.child(index)
                    child_data = child.data(0, Qt.ItemDataRole.UserRole)
                    if child_data and child_data[0] == "placeholder":
                        deck_item.removeChild(child)
                for model_name, count, field_names in model_data:
                    nt_item = QTreeWidgetItem([model_name, str(count)])
                    nt_item.setData(0, Qt.ItemDataRole.UserRole, ("note_type", deck_name, model_name))
                    nt_item.setFlags(nt_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    nt_item.setCheckState(0, Qt.CheckState.Unchecked)
                    deck_item.addChild(nt_item)

                    for field_name in field_names:
                        field_item = QTreeWidgetItem([field_name, ""])
                        field_item.setData(0, Qt.ItemDataRole.UserRole, ("field", deck_name, model_name, field_name))
                        field_item.setFlags(field_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        field_item.setCheckState(0, Qt.CheckState.Unchecked)
                        nt_item.addChild(field_item)
            finally:
                self.scope_tree.blockSignals(previous_block_state)
        except Exception as e:
            log_debug(f"Error populating deck children: {e}")

    def _persist_scope_config(self):
        """Save current scope configuration to config."""
        if getattr(self, '_applying_scope_config', False):
            return
        
        try:
            from ..utils import save_config
            config = load_config()
            config['note_type_filter'] = self._build_scope_config_from_ui()
            save_config(config)
        except Exception as e:
            log_debug(f"Error persisting scope config: {e}")

    def _build_ntf_from_ui(self):
        """Build note_type_filter from the active AI Search scope tree."""
        if hasattr(self, "scope_tree") and self.scope_tree is not None:
            if self.scope_tree.topLevelItemCount() == 0:
                return self._resolve_scope_config_from_config(load_config(), persist=False)
            return self._build_scope_config_from_ui()
        return super()._build_ntf_from_ui()

    def _build_scope_config_from_ui(self):
        """Build scope config dict from the current UI state."""
        mode = "deck"
        
        enabled_scopes = []
        scope_fields = {}
        
        for i in range(self.scope_tree.topLevelItemCount()):
            top_item = self.scope_tree.topLevelItem(i)
            top_data = top_item.data(0, Qt.ItemDataRole.UserRole)
            if top_data and top_data[0] == "deck" and top_item.checkState(0) == Qt.CheckState.Checked:
                self._ensure_deck_tree_children(top_item)
            self._extract_scope_config_recursive(top_item, mode, enabled_scopes, scope_fields)
        
        # Build config with both new and legacy keys for compatibility
        config = {
            'scope_mode': mode,
            'search_all_fields': self.search_all_fields_cb.isChecked(),
            'use_first_field_fallback': self.use_first_field_cb.isChecked(),
        }
        
        if mode == "deck":
            config['enabled_decks'] = enabled_scopes if enabled_scopes else None
            config['enabled_tags'] = None
        
        config['scope_fields'] = scope_fields
        
        # Build legacy keys for backward compatibility
        enabled_note_types = set()
        note_type_fields = {}
        
        for scope, nt_dict in scope_fields.items():
            for nt_name, fields in nt_dict.items():
                enabled_note_types.add(nt_name)
                if fields:
                    note_type_fields[nt_name] = list(set(note_type_fields.get(nt_name, []) + fields))
        
        config['enabled_note_types'] = list(enabled_note_types) if enabled_note_types else None
        config['note_type_fields'] = note_type_fields
        
        return config

    def _extract_scope_config_recursive(self, item, mode, enabled_scopes, scope_fields):
        """Recursively extract scope config from tree items."""
        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not item_data:
            return
        
        item_type = item_data[0]
        
        if item_type == "deck" and item.checkState(0) == Qt.CheckState.Checked:
            scope_name = item_data[1]
            enabled_scopes.append(scope_name)
            
            # Extract note types and fields for this scope
            scope_fields[scope_name] = {}
            for i in range(item.childCount()):
                child = item.child(i)
                child_data = child.data(0, Qt.ItemDataRole.UserRole)
                if child_data and child_data[0] == "note_type" and child.checkState(0) == Qt.CheckState.Checked:
                    nt_name = child_data[2]
                    fields = []
                    for j in range(child.childCount()):
                        field_child = child.child(j)
                        field_data = field_child.data(0, Qt.ItemDataRole.UserRole)
                        if field_data and field_data[0] == "field" and field_child.checkState(0) == Qt.CheckState.Checked:
                            fields.append(field_data[3])
                    scope_fields[scope_name][nt_name] = fields
        
        # Recursively process children
        for i in range(item.childCount()):
            self._extract_scope_config_recursive(item.child(i), mode, enabled_scopes, scope_fields)

    def _apply_scope_config(self, ntf):
        """Apply scope config to the UI."""
        log_debug(f"  [Debug] _apply_scope_config: Starting")
        self._applying_scope_config = True

        try:
            mode = "deck"

            # Set search all fields and fallback
            log_debug(f"  [Debug] _apply_scope_config: Setting search all fields")
            self.search_all_fields_cb.setChecked(bool(ntf.get('search_all_fields', False)))
            log_debug(f"  [Debug] _apply_scope_config: Calling _on_search_all_fields_toggled")
            self._on_search_all_fields_toggled()
            log_debug(f"  [Debug] _apply_scope_config: _on_search_all_fields_toggled completed")
            self.use_first_field_cb.setChecked(bool(ntf.get('use_first_field_fallback', True)))
            log_debug(f"  [Debug] _apply_scope_config: Use first two fields set")

            # Apply scope selections only if tree has items (skip if empty during init)
            log_debug(f"  [Debug] _apply_scope_config: Tree item count = {self.scope_tree.topLevelItemCount()}")
            if self.scope_tree.topLevelItemCount() > 0:
                log_debug(f"  [Debug] _apply_scope_config: Applying scope selections")
                scope_fields = ntf.get('scope_fields', {})
                enabled_scopes = ntf.get('enabled_decks')

                if enabled_scopes:
                    enabled_set = set(enabled_scopes)
                else:
                    enabled_set = None

                for i in range(self.scope_tree.topLevelItemCount()):
                    self._apply_scope_config_recursive(self.scope_tree.topLevelItem(i), mode, enabled_set, scope_fields)
                log_debug(f"  [Debug] _apply_scope_config: Scope selections applied")
            else:
                log_debug(f"  [Debug] _apply_scope_config: Skipping scope selections (tree empty)")

        except Exception as e:
            log_debug(f"Error applying scope config: {e}")

        finally:
            self._applying_scope_config = False
            # Collapse all items by default
            self.scope_tree.collapseAll()
            log_debug(f"  [Debug] _apply_scope_config: Completed")

    def _apply_scope_config_recursive(self, item, mode, enabled_scopes, scope_fields):
        """Recursively apply scope config to tree items."""
        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not item_data:
            return
        
        item_type = item_data[0]
        
        if item_type == "deck":
            scope_name = item_data[1]
            should_check = (enabled_scopes is None) or (scope_name in enabled_scopes)
            item.setCheckState(0, Qt.CheckState.Checked if should_check else Qt.CheckState.Unchecked)

            if scope_name in scope_fields:
                self._ensure_deck_tree_children(item)
            
            # Apply note type and field selections for this scope
            if scope_name in scope_fields:
                for i in range(item.childCount()):
                    child = item.child(i)
                    child_data = child.data(0, Qt.ItemDataRole.UserRole)
                    if child_data and child_data[0] == "note_type":
                        nt_name = child_data[2]
                        if nt_name in scope_fields[scope_name]:
                            child.setCheckState(0, Qt.CheckState.Checked)
                            fields = scope_fields[scope_name][nt_name]
                            for j in range(child.childCount()):
                                field_child = child.child(j)
                                field_data = field_child.data(0, Qt.ItemDataRole.UserRole)
                                if field_data and field_data[0] == "field":
                                    field_name = field_data[3]
                                    field_child.setCheckState(0, Qt.CheckState.Checked if field_name in fields else Qt.CheckState.Unchecked)
                        else:
                            child.setCheckState(0, Qt.CheckState.Unchecked)
        
        # Recursively process children
        for i in range(item.childCount()):
            self._apply_scope_config_recursive(item.child(i), mode, enabled_scopes, scope_fields)

    def _refresh_scope_lists(self):
        """Refresh scope lists and preserve checked state where possible."""
        self._scope_lists_loaded = True
        self._allow_scope_population = True  # Allow population for refresh
        self._invalidate_scope_cache()
        self._set_scope_loading_status(
            "Refreshing scope data. This can take a while on large collections.",
            busy=True,
        )

        # Save current state
        current_config = self._build_scope_config_from_ui()

        self._populate_scope_tree()
        self._apply_scope_config(current_config)
        self._refresh_preset_combos()

        self._set_scope_loading_status("Scope data refreshed.", busy=False)
        QTimer.singleShot(3500, lambda: self._set_scope_loading_status("", busy=False))

        showInfo("Scope lists refreshed.")

    def _set_scope_loading_status(self, message="", busy=False):
        """Set scope loading status message and progress bar."""
        label = getattr(self, "scope_loading_status", None)
        bar = getattr(self, "scope_loading_bar", None)
        
        try:
            if label is not None:
                label.setText(message or "")
                label.setVisible(bool(message))
            if bar is not None:
                bar.setVisible(bool(busy))
            if message:
                log_debug(f"Scope loading status: {message}")
        except RuntimeError:
            pass

    def _invalidate_scope_cache(self):
        """Invalidate scope data cache including deck counts."""
        # Invalidate the lifecycle cache.
        if hasattr(self, '_invalidate_note_type_deck_cache'):
            self._invalidate_note_type_deck_cache()

    def _on_scope_count_notes(self):
        """Handle preview coverage button click for scope selector."""
        try:
            audit = analyze_note_eligibility(self._build_scope_config_from_ui())
            eligible_count = audit.get("eligible_count", 0)
            total_notes = audit.get("total_notes", 0)
            excluded_count = max(0, total_notes - eligible_count)
            showInfo(
                f"Eligible notes: {eligible_count:,}\n"
                f"Excluded notes: {excluded_count:,}\n"
                f"Total matched notes: {total_notes:,}"
            )
        except Exception as e:
            log_debug(f"Error previewing scope coverage: {e}")
            showInfo("Could not preview coverage. Check the debug log for details.")
