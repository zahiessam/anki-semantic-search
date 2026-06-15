"""Search dialog state, history, and option handlers."""

# ============================================================================
# Imports
# ============================================================================

import re

from aqt import mw
from aqt.qt import QDialog, QFrame, QHBoxLayout, QMenu, QPushButton, QScrollArea, QTimer, Qt, QVBoxLayout, QWidget, QWidgetAction
from aqt.utils import askUser, tooltip

from .settings_dialog import SettingsDialog
from .theme import get_addon_theme
from ..core.engine import get_deck_names
from ..utils import (
    clamp_relevance_threshold_percent,
    clear_search_history,
    delete_search_history,
    get_search_history_queries,
    load_config,
    log_debug,
    save_config,
)


# ============================================================================
# Search Dialog State And History Handlers
# ============================================================================

def get_all_note_types():
    try:
        return [model.get('name') for model in mw.col.models.all()] if mw and mw.col else []
    except Exception:
        return []


def _session_debug_log(*args, **kwargs):
    return None


def _qt_cursor_pointing_hand():
    try:
        return Qt.CursorShape.PointingHandCursor
    except AttributeError:
        return Qt.PointingHandCursor


def _qt_elide_right():
    try:
        return Qt.TextElideMode.ElideRight
    except AttributeError:
        return Qt.ElideRight


def _history_popup_dimensions(owner, history_count):
    target_width = 420
    try:
        owner_width = owner.width()
        target_width = min(target_width, max(320, owner_width - 56))
    except Exception:
        pass
    row_height = 38
    header_height = 42 if history_count else 0
    popup_height = min(280, max(54, history_count * row_height + header_height + 14))
    return target_width, popup_height


def _history_popup_styles(theme):
    return f"""
        QMenu {{
            background-color: {theme['panel_bg']};
            color: {theme['text']};
            border: 1px solid {theme['subtle_border']};
            border-radius: 8px;
            padding: 4px;
        }}
        QScrollArea {{
            background-color: {theme['panel_bg']};
            border: none;
        }}
        QWidget#historyPopupContainer {{
            background-color: {theme['panel_bg']};
            border: none;
        }}
        QWidget#historyRow {{
            background-color: transparent;
            border: none;
            border-radius: 6px;
        }}
        QWidget#historyRow:hover {{
            background-color: {theme['field_row_hover_bg']};
        }}
        QPushButton#historyQueryButton {{
            background-color: transparent;
            border: none;
            color: {theme['text']};
            text-align: left;
            padding: 6px 8px;
            font-size: 12px;
            font-weight: 400;
        }}
        QPushButton#historyQueryButton:hover {{
            color: {theme['text']};
        }}
        QPushButton#historyDeleteButton {{
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 10px;
            color: rgba(148, 163, 184, 0.36);
            padding: 0;
            font-size: 13px;
            font-weight: 700;
        }}
        QPushButton#historyDeleteButton:hover {{
            background-color: {theme['control_bg']};
            border-color: {theme['subtle_border']};
            color: {theme['danger']};
        }}
    """


def _history_display_query(query):
    display = str(query or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(display) >= 2 and display[0] == display[-1] and display[0] in ("'", '"'):
        display = display[1:-1].strip()

    useful_lines = []
    answer_markers = (
        "answer",
        "ask ai answer",
        "ask notes answer",
        "searched anki notes",
        "direct answer:",
        "answer from:",
        "relevant_notes:",
    )
    for line in display.split("\n"):
        cleaned = line.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if useful_lines and (
            lowered in answer_markers
            or lowered.startswith(answer_markers)
            or re.match(r"^(?:[ivxlcdm]+\.|\d+\.)\s+", lowered)
        ):
            break
        useful_lines.append(cleaned)
        if len(useful_lines) >= 2:
            break

    display = " ".join(useful_lines) if useful_lines else display
    display = re.sub(r"\s+", " ", display).strip()
    if len(display) > 170:
        display = display[:167].rstrip() + "..."
    return display or str(query or "")


def _position_history_menu(menu, anchor, popup_width, popup_height):
    try:
        below = anchor.mapToGlobal(anchor.rect().bottomLeft())
        above = anchor.mapToGlobal(anchor.rect().topLeft())
        gap = 8
        x = below.x()
        y = below.y() + gap
        screen = anchor.screen() if hasattr(anchor, "screen") else None
        geometry = screen.availableGeometry() if screen else None
        if geometry:
            upward_y = above.y() - popup_height - gap
            if upward_y >= geometry.top() + 4:
                y = upward_y
            elif y + popup_height > geometry.bottom():
                y = max(geometry.top() + 4, geometry.bottom() - popup_height - 4)
            x = max(geometry.left() + 4, min(x, geometry.right() - popup_width - 4))
            y = max(geometry.top() + 4, min(y, geometry.bottom() - popup_height - 4))
        menu.move(x, y)
    except Exception as exc:
        log_debug(f"Could not reposition history menu: {exc}")

def _update_view_all_button_state(self):



    """Update View All button enabled state and tooltip based on whether the results list has rows."""



    if not hasattr(self, 'view_all_btn') or not self.view_all_btn:



        return



    has_rows = self.results_list.rowCount() > 0 if hasattr(self, 'results_list') and self.results_list else False



    self.view_all_btn.setEnabled(has_rows)



    self.view_all_btn.setToolTip(



        "Open all visible notes in the Anki browser"



        if has_rows



        else "No notes in the list \u2014 run a search first."



    )



def showEvent(self, event):



    """Refresh scope banner when dialog is shown (e.g. after Settings changed)."""



    QDialog.showEvent(self, event)



    self._refresh_scope_banner()



def _refresh_scope_banner(self):



    """Update scope banner: X note types, Y fields, Z decks with shortcut to Settings."""



    if not hasattr(self, 'scope_banner') or not self.scope_banner:



        return



    try:



        config = load_config()



        ntf = config.get('note_type_filter') or {}



        enabled_types = ntf.get('enabled_note_types') or []



        ntf_fields = ntf.get('note_type_fields') or {}



        enabled_decks = ntf.get('enabled_decks') or []



        search_all = bool(ntf.get('search_all_fields', False))



        n_types = len(enabled_types) if enabled_types else len(get_all_note_types())



        n_decks = len(enabled_decks) if enabled_decks else len(get_deck_names())



        fields_set = set()



        if search_all:



            fields_set = set(f['name'] for m in (mw.col.models.all() if mw and mw.col else []) for f in m.get('flds', []))



        else:



            for flist in ntf_fields.values():



                fields_set.update(flist or [])



        n_fields = len(fields_set) if fields_set else 1



        txt = f"Searching: {n_types} note types, {n_fields} fields, {n_decks} decks"



        self.scope_banner.setText(txt)



        if hasattr(self, 'sidebar_scope_label') and self.sidebar_scope_label:



            self.sidebar_scope_label.setText(



                f"Scope\n{n_types} note types\n{n_fields} fields\n{n_decks} decks"



            )



    except Exception as e:



        log_debug(f"Scope banner refresh error: {e}")



        self.scope_banner.setText("")



        if hasattr(self, 'sidebar_scope_label') and self.sidebar_scope_label:



            self.sidebar_scope_label.setText("Scope unavailable")



def _on_search_history_selected(self, index):



    """When user selects a recent search from the dropdown, populate the input."""



    if index >= 0 and hasattr(self, 'search_input') and hasattr(self, 'search_history_combo'):



        text = self.search_history_combo.currentText()



        self._set_query_text(text)



def _on_sidebar_history_selected(self, item):



    """Load query from left panel history list."""



    if not item:



        return



    self._set_query_text(item.text())



def _on_search_history_query_selected(self, query):

    """Load a query selected from the history menu."""

    self._set_query_text(query)


def _on_delete_search_history_query(self, query, keep_menu_open=False, deleted_row=None):

    """Delete a single saved search and refresh the history controls."""

    if delete_search_history(query):

        history = get_search_history_queries()

        if keep_menu_open:

            if not history:

                try:

                    menu = getattr(self, 'search_history_menu', None)

                    if menu:

                        menu.close()

                except Exception as exc:

                    log_debug(f"Could not close empty history menu: {exc}")

                self._refresh_search_history()

            elif deleted_row is not None:

                try:

                    deleted_row.setParent(None)

                    deleted_row.deleteLater()

                except Exception as exc:

                    log_debug(f"Could not remove history row from open menu: {exc}")

            if hasattr(self, '_search_history_model'):

                self._search_history_model.setStringList(history)

            if hasattr(self, 'sidebar_history_list') and self.sidebar_history_list:

                self.sidebar_history_list.blockSignals(True)

                self.sidebar_history_list.clear()

                self.sidebar_history_list.addItems(history)

                self.sidebar_history_list.blockSignals(False)

            if hasattr(self, 'search_history_btn') and self.search_history_btn:

                self.search_history_btn.setEnabled(bool(history))

                self.search_history_btn.setToolTip(
                    "Load a recent search"
                    if history else
                    "No recent searches yet."
                )

        else:

            self._refresh_search_history()

        if hasattr(self, 'status_label') and self.status_label:

            self.status_label.setText("Search removed from history.")

            QTimer.singleShot(3000, self._clear_search_removed_status)


def _clear_search_removed_status(self):

    """Clear the single-delete status after a delay."""

    if hasattr(self, 'status_label') and self.status_label:

        if self.status_label.text() == "Search removed from history.":

            self.status_label.setText("Ready")


def _rebuild_search_history_menu(self, history):

    """Rebuild the history popup with per-row delete buttons."""

    if not hasattr(self, 'search_history_btn') or not self.search_history_btn:

        return

    menu = QMenu(self.search_history_btn)

    self.search_history_menu = menu
    theme = get_addon_theme()
    popup_width, popup_height = _history_popup_dimensions(self, len(history or []))
    menu.setStyleSheet(_history_popup_styles(theme))

    def place_menu():
        _position_history_menu(menu, self.search_history_btn, popup_width, popup_height)

    try:
        menu.aboutToShow.connect(lambda: QTimer.singleShot(0, place_menu))
    except Exception:
        pass

    if not history:

        empty_widget = QWidget(menu)
        empty_layout = QVBoxLayout(empty_widget)
        empty_layout.setContentsMargins(12, 10, 12, 10)
        empty_row = QPushButton("No recent searches")
        empty_row.setEnabled(False)
        empty_row.setStyleSheet(
            f"background: transparent; border: none; color: {theme['quiet_text']}; "
            "font-size: 12px; padding: 8px;"
        )
        empty_layout.addWidget(empty_row)
        empty_action = QWidgetAction(menu)
        empty_action.setDefaultWidget(empty_widget)
        menu.addAction(empty_action)

        try:
            empty_widget.setFixedSize(max(240, popup_width - 10), 48)
        except Exception:
            pass

    else:

        scroll = QScrollArea(menu)

        scroll.setWidgetResizable(True)

        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        scroll.setFrameShape(QFrame.Shape.NoFrame if hasattr(QFrame, "Shape") else QFrame.NoFrame)

        scroll.setFixedSize(popup_width, popup_height)

        container = QWidget(scroll)
        container.setObjectName("historyPopupContainer")

        container_layout = QVBoxLayout(container)

        container_layout.setContentsMargins(5, 5, 5, 5)

        container_layout.setSpacing(3)

        header = QWidget(container)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(8, 5, 8, 5)
        header_layout.setSpacing(0)
        title = QPushButton("History")
        title.setEnabled(False)
        title.setStyleSheet(
            f"background: transparent; border: none; color: {theme['text']}; "
            "font-size: 12px; font-weight: 700; text-align: left; padding: 0;"
        )
        subtitle = QPushButton("Recent searches")
        subtitle.setEnabled(False)
        subtitle.setStyleSheet(
            f"background: transparent; border: none; color: {theme['quiet_text']}; "
            "font-size: 10px; font-weight: 400; text-align: left; padding: 0;"
        )
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        container_layout.addWidget(header)

        for query in history:

            row = QWidget(container)
            row.setObjectName("historyRow")
            row.setMinimumHeight(36)

            row_layout = QHBoxLayout(row)

            row_layout.setContentsMargins(4, 2, 4, 2)

            row_layout.setSpacing(4)

            label_width = max(220, popup_width - 62)
            display_query = _history_display_query(query)
            try:
                label = row.fontMetrics().elidedText(display_query, _qt_elide_right(), label_width)
            except Exception:
                label = display_query[:80] + "..." if len(display_query) > 80 else display_query

            query_btn = QPushButton(label)
            query_btn.setObjectName("historyQueryButton")

            query_btn.setFlat(True)

            query_btn.setCursor(_qt_cursor_pointing_hand())

            query_btn.setToolTip("Click to load this search")

            query_btn.setMinimumHeight(32)

            query_btn.setMaximumWidth(label_width + 18)

            def load_query(_checked=False, q=query, m=menu):

                m.close()

                QTimer.singleShot(0, lambda: self._on_search_history_query_selected(q))

            query_btn.clicked.connect(load_query)

            delete_btn = QPushButton("\u00d7")
            delete_btn.setObjectName("historyDeleteButton")

            delete_btn.setFixedSize(20, 20)

            delete_btn.setToolTip("Delete this search")

            delete_btn.setCursor(_qt_cursor_pointing_hand())

            def delete_query(_checked=False, q=query, r=row):

                QTimer.singleShot(0, lambda: self._on_delete_search_history_query(q, keep_menu_open=True, deleted_row=r))

            delete_btn.clicked.connect(delete_query)

            row_layout.addWidget(query_btn, 1)

            row_layout.addWidget(delete_btn)

            container_layout.addWidget(row)

        container_layout.addStretch()

        scroll.setWidget(container)

        action = QWidgetAction(menu)

        action.setDefaultWidget(scroll)

        menu.addAction(action)

    self.search_history_btn.setMenu(menu)

    self.search_history_btn.setEnabled(bool(history))

    self.search_history_btn.setToolTip(
        "Load a recent search"
        if history else
        "No recent searches yet."
    )


def _set_query_text(self, text):



    """Set query text in the editor and show a short status hint."""



    if not text or not hasattr(self, 'search_input'):



        return



    self.search_input.setPlainText(text)



    self.search_input.setFocus()



    if hasattr(self, 'status_label') and self.status_label:



        self.status_label.setText("Query loaded - press Ctrl+Enter to Ask Notes.")



        QTimer.singleShot(3000, self._clear_query_loaded_status)



def _clear_query_loaded_status(self):



    """Clear the 'Query loaded' status after a delay if it wasn't replaced by search results."""



    if hasattr(self, 'status_label') and self.status_label:



        if self.status_label.text().startswith("Query loaded"):



            self.status_label.setText("Ready")



def _on_clear_search_history(self):

    """Clear all search history with a confirmation dialog."""

    from aqt.utils import askUser

    if not askUser("Are you sure you want to permanently delete your entire search history?"):

        return



    if clear_search_history():

        self._refresh_search_history()



def _refresh_search_history(self):



    """Reload the previous-searches list from search_history.json."""



    try:



        history = get_search_history_queries()



        if hasattr(self, '_search_history_model'):



            self._search_history_model.setStringList(history)



        if hasattr(self, 'search_history_btn') and self.search_history_btn:



            self._rebuild_search_history_menu(history)



        if hasattr(self, 'sidebar_history_list') and self.sidebar_history_list:



            self.sidebar_history_list.blockSignals(True)



            self.sidebar_history_list.clear()



            self.sidebar_history_list.addItems(history)



            self.sidebar_history_list.blockSignals(False)



    except Exception:



        pass



def on_item_changed(self, item):



    """Handle item changes - only update count if checkbox column changed"""



    if item.column() == 0:  # Selection checkbox column



        self.update_selection_count()



def _on_relevance_mode_changed(self, _btn, checked):



    """Legacy no-op; relevance mode was replaced by the single threshold slider."""
    return



    # #region agent log



    _session_debug_log(



        "H1",



        "__init__._on_relevance_mode_changed.entry",



        "mode change handler",



        data={"checked": checked, "relevance_mode_before": getattr(self, "relevance_mode", None), "_effective_relevance_mode": getattr(self, "_effective_relevance_mode", None)},



    )



    # #endregion



    if not checked:



        return



    try:



        btn = self.relevance_mode_group.checkedButton()



        mode_key = (btn.property("mode_key") or "balanced").lower()



    except Exception:



        mode_key = "balanced"



    if mode_key not in ("focused", "balanced", "broad"):



        mode_key = "balanced"



    self.relevance_mode = mode_key



    self._effective_relevance_mode = mode_key  # so status bar and "Results from" label show the selected mode

    # A mode switch should apply that mode's own display cutoff. Without
    # recomputing, Focused can inherit a broader mode's row budget.
    self._show_all_dynamic_results = False



    # #region agent log



    _session_debug_log(



        "H1",



        "__init__._on_relevance_mode_changed.after_assign",



        "relevance_mode set",



        data={"mode_key": mode_key, "relevance_mode": self.relevance_mode, "has_all_scored_notes": hasattr(self, "all_scored_notes")},



    )



    # #endregion



    # Persist last-used mode.



    try:



        config = load_config()



        sc = dict(config.get("search_config") or {})



        sc["relevance_mode"] = mode_key



        config["search_config"] = sc



        save_config(config)



    except Exception:



        pass



    # Update "Results from: ... \u00b7 Mode \u00b7 Embeddings: ..." so it shows the new mode



    if hasattr(self, "search_method_result_label") and getattr(self, "_last_search_method", None):



        try:



            sc = load_config().get("search_config") or {}



            mode_display = {"focused": "Focused", "balanced": "Balanced", "broad": "Broad"}.get(mode_key, "Balanced")



            engine = (sc.get("embedding_engine") or "ollama").strip().lower()



            engine_display = {"ollama": "Ollama (local)", "voyage": "Voyage AI", "openai": "OpenAI", "cohere": "Cohere"}.get(engine, engine or "unknown")



            self.search_method_result_label.setText(f"Results from: {self._last_search_method} \u00b7 {mode_display} \u00b7 Embeddings: {engine_display}")



            self.search_method_result_label.setVisible(True)



        except Exception:



            pass



    # Update any existing results with the new mode (will use effective flags)



    if hasattr(self, "all_scored_notes"):
        try:
            config = load_config()
            sc = dict(config.get("search_config") or {})
            sc["relevance_mode"] = mode_key
            scored_notes = list(getattr(self, "all_scored_notes", None) or [])
            if scored_notes and hasattr(self, "_dynamic_note_budget"):
                self._last_dynamic_note_budget = self._dynamic_note_budget(
                    getattr(self, "current_query", "") or "",
                    scored_notes,
                    sc,
                    config.get("provider", "openai"),
                    phase="final",
                    pinned_count=len(
                        (getattr(self, "selected_note_ids", set()) or set())
                        | (getattr(self, "_pinned_note_ids", set()) or set())
                    ),
                    rerank_used=bool(getattr(self, "_last_rerank_success", False)),
                )
        except Exception as exc:
            try:
                log_debug(f"Failed to refresh dynamic note budget after relevance mode change: {exc}")
            except Exception:
                pass




        self.filter_and_display_notes()



def on_sensitivity_changed(self, value):



    if getattr(self, 'sensitivity_value_label', None) is not None:



        self.sensitivity_value_label.setText(f"{value}%")

    self._effective_relevance_threshold_percent = clamp_relevance_threshold_percent(value)
    self._relevance_threshold_source = "user_changed"

    timer = getattr(self, "_relevance_threshold_debounce_timer", None)
    if timer is None:
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: _apply_relevance_threshold_change(self, persist=False))
        self._relevance_threshold_debounce_timer = timer
    timer.start(200)


def _apply_relevance_threshold_change(self, persist=False):
    slider = getattr(self, "sensitivity_slider", None)
    if slider is None:
        return
    value = clamp_relevance_threshold_percent(slider.value())
    self._effective_relevance_threshold_percent = value

    self._relevance_threshold_source = "user_changed"

    if persist:
        try:
            config = load_config()
            sc = dict(config.get("search_config") or {})
            sc["relevance_threshold_percent"] = value
            config["search_config"] = sc
            save_config(config)
            self._relevance_threshold_source = "config"
        except Exception as exc:
            try:
                log_debug(f"Failed to persist relevance threshold: {exc}")
            except Exception:
                pass

    if hasattr(self, 'all_scored_notes'):
        self._show_all_dynamic_results = False
        self.filter_and_display_notes()


def on_relevance_threshold_released(self):
    timer = getattr(self, "_relevance_threshold_debounce_timer", None)
    if timer is not None:
        timer.stop()
    _apply_relevance_threshold_change(self, persist=True)



def _on_show_only_cited_changed(self, _state):



    """Persist 'Show only cited notes' and refresh the table."""



    try:



        config = load_config()



        sc = dict(config.get('search_config') or {})



        sc['show_only_cited'] = getattr(self, 'show_only_cited_cb', None) and self.show_only_cited_cb.isChecked()



        config['search_config'] = sc



        save_config(config)



    except Exception:



        pass



    if hasattr(self, 'all_scored_notes'):



        self.filter_and_display_notes()



def open_settings(self):



    """Open the settings dialog in a non-modal window so Anki stays usable."""



    dialog = SettingsDialog(self)



    dialog.setWindowModality(Qt.WindowModality.NonModal)



    dialog.show()



    dialog.raise_()



    dialog.activateWindow()



def get_config(self):



    config = load_config()



    if not config or 'api_key' not in config:



        return None



    return config


def set_search_mode(self, search_mode=None):
    self._search_mode = (search_mode or "ask_ai").strip() or "ask_ai"


def set_review_context(self, review_card=None, review_note_id=None, review_context=None):
    self._review_card = review_card
    self._review_note_id = review_note_id
    self._review_context = review_context or {}
    self._review_context_text = (self._review_context.get("text") or "").strip()
    if self._review_note_id is None and review_card is not None:
        try:
            self._review_note_id = getattr(review_card.note(), "id", None)
        except Exception:
            pass
    if self._review_note_id is not None:
        try:
            self._pinned_note_ids = set(getattr(self, "_pinned_note_ids", set()) or set())
            self._pinned_note_ids.add(int(self._review_note_id))
        except Exception:
            pass
    self._refresh_review_context_ui()


def _refresh_review_context_ui(self):
    has_review = bool(getattr(self, "_review_note_id", None) or getattr(self, "_review_context_text", ""))
    banner = getattr(self, "review_context_banner", None)
    if banner is not None:
        if has_review:
            note_id = getattr(self, "_review_note_id", None)
            suffix = f"Note {note_id}" if note_id else "Current note"
            banner.setText(f"Review context active: {suffix}")
            banner.setVisible(True)
        else:
            banner.setText("")
            banner.setVisible(False)
    related_btn = getattr(self, "find_related_btn", None)
    if related_btn is not None:
        has_image = bool(getattr(self, "_composer_image_payloads", []) or [])
        related_btn.setVisible(has_review or has_image)
        related_btn.setEnabled(has_review or has_image)
        if has_image and not has_review:
            related_btn.setToolTip("Search for notes related to the attached image.")
        else:
            related_btn.setToolTip("Search for notes related to the current review note.")


def _review_context_note_for_answer(self):
    text = (getattr(self, "_review_context_text", "") or "").strip()
    note_id = getattr(self, "_review_note_id", None)
    if not text or note_id is None:
        return None
    try:
        note_id = int(note_id)
    except Exception:
        pass
    return {
        "id": note_id,
        "chunk_index": None,
        "content_hash": "review-context",
        "content": text,
        "display_content": text,
        "_full_content": text,
        "_full_display_content": text,
        "why_ref": "Current review note",
        "matching_terms": [],
        "_passes_broad": True,
        "_display_relevance": 100,
        "_review_context_note": True,
    }


def _review_related_query(self):
    text = (getattr(self, "_review_context_text", "") or "").strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)[:1200].strip()


def find_related_notes_from_review(self):
    query = self._review_related_query()
    has_image = bool(getattr(self, "_composer_image_payloads", []) or [])
    if not query and not has_image:
        tooltip("Attach an image or open from a review note first.")
        return
    if hasattr(self, "_expand_sources_panel"):
        self._expand_sources_panel(manual=True)
    self.set_search_mode("related_notes")
    self._pending_related_notes_query = query
    self.perform_search()


class SearchDialogStateMixin:
    """Owns search history, scope banner, display options, and config access."""

    _update_view_all_button_state = _update_view_all_button_state
    showEvent = showEvent
    _refresh_scope_banner = _refresh_scope_banner
    _on_search_history_selected = _on_search_history_selected
    _on_sidebar_history_selected = _on_sidebar_history_selected
    _on_search_history_query_selected = _on_search_history_query_selected
    _on_delete_search_history_query = _on_delete_search_history_query
    _clear_search_removed_status = _clear_search_removed_status
    _rebuild_search_history_menu = _rebuild_search_history_menu
    _set_query_text = _set_query_text
    _clear_query_loaded_status = _clear_query_loaded_status
    _on_clear_search_history = _on_clear_search_history
    _refresh_search_history = _refresh_search_history
    on_item_changed = on_item_changed
    _on_relevance_mode_changed = _on_relevance_mode_changed
    on_sensitivity_changed = on_sensitivity_changed
    on_relevance_threshold_released = on_relevance_threshold_released
    _on_show_only_cited_changed = _on_show_only_cited_changed
    open_settings = open_settings
    get_config = get_config
    set_search_mode = set_search_mode
    set_review_context = set_review_context
    _refresh_review_context_ui = _refresh_review_context_ui
    _review_context_note_for_answer = _review_context_note_for_answer
    _review_related_query = _review_related_query
    find_related_notes_from_review = find_related_notes_from_review
