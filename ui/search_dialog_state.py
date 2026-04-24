"""Search dialog state, history, and option handlers."""

# ============================================================================
# Imports
# ============================================================================

from aqt import mw
from aqt.qt import QDialog, QTimer, Qt
from aqt.utils import askUser

from .settings_dialog import SettingsDialog
from ..core.engine import get_deck_names
from ..utils import (
    clear_search_history,
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



        txt = f"Searching: {n_types} note types, {n_fields} fields, {n_decks} decks \u2014 <a href='#settings' style='color:#3498db;'>Settings</a>"



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



def _set_query_text(self, text):



    """Set query text in the editor and show a short status hint."""



    if not text or not hasattr(self, 'search_input'):



        return



    self.search_input.setPlainText(text)



    self.search_input.setFocus()



    if hasattr(self, 'status_label') and self.status_label:



        self.status_label.setText("Query loaded - press Ctrl+Enter to search.")



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



        if hasattr(self, 'sidebar_history_list') and self.sidebar_history_list:



            self.sidebar_history_list.blockSignals(True)



            self.sidebar_history_list.clear()



            self.sidebar_history_list.addItems(history)



            self.sidebar_history_list.blockSignals(False)



    except Exception:



        pass



def on_item_changed(self, item):



    """Handle item changes - only update count if checkbox column changed"""



    if item.column() == 1:  # Only process changes in content column (checkbox)



        self.update_selection_count()



def _on_relevance_mode_changed(self, _btn, checked):



    """Persist relevance_mode (Focused/Balanced/Broad) and refresh current view."""



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



    # #region agent log



    _session_debug_log(



        "H1",



        "__init__._on_relevance_mode_changed.after_assign",



        "relevance_mode set",



        data={"mode_key": mode_key, "relevance_mode": self.relevance_mode, "has_all_scored_notes": hasattr(self, "all_scored_notes")},



    )



    # #endregion



    # Persist last-used mode and keep strict_relevance in sync for compatibility



    try:



        config = load_config()



        sc = dict(config.get("search_config") or {})



        sc["relevance_mode"] = mode_key



        sc["strict_relevance"] = True if mode_key == "focused" else False



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



        self.filter_and_display_notes()



def on_sensitivity_changed(self, value):



    if getattr(self, 'sensitivity_value_label', None) is not None:



        self.sensitivity_value_label.setText(f"{value}%")



    # Persist so next time the add-on opens with the same choice (keep full search_config)



    try:



        config = load_config()



        sc = dict(config.get('search_config') or {})  # copy so we don't lose other keys



        sc['sensitivity_percent'] = value



        config['search_config'] = sc



        save_config(config)



    except Exception:



        pass



    if hasattr(self, 'all_scored_notes'):



        self.filter_and_display_notes()



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
