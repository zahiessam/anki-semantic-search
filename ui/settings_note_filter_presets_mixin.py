# ============================================================================
# Imports
# ============================================================================

import glob
import os
import sqlite3
import subprocess
import time

try:
    import sip
except ImportError:
    try:
        from PyQt6 import sip
    except ImportError:
        try:
            from PyQt5 import sip
        except ImportError:
            sip = None

from aqt import dialogs, mw
from aqt.qt import *
from aqt.utils import showInfo, showText, tooltip

from .dependency_install import _resolve_external_python_exe, install_dependencies
from .settings_constants import (
    ANSWER_CLOUD_PROVIDERS,
    ANSWER_KEY_PROVIDER_PREFIXES,
    EMBEDDING_CLOUD_PROVIDERS,
    EMBEDDING_KEY_PROVIDER_PREFIXES,
)
from .settings_rerank_workers import RerankModelDownloadWorker, RerankModelVerifyWorker
from .theme import (
    get_addon_theme,
    settings_button_style,
    settings_panel_style,
    settings_status_label_style,
    settings_text_style,
)
from .widgets import CollapsibleSection, apply_setting_row_tooltip, settings_field_row, sync_setting_row_tooltips
from ..core.cloud_diagnostics import (
    classify_provider_error,
    provider_status_message,
    test_cloud_answer_connection,
)
from ..core.engine import (
    analyze_note_eligibility,
    clear_checkpoint,
    count_notes_matching_config,
    get_deck_names,
    get_embedding_engine_id,
    get_embedding_for_query,
    get_models_with_fields,
    get_notes_count_per_deck,
    get_notes_count_per_model,
    get_ollama_models,
    load_checkpoint,
    load_embedding_engine_counts,
    load_embeddings_bulk,
    make_embedding_scope_id,
    _normalize_ollama_base_url,
)
from ..core.workers import EmbeddingWorker, RerankCheckWorker
from ..utils import (
    EmbeddingsTabMessages,
    format_partial_failure_completion,
    format_partial_failure_progress,
    get_effective_embedding_config,
    get_embeddings_db_path,
    get_embeddings_storage_path_for_read,
    get_retrieval_config,
    load_config,
    log_debug,
    save_config,
    validate_embedding_config,
)
from ..utils.config import (
    DEFAULT_RERANK_MODEL,
    RERANK_TIMEOUT_SECONDS_DEFAULT,
    RERANK_TIMEOUT_SECONDS_MAX,
    RERANK_TIMEOUT_SECONDS_MIN,
    RERANK_TOP_K_DEFAULT,
    get_rerank_config,
)

_addon_theme = get_addon_theme

class SettingsNoteFilterPresetsMixin:
    def _refresh_preset_combos_with_timing(self):



        """Refresh preset combos (with timing)."""



        # Check if widget still exists (dialog might have been closed)



        try:



            if not hasattr(self, 'load_preset_combo') or self.load_preset_combo is None:



                return



            import time



            start = time.time()



            self._refresh_preset_combos()



            elapsed = time.time() - start



            log_debug(f"  [Timing] _refresh_preset_combos(): {elapsed:.3f}s")



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _refresh_preset_combos_with_timing: {e}")


    def _refresh_preset_combos(self):



        # Check if widgets still exist



        if not hasattr(self, 'load_preset_combo') or self.load_preset_combo is None:



            return



        try:



            config = load_config()



            presets = config.get('saved_presets') or {}



            current_name = config.get('current_preset_name')



            names = sorted(presets.keys())



            self.load_preset_combo.clear()



            self.load_preset_combo.addItem("-- Select --", None)



            for n in names:



                self.load_preset_combo.addItem(n, n)



            selected_index = self.load_preset_combo.findData(current_name)



            if selected_index >= 0:



                self.load_preset_combo.setCurrentIndex(selected_index)



            if hasattr(self, 'delete_preset_combo') and self.delete_preset_combo is not None:



                try:



                    self.delete_preset_combo.clear()



                    self.delete_preset_combo.addItem("-- Select --", None)



                    for n in names:



                        self.delete_preset_combo.addItem(n, n)



                    delete_index = self.delete_preset_combo.findData(current_name)



                    if delete_index >= 0:



                        self.delete_preset_combo.setCurrentIndex(delete_index)



                except RuntimeError:



                    # Widget was deleted, ignore



                    pass



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _refresh_preset_combos: {e}")


    def _on_save_preset(self):



        name = self.preset_name_edit.text().strip()



        if not name:



            showInfo("Enter a preset name.")



            return



        config = load_config()



        presets = config.get('saved_presets') or {}



        presets[name] = self._build_scope_config_from_ui()



        config['saved_presets'] = presets



        config['current_preset_name'] = name



        if save_config(config):



            self.preset_name_edit.clear()



            self._refresh_preset_combos()



            selected_index = self.load_preset_combo.findData(name)



            if selected_index >= 0:



                self.load_preset_combo.setCurrentIndex(selected_index)



            showInfo(f"Preset '{name}' saved.")


    def _on_load_preset(self):



        name = self.load_preset_combo.currentData()



        if not name:



            showInfo("Select a preset to load.")



            return



        config = load_config()



        presets = config.get('saved_presets') or {}



        if name not in presets:



            showInfo("Preset not found.")



            return



        self._apply_scope_config(presets[name])



        config['current_preset_name'] = name
        config['note_type_filter'] = dict(presets[name] or {})



        save_config(config)



        selected_index = self.load_preset_combo.findData(name)



        if selected_index >= 0:



            self.load_preset_combo.setCurrentIndex(selected_index)



        if hasattr(self, 'delete_preset_combo') and self.delete_preset_combo is not None:



            delete_index = self.delete_preset_combo.findData(name)



            if delete_index >= 0:



                self.delete_preset_combo.setCurrentIndex(delete_index)



        showInfo(f"Loaded preset '{name}'.")


    def _on_delete_preset(self):



        name = self.delete_preset_combo.currentData()



        if not name:



            showInfo("Select a preset to delete.")



            return



        config = load_config()



        presets = config.get('saved_presets') or {}



        if name in presets:



            del presets[name]



            config['saved_presets'] = presets



            if config.get('current_preset_name') == name:



                config['current_preset_name'] = None



            save_config(config)



            self._refresh_preset_combos()



            showInfo(f"Preset '{name}' deleted.")


