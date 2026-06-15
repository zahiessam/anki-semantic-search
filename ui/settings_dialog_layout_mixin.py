"""Top-level Settings dialog UI construction."""

# ============================================================================
# Imports
# ============================================================================

import time

from aqt.qt import *

from .theme import get_addon_theme
from ..utils import load_config, log_debug

_addon_theme = get_addon_theme


class SettingsDialogLayoutMixin:
    """Owns top-level Settings UI assembly and final load/apply sequencing."""

    def setup_ui(self):

        # --- RADICAL STABILIZATION: Guard against re-initialization ---

        if hasattr(self, "_ui_initialized") and self._ui_initialized:

            return

        self._ui_initialized = True



        current_config = load_config()

        self.current_config = current_config # Store for access in other methods


        start_time = time.time()



        log_debug("=== Settings Dialog UI Setup Started ===")



        palette = QApplication.palette()



        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128



        theme = _addon_theme(is_dark)







        # Main layout with proper spacing - fix layout issues



        main_layout = QVBoxLayout(self)



        main_layout.setSpacing(10)



        main_layout.setContentsMargins(15, 15, 15, 15)







        elapsed = time.time() - start_time



        log_debug(f"  [Timing] Layout setup: {elapsed:.3f}s")







        tabs = QTabWidget()

        # Size to content; scroll_content minimum and scroll area give enough height for scrolling when needed
        tabs.setMinimumHeight(0)

        api_tab = self._build_api_settings_tab(theme, current_config)
        tabs.addTab(api_tab, "\U0001F511 API settings")

        style_tab = self._build_styling_tab(theme)
        tabs.addTab(style_tab, "\U0001F3A8 Styling")

        try:
            scope_tab = self._build_scope_selector_tab(theme)
            tabs.addTab(scope_tab, "\U0001F4CB AI search scope")
            note_filter_tab = scope_tab
        except Exception as e:
            log_debug(f"Error building scope selector tab: {e}")
            # Fallback to old note types tab if scope selector fails
            nt_tab = self._build_note_types_tab(theme)
            tabs.addTab(nt_tab, "\U0001F4CB Note Types & Fields")
            note_filter_tab = nt_tab

        search_tab = self._build_search_embeddings_tab(theme)
        tabs.addTab(search_tab, "\U0001F50D Search & embeddings")

        self._finalize_settings_layout(main_layout, tabs, search_tab, note_filter_tab, start_time, current_config)
