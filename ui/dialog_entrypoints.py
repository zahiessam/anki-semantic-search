"""Public dialog launcher/coordinator functions.

This module owns the dialog-opening logic while ui.dialogs keeps thin wrappers for
backward compatibility and sidebar_bootstrap monkey-patching.
"""

# ============================================================================
# Imports
# ============================================================================

import os
import subprocess
import sys

from aqt import mw
from aqt.qt import QTimer, Qt
from aqt.utils import showInfo


# ============================================================================
# Dialog Singleton State
# ============================================================================

_ai_search_dialog_instance = None


def _dialogs_module(dialogs_module=None):
    if dialogs_module is not None:
        return dialogs_module
    from . import dialogs as imported_dialogs
    return imported_dialogs


def toggle_sidebar_visibility(visible: bool, dialogs_module=None):
    """Safely show or hide the sidebar drawer."""
    dialogs_module = _dialogs_module(dialogs_module)
    sidebar = getattr(dialogs_module, "_sidebar_instance", None)
    updater = getattr(dialogs_module, "update_sidebar_position", None)

    if visible:
        if callable(updater):
            updater()
        return

    if sidebar and hasattr(sidebar, "hide"):
        sidebar.hide()


def show_search_dialog(dialogs_module=None):
    global _ai_search_dialog_instance
    dialogs_module = _dialogs_module(dialogs_module)
    sidebar_toggle = getattr(dialogs_module, "toggle_sidebar_visibility", toggle_sidebar_visibility)

    if hasattr(mw, "_is_spawning_dialog") and mw._is_spawning_dialog:
        return
    mw._is_spawning_dialog = True

    try:
        sidebar_toggle(False)

        if _ai_search_dialog_instance:
            try:
                if _ai_search_dialog_instance.isVisible():
                    _ai_search_dialog_instance.raise_()
                    _ai_search_dialog_instance.activateWindow()
                    return
                _ai_search_dialog_instance.show()
                return
            except (RuntimeError, AttributeError):
                _ai_search_dialog_instance = None

        _ai_search_dialog_instance = dialogs_module.AISearchDialog(mw)
        _ai_search_dialog_instance.finished.connect(lambda: _on_search_dialog_closed(dialogs_module))
        _ai_search_dialog_instance.show()
    finally:
        QTimer.singleShot(400, lambda: setattr(mw, "_is_spawning_dialog", False))


def _on_search_dialog_closed(dialogs_module=None):
    global _ai_search_dialog_instance
    dialogs_module = _dialogs_module(dialogs_module)
    sidebar_toggle = getattr(dialogs_module, "toggle_sidebar_visibility", toggle_sidebar_visibility)
    _ai_search_dialog_instance = None
    QTimer.singleShot(150, lambda: sidebar_toggle(True))


def show_settings_dialog(open_to_embeddings=False, auto_start_indexing=False, dialogs_module=None):
    dialogs_module = _dialogs_module(dialogs_module)
    sidebar_toggle = getattr(dialogs_module, "toggle_sidebar_visibility", toggle_sidebar_visibility)
    sidebar_toggle(False)

    dialog = dialogs_module.SettingsDialog(mw, open_to_embeddings=open_to_embeddings)
    dialog.setWindowModality(Qt.WindowModality.NonModal)
    dialog.finished.connect(lambda _: sidebar_toggle(True))
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()

    if auto_start_indexing:
        QTimer.singleShot(500, lambda: dialog._create_or_update_embeddings())


def show_debug_log():
    try:
        addon_dir = os.path.dirname(__file__)
        log_file = os.path.join(addon_dir, "debug_log.txt")

        if os.path.exists(log_file):
            if os.name == 'nt':
                os.startfile(log_file)
            else:
                subprocess.call(['open' if sys.platform == 'darwin' else 'xdg-open', log_file])
        else:
            showInfo("Debug log file not found. Try using the add-on first.")
    except Exception as e:
        showInfo(f"Error opening log file: {e}")
