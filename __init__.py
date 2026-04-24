# Anki Semantic Search Addon Entry Point

from aqt import gui_hooks, mw
from aqt.qt import QTimer

from .utils import log_debug

try:
    from .ui.sidebar_bootstrap import (
        MainWindowMoveFilter,
        show_sidebar,
        update_sidebar_position,
    )
except Exception as exc:
    log_debug(f"Semantic Search Addon: failed to import sidebar bootstrap: {exc}")
    show_sidebar = None
    update_sidebar_position = None
    MainWindowMoveFilter = None


def initialize_addons() -> None:
    """Load addon UI bootstrap and register main-window hooks."""
    log_debug("Semantic Search Addon: Initializing...")


def _on_main_window_did_init() -> None:
    """Show the floating sidebar after Anki finishes creating the main window."""
    if not show_sidebar or not update_sidebar_position:
        return

    QTimer.singleShot(1000, show_sidebar)

    try:
        gui_hooks.state_did_change.append(
            lambda _new_state, _old_state: update_sidebar_position()
        )
    except Exception as exc:
        log_debug(f"Semantic Search Addon: could not hook state updates: {exc}")

    if (
        MainWindowMoveFilter
        and hasattr(mw, "installEventFilter")
        and not hasattr(mw, "_semantic_search_move_filter")
    ):
        mw._semantic_search_move_filter = MainWindowMoveFilter(mw)
        mw.installEventFilter(mw._semantic_search_move_filter)


initialize_addons()

if show_sidebar and update_sidebar_position:
    try:
        gui_hooks.main_window_did_init.append(_on_main_window_did_init)
    except Exception as exc:
        log_debug(f"Semantic Search Addon: could not register main window hook: {exc}")
