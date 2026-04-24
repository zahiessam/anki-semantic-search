from aqt import mw
from aqt.qt import QEvent, QObject
from aqt.utils import tooltip

from ..utils import log_debug
from .widgets import SemanticSearchSideBar

_sidebar_instance = None
_sidebar_hidden_for_dialog = False


def _with_dialogs(action_name: str, callback):
    try:
        from . import dialogs as dialogs_module
        dialogs_module.toggle_sidebar_visibility = toggle_sidebar_visibility
        dialogs_module.update_sidebar_position = update_sidebar_position
        dialogs_module.show_sidebar = show_sidebar

        callback(dialogs_module)
    except Exception as exc:
        toggle_sidebar_visibility(True)
        log_debug(f"Semantic Search Addon: failed to open {action_name}: {exc}", is_error=True)
        tooltip(f"Semantic Search {action_name} is unavailable. Check the add-on log.")


def _open_search_dialog():
    global _sidebar_hidden_for_dialog
    _sidebar_hidden_for_dialog = True
    toggle_sidebar_visibility(False)
    _with_dialogs("search", lambda dialogs_module: dialogs_module.show_search_dialog())


def _open_embed_dialog():
    global _sidebar_hidden_for_dialog
    _sidebar_hidden_for_dialog = True
    toggle_sidebar_visibility(False)
    _with_dialogs(
        "embedding settings",
        lambda dialogs_module: dialogs_module.show_settings_dialog(
            open_to_embeddings=True,
            auto_start_indexing=True,
        ),
    )


def _open_settings_dialog():
    global _sidebar_hidden_for_dialog
    _sidebar_hidden_for_dialog = True
    toggle_sidebar_visibility(False)
    _with_dialogs("settings", lambda dialogs_module: dialogs_module.show_settings_dialog(False))


def show_sidebar():
    global _sidebar_instance
    if not mw:
        return
    if not _sidebar_instance:
        _sidebar_instance = SemanticSearchSideBar(
            mw,
            callbacks={
                "search": _open_search_dialog,
                "embed": _open_embed_dialog,
                "settings": _open_settings_dialog,
            },
        )
    update_sidebar_position()


def update_sidebar_position():
    if not _sidebar_instance or not mw or not mw.isVisible():
        return

    if _sidebar_hidden_for_dialog:
        _sidebar_instance.hide()
        return

    geom = mw.geometry()
    padding = 20
    x = geom.x() + padding
    y = geom.y() + geom.height() - _sidebar_instance.height() - padding - 40
    _sidebar_instance.move(x, y)
    _sidebar_instance.show()
    _sidebar_instance.raise_()


def toggle_sidebar_visibility(visible: bool):
    global _sidebar_hidden_for_dialog
    if visible:
        _sidebar_hidden_for_dialog = False
        update_sidebar_position()
        return

    _sidebar_hidden_for_dialog = True
    if _sidebar_instance:
        _sidebar_instance.hide()


class MainWindowMoveFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show):
            update_sidebar_position()
        return False
