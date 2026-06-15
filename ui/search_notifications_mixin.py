"""Non-modal completion notifications for long search jobs."""

from aqt.qt import QApplication
from aqt.utils import tooltip

from ..utils import log_debug


def _play_system_completion_sound(kind="info"):
    """Use the OS notification sound, with Qt's beep as a quiet fallback."""
    try:
        import winsound

        sound = (
            winsound.MB_ICONHAND
            if kind == "error"
            else winsound.MB_ICONASTERISK
        )
        winsound.MessageBeep(sound)
        return
    except Exception as exc:
        log_debug(f"Windows completion sound skipped: {exc}")

    try:
        QApplication.beep()
    except Exception as exc:
        log_debug(f"Qt completion beep skipped: {exc}")


class SearchNotificationsMixin:
    def _notify_long_job_done(self, title, detail="", kind="info"):
        """Flash Anki, play a system sound, and show a short tooltip when work finishes."""
        title = (title or "Job complete").strip()
        detail = (detail or "").strip()
        message = f"{title}: {detail}" if detail else title
        _play_system_completion_sound(kind)
        try:
            QApplication.alert(self, 0)
        except Exception as exc:
            log_debug(f"Completion alert skipped: {exc}")
        try:
            tooltip(message, period=5000)
        except Exception as exc:
            log_debug(f"Completion tooltip skipped: {exc}")
