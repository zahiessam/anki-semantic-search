"""Progress/status helpers for the AI search dialog."""

from aqt.qt import QTimer


class SearchProgressMixin:
    """Owns search progress bar, progress label, and estimated timer helpers."""

    def _on_embedding_search_progress(self, current, total, message):
        """Update status and progress bar while embedding search runs in background."""
        try:
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText(message)

            if hasattr(self, 'search_progress_bar') and self.search_progress_bar and total > 0:
                self.search_progress_bar.setRange(0, total)
                self.search_progress_bar.setValue(current)
                self.search_progress_bar.setVisible(True)

            if hasattr(self, 'search_progress_label') and self.search_progress_label:
                self.search_progress_label.setText(f"{current}/{total}")
                self.search_progress_label.setVisible(True)

        except Exception:
            pass

    def _show_busy_progress(self, message=""):
        """Show indeterminate progress bar and optional label during long operations (re-rank, AI call, load)."""
        self._show_centile_progress(message, 0)

    def _show_centile_progress(self, message="", percent=0):
        """Show 0-100% progress bar and label. Use for estimated or real progress during long operations."""
        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:
            self.search_progress_bar.setRange(0, 100)
            self.search_progress_bar.setValue(max(0, min(100, round(percent))))
            self.search_progress_bar.setVisible(True)

        if hasattr(self, 'search_progress_label') and self.search_progress_label:
            self.search_progress_label.setText(message)
            self.search_progress_label.setVisible(True)

        self._last_progress_message = message

    def _start_estimated_progress_timer(self, duration_sec, start_pct=5, end_pct=95):
        """Advance progress bar from start_pct to end_pct over duration_sec (est. wait). Call _stop_estimated_progress_timer when done."""
        import time

        self._stop_estimated_progress_timer()

        self._progress_estimate_active = True
        self._progress_estimate_start = time.time()
        self._progress_estimate_duration = max(1, duration_sec)
        self._progress_estimate_start_pct = start_pct
        self._progress_estimate_end_pct = end_pct

        def _tick():
            if not getattr(self, '_progress_estimate_active', False):
                return

            elapsed = time.time() - getattr(self, '_progress_estimate_start', 0)
            dur = getattr(self, '_progress_estimate_duration', 30)
            s = getattr(self, '_progress_estimate_start_pct', 5)
            e = getattr(self, '_progress_estimate_end_pct', 95)
            pct = s + (elapsed / dur) * (e - s)
            pct = max(s, min(e, pct))
            msg = getattr(self, '_last_progress_message', '')
            self._show_centile_progress(msg, pct)

            if elapsed < dur:
                QTimer.singleShot(500, _tick)

        QTimer.singleShot(300, _tick)

    def _stop_estimated_progress_timer(self):
        """Stop the estimated progress timer (e.g. when the long operation finishes)."""
        self._progress_estimate_active = False

    def _hide_busy_progress(self):
        """Hide progress bar and label; reset bar to deterministic range for next use."""
        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:
            self.search_progress_bar.setRange(0, 100)
            self.search_progress_bar.setValue(0)
            self.search_progress_bar.setVisible(False)

        if hasattr(self, 'search_progress_label') and self.search_progress_label:
            self.search_progress_label.setText("")
            self.search_progress_label.setVisible(False)
