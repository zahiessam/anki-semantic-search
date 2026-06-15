"""Note preview hover callback for search results."""

from aqt.qt import QCursor, QRect, QTimer, Qt


NOTE_PREVIEW_SHOW_DELAY_MS = 650


class SearchNotePreviewMixin:
    """Owns showing and hiding the result note preview popup."""

    def _show_note_preview_for_cell(self, row, column):
        self._cancel_pending_note_preview()
        if column != 2 or not hasattr(self, '_note_preview_popup'):
            if hasattr(self, '_note_preview_popup'):
                self._note_preview_popup.schedule_hide_if_cursor_outside()
            return
        if getattr(self._note_preview_popup, "is_pinned", lambda: False)():
            return

        item = self.results_list.item(row, column)
        if not item:
            self._note_preview_popup.schedule_hide_if_cursor_outside()
            return

        note_info = item.data(Qt.ItemDataRole.UserRole + 3)
        if not isinstance(note_info, dict):
            self._note_preview_popup.schedule_hide_if_cursor_outside()
            return

        self._pending_note_preview_info = note_info
        self._pending_note_preview_cell = (row, column)

        timer = getattr(self, "_note_preview_show_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._show_pending_note_preview)
            self._note_preview_show_timer = timer
        timer.start(NOTE_PREVIEW_SHOW_DELAY_MS)

    def _cancel_pending_note_preview(self):
        timer = getattr(self, "_note_preview_show_timer", None)
        if timer is not None:
            timer.stop()
        self._pending_note_preview_info = None
        self._pending_note_preview_cell = None

    def _show_pending_note_preview(self):
        note_info = getattr(self, "_pending_note_preview_info", None)
        cell = getattr(self, "_pending_note_preview_cell", None)
        self._pending_note_preview_info = None
        self._pending_note_preview_cell = None
        if not isinstance(note_info, dict) or not cell:
            return
        if not hasattr(self, "_note_preview_popup"):
            return
        try:
            row, column = cell
            current = self.results_list.indexAt(self.results_list.viewport().mapFromGlobal(QCursor.pos()))
            if not current.isValid() or current.row() != row or current.column() != column:
                return
            item = self.results_list.item(row, column)
            index = self.results_list.indexFromItem(item)
            local_rect = self.results_list.visualRect(index)
            top_left = self.results_list.viewport().mapToGlobal(local_rect.topLeft())
            bottom_right = self.results_list.viewport().mapToGlobal(local_rect.bottomRight())
            anchor_rect = QRect(top_left, bottom_right)
            preview_container = getattr(self, "results_container", None) or self
            container_rect = QRect(
                preview_container.mapToGlobal(preview_container.rect().topLeft()),
                preview_container.mapToGlobal(preview_container.rect().bottomRight()),
            )
            self._note_preview_popup.watch_hover_object(self.results_list.viewport())
        except Exception:
            anchor_rect = None
            container_rect = None
        self._note_preview_popup.show_note(note_info, anchor_rect=anchor_rect, container_rect=container_rect)

    def _reset_note_preview_popup(self):
        self._cancel_pending_note_preview()
        popup = getattr(self, "_note_preview_popup", None)
        if popup is not None:
            if hasattr(popup, "reset_pinned"):
                popup.reset_pinned()
            else:
                popup.hide()
