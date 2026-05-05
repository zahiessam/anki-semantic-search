"""Note preview hover callback for search results."""

from aqt.qt import Qt


class SearchNotePreviewMixin:
    """Owns showing and hiding the result note preview popup."""

    def _show_note_preview_for_cell(self, row, column):
        if column != 2 or not hasattr(self, '_note_preview_popup'):
            if hasattr(self, '_note_preview_popup'):
                self._note_preview_popup.hide()
            return

        item = self.results_list.item(row, column)
        if not item:
            self._note_preview_popup.hide()
            return

        note_info = item.data(Qt.ItemDataRole.UserRole + 3)
        if not isinstance(note_info, dict):
            self._note_preview_popup.hide()
            return

        self._note_preview_popup.show_note(note_info)
