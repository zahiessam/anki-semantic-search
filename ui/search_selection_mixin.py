"""Selection state helpers for search result rows."""

from aqt.qt import Qt
from aqt.utils import tooltip


class SearchSelectionMixin:
    """Owns result row selection, selection persistence, and selection UI state."""

    def update_selection_count(self):
        """Update the selected count display and toggle button text."""
        if not hasattr(self, 'results_list'):
            return

        checked_count = 0
        total_count = self.results_list.rowCount()

        if not hasattr(self, 'selected_note_ids'):
            self.selected_note_ids = set()

        for row in range(total_count):
            item = self.results_list.item(row, 0)

            if item:
                note_id = item.data(Qt.ItemDataRole.UserRole)

                if item.checkState() == Qt.CheckState.Checked:
                    checked_count += 1

                    if note_id:
                        self.selected_note_ids.add(note_id)
                else:
                    if note_id:
                        self.selected_note_ids.discard(note_id)

        if hasattr(self, 'selected_count_label'):
            if total_count > 0:
                self.selected_count_label.setText(f"({checked_count} of {total_count} selected)")
            else:
                self.selected_count_label.setText("(0 selected)")

        if hasattr(self, 'toggle_select_btn'):
            if checked_count == total_count and total_count > 0:
                self.toggle_select_btn.setText("Deselect All")
            else:
                self.toggle_select_btn.setText("Select All")

        if hasattr(self, 'view_btn'):
            self.view_btn.setEnabled(checked_count > 0)

    def toggle_select_all(self):
        """Toggle between selecting all and deselecting all notes."""
        if not hasattr(self, 'results_list') or self.results_list.rowCount() == 0:
            return

        all_selected = True

        for row in range(self.results_list.rowCount()):
            item = self.results_list.item(row, 0)

            if item and item.checkState() != Qt.CheckState.Checked:
                all_selected = False
                break

        if all_selected:
            self.deselect_all_notes()
        else:
            self.select_all_notes()

    def select_all_notes(self):
        """Select all notes in the results list."""
        if not hasattr(self, 'results_list'):
            return

        self.results_list.blockSignals(True)

        for row in range(self.results_list.rowCount()):
            item = self.results_list.item(row, 0)

            if item:
                item.setCheckState(Qt.CheckState.Checked)
                note_id = item.data(Qt.ItemDataRole.UserRole)

                if note_id:
                    self.selected_note_ids.add(note_id)

        self.results_list.blockSignals(False)

        self.update_selection_count()

        tooltip(f"\u2713 Selected all {self.results_list.rowCount()} notes")

    def deselect_all_notes(self):
        """Deselect all notes in the results list."""
        if not hasattr(self, 'results_list'):
            return

        self.results_list.blockSignals(True)

        for row in range(self.results_list.rowCount()):
            item = self.results_list.item(row, 0)

            if item:
                item.setCheckState(Qt.CheckState.Unchecked)
                note_id = item.data(Qt.ItemDataRole.UserRole)

                if note_id:
                    self.selected_note_ids.discard(note_id)

        self.results_list.blockSignals(False)

        self.update_selection_count()

        tooltip(f"\xe2\u0153\u2014 Deselected all notes")

    def restore_selections(self):
        """Restore selections from stored note IDs."""
        if not hasattr(self, 'selected_note_ids') or not self.selected_note_ids:
            return

        self.results_list.blockSignals(True)

        for row in range(self.results_list.rowCount()):
            item = self.results_list.item(row, 0)

            if item:
                note_id = item.data(Qt.ItemDataRole.UserRole)

                if note_id in self.selected_note_ids:
                    item.setCheckState(Qt.CheckState.Checked)

        self.results_list.blockSignals(False)
