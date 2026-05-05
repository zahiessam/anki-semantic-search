"""Browser-opening actions for search result rows."""

from aqt import dialogs, mw
from aqt.qt import QTimer, Qt
from aqt.utils import tooltip


class SearchBrowserActionsMixin:
    """Owns opening selected, all, or double-clicked search results in Anki Browser."""

    def open_selected_in_browser(self):
        checked_ids = []

        for row in range(self.results_list.rowCount()):
            item = self.results_list.item(row, 0)

            if item and item.checkState() == Qt.CheckState.Checked:
                note_id = item.data(Qt.ItemDataRole.UserRole)
                checked_ids.append(str(note_id))

        if not checked_ids:
            tooltip("Please check at least one note to view")
            return

        browser = dialogs.open("Browser", mw)
        search_query = "nid:" + ",".join(checked_ids)
        browser.form.searchEdit.lineEdit().setText(search_query)
        browser.onSearchActivated()
        QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))
        tooltip(f"\u2713 Opened {len(checked_ids)} selected notes in browser")

    def open_all_in_browser(self):
        if self.results_list.rowCount() == 0:
            tooltip("No notes to view")
            return

        note_ids = []

        for row in range(self.results_list.rowCount()):
            item = self.results_list.item(row, 0)

            if item:
                note_id = item.data(Qt.ItemDataRole.UserRole)
                note_ids.append(str(note_id))

        browser = dialogs.open("Browser", mw)
        search_query = "nid:" + ",".join(note_ids)
        browser.form.searchEdit.lineEdit().setText(search_query)
        browser.onSearchActivated()
        QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))
        tooltip(f"\u2713 Opened {len(note_ids)} notes in browser")

    def open_in_browser(self, item):
        """Open note in browser when double-clicked."""
        row = item.row()
        content_item = self.results_list.item(row, 0)

        if content_item:
            note_id = content_item.data(Qt.ItemDataRole.UserRole)
            browser = dialogs.open("Browser", mw)
            browser.form.searchEdit.lineEdit().setText(f"nid:{note_id}")
            browser.onSearchActivated()
            QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))
            tooltip("\u2713 Note opened in browser")
