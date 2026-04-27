"""Citation navigation, browser opening, and answer clipboard helpers."""

# ============================================================================
# Imports
# ============================================================================

import time

from aqt import dialogs, mw
from aqt.qt import QApplication, QItemSelectionModel, QMimeData, QTimer, Qt
from aqt.utils import tooltip

from ..utils import log_debug


# ============================================================================
# Answer HTML, Citation Navigation, And Clipboard Helpers
# ============================================================================

def _restore_answer_html(self, html):



    """Restore the answer box HTML (used after link click so the AI answer does not disappear)."""



    if html and hasattr(self, 'answer_box'):



        self.answer_box.setHtml(html)


def _selected_result_note_ids(self):
    """Return note ids for currently highlighted rows in the results table."""

    if not hasattr(self, 'results_list') or not self.results_list.selectionModel():
        return set()

    note_ids = set()
    for index in self.results_list.selectionModel().selectedRows():
        content_item = self.results_list.item(index.row(), 2)
        if content_item:
            note_id = content_item.data(Qt.ItemDataRole.UserRole)
            if note_id:
                note_ids.add(note_id)
    return note_ids


def _highlight_result_notes(self, note_ids, scroll_to_note_id=None, scroll_to_ref=None):
    """Highlight only the requested citation result rows."""

    if not hasattr(self, 'results_list') or not self.results_list.selectionModel():
        return

    selection_model = self.results_list.selectionModel()
    selection_model.clearSelection()
    target_item = None
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows

    for row in range(self.results_list.rowCount()):
        ref_item = self.results_list.item(row, 1)
        content_item = self.results_list.item(row, 2)
        if not content_item:
            continue

        note_id = content_item.data(Qt.ItemDataRole.UserRole)
        is_target_ref = scroll_to_ref is not None and ref_item and str(ref_item.text()) == str(scroll_to_ref)
        is_target_note = scroll_to_note_id is not None and note_id == scroll_to_note_id

        if note_id in note_ids or is_target_ref or is_target_note:
            selection_model.select(self.results_list.model().index(row, 0), flags)
            if is_target_ref or is_target_note:
                target_item = content_item

    if target_item:
        self.results_list.scrollToItem(target_item)



def _on_answer_link_clicked(self, url):



    """Citation links: single-click highlights note in Matching notes; double-click opens in Anki Browser (over add-on). Supports #cite-N, anki:goto_note:{note_id}, and legacy note:N."""



    import time



    saved_html = getattr(self, '_last_formatted_answer', None) or (self.answer_box.toHtml() if hasattr(self.answer_box, 'toHtml') else None)



    s = url.toString() if hasattr(url, 'toString') else str(url)



    if not s:



        if saved_html:



            self.answer_box.setHtml(saved_html)



        return



    ctx = getattr(self, '_context_note_ids', None) or []



    note_id = None



    num = None



    if s.startswith('#cite-'):



        try:



            num = int(s.replace('#cite-', '').strip())



            if 1 <= num <= len(ctx):



                note_id = ctx[num - 1]



        except (ValueError, TypeError):



            pass



    elif s.startswith('anki:goto_note:'):



        try:



            note_id = int(s.split(':', 2)[2].strip())



            if ctx and note_id in ctx:



                num = ctx.index(note_id) + 1



            else:



                num = note_id



        except (ValueError, IndexError):



            pass



    elif s.startswith('note:'):



        try:



            num = int(s.split(':', 1)[1].strip())



            if 1 <= num <= len(ctx):



                note_id = ctx[num - 1]



        except (ValueError, IndexError):



            pass



    if note_id is None:



        if saved_html:



            self.answer_box.setHtml(saved_html)



        return



    if num is None:



        num = (ctx.index(note_id) + 1) if note_id in ctx else note_id







    # Single vs double click: second click on same link within 400ms = open browser; else only highlight



    now = time.time()



    last = getattr(self, '_citation_last_click', None)



    is_double = last is not None and last[0] == s and (now - last[1]) < 0.4



    self._citation_last_click = None if is_double else (s, now)







    if is_double:



        self._open_note_in_browser(note_id, num)







    # Always highlight the corresponding row in the results list



    if hasattr(self, 'all_scored_notes') and self.all_scored_notes:



        self._pinned_note_ids = {note_id}



        max_score = self.all_scored_notes[0][0]



        thresh = self.sensitivity_slider.value() if self.sensitivity_slider else 0



        min_score = (thresh / 100.0) * max_score if max_score > 0 else 0



        id_to_score = {n['id']: s for s, n in self.all_scored_notes}



        pinned_orig_scores = [id_to_score.get(nid, 0) for nid in self._pinned_note_ids]



        any_filtered = any(orig < min_score for orig in pinned_orig_scores)



        if any_filtered and self.sensitivity_slider is not None:



            self.sensitivity_slider.blockSignals(True)



            self.sensitivity_slider.setValue(0)



            if self.sensitivity_value_label is not None:



                self.sensitivity_value_label.setText("0%")



            self.sensitivity_slider.blockSignals(False)



        order = {nid: i for i, nid in enumerate(ctx)}



        pinned = []



        rest = []



        for score, note in self.all_scored_notes:



            if note['id'] in self._pinned_note_ids:



                pinned.append((max_score, note))



            else:



                rest.append((score, note))



        pinned.sort(key=lambda x: order.get(x[1]['id'], 999))



        self.all_scored_notes = pinned + rest



        self.filter_and_display_notes()







        # Scroll to and highlight only the clicked citation row.
        _highlight_result_notes(self, {note_id}, scroll_to_note_id=note_id, scroll_to_ref=num)







    if saved_html:



        QTimer.singleShot(0, lambda h=saved_html: self._restore_answer_html(h))



def _citation_timer_clear(self):



    self._citation_last_click = None



def _bring_browser_to_front(self, browser):



    """Raise browser window after a short delay so it stays on top of the add-on dialog."""



    if browser and hasattr(browser, 'activateWindow'):



        browser.activateWindow()



        browser.raise_()



def _open_note_in_browser(self, note_id, num):



    """Open note in Anki Browser (used when user double-clicks a citation link). Brings browser to front over add-on."""



    try:



        browser = dialogs.open("Browser", mw)



        if browser:



            browser.form.searchEdit.lineEdit().setText(f"nid:{note_id}")



            browser.onSearchActivated()



            QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))



            tooltip(f"Opened note [{num}] (ID: {note_id}) in browser")



    except Exception as e:



        log_debug(f"Error opening note in browser: {e}")



        tooltip(f"Could not open note [{num}] in browser")



def copy_answer_to_clipboard(self):



    html = getattr(self, '_last_formatted_answer', None) or ""



    plain = self.answer_box.toPlainText().strip()



    if html or plain:



        cb = QApplication.clipboard()



        if cb:



            mime = QMimeData()



            if html:



                mime.setHtml(html)



            mime.setText(plain if plain else "")



            cb.setMimeData(mime)



            tooltip("Copied (paste into Word for bullets and formatting)")



    else:



        tooltip("No answer to copy")
