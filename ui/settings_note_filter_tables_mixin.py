# ============================================================================
# Imports
# ============================================================================

import glob
import os
import sqlite3
import subprocess
import time

try:
    import sip
except ImportError:
    try:
        from PyQt6 import sip
    except ImportError:
        try:
            from PyQt5 import sip
        except ImportError:
            sip = None

from aqt import dialogs, mw
from aqt.qt import *
from aqt.utils import showInfo, showText, tooltip

from .dependency_install import _resolve_external_python_exe, install_dependencies
from .settings_constants import (
    ANSWER_CLOUD_PROVIDERS,
    ANSWER_KEY_PROVIDER_PREFIXES,
    EMBEDDING_CLOUD_PROVIDERS,
    EMBEDDING_KEY_PROVIDER_PREFIXES,
)
from .settings_rerank_workers import RerankModelDownloadWorker, RerankModelVerifyWorker
from .theme import (
    get_addon_theme,
    settings_button_style,
    settings_panel_style,
    settings_status_label_style,
    settings_text_style,
)
from .widgets import CollapsibleSection, apply_setting_row_tooltip, settings_field_row, sync_setting_row_tooltips
from ..core.cloud_diagnostics import (
    classify_provider_error,
    provider_status_message,
    test_cloud_answer_connection,
)
from ..core.engine import (
    analyze_note_eligibility,
    clear_checkpoint,
    count_notes_matching_config,
    get_deck_names,
    get_embedding_engine_id,
    get_embedding_for_query,
    get_models_with_fields,
    get_notes_count_per_deck,
    get_notes_count_per_model,
    get_ollama_models,
    load_checkpoint,
    load_embedding_engine_counts,
    load_embeddings_bulk,
    make_embedding_scope_id,
    _normalize_ollama_base_url,
)
from ..core.workers import EmbeddingWorker, RerankCheckWorker
from ..utils import (
    EmbeddingsTabMessages,
    format_partial_failure_completion,
    format_partial_failure_progress,
    get_effective_embedding_config,
    get_embeddings_db_path,
    get_embeddings_storage_path_for_read,
    get_retrieval_config,
    load_config,
    log_debug,
    save_config,
    validate_embedding_config,
)
from ..utils.config import (
    DEFAULT_RERANK_MODEL,
    RERANK_TIMEOUT_SECONDS_DEFAULT,
    RERANK_TIMEOUT_SECONDS_MAX,
    RERANK_TIMEOUT_SECONDS_MIN,
    RERANK_TOP_K_DEFAULT,
    get_rerank_config,
)

_addon_theme = get_addon_theme

class SettingsNoteFilterTablesMixin:
    def _populate_note_type_lists_with_timing(self):



        """Fill note types table with name and count columns (with timing)."""



        # Check if widget still exists (dialog might have been closed)



        try:



            if not hasattr(self, 'note_types_table') or self.note_types_table is None:



                return



            # Check if the C++ object is still valid



            if not sip.isdeleted(self.note_types_table) if hasattr(sip, 'isdeleted') else True:



                import time



                start = time.time()



                self._populate_note_type_lists()



                elapsed = time.time() - start



                log_debug(f"  [Timing] _populate_note_type_lists(): {elapsed:.3f}s")



                # After table is populated, apply saved note/deck/field config



                try:



                    cfg = load_config()



                    ntf = cfg.get('note_type_filter', {})



                    self._apply_note_type_config(ntf)



                except Exception as e:



                    log_debug(f"Error re-applying note_type_filter after note type populate: {e}")



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_note_type_lists_with_timing: {e}")


    def _populate_note_type_lists(self):



        """Fill note types table with name and count columns."""



        # Check if widget still exists



        if not hasattr(self, 'note_types_table') or self.note_types_table is None:



            return



        was_applying = getattr(self, '_applying_note_type_config', False)
        self._applying_note_type_config = True

        try:



            self.note_types_table.setRowCount(0)



            counts = get_notes_count_per_model()



            for name in sorted(counts.keys()):



                c = counts.get(name, 0)



                row = self.note_types_table.rowCount()



                self.note_types_table.insertRow(row)







                # Name column with checkbox



                name_item = QTableWidgetItem(name)



                name_item.setData(Qt.ItemDataRole.UserRole, name)



                name_item.setFlags(name_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)



                name_item.setCheckState(Qt.CheckState.Unchecked)



                self.note_types_table.setItem(row, 0, name_item)







                # Count column



                count_item = QTableWidgetItem()



                count_item.setData(Qt.ItemDataRole.DisplayRole, c)  # Store numeric value for proper sorting



                count_item.setText(str(c))



                count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)



                count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make read-only



                self.note_types_table.setItem(row, 1, count_item)







            # Sort by count descending by default



            self.note_types_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_note_type_lists: {e}")

        finally:

            self._applying_note_type_config = was_applying


    def _populate_fields_by_note_type(self):



        """Build collapsible field sections per note type and repopulate them."""



        # Check if widget still exists



        if not hasattr(self, 'fields_by_note_type_layout') or self.fields_by_note_type_layout is None:



            return



        try:



            while self.fields_by_note_type_layout.count() > 0:



                it = self.fields_by_note_type_layout.takeAt(0)



                if it and it.widget():



                    it.widget().deleteLater()



            self._field_cbs.clear()



            self._field_groupboxes.clear()



            included = set()



            for i in range(self.note_types_table.rowCount()):



                it = self.note_types_table.item(i, 0)



                if it and it.checkState() == Qt.CheckState.Checked:



                    included_name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()



                    included.add(included_name)



            for model_name, count, field_names in get_models_with_fields():



                is_included = model_name in included



                gb = CollapsibleSection(
                    f"{model_name}  ({count} notes)",
                    parent=self.fields_by_note_type_inner,
                    is_expanded=is_included,
                )



                cbs = {}



                for fn in field_names:



                    cb = QCheckBox(fn, gb)



                    cb.installEventFilter(self)



                    if hasattr(self, "_wheel_guarded_widgets"):



                        self._wheel_guarded_widgets.add(cb)



                    cb.stateChanged.connect(lambda *_: self._persist_note_type_filter())



                    cbs[fn] = cb



                    gb.addWidget(cb)



                self._field_cbs[model_name] = cbs



                self._field_groupboxes[model_name] = gb



                self.fields_by_note_type_layout.addWidget(gb)



            self._update_field_groups_enabled()



            self._sync_field_groups_to_note_type_order()



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_fields_by_note_type: {e}")


    def _populate_fields_by_note_type_with_timing(self):



        """Populate fields by note type (with timing)."""



        import time



        start = time.time()



        self._populate_fields_by_note_type()



        elapsed = time.time() - start



        if elapsed > 0.1:  # Only log if it takes significant time



            log_debug(f"  [Timing] _populate_fields_by_note_type(): {elapsed:.3f}s")



        # Ensure field checkboxes match saved configuration



        try:



            cfg = load_config()



            ntf = cfg.get('note_type_filter', {})



            self._apply_note_type_config(ntf)



        except Exception as e:



            log_debug(f"Error re-applying note_type_filter after field populate: {e}")


    def _populate_decks_list(self, force_refresh=False):



        import time



        deck_start = time.time()







        # Check if widget still exists



        if not hasattr(self, 'decks_list') or self.decks_list is None:



            return



        try:



            self.decks_list.clear()







            if not mw or not mw.col:



                return







            deck_data_start = time.time()



            counts, deck_names, used_cache = self._get_cached_deck_data(force_refresh=force_refresh)



            deck_data_elapsed = time.time() - deck_data_start



            log_debug(
                f"  [Timing] deck data ({'cache' if used_cache else 'fresh'}): {deck_data_elapsed:.3f}s"
            )







            # Build hierarchical deck structure



            deck_tree = {}  # parent_name -> [child_decks]



            top_level_decks = []







            for name in deck_names:



                if '::' in name:



                    # Sub-deck



                    parts = name.split('::')



                    parent = '::'.join(parts[:-1])



                    if parent not in deck_tree:



                        deck_tree[parent] = []



                    deck_tree[parent].append(name)



                else:



                    # Top-level deck



                    top_level_decks.append(name)







            # Get card counts for each deck (new, learn, due)

            card_counts = {}

            try:

                # In modern Anki, we don't need to import Scheduler; mw.col.sched is ready.

                # Use the safer get_deck_stats if available, or fallback.

                deck_ids = {name: mw.col.decks.id(name) for name in deck_names if mw.col.decks.by_name(name)}



                for name, deck_id in deck_ids.items():

                    try:

                        # Modern scheduler fallback

                        if hasattr(mw.col.sched, "counts"):

                            counts_info = mw.col.sched.counts(deck_id)

                            card_counts[name] = {

                                'new': getattr(counts_info, 'new', 0),

                                'learn': getattr(counts_info, 'learn', 0),

                                'due': getattr(counts_info, 'review', 0)

                            }

                        else:

                            # If counts() is missing, just use 0s for now

                            card_counts[name] = {'new': 0, 'learn': 0, 'due': 0}

                    except Exception:

                        card_counts[name] = {'new': 0, 'learn': 0, 'due': 0}



            except Exception as e:

                log_debug(f"Error getting card counts: {e}")

                # Fallback: set all to 0

                for name in deck_names:

                    card_counts[name] = {'new': 0, 'learn': 0, 'due': 0}







            # Sort decks



            top_level_decks.sort()



            for parent in deck_tree:



                deck_tree[parent].sort()







            # Create tree items



            def create_deck_item(name, is_parent=False):



                """Create a tree item for a deck (showing only name + total notes)."""



                note_count = counts.get(name, 0)



                # Hide the built-in empty 'Default' deck, which many users don't use



                if name == "Default" and note_count == 0:



                    return None







                # Extract display name (without parent prefix for sub-decks)



                display_name = name.split('::')[-1] if '::' in name else name







                item = QTreeWidgetItem([display_name, str(note_count)])







                # Store full deck name in item data for later retrieval



                item.setData(0, Qt.ItemDataRole.UserRole, name)







                # Make checkable



                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)



                item.setCheckState(0, Qt.CheckState.Unchecked)







                # Style parent decks (bold)



                if is_parent:



                    font = item.font(0)



                    font.setBold(True)



                    item.setFont(0, font)







                return item







            # Add top-level decks



            for deck_name in top_level_decks:



                parent_item = create_deck_item(deck_name, is_parent=(deck_name in deck_tree))



                if parent_item is None:



                    continue



                self.decks_list.addTopLevelItem(parent_item)







                # Add children if any



                if deck_name in deck_tree:



                    for child_name in deck_tree[deck_name]:



                        child_item = create_deck_item(child_name)



                        parent_item.addChild(child_item)







                    # Collapsed by default; user expands on click



                    parent_item.setExpanded(False)







            # Also add decks that are parents but not top-level (nested hierarchies)



            for parent_name in sorted(deck_tree.keys()):



                if parent_name not in top_level_decks:



                    # This is a nested parent, find its position in hierarchy



                    parts = parent_name.split('::')



                    if len(parts) > 1:



                        # Find parent item



                        grandparent_name = '::'.join(parts[:-1])



                        # Search for grandparent in tree



                        for i in range(self.decks_list.topLevelItemCount()):



                            parent_item = self._find_deck_item_recursive(self.decks_list.topLevelItem(i), grandparent_name)



                            if parent_item:



                                child_item = create_deck_item(parent_name, is_parent=True)



                                if child_item is None:



                                    break



                                parent_item.addChild(child_item)



                                # Add its children



                                for child_name in deck_tree[parent_name]:



                                    grandchild_item = create_deck_item(child_name)



                                    child_item.addChild(grandchild_item)



                                child_item.setExpanded(False)



                                break







            deck_elapsed = time.time() - deck_start



            log_debug(f"  [Timing] _populate_decks_list(): {deck_elapsed:.3f}s")
            return used_cache







        except RuntimeError:



            # Widget was deleted, ignore



            return False



        except Exception as e:



            log_debug(f"Error in _populate_decks_list: {e}")



            import traceback



            log_debug(traceback.format_exc())
            return False


    def _find_deck_item_recursive(self, item, deck_name):



        """Recursively find a deck item by its full name"""



        if item is None:



            return None



        if item.data(0, Qt.ItemDataRole.UserRole) == deck_name:



            return item



        for i in range(item.childCount()):



            found = self._find_deck_item_recursive(item.child(i), deck_name)



            if found:



                return found



        return None


    def _iterate_all_deck_items(self):



        """Generator that yields all deck items (top-level and children)"""



        for i in range(self.decks_list.topLevelItemCount()):



            item = self.decks_list.topLevelItem(i)



            yield item



            # Recursively yield children



            for j in range(item.childCount()):



                yield from self._iterate_all_deck_items_recursive(item.child(j))


    def _iterate_all_deck_items_recursive(self, item):



        """Recursively yield item and all its children"""



        if item:



            yield item



            for i in range(item.childCount()):



                yield from self._iterate_all_deck_items_recursive(item.child(i))


