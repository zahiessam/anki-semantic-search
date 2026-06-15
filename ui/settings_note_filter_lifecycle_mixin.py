"""Note type/deck loading lifecycle helpers for Settings."""

import time

from aqt import mw
from aqt.qt import QTimer

from ..core.engine import get_deck_names, get_notes_count_per_deck
from ..utils import load_config, save_config, log_debug


_note_type_deck_data_cache = {
    "collection_key": None,
    "deck_names": None,
    "deck_note_counts": None,
    "tag_counts": None,
}


class SettingsNoteFilterLifecycleMixin:
    """Owns note/deck cache state and lazy loading coordination."""

    def _set_note_type_loading_status(self, message="", busy=False):
        label = getattr(self, "note_type_loading_status", None)
        bar = getattr(self, "note_type_loading_bar", None)

        try:
            if label is not None:
                label.setText(message or "")
                label.setVisible(bool(message))
            if bar is not None:
                bar.setVisible(bool(busy))
            if message:
                log_debug(f"Note Types loading status: {message}")
        except RuntimeError:
            pass

    def _collection_cache_key(self):
        try:
            col = getattr(mw, "col", None)
            if col is None:
                return None
            path = getattr(col, "path", None)
            if callable(path):
                return path()
            return path or id(col)
        except Exception:
            return None

    def _get_cached_deck_data(self, force_refresh=False):
        global _note_type_deck_data_cache

        collection_key = self._collection_cache_key()
        cache_matches = (
            _note_type_deck_data_cache.get("collection_key") == collection_key
            and _note_type_deck_data_cache.get("deck_names") is not None
            and _note_type_deck_data_cache.get("deck_note_counts") is not None
        )

        if cache_matches and not force_refresh:
            return (
                dict(_note_type_deck_data_cache.get("deck_note_counts") or {}),
                list(_note_type_deck_data_cache.get("deck_names") or []),
                True,
            )

        counts = get_notes_count_per_deck()
        deck_names = get_deck_names()
        _note_type_deck_data_cache = {
            "collection_key": collection_key,
            "deck_names": list(deck_names),
            "deck_note_counts": dict(counts),
            "tag_counts": None,
        }
        return dict(counts), list(deck_names), False

    def _invalidate_note_type_deck_cache(self):
        global _note_type_deck_data_cache
        _note_type_deck_data_cache = {
            "collection_key": None,
            "deck_names": None,
            "deck_note_counts": None,
            "tag_counts": None,
        }

    def _on_settings_tab_changed(self, index):
        """Lazy-load expensive scope lists only when that tab is opened."""
        try:
            tabs = getattr(self, "_settings_tabs", None)
            scope_tab = getattr(self, "_scope_tab", None)
            if tabs is not None and scope_tab is not None and tabs.widget(index) == scope_tab:
                self._schedule_scope_lists_load()
        except Exception as e:
            log_debug(f"Error handling settings tab change: {e}")

    def _schedule_scope_lists_load(self, force=False):
        if not force and getattr(self, "_scope_lists_loaded", False):
            return

        self._scope_lists_loaded = True
        # Allow scope tree population when tab is opened
        self._allow_scope_population = True
        cache_ready = (
            _note_type_deck_data_cache.get("collection_key") == self._collection_cache_key()
            and _note_type_deck_data_cache.get("deck_names") is not None
            and _note_type_deck_data_cache.get("deck_note_counts") is not None
        )
        if cache_ready and not force:
            self._set_scope_loading_status(
                "Loading scope data. Reusing deck counts from this Anki session.",
                busy=False,
            )
        else:
            self._set_scope_loading_status(
                "Loading scope data. This can take a while on large collections.",
                busy=True,
            )
        QTimer.singleShot(50, lambda: self._populate_scope_tab_with_timing(force_refresh=force))

    def _populate_scope_tab_with_timing(self, force_refresh=False):
        containers = [
            getattr(self, "scope_tree", None),
        ]
        containers = [widget for widget in containers if widget is not None]
        try:
            start = time.time()
            cache_ready = (
                _note_type_deck_data_cache.get("collection_key") == self._collection_cache_key()
                and _note_type_deck_data_cache.get("deck_names") is not None
                and _note_type_deck_data_cache.get("deck_note_counts") is not None
            )
            for widget in containers:
                widget.setUpdatesEnabled(False)
            self._populate_scope_tree()
            # Skip preset refresh for now since we removed preset controls from scope tab
            # self._refresh_preset_combos()
            try:
                cfg = load_config()
                self._apply_scope_config(self._resolve_scope_config_from_config(cfg, persist=True))
                if hasattr(self, "_refresh_chatbot_fields_after_scope_apply"):
                    self._refresh_chatbot_fields_after_scope_apply(cfg)
            except Exception as e:
                log_debug(f"Error applying scope config after scope tab populate: {e}")
            elapsed = time.time() - start
            log_debug(f"  [Timing] _populate_scope_tab(): {elapsed:.3f}s")
            status = (
                f"Scope data loaded from cache in {elapsed:.1f}s."
                if cache_ready and not force_refresh
                else f"Scope data loaded in {elapsed:.1f}s."
            )
            self._set_scope_loading_status(status, busy=False)
            QTimer.singleShot(3500, lambda: self._set_scope_loading_status("", busy=False))
        except RuntimeError:
            pass
        except Exception as e:
            log_debug(f"Error in _populate_scope_tab_with_timing: {e}")
            self._set_scope_loading_status(
                "Could not load scope data. Use Refresh lists to try again.",
                busy=False,
            )
        finally:
            try:
                for widget in containers:
                    widget.setUpdatesEnabled(True)
                    widget.update()
            except RuntimeError:
                pass

    def _populate_note_type_tab_with_timing(self, force_refresh=False):
        containers = [
            getattr(self, "note_types_table", None),
            getattr(self, "fields_by_note_type_scroll", None),
            getattr(self, "decks_list", None),
        ]
        containers = [widget for widget in containers if widget is not None]
        try:
            start = time.time()
            for widget in containers:
                widget.setUpdatesEnabled(False)
            self._populate_note_type_lists()
            self._populate_fields_by_note_type()
            used_cache = self._populate_decks_list(force_refresh=force_refresh)
            self._refresh_preset_combos()
            try:
                cfg = load_config()
                self._apply_note_type_config(self._resolve_note_type_filter_from_config(cfg, persist=True))
            except Exception as e:
                log_debug(f"Error applying note_type_filter after Note Types tab populate: {e}")
            elapsed = time.time() - start
            log_debug(f"  [Timing] _populate_note_type_tab(): {elapsed:.3f}s")
            status = (
                f"Note type and deck data loaded from cache in {elapsed:.1f}s."
                if used_cache
                else f"Note type and deck data loaded in {elapsed:.1f}s."
            )
            self._set_note_type_loading_status(status, busy=False)
            QTimer.singleShot(3500, lambda: self._set_note_type_loading_status("", busy=False))
        except RuntimeError:
            pass
        except Exception as e:
            log_debug(f"Error in _populate_note_type_tab_with_timing: {e}")
            self._set_note_type_loading_status(
                "Could not load note type and deck data. Use Refresh lists to try again.",
                busy=False,
            )
        finally:
            try:
                for widget in containers:
                    widget.setUpdatesEnabled(True)
                    widget.update()
            except RuntimeError:
                pass

    def _resolve_scope_config_from_config(self, config, persist=False):
        """Resolve scope config from config, migrating legacy format if needed."""
        ntf = config.get('note_type_filter', {})
        
        if ntf.get('scope_mode') == 'deck':
            resolved = dict(ntf)
            resolved['enabled_tags'] = None
            return resolved
        
        migrated = self._migrate_legacy_config_to_scope(ntf)
        
        if persist and migrated:
            config['note_type_filter'] = migrated
            save_config(config)
        
        return migrated

    def _migrate_legacy_config_to_scope(self, ntf):
        """Migrate legacy note_type_filter config to new scope format."""
        enabled_decks = ntf.get('enabled_decks') if ntf.get('scope_mode') != 'tag' else None
        enabled_note_types = ntf.get('enabled_note_types')
        note_type_fields = ntf.get('note_type_fields', {})
        search_all_fields = ntf.get('search_all_fields', False)
        use_first_field_fallback = ntf.get('use_first_field_fallback', True)
        
        # Build new scope format
        scope_mode = 'deck'  # Default to deck mode for legacy configs
        scope_fields = {}
        
        if enabled_decks:
            # For each enabled deck, apply the note type fields
            for deck in enabled_decks:
                scope_fields[deck] = {}
                if enabled_note_types:
                    for nt in enabled_note_types:
                        fields = note_type_fields.get(nt, [])
                        if fields:
                            scope_fields[deck][nt] = fields
        
        # Build migrated config
        migrated = {
            'scope_mode': scope_mode,
            'enabled_decks': enabled_decks,
            'enabled_tags': None,
            'scope_fields': scope_fields,
            'search_all_fields': search_all_fields,
            'use_first_field_fallback': use_first_field_fallback,
            # Keep legacy keys for compatibility
            'enabled_note_types': enabled_note_types,
            'note_type_fields': note_type_fields,
        }
        
        return migrated

    def _populate_decks_list_with_timing(self, force_refresh=False):
        """Fill decks tree with deck names/counts while reporting timing."""
        try:
            if not hasattr(self, 'decks_list') or self.decks_list is None:
                return

            start = time.time()
            cache_ready = (
                _note_type_deck_data_cache.get("collection_key") == self._collection_cache_key()
                and _note_type_deck_data_cache.get("deck_names") is not None
                and _note_type_deck_data_cache.get("deck_note_counts") is not None
            )
            if cache_ready and not force_refresh:
                self._set_note_type_loading_status(
                    "Loading decks from this Anki session's cached counts.",
                    busy=False,
                )
            else:
                self._set_note_type_loading_status(
                    "Loading deck counts. This can take a while on large collections or many subdecks.",
                    busy=True,
                )

            used_cache = self._populate_decks_list(force_refresh=force_refresh)
            elapsed = time.time() - start
            log_debug(f"  [Timing] _populate_decks_list(): {elapsed:.3f}s")

            try:
                cfg = load_config()
                ntf = cfg.get('note_type_filter', {})
                self._apply_note_type_config(ntf)
            except Exception as e:
                log_debug(f"Error re-applying note_type_filter after deck populate: {e}")

            status = (
                f"Deck data loaded from cache in {elapsed:.1f}s."
                if used_cache
                else f"Deck data loaded in {elapsed:.1f}s."
            )
            self._set_note_type_loading_status(status, busy=False)
            QTimer.singleShot(3500, lambda: self._set_note_type_loading_status("", busy=False))
        except RuntimeError:
            pass
        except Exception as e:
            log_debug(f"Error in _populate_decks_list_with_timing: {e}")
            self._set_note_type_loading_status(
                "Could not load deck data. Use Refresh lists to try again.",
                busy=False,
            )
