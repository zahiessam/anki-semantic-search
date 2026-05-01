# ============================================================================
# Imports
# ============================================================================

import time
import os
import sys
try:
    import sip
except ImportError:
    try:
        from PyQt6 import sip
    except ImportError:
        try:
            from PyQt5 import sip
        except ImportError:
            # Fallback for older versions or other environments
            sip = None
import re
import datetime
import json
import hashlib
import urllib.request
import urllib.error
import subprocess
from aqt.qt import *
from aqt import mw, gui_hooks, dialogs
from aqt.utils import askUser, showInfo, showText, tooltip
from anki.hooks import addHook
from . import dialog_entrypoints as _dialog_entrypoints
from . import answer_formatting
from . import answer_navigation
from . import embedding_helpers
from . import note_content
from . import query_enhancement
from . import search_dialog_state
from . import search_dialog_ui
from . import search_dialog_lifecycle

# Local imports
from .theme import get_addon_theme
from .widgets import (
    SpellCheckPlainTextEdit,
    CollapsibleSection,
    SemanticSearchSideBar,
    _get_spell_checker
)
from .search_dialog import RelevanceBarDelegate, ContentDelegate
from ..utils import (
    log_debug,
    load_config,
    save_config,
    get_config_value,
    get_effective_embedding_config,
    validate_embedding_config,
    get_retrieval_config,
    get_addon_name,
    get_embeddings_db_path,
    get_embeddings_storage_path_for_read,
    get_checkpoint_path,
    EmbeddingsTabMessages,
    ErrorAndEngineMessages,
    format_partial_failure_progress,
    format_partial_failure_completion,
    format_dimension_mismatch_hint,
    get_search_history_queries,
    load_search_history,
    save_search_history,
    clear_search_history,
)
from ..utils.text import unescape_string, clean_html, clean_html_for_display, reveal_cloze, semantic_chunk_text
from ..utils.config import VOYAGE_EMBEDDING_MODELS
from ..core.engine import (
    get_ollama_models,
    estimate_tokens,
    get_notes_count_per_model,
    get_models_with_fields,
    get_deck_names,
    get_notes_count_per_deck,
    _build_deck_query,
    count_notes_matching_config,
    analyze_note_eligibility,
    get_embedding_engine_id,
    migrate_embeddings_json_to_db,
    get_embedding_for_query,
    load_checkpoint,
    clear_checkpoint,
    extract_keywords_improved,
    compute_tfidf_scores,
    aggregate_scored_notes_by_note_id,
    load_embedding,
    save_embedding,
    flush_embedding_batch,
    save_checkpoint as core_save_checkpoint
)
from ..core.errors import _is_embedding_dimension_mismatch

# ============================================================================
# Module Configuration And Shared Helpers
# ============================================================================

_addon_theme = get_addon_theme
ADDON_NAME = get_addon_name()
_semantic_chunk_text = semantic_chunk_text


def _prefilter_safety_reason(col, ntf, config):
    """Return None when V2 can safely prefilter before loading full note text.

    Unsafe cases deliberately fall back to the historical full loading path:
    ambiguous/missing decks, unknown note types, missing selected fields without
    first-field fallback, all-fields search across unknown/mixed types, or cached
    history replay paths where note context must match prior results.
    """
    retrieval = get_retrieval_config(config)
    if retrieval.get("retrieval_version") != "v2":
        return "retrieval_version is legacy"
    if not ntf:
        return "no note_type_filter configured"
    if config.get("_history_replay"):
        return "history replay is active"

    enabled_decks = ntf.get("enabled_decks") or []
    if enabled_decks:
        try:
            all_decks = set(col.decks.all_names())
        except Exception:
            return "could not resolve deck names"
        missing_decks = [deck for deck in enabled_decks if deck not in all_decks]
        if missing_decks:
            return f"selected deck(s) not found: {missing_decks[:3]}"

    models = col.models.all()
    model_by_name = {model.get("name"): model for model in models}
    enabled_types = ntf.get("enabled_note_types") or []
    if enabled_types:
        missing_types = [name for name in enabled_types if name not in model_by_name]
        if missing_types:
            return f"selected note type(s) not found: {missing_types[:3]}"

    search_all = bool(ntf.get("search_all_fields", False))
    if search_all and not enabled_types:
        return "all-fields search across unknown/mixed note types"

    if not search_all:
        ntf_fields = ntf.get("note_type_fields") or {}
        use_first = bool(ntf.get("use_first_field_fallback", True))
        type_names = enabled_types or list(model_by_name)
        for model_name in type_names:
            model = model_by_name.get(model_name)
            if not model:
                continue
            available = {field.get("name", "").lower() for field in model.get("flds", [])}
            wanted = {field.lower() for field in (ntf_fields.get(model_name) or [])}
            if wanted and not wanted.issubset(available):
                return f"selected field missing on {model_name}"
            if not wanted and not use_first:
                return f"no fields selected for {model_name} and first-field fallback disabled"

    return None


def get_safe_config(value):
    """Return config-like data with secrets redacted for logging."""
    secret_markers = ("api_key", "token", "secret", "password")

    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            key_str = str(key).lower()
            if any(marker in key_str for marker in secret_markers):
                safe[key] = "***REDACTED***" if item else item
            else:
                safe[key] = get_safe_config(item)
        return safe

    if isinstance(value, list):
        return [get_safe_config(item) for item in value]

    return value


def get_notes_content_with_col(col, config):
    """Load searchable note content in a background QueryOp using the provided collection."""
    notes_data = []
    ntf = (config or {}).get('note_type_filter', {}) or {}

    if ntf.get('fields_to_search') and not ntf.get('note_type_fields'):
        global_flds = set(f.lower() for f in ntf['fields_to_search'])
        ntf = dict(ntf)
        ntf['note_type_fields'] = {}
        for model in col.models.all():
            field_names = [fld['name'] for fld in model.get('flds', [])]
            ntf['note_type_fields'][model['name']] = [
                field_name for field_name in field_names if field_name.lower() in global_flds
            ]

    legacy_fields = None
    if not ntf:
        enabled_set = None
        search_all = False
        ntf_fields = {}
        use_first = False
        legacy_fields = {'text', 'extra'}
        fields_description = "Text & Extra"
    else:
        enabled = ntf.get('enabled_note_types')
        enabled_set = set(enabled) if enabled else None
        search_all = bool(ntf.get('search_all_fields', False))
        ntf_fields = ntf.get('note_type_fields') or {}
        use_first = bool(ntf.get('use_first_field_fallback', True))
        fields_description = "all fields" if search_all else "per-type"

    safety_reason = _prefilter_safety_reason(col, ntf, config)
    retrieval = get_retrieval_config(config)
    if retrieval.get("retrieval_version") == "v2":
        if safety_reason:
            log_debug(f"V2 note prefilter fallback: {safety_reason}")
        else:
            log_debug("V2 note prefilter safe: deck/note-type/field filters resolved")

    deck_q = _build_deck_query(ntf.get('enabled_decks') if ntf else None)
    note_ids = col.find_notes(deck_q) if deck_q else col.find_notes("")
    total_notes = len(note_ids)
    cache_key = (
        deck_q or '',
        frozenset(enabled_set) if enabled_set is not None else None,
        search_all,
        tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in ntf_fields.items())),
        total_notes,
    )

    model_map = {}
    for model in col.models.all():
        mid = model['id']
        model_name = model['name']
        if enabled_set is not None and model_name not in enabled_set:
            continue
        flds = model.get('flds', [])
        if search_all:
            indices = list(range(len(flds)))
        else:
            if legacy_fields is not None:
                wanted = legacy_fields
            else:
                wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))
                if not wanted and use_first and flds:
                    wanted = {flds[0]['name'].lower()}
            indices = [i for i, field in enumerate(flds) if field['name'].lower() in wanted]
        if indices:
            model_map[mid] = (model_name, indices)

    if not note_ids:
        return notes_data, fields_description, cache_key

    chunk_target = 500
    id_list = ",".join(map(str, note_ids))
    for nid, mid, flds_str in col.db.execute(f"select id, mid, flds from notes where id in ({id_list})"):
        model_info = model_map.get(mid)
        if not model_info:
            continue
        model_name, indices = model_info
        fields = (flds_str or "").split("\x1f")
        content_parts = []
        for index in indices:
            if index < len(fields):
                value = fields[index].strip()
                if value:
                    content_parts.append(value)
        if not content_parts:
            continue

        raw_content = " | ".join(content_parts)
        content = clean_html(raw_content).strip()
        if not content:
            continue
        display_parts = []
        for part in content_parts:
            display_part = clean_html_for_display(part).strip()
            if display_part:
                display_parts.append(display_part)
        display_content = " | ".join(display_parts) or content

        chunks = _semantic_chunk_text(content, chunk_target)
        if len(chunks) <= 1:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            notes_data.append({
                'id': nid,
                'content': content,
                'content_hash': content_hash,
                'model': model_name,
                'display_content': display_content,
            })
            continue

        for chunk_idx, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            notes_data.append({
                'id': nid,
                'content': chunk,
                'content_hash': chunk_hash,
                'model': model_name,
                'display_content': clean_html_for_display(chunk).strip() or chunk,
                'chunk_index': chunk_idx,
                '_full_content': content,
                '_full_display_content': display_content,
            })

    return notes_data, fields_description, cache_key

# --- Regular Expression Constants ---

# Constants from original __init__.py
HTML_TAG_RE = re.compile(r'<.*?>', re.DOTALL)
CLOZE_RE = re.compile(r'\{\{c\d+::(.*?)(?=\}\}|::)(?:::[^}]*)?\}\}')
SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?\n])\s+|\n+')
CITATION_RE = re.compile(r'\[([\d,\s]+)\]')
CITATION_N_RE = re.compile(r'\[N([\d,\sN]+)\]')
MD_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
MD_BOLD_ALT_RE = re.compile(r'__(.+?)__')
MD_HIGHLIGHT_RE = re.compile(r'~~(.+?)~~')
MD_HEADER_RE = re.compile(r'^(.{1,50}):(\s*)$', re.MULTILINE)
MD_UNTERMINATED_BOLD_RE = re.compile(r'\*\*([^*]+)$')
WORD_BOUNDARY_RE = re.compile(r'\b\w+\b')
DIGIT_RE = re.compile(r'\d+')

# --- Debug Logging Helpers ---

def _session_debug_log(hypothesis_id, location, message, data=None):
    """Write NDJSON to session log file for debug."""
    try:
        entry = {
            "sessionId": "85902e",
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
        }
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(addon_dir, "user_files", "debug-85902e.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _agent_debug_log(run_id, hypothesis_id, location, message, data=None):
    """Lightweight debug logger for agent-driven investigations."""
    try:
        entry = {
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
        }
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(addon_dir, "user_files", "debug.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ============================================================================
# AI Search Dialog
# ============================================================================

class AISearchDialog(QDialog):



    # --- Lifecycle And Window Setup ---

    def __init__(self, parent=None):
        search_dialog_lifecycle.initialize_search_dialog(self, parent)







    # --- Search Entry Points ---

    def perform_search(self):



        """



        Bridge method so AISearchDialog always has perform_search.



        Delegates to the implementation that was originally defined on EmbeddingSearchWorker.



        """



        return EmbeddingSearchWorker.perform_search(self)







    # --- Selection Controls ---

    def toggle_select_all(self):



        """Bridge to shared toggle_select_all implementation."""



        return EmbeddingSearchWorker.toggle_select_all(self)







    def select_all_notes(self):



        """Bridge to shared select_all_notes implementation."""



        return EmbeddingSearchWorker.select_all_notes(self)







    def deselect_all_notes(self):



        """Bridge to shared deselect_all_notes implementation."""



        return EmbeddingSearchWorker.deselect_all_notes(self)







    # --- Defaults And UI Construction ---

    def reset_to_medical_defaults(self):
        return search_dialog_ui.reset_to_medical_defaults(self)



    def setup_ui(self):
        return search_dialog_ui.setup_ui(self)







    # --- Search History And Scope Banner ---

    def _update_view_all_button_state(self):
        return search_dialog_state._update_view_all_button_state(self)







    def showEvent(self, event):
        return search_dialog_state.showEvent(self, event)







    def _refresh_scope_banner(self):
        return search_dialog_state._refresh_scope_banner(self)







    def _on_search_history_selected(self, index):
        return search_dialog_state._on_search_history_selected(self, index)







    def _on_sidebar_history_selected(self, item):
        return search_dialog_state._on_sidebar_history_selected(self, item)







    def _set_query_text(self, text):
        return search_dialog_state._set_query_text(self, text)







    def _clear_query_loaded_status(self):
        return search_dialog_state._clear_query_loaded_status(self)







    def _on_clear_search_history(self):
        return search_dialog_state._on_clear_search_history(self)







    def _refresh_search_history(self):
        return search_dialog_state._refresh_search_history(self)







    # --- Relevance And Display Options ---

    def on_item_changed(self, item):
        return search_dialog_state.on_item_changed(self, item)







    def _on_relevance_mode_changed(self, mode_key, checked):
        return search_dialog_state._on_relevance_mode_changed(self, mode_key, checked)







    def on_sensitivity_changed(self, value):
        return search_dialog_state.on_sensitivity_changed(self, value)







    def _on_show_only_cited_changed(self, checked):
        return search_dialog_state._on_show_only_cited_changed(self, checked)







    # --- Answer Formatting And Citation Navigation ---

    def _restore_answer_html(self, html):
        return answer_navigation._restore_answer_html(self, html)







    def _on_answer_link_clicked(self, url):
        return answer_navigation._on_answer_link_clicked(self, url)







    def _citation_timer_clear(self):
        return answer_navigation._citation_timer_clear(self)







    def _bring_browser_to_front(self, browser):
        return answer_navigation._bring_browser_to_front(self, browser)







    def _open_note_in_browser(self, note_id):
        return answer_navigation._open_note_in_browser(self, note_id)







    def _spacing_styles(self):
        return answer_formatting.spacing_styles(self.styling_config.get('answer_spacing', 'normal'))





    def format_answer(self, answer):
        return answer_formatting.format_answer_html(
            answer,
            getattr(self, '_context_note_ids', None) or [],
            self.styling_config.get('answer_spacing', 'normal'),
            {
                'citation_n': CITATION_N_RE,
                'citation': CITATION_RE,
                'md_bold': MD_BOLD_RE,
                'md_unterminated_bold': MD_UNTERMINATED_BOLD_RE,
                'md_bold_alt': MD_BOLD_ALT_RE,
                'md_header': MD_HEADER_RE,
                'md_highlight': MD_HIGHLIGHT_RE,
            },
        )







    # --- Clipboard, Settings, And Config Access ---

    def copy_answer_to_clipboard(self):
        return answer_navigation.copy_answer_to_clipboard(self)







    def open_settings(self):
        return search_dialog_state.open_settings(self)







    def get_config(self):
        return search_dialog_state.get_config(self)







    # --- Note Collection And Text Preparation ---

    def get_all_notes_content(self):
        return note_content.get_all_notes_content(self)





    # --- Keyword And TF-IDF Helpers ---

    def _aggregate_scored_notes_by_note_id(self, scored_notes):
        return note_content.aggregate_scored_notes_by_note_id(scored_notes)





    def strip_html(self, text):
        return note_content.strip_html(text)





    def reveal_cloze_for_display(self, text):
        return note_content.reveal_cloze_for_display(text)





    def _simple_stem(self, word):
        return note_content._simple_stem(word)





    def _get_extended_stop_words(self):
        search_config = (load_config() or {}).get('search_config') or {}
        return note_content.get_extended_stop_words(search_config)





    def _extract_keywords_improved(self, query):
        return note_content.extract_keywords_for_dialog(self, query, _agent_debug_log)





    def _compute_tfidf_scores(self, notes, query_keywords):
        return note_content.compute_tfidf_scores_for_dialog(self, notes, query_keywords)







    # --- Embedding Search Helpers ---

    def _get_note_embedding(self, note_content, note_id=None):
        return embedding_helpers._get_note_embedding(self, note_content, note_id)







    def _embedding_search(self, query, notes):
        return embedding_helpers._embedding_search(self, query, notes)







    # --- Metadata And Context Boosting ---

    def _get_note_metadata(self, note_id):
        return embedding_helpers._get_note_metadata(self, note_id)







    def _context_aware_boost(self, note, base_score):
        return embedding_helpers._context_aware_boost(self, note, base_score)







    # --- AI Query Expansion And Relevance Filtering ---

    def _expand_query(self, query, config):
        return query_enhancement._expand_query(self, query, config)







    def _get_ai_excluded_terms(self, query, search_config):
        return query_enhancement._get_ai_excluded_terms(self, query, search_config)







    def _generate_hyde_document(self, query, config):
        return query_enhancement._generate_hyde_document(self, query, config)







    def _passes_focused_balanced_broad(
        self,
        matched_keywords,
        final_score,
        emb_score,
        max_emb_score,
        keywords,
        search_method,
        embeddings_available,
        min_emb_frac=0.25,
        very_high_emb_frac=0.9,
    ):
        return query_enhancement._passes_focused_balanced_broad(
            self,
            matched_keywords,
            final_score,
            emb_score,
            max_emb_score,
            keywords,
            search_method,
            embeddings_available,
            min_emb_frac,
            very_high_emb_frac,
        )











# ============================================================================
# Search Worker Compatibility And Standalone Search Helpers
# ============================================================================

# --- Embedding Search Execution ---




# ============================================================================
# Search Worker Compatibility And Standalone Search Helpers
# ============================================================================

from .search_workers import (
    MAX_RERANK_COUNT,
    RRF_K,
    AskAIWorker,
    EmbeddingSearchWorker,
    KeywordFilterContinueWorker,
    KeywordFilterWorker,
    RelevanceRerankWorker,
    RerankCheckWorker,
    RerankWorker,
    _do_rerank,
    _run_embedding_search_sync,
)











# --- Anthropic Prompt Construction ---

# ============================================================================
# Anthropic Prompt Construction
# ============================================================================

from .answer_prompts import _build_anthropic_prompt_parts, _normalize_query_space











# --- Search Workflow Methods Copied Onto AISearchDialog ---

# ============================================================================
# Search Workflow Compatibility Import And Wiring
# ============================================================================

from .search_workflow import AnthropicStreamWorker, configure_search_workflow_globals, install_search_workflow_methods

configure_search_workflow_globals(
    DIGIT_RE=DIGIT_RE,
    _agent_debug_log=_agent_debug_log,
    _session_debug_log=_session_debug_log,
    get_notes_content_with_col=get_notes_content_with_col,
    get_safe_config=get_safe_config,
)
install_search_workflow_methods(AISearchDialog)

# ============================================================================
# Search Dialog Singleton And Sidebar Controls
# ============================================================================

def toggle_sidebar_visibility(visible: bool):
    return _dialog_entrypoints.toggle_sidebar_visibility(visible, dialogs_module=sys.modules[__name__])


# ============================================================================
# Settings Dialog Compatibility Import
# ============================================================================

from .settings_dialog import SettingsDialog







# END OF PART 1 - Continue to PART 2

# PART 2 OF 3 - Continue from PART 1



from aqt.qt import *

# Use pyqtSignal directly as it is the most stable across Anki versions

try:

    from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

except ImportError:

    from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot



# ============================================================================
# Public Dialog Entry Points
# ============================================================================

def show_search_dialog():
    return _dialog_entrypoints.show_search_dialog(dialogs_module=sys.modules[__name__])



def _on_search_dialog_closed():
    return _dialog_entrypoints._on_search_dialog_closed(dialogs_module=sys.modules[__name__])



def show_settings_dialog(open_to_embeddings=False, auto_start_indexing=False):
    return _dialog_entrypoints.show_settings_dialog(
        open_to_embeddings=open_to_embeddings,
        auto_start_indexing=auto_start_indexing,
        dialogs_module=sys.modules[__name__],
    )







def show_debug_log():
    return _dialog_entrypoints.show_debug_log()







# ============================================================================
# Dependency Detection And Installation Helpers
# ============================================================================

from .dependency_install import (
    _check_sentence_transformers_installed_subprocess,
    _patch_colorama_for_transformers,
    _resolve_external_python_exe,
    check_dependency_installed,
    check_vc_redistributables,
    fix_pytorch_dll_issue,
    get_pytorch_dll_error_guidance,
    install_dependencies,
    install_vc_redistributables,
    try_alternative_pytorch_install,
)







log_debug("=== Anki Semantic Search Add-on Loaded ===")



log_debug(f"Addon directory: {os.path.dirname(__file__)}")



log_debug(f"Addon folder name: {ADDON_NAME}")







# Note: colorama is already patched at module load time via _patch_colorama_early()







# ============================================================================
# Add-on Menu And Hook Registration
# ============================================================================

# Add menu items



ai_search_menu = QMenu("\U0001F50D Anki Semantic Search", mw)







# search_action = QAction("Search Notes", mw)



# search_action.triggered.connect(show_search_dialog)



# search_action.setToolTip("Open Anki Semantic Search window")



# search_action.setShortcut("Ctrl+Shift+S")



# mw.form.menuTools.addAction(search_action)







mw.addonManager.setConfigAction(ADDON_NAME, show_settings_dialog)







# Background indexer: re-enabled using QueryOp \u2014 collection access only on main thread via QueryOp; worker thread does API + save only (no mw.col).


