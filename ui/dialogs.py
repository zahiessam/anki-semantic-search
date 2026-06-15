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
from . import search_dialog_lifecycle
from .answer_formatting import SearchAnswerFormattingMixin
from .answer_navigation import SearchAnswerNavigationMixin
from .search_chat_mixin import SearchChatMixin
from .search_answer_display_mixin import SearchAnswerDisplayMixin
from .search_answer_workflow_mixin import SearchAnswerWorkflowMixin
from .search_anthropic_streaming_mixin import SearchAnthropicStreamingMixin
from .search_browser_actions_mixin import SearchBrowserActionsMixin
from .search_context_planning_mixin import SearchContextPlanningMixin
from .search_cross_encoder_rerank_mixin import SearchCrossEncoderRerankMixin
from .search_dialog_state import SearchDialogStateMixin
from .search_dialog_ui import SearchDialogUiMixin
from .embedding_helpers import SearchEmbeddingHelpersMixin
from .search_keyword_filter_mixin import SearchKeywordFilterMixin
from .search_keyword_continue_mixin import SearchKeywordContinueMixin
from .search_mmr_mixin import SearchMMRMixin
from .search_model_selection_mixin import SearchModelSelectionMixin
from .search_note_preview_mixin import SearchNotePreviewMixin
from .search_notifications_mixin import SearchNotificationsMixin
from .search_continuation_mixin import SearchContinuationMixin
from .search_execution_mixin import SearchExecutionMixin
from .search_image_attachment_mixin import SearchImageAttachmentMixin
from .search_orchestration_mixin import SearchOrchestrationMixin
from .search_progress_mixin import SearchProgressMixin
from .search_provider_calls_mixin import SearchProviderCallsMixin
from .search_analytics_mixin import SearchAnalyticsMixin
from .search_result_display_mixin import SearchResultDisplayMixin
from .search_result_explainability_mixin import SearchResultExplainabilityMixin
from .search_selection_mixin import SearchSelectionMixin
from .search_source_labels_mixin import SearchSourceLabelsMixin
from .note_content import SearchNoteContentMixin
from .query_enhancement import SearchQueryEnhancementMixin
from .search_dialog_shared import (
    _agent_debug_log,
    _prefilter_safety_reason,
    _session_debug_log,
    get_notes_content_with_col,
    get_safe_config,
)

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
from ..utils.text import unescape_string, clean_html, clean_html_for_display, reveal_cloze, semantic_chunk_text, build_searchable_note_chunks
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
    get_embedding_for_query,
    load_checkpoint,
    clear_checkpoint,
    extract_keywords_improved,
    compute_bm25_scores,
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


# Search-dialog shared helpers are imported from search_dialog_shared and
# re-exported here for compatibility with older import paths.

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

# ============================================================================
# AI Search Dialog
# ============================================================================

class AISearchDialog(
    SearchImageAttachmentMixin,
    SearchDialogUiMixin,
    SearchDialogStateMixin,
    SearchAnswerFormattingMixin,
    SearchChatMixin,
    SearchAnswerNavigationMixin,
    SearchNoteContentMixin,
    SearchEmbeddingHelpersMixin,
    SearchQueryEnhancementMixin,
    SearchAnalyticsMixin,
    SearchProgressMixin,
    SearchBrowserActionsMixin,
    SearchSelectionMixin,
    SearchNotePreviewMixin,
    SearchNotificationsMixin,
    SearchResultExplainabilityMixin,
    SearchResultDisplayMixin,
    SearchAnswerDisplayMixin,
    SearchAnswerWorkflowMixin,
    SearchAnthropicStreamingMixin,
    SearchKeywordFilterMixin,
    SearchKeywordContinueMixin,
    SearchExecutionMixin,
    SearchContinuationMixin,
    SearchOrchestrationMixin,
    SearchMMRMixin,
    SearchCrossEncoderRerankMixin,
    SearchProviderCallsMixin,
    SearchSourceLabelsMixin,
    SearchModelSelectionMixin,
    SearchContextPlanningMixin,
    QDialog,
):



    # --- Lifecycle And Window Setup ---

    def __init__(self, parent=None, review_card=None, review_note_id=None, review_context=None):
        initial_review_context = {
            "review_card": review_card,
            "review_note_id": review_note_id,
            "review_context": review_context,
        }
        search_dialog_lifecycle.initialize_search_dialog(self, parent)
        if hasattr(self, "set_review_context"):
            self.set_review_context(**initial_review_context)

    def closeEvent(self, event):
        try:
            session_memory = getattr(self, "_agentic_session_memory", None)
            if session_memory is not None and hasattr(session_memory, "clear"):
                session_memory.clear()
        except Exception:
            pass
        super().closeEvent(event)




# ============================================================================
# Search Dialog Singleton And Sidebar Controls
# ============================================================================

def toggle_sidebar_visibility(visible: bool):
    return _dialog_entrypoints.toggle_sidebar_visibility(visible, dialogs_module=sys.modules[__name__])


# ============================================================================
# Settings Dialog Compatibility Import
# ============================================================================

from .settings_dialog import SettingsDialog

# ============================================================================
# Public Dialog Entry Points
# ============================================================================

def show_search_dialog(
    initial_query=None,
    auto_search=False,
    review_card=None,
    review_note_id=None,
    review_context=None,
    search_mode=None,
):
    return _dialog_entrypoints.show_search_dialog(
        initial_query=initial_query,
        auto_search=auto_search,
        review_card=review_card,
        review_note_id=review_note_id,
        review_context=review_context,
        search_mode=search_mode,
        dialogs_module=sys.modules[__name__],
    )



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
