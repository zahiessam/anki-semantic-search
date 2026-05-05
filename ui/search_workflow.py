"""Search workflow compatibility wiring.

Historically, the add-on copied a large set of methods defined on worker classes
onto `AISearchDialog` at runtime. We keep that behavior via
`install_search_workflow_methods()`.

Implementation now lives in:
- `ui.search_workflow_streaming` (Anthropic streaming worker)
- `ui.search_workflow_methods` (the copied dialog/worker methods)
"""

from __future__ import annotations

from . import search_workflow_methods as _methods
from .search_workers import EmbeddingSearchWorker, RerankWorker
from .search_workflow_streaming import AnthropicStreamWorker


# ============================================================================
# Dynamic Method Compatibility Wiring
# ============================================================================

_AISEARCH_METHODS_FROM_WORKER = (
'_on_embedding_search_finished',
    '_on_keyword_filter_continue_done', '_on_embedding_search_error', '_on_rerank_done', '_on_get_notes_done', '_on_keyword_filter_done',
    '_perform_search_continue', 'perform_search',
    '_mmr_token_set', '_apply_mmr_diversity',
    '_on_relevance_rerank_done', '_display_answer_and_notes_after_rerank', '_on_relevance_rerank_done_stream', '_finish_anthropic_stream_display',
    '_on_ask_ai_success', '_on_ask_ai_error', '_on_ask_ai_worker_finished',
    '_rerank_with_cross_encoder', 'keyword_filter', 'keyword_filter_continue',
    '_local_context_usage_plan', '_fit_context_lines_to_token_budget',
    'ask_ai', 'call_ollama', 'call_anthropic', 'call_openai', 'call_google', 'call_openrouter',
    'call_custom', '_openai_compatible_chat_url', 'make_request', 'parse_response', '_rerank_by_relevance_to_answer', 'filter_and_display_notes', '_get_matching_terms_for_note',
    '_start_anthropic_stream', '_append_stream_chunk', '_on_anthropic_stream_done', '_on_anthropic_stream_error',
)


def _attach_methods_to_workers() -> None:
    """Attach copied method functions onto legacy worker classes."""
    for name in _AISEARCH_METHODS_FROM_WORKER:
        fn = getattr(_methods, name, None)
        if fn is not None:
            setattr(EmbeddingSearchWorker, name, fn)


_attach_methods_to_workers()


def install_search_workflow_methods(search_dialog_cls):
    for name in _AISEARCH_METHODS_FROM_WORKER:
        method = (
            getattr(EmbeddingSearchWorker, name, None)
            or getattr(RerankWorker, name, None)
            or getattr(AnthropicStreamWorker, name, None)
        )
        if method is not None:
            setattr(search_dialog_cls, name, method)


def configure_search_workflow_globals(**values):
    globals().update(values)
    _methods.__dict__.update(values)
