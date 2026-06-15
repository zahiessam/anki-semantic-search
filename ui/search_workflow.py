"""Compatibility module for historical search workflow imports.

Search workflow behavior now lives on explicit AISearchDialog mixins.
"""

from .search_anthropic_streaming_mixin import AnthropicStreamWorker

__all__ = ["AnthropicStreamWorker"]
