"""Helpers for parsing Anthropic message responses safely."""

from .log import log_debug


class AnthropicResponseTextError(ValueError):
    """Raised when an Anthropic response does not contain extractable text."""


def _response_shape(result):
    if not isinstance(result, dict):
        return {
            "top_level_type": type(result).__name__,
            "top_level_keys": [],
            "content_type": None,
            "content_count": None,
            "content_types": [],
        }

    content = result.get("content")
    shape = {
        "top_level_type": type(result).__name__,
        "top_level_keys": sorted(str(key) for key in result.keys()),
        "content_type": type(content).__name__ if content is not None else "missing",
        "content_count": len(content) if isinstance(content, list) else None,
        "content_types": [],
    }

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                shape["content_types"].append(str(block.get("type", "missing")))
            else:
                shape["content_types"].append(type(block).__name__)

    return shape


def extract_anthropic_text(result, source="Anthropic"):
    """Join all text blocks in an Anthropic messages response.

    Anthropic usually returns {"content": [{"type": "text", "text": "..."}]},
    but error or tool-oriented responses may omit content, return non-list
    content, or contain non-text blocks. Keep all of those failures explicit.
    """

    content = result.get("content") if isinstance(result, dict) else None
    texts = []

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

    if texts:
        return "\n".join(texts)

    shape = _response_shape(result)
    log_debug(f"{source} response text extraction failed: {shape}")
    raise AnthropicResponseTextError(
        f"{source} response contained no text blocks; response_shape={shape}"
    )
