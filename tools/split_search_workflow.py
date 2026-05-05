from __future__ import annotations

import pathlib
import re


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    ui_dir = repo_root / "ui"
    src_path = ui_dir / "search_workflow.py"

    text = src_path.read_text(encoding="utf-8")

    marker_methods_start = (
        "# END OF PART 2 - PART 3: Methods below are indented under EmbeddingSearchWorker"
    )
    idx_start = text.find(marker_methods_start)
    if idx_start == -1:
        raise RuntimeError("Could not find methods start marker")

    start_line_end = text.find("\n", idx_start)
    if start_line_end == -1:
        raise RuntimeError("Malformed file around methods start marker")

    marker_dynamic = (
        "# ============================================================================\n"
        "# Dynamic Method Compatibility Wiring\n"
        "# ============================================================================"
    )
    idx_dynamic = text.find(marker_dynamic)
    if idx_dynamic == -1:
        raise RuntimeError("Could not find dynamic wiring marker")

    header_and_streaming = text[: start_line_end + 1]
    methods_block = text[start_line_end + 1 : idx_dynamic]
    dynamic_block = text[idx_dynamic:]

    m_cls = re.search(r"\nclass AnthropicStreamWorker\(QThread\):\n", header_and_streaming)
    if not m_cls:
        raise RuntimeError("Could not find AnthropicStreamWorker class in header")
    cls_start = m_cls.start() + 1  # at 'class'
    imports_block = header_and_streaming[:cls_start]
    streaming_block = header_and_streaming[cls_start:]

    # Unindent the copied methods block by 4 spaces.
    out_lines: list[str] = []
    for ln in methods_block.splitlines(True):
        out_lines.append(ln[4:] if ln.startswith("    ") else ln)
    methods_block_unindented = "".join(out_lines)

    streaming_path = ui_dir / "search_workflow_streaming.py"
    methods_path = ui_dir / "search_workflow_methods.py"

    streaming_text = (
        '"""Anthropic streaming worker for the search dialog."""\n\n'
        "from aqt.qt import QThread, pyqtSignal\n\n"
        "from ..utils import log_debug\n\n\n"
        + streaming_block.strip()
        + "\n"
    )

    methods_text = (
        '"""Search workflow methods copied onto the AI search dialog.\n\n'
        "This module is imported by `ui.search_workflow` and its functions are\n"
        "attached onto worker classes for legacy dynamic method-copy compatibility.\n"
        '"""\n\n'
        + imports_block.strip()
        + "\n\n"
        + methods_block_unindented.strip()
        + "\n"
    )

    # Extract the tuple body from the original dynamic block.
    m_tuple = re.search(
        r"_AISEARCH_METHODS_FROM_WORKER\s*=\s*\((.*?)\)\n\n",
        dynamic_block,
        re.S,
    )
    if not m_tuple:
        raise RuntimeError("Could not extract _AISEARCH_METHODS_FROM_WORKER tuple")
    tuple_body = m_tuple.group(1).strip()

    new_search_workflow = f'''"""Search workflow compatibility wiring.

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
{tuple_body}
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
'''

    streaming_path.write_text(streaming_text, encoding="utf-8", newline="\n")
    methods_path.write_text(methods_text, encoding="utf-8", newline="\n")
    src_path.write_text(new_search_workflow, encoding="utf-8", newline="\n")

    print(f"Wrote: {streaming_path.relative_to(repo_root)}")
    print(f"Wrote: {methods_path.relative_to(repo_root)}")
    print(f"Wrote: {src_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()

