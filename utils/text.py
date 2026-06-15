import hashlib
import html
import re

# Compiled regex patterns for performance and consistency
HTML_TAG_RE = re.compile(r'<.*?>', re.DOTALL)
DISPLAY_BLOCK_TAG_RE = re.compile(
    r'</?(?:div|p|li|ul|ol|tr|table|tbody|thead|tfoot|section|article|h[1-6])\b[^>]*>',
    re.IGNORECASE,
)
DISPLAY_BREAK_TAG_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)
DISPLAY_CELL_TAG_RE = re.compile(r'</?(?:td|th)\b[^>]*>', re.IGNORECASE)
DISPLAY_SEPARATOR_RE = re.compile(r'\s*(?:\|\s*){1,}')
DISPLAY_SPACE_RE = re.compile(r'[ \t\r\f\v]+')
DISPLAY_NEWLINE_RE = re.compile(r'\s*\n+\s*')
CLOZE_RE = re.compile(r'\{\{c\d+::(.*?)(?=\}\}|::)(?:::[^}]*)?\}\}')
INNER_CLOZE_RE = re.compile(r'\{\{c\d+::([^{}]*?)(?=\}\}|::)(?:::[^{}]*)?\}\}')
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
SEARCH_CHUNK_TARGET_SIZE = 500
SEARCH_CHUNK_OVERLAP_RATIO = 0.15
EMBEDDING_CONTENT_VERSION = 2

def unescape_string(s):
    """Convert Unicode escapes in string literals to literal characters."""
    if not isinstance(s, str):
        return s
    try:
        # Match \xHH and \uHHHH
        return s.encode('utf-8').decode('unicode-escape')
    except Exception:
        return s

def clean_html(text):
    """Strip HTML tags (for use in background thread)."""
    if not text:
        return ""
    return HTML_TAG_RE.sub('', text)

def clean_html_for_display(text):
    """Strip HTML while preserving readable display boundaries."""
    if not text:
        return ""
    cleaned = html.unescape(str(text))
    cleaned = DISPLAY_BREAK_TAG_RE.sub(" ", cleaned)
    cleaned = DISPLAY_CELL_TAG_RE.sub(" ", cleaned)
    cleaned = DISPLAY_BLOCK_TAG_RE.sub(" ", cleaned)
    cleaned = HTML_TAG_RE.sub("", cleaned)
    cleaned = DISPLAY_SEPARATOR_RE.sub(" | ", cleaned)
    cleaned = DISPLAY_SPACE_RE.sub(" ", cleaned)
    cleaned = DISPLAY_NEWLINE_RE.sub(" | ", cleaned)
    cleaned = DISPLAY_SEPARATOR_RE.sub(" | ", cleaned)
    return cleaned.strip(" |")

def reveal_cloze(text):
    """Reveal cloze deletions for display: {{cN::answer}} -> answer."""
    if not text:
        return text
    previous = str(text)
    for _ in range(20):
        revealed = INNER_CLOZE_RE.sub(r'\1', previous)
        if revealed == previous:
            revealed = CLOZE_RE.sub(r'\1', previous)
        if revealed == previous:
            return revealed
        previous = revealed
    return previous

def semantic_chunk_text(text, target_size=500, overlap_ratio=0.15):
    """Split text into chunks of ~target_size chars with sentence-aware overlap."""
    text = (text or "").strip()
    if not text or len(text) <= target_size:
        return [text] if text else []
    try:
        overlap_ratio = float(overlap_ratio)
    except Exception:
        overlap_ratio = 0.15
    overlap_size = int(target_size * max(0.0, min(0.5, overlap_ratio)))
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + target_size, len(text))
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        # Find last sentence boundary in this window
        segment = text[start:end]
        last_dot = segment.rfind('.')
        last_excl = segment.rfind('!')
        last_q = segment.rfind('?')
        last_nl = segment.rfind('\n')
        best = max(last_dot, last_excl, last_q, last_nl)
        if best >= target_size // 2:
            end = start + best + 1
            chunk = text[start:end].strip()
        else:
            chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if overlap_size <= 0:
            next_start = end
        else:
            next_start = max(0, end - overlap_size)
            if next_start <= start:
                next_start = end
            if next_start < len(text):
                boundary_start = next_start
                overlap_segment = text[next_start:end]
                first_space = overlap_segment.find(" ")
                if 0 <= first_space < max(1, overlap_size // 3):
                    boundary_start = next_start + first_space + 1
                next_start = min(boundary_start, end)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks if chunks else [text]


def _versioned_content_hash(text, content_version=EMBEDDING_CONTENT_VERSION):
    return hashlib.md5(f"v{int(content_version)}:{text}".encode()).hexdigest()


def get_canonical_search_text(
    content_parts,
    target_size=SEARCH_CHUNK_TARGET_SIZE,
    overlap_ratio=SEARCH_CHUNK_OVERLAP_RATIO,
    content_version=EMBEDDING_CONTENT_VERSION,
):
    """Build the canonical searchable text contract for indexing, lookup, and display."""
    parts = [str(part).strip() for part in (content_parts or []) if str(part or "").strip()]
    if not parts:
        return {
            "content_version": int(content_version),
            "canonical_text": "",
            "display_text": "",
            "chunks": [],
        }

    raw_content = reveal_cloze(" | ".join(parts))
    content = clean_html(raw_content).strip()
    if not content:
        return {
            "content_version": int(content_version),
            "canonical_text": "",
            "display_text": "",
            "chunks": [],
        }

    display_parts = []
    for part in parts:
        display_part = clean_html_for_display(reveal_cloze(part)).strip()
        if display_part:
            display_parts.append(display_part)
    display_content = " | ".join(display_parts) or content

    chunks = semantic_chunk_text(content, target_size, overlap_ratio)
    if len(chunks) <= 1:
        return {
            "content_version": int(content_version),
            "canonical_text": content,
            "display_text": display_content,
            "chunks": [
                {
                    "chunk_index": None,
                    "text": content,
                    "display_text": display_content,
                    "content_hash": _versioned_content_hash(content, content_version),
                    "content": content,
                    "display_content": display_content,
                    "_full_content": None,
                    "_full_display_content": None,
                }
            ],
        }

    chunk_items = []
    for chunk_idx, chunk in enumerate(chunks):
        chunk_items.append(
            {
                "chunk_index": chunk_idx,
                "text": chunk,
                "display_text": clean_html_for_display(chunk).strip() or chunk,
                "content_hash": _versioned_content_hash(chunk, content_version),
                "content": chunk,
                "display_content": clean_html_for_display(chunk).strip() or chunk,
                "_full_content": content,
                "_full_display_content": display_content,
            }
        )

    return {
        "content_version": int(content_version),
        "canonical_text": content,
        "display_text": display_content,
        "chunks": chunk_items,
    }


def build_searchable_note_chunks(content_parts, target_size=SEARCH_CHUNK_TARGET_SIZE, overlap_ratio=SEARCH_CHUNK_OVERLAP_RATIO):
    """Compatibility wrapper around the canonical searchable content contract."""
    canonical = get_canonical_search_text(content_parts, target_size, overlap_ratio)
    return {
        "content_version": canonical["content_version"],
        "content": canonical["canonical_text"],
        "canonical_text": canonical["canonical_text"],
        "display_content": canonical["display_text"],
        "display_text": canonical["display_text"],
        "chunks": canonical["chunks"],
    }


def find_text_overlap(left, right, min_chars=40, min_ratio=0.10):
    """Return exact suffix/prefix overlap length when two chunks share context."""
    left = (left or "").strip()
    right = (right or "").strip()
    if not left or not right:
        return 0
    limit = min(len(left), len(right))
    threshold = max(int(min(len(left), len(right)) * min_ratio), int(min_chars))
    if limit < threshold:
        return 0
    for size in range(limit, threshold - 1, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def merge_overlapping_note_chunks(notes, min_overlap_chars=40, min_overlap_ratio=0.10):
    """Merge same-note chunks that share exact overlap before building LLM context."""
    merged = []
    index_by_note_id = {}
    for note in notes or []:
        note_id = note.get("id") if isinstance(note, dict) else None
        if note_id is None or note_id not in index_by_note_id:
            copy = dict(note)
            if copy.get("chunk_index") is not None:
                copy["_merged_chunk_indexes"] = [copy.get("chunk_index")]
            merged.append(copy)
            if note_id is not None:
                index_by_note_id[note_id] = len(merged) - 1
            continue

        existing = merged[index_by_note_id[note_id]]
        existing_text = existing.get("content", "")
        current_text = note.get("content", "")
        overlap = find_text_overlap(
            existing_text,
            current_text,
            min_chars=min_overlap_chars,
            min_ratio=min_overlap_ratio,
        )
        if overlap:
            existing["content"] = existing_text + current_text[overlap:]
            if existing.get("display_content") and note.get("display_content"):
                display_overlap = find_text_overlap(
                    existing.get("display_content", ""),
                    note.get("display_content", ""),
                    min_chars=min_overlap_chars,
                    min_ratio=min_overlap_ratio,
                )
                if display_overlap:
                    existing["display_content"] = (
                        existing.get("display_content", "") + note.get("display_content", "")[display_overlap:]
                    )
            indexes = existing.setdefault("_merged_chunk_indexes", [])
            for idx in (existing.get("chunk_index"), note.get("chunk_index")):
                if idx is not None and idx not in indexes:
                    indexes.append(idx)
            indexes.sort()
            continue

        copy = dict(note)
        if copy.get("chunk_index") is not None:
            copy["_merged_chunk_indexes"] = [copy.get("chunk_index")]
        merged.append(copy)
    return merged
