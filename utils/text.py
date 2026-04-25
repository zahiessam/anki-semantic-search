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
    return CLOZE_RE.sub(r'\1', text)

def semantic_chunk_text(text, target_size=500):
    """Split text into chunks of ~target_size chars at sentence boundaries."""
    text = (text or "").strip()
    if not text or len(text) <= target_size:
        return [text] if text else []
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
        start = end
    return chunks if chunks else [text]
