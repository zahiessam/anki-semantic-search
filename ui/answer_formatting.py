"""Answer HTML formatting helpers."""

# ============================================================================
# Imports
# ============================================================================

import html
import re


# ============================================================================
# Answer HTML Formatting
# ============================================================================

def spacing_styles(mode):
    if mode == 'compact':
        return {'lh': '1.2', 'p': '0.15em 0 0.3em 0', 'ul': '0.15em 0 0.3em 0', 'li': '0.08em 0'}
    if mode == 'comfortable':
        return {'lh': '1.5', 'p': '0.3em 0 0.5em 0', 'ul': '0.3em 0 0.5em 0', 'li': '0.15em 0'}
    return {'lh': '1.35', 'p': '0.2em 0 0.4em 0', 'ul': '0.2em 0 0.4em 0', 'li': '0.1em 0'}


_ANSWER_SECTION_LABELS = {
    "direct answer": "direct",
    "key points": "heading",
    "details": "heading",
    "table": "heading",
    "side note": "side",
}


_INLINE_MATH_COMMANDS = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "Delta": "Delta",
    "mu": "mu",
    "uparrow": "↑",
    "downarrow": "↓",
    "rightarrow": "→",
    "leftarrow": "←",
    "to": "→",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "pm": "±",
}


def normalize_inline_math(text):
    cleaned = html.unescape(str(text or ""))
    for command, target in _INLINE_MATH_COMMANDS.items():
        cleaned = re.sub(rf"\\+{re.escape(command)}\b", target, cleaned)
    entity_replacements = {
        "&leftarrow;": "←",
        "&rightarrow;": "→",
        "&uparrow;": "↑",
        "&downarrow;": "↓",
    }
    for source, target in entity_replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"\$([^$]{1,80})\$", r"\1", cleaned)
    cleaned = re.sub(r"\\\(([^)]{1,120})\\\)", r"\1", cleaned)
    cleaned = re.sub(r"\\\[([^]]{1,240})\\\]", r"\1", cleaned)
    return cleaned


def format_markdown_answer_html(answer, spacing_mode, patterns, citation_transform=None):
    if not answer:
        return ""

    styles = spacing_styles(spacing_mode)
    md_bold_re = patterns['md_bold']
    md_unterminated_bold_re = patterns['md_unterminated_bold']
    md_bold_alt_re = patterns['md_bold_alt']
    md_header_re = patterns['md_header']
    md_highlight_re = patterns['md_highlight']
    md_code_re = patterns.get('md_code')
    md_italic_re = patterns.get('md_italic')

    def rich_escape(text):
        text = normalize_inline_math(text)
        escaped = html.escape(text)
        if md_code_re:
            escaped = md_code_re.sub(
                r'<code style="background: rgba(148, 163, 184, 0.18); padding: 1px 4px; border-radius: 4px;">\1</code>',
                escaped,
            )
        escaped = md_bold_re.sub(r'<strong style="font-weight: bold;">\1</strong>', escaped)
        escaped = md_unterminated_bold_re.sub(r'<strong style="font-weight: bold;">\1</strong>', escaped)
        escaped = md_bold_alt_re.sub(r'<strong style="font-weight: bold;">\1</strong>', escaped)
        if md_italic_re:
            escaped = md_italic_re.sub(r'<em>\1</em>', escaped)
        escaped = md_header_re.sub(r'<strong style="font-weight: bold;">\1</strong>:\2', escaped)
        escaped = md_highlight_re.sub(
            r'<span style="background-color: rgba(255,235,59,0.45); padding: 0 2px;">\1</span>',
            escaped,
        )

        for open_char, close_char in (('\uFF3B', '\uFF3D'), ('\u3010', '\u3011'), ('\u301A', '\u301B')):
            escaped = escaped.replace(open_char, '[').replace(close_char, ']')

        if citation_transform:
            escaped = citation_transform(escaped)
        return escaped

    def is_table_separator(line):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)

    def table_html(table_lines):
        header = [cell.strip() for cell in table_lines[0].strip().strip("|").split("|")]
        rows = [
            [cell.strip() for cell in row.strip().strip("|").split("|")]
            for row in table_lines[2:]
        ]
        out = [
                '<table style="border-collapse: collapse; margin: 0.55em 0; width: 100%; font-size: 0.96em; background-color: transparent;">',
            "<thead><tr>",
        ]
        for cell in header:
            out.append(
                '<th style="border: 1px solid rgba(148, 163, 184, 0.55); '
                'background: rgba(148, 163, 184, 0.14); padding: 5px 7px; '
                'text-align: left; font-weight: 700;">'
                + rich_escape(cell)
                + "</th>"
            )
        out.append("</tr></thead><tbody>")
        for row in rows:
            out.append("<tr>")
            for column in range(len(header)):
                cell = row[column] if column < len(row) else ""
                out.append(
                    '<td style="border: 1px solid rgba(148, 163, 184, 0.45); '
                    'padding: 5px 7px; vertical-align: top;">'
                    + rich_escape(cell)
                    + "</td>"
                )
            out.append("</tr>")
        out.append("</tbody></table>")
        return "".join(out)

    def extract_table_blocks(text):
        lines = (text or "").split("\n")
        out_lines = []
        blocks = {}
        index = 0
        table_index = 0
        while index < len(lines):
            if (
                index + 1 < len(lines)
                and "|" in lines[index]
                and "|" in lines[index + 1]
                and is_table_separator(lines[index + 1])
            ):
                table_lines = [lines[index], lines[index + 1]]
                index += 2
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    table_lines.append(lines[index])
                    index += 1
                placeholder = f"__SEMANTIC_TABLE_{table_index}__"
                blocks[placeholder] = table_html(table_lines)
                out_lines.append(placeholder)
                table_index += 1
                continue
            out_lines.append(lines[index])
            index += 1
        return "\n".join(out_lines), blocks

    answer, table_blocks = extract_table_blocks(answer)
    result_lines = []
    in_list = False
    in_ordered_list = False
    list_depth = 0
    category_sublist = False
    lines = answer.split('\n')

    def close_open_list():
        nonlocal in_list, in_ordered_list, list_depth, category_sublist
        if in_list:
            if list_depth == 2:
                result_lines.append('</ul></li></ul>')
            elif list_depth == 1:
                result_lines.append('</ul>')
        if in_ordered_list:
            result_lines.append('</ol>')
        in_list = False
        in_ordered_list = False
        list_depth = 0
        category_sublist = False

    def is_bullet_line(value):
        stripped_value = value.lstrip()
        return stripped_value.startswith('\u2022') or stripped_value.startswith('-') or stripped_value.startswith('*')

    def is_category_heading(value, index):
        stripped_value = value.strip()
        if not stripped_value or is_bullet_line(value):
            return False
        if stripped_value.startswith('#') or stripped_value in table_blocks:
            return False
        if stripped_value.lower().startswith(("side note", "relevant_notes")):
            return False
        if len(stripped_value) > 80 or re.search(r"[.;!?]$", stripped_value):
            return False
        plain = re.sub(r"^\*\*(.*?)\*\*$", r"\1", stripped_value).strip()
        if not plain or len(plain.split()) > 7:
            return False
        next_index = index + 1
        return next_index < len(lines) and is_bullet_line(lines[next_index])

    def section_line(value):
        stripped_value = value.strip()
        plain = re.sub(r"^\*\*(.*?)\*\*$", r"\1", stripped_value).strip()
        match = re.match(r"^(direct answer|key points|details|table|side note)\s*:?\s*(.*)$", plain, re.IGNORECASE)
        if not match:
            return None
        label = match.group(1).lower()
        content = match.group(2).strip()
        if not content and label not in _ANSWER_SECTION_LABELS:
            return None
        return label, content, _ANSWER_SECTION_LABELS[label]

    def render_section(label, content, role):
        display = label[0].upper() + label[1:] if label else label
        if role == "direct":
            body = rich_escape(content) if content else ""
            return (
                '<p style="margin: 0.15em 0 0.55em 0; padding: 7px 9px; '
                'border-left: 3px solid rgba(52, 152, 219, 0.62); '
                'background-color: transparent; border-radius: 5px;">'
                f'<strong style="font-weight: 700;">{display}:</strong> {body}</p>'
            )
        if role == "side":
            body = rich_escape(content) if content else ""
            return (
                '<p style="margin: 0.55em 0 0.15em 0; padding-top: 0.45em; background-color: transparent; '
                'border-top: 1px solid rgba(148, 163, 184, 0.28); '
                'color: #95a5a6; font-size: 0.95em;">'
                f'<strong style="font-weight: 700;">{display}:</strong> {body}</p>'
            )
        if content:
            return (
                '<p style="margin: 0.65em 0 0.25em 0; font-weight: 700; font-size: 1.03em; background-color: transparent;">'
                f'{rich_escape(display + ": " + content)}</p>'
            )
        return (
            '<p style="margin: 0.65em 0 0.25em 0; font-weight: 700; font-size: 1.03em; background-color: transparent;">'
            f'{rich_escape(display)}</p>'
        )

    for index, raw in enumerate(lines):
        line = raw.rstrip()
        if not line:
            close_open_list()
            if result_lines and result_lines[-1] != '<div style="height: 0.25em;"></div>':
                result_lines.append('<div style="height: 0.25em;"></div>')
            continue

        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        is_sub = indent >= 2 and is_bullet_line(line)

        if stripped in table_blocks:
            close_open_list()
            result_lines.append(table_blocks[stripped])
            continue

        section = section_line(stripped)
        if section:
            close_open_list()
            result_lines.append(render_section(*section))
            continue

        if stripped.startswith('##'):
            close_open_list()
            title = rich_escape(stripped.lstrip('#').strip())
            result_lines.append(f'<p style="margin: 0.75em 0 0.4em 0; font-weight: bold; font-size: 1.22em; background-color: transparent;">● {title}</p>')
            continue

        if stripped.startswith('#') and not stripped.startswith('##'):
            close_open_list()
            title = rich_escape(stripped.lstrip('#').strip())
            result_lines.append(f'<p style="margin: 0.75em 0 0.4em 0; font-weight: bold; font-size: 1.22em; background-color: transparent;">● {title}</p>')
            continue

        if is_category_heading(line, index):
            close_open_list()
            result_lines.append(f'<ul style="margin: {styles["ul"]}; padding-left: 1.3em; list-style-type: disc; background-color: transparent;">')
            result_lines.append(
                f'<li style="margin: {styles["li"]}; font-weight: bold; background-color: transparent;">'
                f'{rich_escape(stripped)}'
                '<ul style="margin: 0.2em 0 0.2em 0.8em; padding-left: 1.2em; list-style-type: circle; background-color: transparent;">'
            )
            in_list = True
            list_depth = 2
            category_sublist = True
            continue

        if is_bullet_line(line):
            content = stripped.lstrip('\u2022-*').strip()
            bullet_chars = '\u2022-*\u00b7\u25cf\u25e6\u2219\u2023'
            while content and content[0] in bullet_chars:
                content = content[1:].lstrip()
            if content.endswith("**") and "**" not in content[:-2]:
                content = "**" + content[:-2].strip() + "**"
            content = rich_escape(content)
            if not content or not content.strip():
                continue

            if category_sublist and in_list and list_depth == 2:
                result_lines.append(f'<li style="margin: {styles["li"]}; font-weight: normal; background-color: transparent;">{content}</li>')
            elif is_sub and in_list and list_depth == 1:
                if result_lines and result_lines[-1].strip().endswith('</li>'):
                    result_lines[-1] = (
                        result_lines[-1].rstrip().removesuffix('</li>').rstrip()
                        + '<ul style="margin: 0.2em 0 0.2em 0.8em; padding-left: 1.2em; list-style-type: circle; background-color: transparent;">'
                    )
                else:
                    result_lines.append('<ul style="margin: 0.2em 0 0.2em 0.8em; padding-left: 1.2em; list-style-type: circle; background-color: transparent;">')
                result_lines.append(f'<li style="margin: {styles["li"]}; background-color: transparent;">{content}</li>')
                list_depth = 2
            elif is_sub and in_list and list_depth == 2:
                result_lines.append(f'<li style="margin: {styles["li"]}; background-color: transparent;">{content}</li>')
            elif not in_list:
                result_lines.append(f'<ul style="margin: {styles["ul"]}; padding-left: 1.3em; list-style-type: disc; background-color: transparent;">')
                result_lines.append(f'<li style="margin: {styles["li"]}; background-color: transparent;">{content}</li>')
                in_list = True
                list_depth = 1
            else:
                if list_depth == 2:
                    result_lines.append('</ul></li>')
                    list_depth = 1
                    category_sublist = False
                result_lines.append(f'<li style="margin: {styles["li"]}; background-color: transparent;">{content}</li>')
            continue

        ordered = re.match(r"^\d+[\.)]\s+(.+)$", stripped)
        if ordered:
            if in_list:
                close_open_list()
            if not in_ordered_list:
                result_lines.append(f'<ol style="margin: {styles["ul"]}; padding-left: 1.3em; background-color: transparent;">')
                in_ordered_list = True
            result_lines.append(
                f'<li style="margin: {styles["li"]}; background-color: transparent;">'
                f'{rich_escape(ordered.group(1).strip())}</li>'
            )
            continue

        close_open_list()
        if stripped.endswith("**") and "**" not in stripped[:-2]:
            line = "**" + stripped[:-2].strip() + "**"
        result_lines.append(f'<p style="margin: {styles["p"]}; background-color: transparent;">{rich_escape(line)}</p>')

    close_open_list()

    html_content = ''.join(result_lines)
    link_style = "<style>a { color: #3498db !important; }</style>"
    if citation_transform:
        link_style = (
            "<style>"
            "a { color: #3498db !important; } "
            "a.semantic-citation { color: #8ab4f8 !important; text-decoration: none !important; } "
            "a.semantic-citation:hover { color: #d6ecff !important; border-color: rgba(93,173,226,0.72) !important; text-decoration: underline !important; }"
            "</style>"
        )
    return f'{link_style}<div style="line-height: {styles["lh"]}; margin: 0; background-color: transparent;">{html_content}</div>'


def _build_citation_transform(context_note_ids, patterns, citation_scope=None):
    citation_n_re = patterns['citation_n']
    citation_re = patterns['citation']
    context_ids = context_note_ids or []
    context_len = len(context_ids)

    def citation_position(label):
        m = re.match(r'\s*N?(\d+)[A-Za-z]?\s*$', str(label or ""))
        return int(m.group(1)) if m else None

    def cite_link(match):
        raw = match.group(1)
        pairs = []
        for part in raw.split(','):
            display = part.strip()
            position = citation_position(display)
            if position is not None:
                if 1 <= position <= context_len:
                    note_id = context_ids[position - 1]
                    pairs.append((note_id, display, position))
                else:
                    pairs.append((None, display, 0))
        if not pairs:
            return match.group(0)

        links = []
        for note_id, display, position in pairs:
            if note_id is not None:
                href = f"cite:{citation_scope}:{position}" if citation_scope else f"cite:{position}"
                links.append(
                    f'<a class="semantic-citation" href="{href}" '
                    f'style="color:#8ab4f8;background:rgba(52,152,219,0.12);'
                    f'border:1px solid rgba(52,152,219,0.32);border-radius:4px;'
                    f'padding:0 3px;font-size:0.86em;font-weight:700;'
                    f'text-decoration:none;white-space:nowrap;cursor:pointer;" '
                    f'title="Single-click: highlight in list. Double-click: open in browser.">[{display}]</a>'
                )
            else:
                links.append(
                    f'<span title="Citation out of range (max {context_len})" '
                    f'style="color:#95a5a6;background:rgba(149,165,166,0.10);'
                    f'border:1px solid rgba(149,165,166,0.24);border-radius:4px;'
                    f'padding:0 3px;font-size:0.86em;white-space:nowrap;">'
                    f'[{display}]</span>'
                )
        return ', '.join(links)

    def transform(escaped):
        escaped = citation_n_re.sub(cite_link, escaped)
        escaped = citation_re.sub(cite_link, escaped)
        return escaped

    return transform


def _default_patterns():
    return {
        'citation_n': _CITATION_N_RE,
        'citation': _CITATION_RE,
        'md_bold': _MD_BOLD_RE,
        'md_unterminated_bold': _MD_UNTERMINATED_BOLD_RE,
        'md_bold_alt': _MD_BOLD_ALT_RE,
        'md_header': _MD_HEADER_RE,
        'md_highlight': _MD_HIGHLIGHT_RE,
        'md_code': _MD_CODE_RE,
        'md_italic': _MD_ITALIC_RE,
    }


def format_answer_html(answer, context_note_ids, spacing_mode, patterns, citation_scope=None):
    citation_transform = _build_citation_transform(context_note_ids, patterns, citation_scope=citation_scope)
    return format_markdown_answer_html(answer, spacing_mode, patterns, citation_transform=citation_transform)


def format_direct_ai_answer_html(answer, spacing_mode="normal"):
    return format_markdown_answer_html(answer, spacing_mode, _default_patterns(), citation_transform=None)


_CITATION_RE = re.compile(r'\[((?:\d+[A-Za-z]?\s*,\s*)*\d+[A-Za-z]?)\]')
_CITATION_N_RE = re.compile(r'\[N((?:\d+[A-Za-z]?\s*,\s*N?)*\d+[A-Za-z]?)\]')
_MD_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
_MD_BOLD_ALT_RE = re.compile(r'__(.+?)__')
_MD_HIGHLIGHT_RE = re.compile(r'~~(.+?)~~')
_MD_HEADER_RE = re.compile(r'^(.{1,50}):(\s*)$', re.MULTILINE)
_MD_UNTERMINATED_BOLD_RE = re.compile(r'\*\*([^*]+)$')
_MD_CODE_RE = re.compile(r'`(.+?)`')
_MD_ITALIC_RE = re.compile(r'(?<!\*)\*([^*\n]+)\*(?!\*)')


class SearchAnswerFormattingMixin:
    """Owns answer HTML formatting and spacing helpers."""

    def _spacing_styles(self):
        return spacing_styles(self.styling_config.get('answer_spacing', 'normal'))

    def format_answer(self, answer, context_note_ids=None, citation_scope=None):
        return format_answer_html(
            answer,
            context_note_ids if context_note_ids is not None else (getattr(self, '_context_note_ids', None) or []),
            self.styling_config.get('answer_spacing', 'normal'),
            {
                'citation_n': _CITATION_N_RE,
                'citation': _CITATION_RE,
                'md_bold': _MD_BOLD_RE,
                'md_unterminated_bold': _MD_UNTERMINATED_BOLD_RE,
                'md_bold_alt': _MD_BOLD_ALT_RE,
                'md_header': _MD_HEADER_RE,
                'md_highlight': _MD_HIGHLIGHT_RE,
                'md_code': _MD_CODE_RE,
                'md_italic': _MD_ITALIC_RE,
            },
            citation_scope=citation_scope,
        )
