"""Answer HTML formatting helpers."""

# ============================================================================
# Imports
# ============================================================================

import html


# ============================================================================
# Answer HTML Formatting
# ============================================================================

def spacing_styles(mode):
    if mode == 'compact':
        return {'lh': '1.2', 'p': '0.15em 0 0.3em 0', 'ul': '0.15em 0 0.3em 0', 'li': '0.08em 0'}
    if mode == 'comfortable':
        return {'lh': '1.5', 'p': '0.3em 0 0.5em 0', 'ul': '0.3em 0 0.5em 0', 'li': '0.15em 0'}
    return {'lh': '1.35', 'p': '0.2em 0 0.4em 0', 'ul': '0.2em 0 0.4em 0', 'li': '0.1em 0'}


def format_answer_html(answer, context_note_ids, spacing_mode, patterns):
    if not answer:
        return ""

    styles = spacing_styles(spacing_mode)
    citation_n_re = patterns['citation_n']
    citation_re = patterns['citation']
    md_bold_re = patterns['md_bold']
    md_unterminated_bold_re = patterns['md_unterminated_bold']
    md_bold_alt_re = patterns['md_bold_alt']
    md_header_re = patterns['md_header']
    md_highlight_re = patterns['md_highlight']

    def rich_escape(text):
        escaped = html.escape(text)
        escaped = md_bold_re.sub(r'<strong style="font-weight: bold;">\1</strong>', escaped)
        escaped = md_unterminated_bold_re.sub(r'<strong style="font-weight: bold;">\1</strong>', escaped)
        escaped = md_bold_alt_re.sub(r'<strong style="font-weight: bold;">\1</strong>', escaped)
        escaped = md_header_re.sub(r'<strong style="font-weight: bold;">\1</strong>:\2', escaped)
        escaped = md_highlight_re.sub(
            r'<span style="background-color: rgba(255,235,59,0.45); padding: 0 2px;">\1</span>',
            escaped,
        )

        for open_char, close_char in (('\uFF3B', '\uFF3D'), ('\u3010', '\u3011'), ('\u301A', '\u301B')):
            escaped = escaped.replace(open_char, '[').replace(close_char, ']')

        context_ids = context_note_ids or []
        context_len = len(context_ids)

        def cite_link(match):
            raw = match.group(1)
            pairs = []
            for part in raw.split(','):
                display = part.strip()
                numeric = display.lstrip('N').strip()
                if numeric.isdigit():
                    position = int(numeric)
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
                    links.append(
                        f'<a href="cite:{position}" style="color:#3498db;text-decoration:underline;cursor:pointer;" '
                        f'title="Single-click: highlight in list. Double-click: open in browser.">[{display}]</a>'
                    )
                else:
                    links.append(
                        f'<span title="Citation out of range (max {context_len})" style="color:#95a5a6;">'
                        f'[{display}]</span>'
                    )
            return ', '.join(links)

        escaped = citation_n_re.sub(cite_link, escaped)
        escaped = citation_re.sub(cite_link, escaped)
        return escaped

    result_lines = []
    in_list = False
    list_depth = 0

    for raw in answer.split('\n'):
        line = raw.rstrip()
        if not line:
            if in_list:
                if list_depth == 2:
                    result_lines.append('</ul></li></ul>')
                elif list_depth == 1:
                    result_lines.append('</ul>')
                in_list = False
                list_depth = 0
            result_lines.append('<br>')
            continue

        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        is_sub = indent >= 2 and (stripped.startswith('\u2022') or stripped.startswith('-') or stripped.startswith('*'))

        if stripped.startswith('##'):
            if in_list:
                if list_depth == 2:
                    result_lines.append('</ul></li></ul>')
                elif list_depth == 1:
                    result_lines.append('</ul>')
                in_list = False
                list_depth = 0
            title = rich_escape(stripped.lstrip('#').strip())
            result_lines.append(f'<p style="margin: 0.75em 0 0.4em 0; font-weight: bold; font-size: 1.22em;">● {title}</p>')
            continue

        if stripped.startswith('#') and not stripped.startswith('##'):
            if in_list:
                if list_depth == 2:
                    result_lines.append('</ul></li></ul>')
                elif list_depth == 1:
                    result_lines.append('</ul>')
                in_list = False
                list_depth = 0
            title = rich_escape(stripped.lstrip('#').strip())
            result_lines.append(f'<p style="margin: 0.75em 0 0.4em 0; font-weight: bold; font-size: 1.22em;">● {title}</p>')
            continue

        if stripped.startswith('\u2022') or stripped.startswith('-') or stripped.startswith('*'):
            content = stripped.lstrip('\u2022-*').strip()
            bullet_chars = '\u2022-*\u00b7\u25cf\u25e6\u2219\u2023'
            while content and content[0] in bullet_chars:
                content = content[1:].lstrip()
            content = rich_escape(content)
            if not content or not content.strip():
                continue

            if is_sub and in_list and list_depth == 1:
                if result_lines and result_lines[-1].strip().endswith('</li>'):
                    result_lines[-1] = (
                        result_lines[-1].rstrip().removesuffix('</li>').rstrip()
                        + '<ul style="margin: 0.2em 0 0.2em 0.8em; padding-left: 1.2em; list-style-type: circle;">'
                    )
                else:
                    result_lines.append('<ul style="margin: 0.2em 0 0.2em 0.8em; padding-left: 1.2em; list-style-type: circle;">')
                result_lines.append(f'<li style="margin: {styles["li"]};">{content}</li>')
                list_depth = 2
            elif is_sub and in_list and list_depth == 2:
                result_lines.append(f'<li style="margin: {styles["li"]};">{content}</li>')
            elif not in_list:
                result_lines.append(f'<ul style="margin: {styles["ul"]}; padding-left: 1.3em; list-style-type: disc;">')
                result_lines.append(f'<li style="margin: {styles["li"]};">{content}</li>')
                in_list = True
                list_depth = 1
            else:
                if list_depth == 2:
                    result_lines.append('</ul></li>')
                    list_depth = 1
                result_lines.append(f'<li style="margin: {styles["li"]};">{content}</li>')
            continue

        if in_list:
            if list_depth == 2:
                result_lines.append('</ul></li></ul>')
            elif list_depth == 1:
                result_lines.append('</ul>')
            in_list = False
            list_depth = 0
        result_lines.append(f'<p style="margin: {styles["p"]};">{rich_escape(line)}</p>')

    if in_list:
        if list_depth == 2:
            result_lines.append('</ul></li></ul>')
        elif list_depth == 1:
            result_lines.append('</ul>')

    html_content = ''.join(result_lines)
    link_style = (
        "<style>a { color: #3498db !important; text-decoration: underline !important; } "
        "a:hover { color: #5dade2 !important; }</style>"
    )
    return f'{link_style}<div style="line-height: {styles["lh"]}; margin: 0;">{html_content}</div>'
