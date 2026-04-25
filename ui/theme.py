from aqt.qt import QApplication, QColor, QPalette


def _hex(color):
    return color.name()


def _blend(a, b, amount):
    """Blend QColor a toward QColor b by amount 0..1."""
    amount = max(0.0, min(1.0, amount))
    return QColor(
        round(a.red() + (b.red() - a.red()) * amount),
        round(a.green() + (b.green() - a.green()) * amount),
        round(a.blue() + (b.blue() - a.blue()) * amount),
    )


def _rgba(color, alpha):
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha})"


def get_addon_theme(is_dark=None):
    """Shared Anki-aware visual tokens for Settings/Search dialogs and sidebar."""
    palette = QApplication.palette()
    window = palette.color(QPalette.ColorRole.Window)
    base = palette.color(QPalette.ColorRole.Base)
    text = palette.color(QPalette.ColorRole.WindowText)
    input_text = palette.color(QPalette.ColorRole.Text)
    button = palette.color(QPalette.ColorRole.Button)
    highlight = palette.color(QPalette.ColorRole.Highlight)

    if is_dark is None:
        is_dark = window.lightness() < 128

    white = QColor("#ffffff")
    black = QColor("#000000")
    accent = highlight if highlight.isValid() else QColor("#3498db")
    accent_hover = _blend(accent, white if is_dark else black, 0.18)
    panel_bg = _blend(window, white if is_dark else black, 0.07)
    section_bg = _blend(window, white if is_dark else black, 0.11)
    section_header_bg = _blend(window, white if is_dark else black, 0.08)
    section_header_checked = _blend(window, white if is_dark else black, 0.13)
    control_bg = QColor("#202020") if is_dark else _blend(base, white, 0.72)
    field_row_bg = _blend(section_bg, control_bg, 0.70) if is_dark else _blend(base, black, 0.04)
    field_row_hover_bg = _blend(field_row_bg, white if is_dark else black, 0.06)
    checkbox_unchecked_bg = control_bg
    checkbox_checked_bg = accent
    control_border = QColor("#626262") if is_dark else _blend(window, black, 0.26)
    field_row_border = _blend(control_border, window, 0.25)
    control_hover_border = _blend(control_border, white if is_dark else black, 0.22)
    panel_border = _blend(window, white if is_dark else black, 0.22)
    subtle_border = _blend(window, white if is_dark else black, 0.14)
    subtext = _blend(text, window, 0.35)
    quiet_text = _blend(text, window, 0.52)
    muted_btn = _blend(button, white if is_dark else black, 0.16)
    muted_btn_hover = _blend(muted_btn, white if is_dark else black, 0.12)

    success = QColor("#2ecc71" if is_dark else "#27ae60")
    warn = QColor("#f39c12" if is_dark else "#d35400")
    danger = QColor("#e74c3c" if is_dark else "#c0392b")
    teal = QColor("#1abc9c" if is_dark else "#16a085")

    return {
        "bg": _hex(window),
        "text": _hex(text),
        "subtext": _hex(subtext),
        "input_bg": _hex(base),
        "control_bg": _hex(control_bg),
        "field_row_bg": _hex(field_row_bg),
        "field_row_hover_bg": _hex(field_row_hover_bg),
        "field_row_border": _hex(field_row_border),
        "checkbox_unchecked_bg": _hex(checkbox_unchecked_bg),
        "checkbox_checked_bg": _hex(checkbox_checked_bg),
        "input_text": _hex(input_text),
        "border": _hex(accent),
        "accent": _hex(accent),
        "accent_hover": _hex(accent_hover),
        "accent_border": _hex(_blend(accent, black, 0.18)),
        "focus_border": _hex(accent),
        "button_text": "#ffffff",
        "selected_text": _hex(palette.color(QPalette.ColorRole.HighlightedText)),
        "success": _hex(success),
        "success_hover": _hex(_blend(success, white if is_dark else black, 0.12)),
        "success_border": _hex(_blend(success, black, 0.24)),
        "warning": _hex(warn),
        "warning_hover": _hex(_blend(warn, white if is_dark else black, 0.12)),
        "danger": _hex(danger),
        "danger_hover": _hex(_blend(danger, white if is_dark else black, 0.12)),
        "danger_border": _hex(_blend(danger, black, 0.24)),
        "teal": _hex(teal),
        "teal_hover": _hex(_blend(teal, white if is_dark else black, 0.12)),
        "teal_border": _hex(_blend(teal, black, 0.24)),
        "muted_btn": _hex(muted_btn),
        "muted_btn_hover": _hex(muted_btn_hover),
        "panel_bg": _hex(panel_bg),
        "section_bg": _hex(section_bg),
        "section_header_bg": _hex(section_header_bg),
        "section_header_checked": _hex(section_header_checked),
        "panel_border": _hex(panel_border),
        "subtle_border": _hex(subtle_border),
        "control_border": _hex(control_border),
        "control_hover_border": _hex(control_hover_border),
        "header_bg": _hex(window),
        "quiet_text": _hex(quiet_text),
        "warn_bg": _rgba(warn, 0.12),
        "warn_text": _hex(warn),
    }


def settings_dialog_stylesheet(theme):
    return f"""
        QDialog {{ background-color: {theme['bg']}; }}
        QLabel {{ color: {theme['text']}; }}
        QScrollArea, QScrollArea > QWidget > QWidget {{ background-color: {theme['bg']}; border: none; }}
        QLineEdit, QTextEdit, QPlainTextEdit {{
            min-height: 20px;
            padding: 6px 8px;
            border: 1px solid {theme['control_border']};
            border-radius: 6px;
            background-color: {theme['control_bg']};
            color: {theme['input_text']};
            font-size: 12px;
        }}
        QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover {{
            border-color: {theme['control_hover_border']};
        }}
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border: 1px solid {theme['focus_border']};
        }}
        QPushButton {{
            padding: 8px 14px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
            color: {theme['button_text']};
            border: 1px solid transparent;
        }}
        QPushButton#saveBtn {{ background-color: {theme['success']}; border: none; }}
        QPushButton#saveBtn:hover {{ background-color: {theme['success_hover']}; }}
        QPushButton#cancelBtn {{ background-color: {theme['muted_btn']}; border: none; }}
        QPushButton#cancelBtn:hover {{ background-color: {theme['muted_btn_hover']}; }}
        QComboBox {{
            min-height: 20px;
            padding: 6px 8px;
            border: 1px solid {theme['control_border']};
            border-radius: 6px;
            background-color: {theme['control_bg']};
            color: {theme['input_text']};
        }}
        QComboBox:hover {{ background-color: {theme['control_bg']}; border-color: {theme['control_hover_border']}; }}
        QComboBox:focus {{ border: 1px solid {theme['focus_border']}; }}
        QComboBox QAbstractItemView {{
            background-color: {theme['control_bg']};
            color: {theme['input_text']};
            selection-background-color: {theme['accent']};
            selection-color: {theme['selected_text']};
            border: 1px solid {theme['control_border']};
        }}
        QComboBox::drop-down {{
            background-color: {theme['control_bg']};
            border-left: 1px solid {theme['control_border']};
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
        }}
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {theme['panel_border']};
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            color: {theme['text']};
        }}
        QGroupBox:disabled {{ color: {theme['subtext']}; border-color: {theme['panel_border']}; }}
        QSpinBox {{
            min-height: 20px;
            padding: 6px 8px;
            border: 1px solid {theme['control_border']};
            border-radius: 4px;
            background-color: {theme['control_bg']};
            color: {theme['input_text']};
            max-width: 160px;
        }}
        QSpinBox:hover {{ border-color: {theme['control_hover_border']}; }}
        QSpinBox:focus {{ border: 1px solid {theme['focus_border']}; }}
        QSpinBox::up-button, QSpinBox::down-button {{
            background-color: {theme['control_bg']};
            border-left: 1px solid {theme['control_border']};
        }}
        QTabWidget::pane {{ border: 1px solid {theme['subtle_border']}; background-color: {theme['bg']}; }}
        QTabBar::tab {{
            background-color: {theme['section_header_bg']};
            color: {theme['text']};
            padding: 9px 18px;
            border: 1px solid {theme['subtle_border']};
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        QTabBar::tab:hover {{ background-color: {theme['panel_bg']}; }}
        QTabBar::tab:selected {{
            background-color: {theme['accent']};
            color: {theme['selected_text']};
            border-color: {theme['accent']};
        }}
        QCheckBox {{
            spacing: 8px;
            padding: 4px 6px;
            border-radius: 4px;
        }}
        QCheckBox:hover {{ background-color: {theme['panel_bg']}; }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {theme['control_border']};
            border-radius: 4px;
            background-color: {theme['checkbox_unchecked_bg']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {theme['checkbox_checked_bg']};
            border-color: {theme['checkbox_checked_bg']};
        }}
    """


def settings_field_row_stylesheet(theme):
    return f"""
        QFrame#settingsFieldRow {{
            background-color: {theme['field_row_bg']};
            border: 1px solid {theme['field_row_border']};
            border-radius: 5px;
        }}
        QFrame#settingsFieldRow:hover {{
            background-color: {theme['field_row_hover_bg']};
            border-color: {theme['control_border']};
        }}
        QFrame#settingsFieldRow:disabled {{
            color: {theme['subtext']};
            border-color: {theme['subtle_border']};
        }}
    """


def collapsible_section_button_stylesheet(theme):
    return f"""
        QPushButton {{
            text-align: left;
            font-weight: bold;
            font-size: 14px;
            padding: 11px 16px;
            background-color: {theme['section_header_bg']};
            color: {theme['text']};
            border: 1px solid {theme['panel_border']};
            border-radius: 6px;
            margin-top: 8px;
        }}
        QPushButton:hover {{
            background-color: {theme['section_header_checked']};
            border-color: {theme['control_border']};
        }}
        QPushButton:checked {{
            border-bottom-left-radius: 0px;
            border-bottom-right-radius: 0px;
            background-color: {theme['section_header_checked']};
            border-color: {theme['panel_border']};
        }}
    """


def collapsible_section_content_stylesheet(theme):
    return f"""
        QWidget {{
            background-color: {theme['section_bg']};
            border: 1px solid {theme['panel_border']};
            border-top: none;
            border-bottom-left-radius: 6px;
            border-bottom-right-radius: 6px;
        }}
    """


def settings_button_style(theme, variant="muted"):
    variants = {
        "primary": ("accent", "accent_hover", "subtle_border"),
        "accent": ("accent", "accent_hover", "subtle_border"),
        "muted": ("muted_btn", "muted_btn_hover", "subtle_border"),
        "success": ("success", "success_hover", "success_border"),
        "warning": ("warning", "warning_hover", "subtle_border"),
        "danger": ("danger", "danger_hover", "danger_border"),
    }
    bg_key, hover_key, border_key = variants.get(variant, variants["muted"])
    return (
        "QPushButton { "
        f"background-color: {theme[bg_key]}; color: {theme['button_text']}; "
        f"padding: 8px; font-weight: bold; border: 1px solid {theme[border_key]}; "
        "} "
        f"QPushButton:hover {{ background-color: {theme[hover_key]}; }}"
    )


def settings_status_label_style(theme, state="info"):
    colors = {
        "success": theme["success"],
        "ready": theme["success"],
        "warning": theme["warning"],
        "error": theme["danger"],
        "danger": theme["danger"],
        "info": theme["accent"],
    }
    color = colors.get(state, theme["accent"])
    return (
        f"padding: 10px; border: 1px solid {color}; border-radius: 4px; "
        f"background: {theme['panel_bg']}; color: {color}; font-weight: bold;"
    )


def settings_text_style(theme, role="body"):
    styles = {
        "heading": f"font-size: 17px; font-weight: bold; color: {theme['text']}; margin-bottom: 4px;",
        "section_heading": f"font-size: 14px; font-weight: bold; color: {theme['text']};",
        "subtitle": f"font-size: 12px; color: {theme['subtext']}; margin-bottom: 10px;",
        "hint": f"font-size: 10px; color: {theme['subtext']};",
        "subtle": f"color: {theme['subtext']};",
        "summary": f"background-color: {theme['header_bg']}; color: {theme['accent']}; padding: 8px; border-radius: 5px; font-weight: bold;",
        "body": f"color: {theme['text']};",
    }
    return styles.get(role, styles["body"])


def settings_panel_style(theme, variant="panel"):
    if variant == "warning":
        return f"font-size: 11px; color: {theme['warn_text']}; margin-bottom: 10px; padding: 8px; background-color: {theme['warn_bg']}; border-radius: 4px;"
    if variant == "index":
        return f"QFrame {{ background-color: {theme['section_bg']}; border-radius: 8px; border: 1px solid {theme['accent']}; }}"
    return f"background-color: {theme['panel_bg']}; color: {theme['text']}; padding: 10px; border: 1px solid {theme['panel_border']}; border-radius: 5px;"
