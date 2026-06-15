"""AISearchDialog lifecycle and constructor setup."""

# ============================================================================
# Imports
# ============================================================================

from aqt.qt import QApplication, QDialog, QPalette, Qt

from .theme import get_addon_theme
from ..utils import load_config


# ============================================================================
# Search Dialog Lifecycle
# ============================================================================

_addon_theme = get_addon_theme

def initialize_search_dialog(dialog, parent=None):



    QDialog.__init__(dialog, parent)



    dialog.setWindowTitle("Anki Semantic Search")







    config = load_config()



    styling = config.get('styling', {})



    default_width = styling.get('window_width', 1100)



    default_height = styling.get('window_height', 800)







    dialog.setMinimumWidth(1000)



    dialog.setMinimumHeight(750)



    dialog.resize(default_width, default_height)







    # Behave like a normal window so minimize/maximize work (don't use dialog-only flags)



    dialog.setWindowFlags(



        Qt.WindowType.Window



        | Qt.WindowType.WindowMinimizeButtonHint



        | Qt.WindowType.WindowMaximizeButtonHint



        | Qt.WindowType.WindowCloseButtonHint



    )







    dialog.styling_config = styling



    dialog.sensitivity_slider = None







    palette = QApplication.palette()



    is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128



    theme = _addon_theme(is_dark)
    primary_btn_bg = "#2f81f7" if is_dark else theme["accent"]
    primary_btn_hover = "#4f9cff" if is_dark else theme["accent_hover"]
    primary_btn_border = "#1f6feb" if is_dark else theme["accent_border"]
    primary_btn_text = "#ffffff" if is_dark else theme["accent_text"]



    dialog.setStyleSheet(



        f"""



        QDialog {{ background-color: {theme['bg']}; }}



        QLabel {{ color: {theme['text']}; }}



        QLineEdit, QTextEdit, QPlainTextEdit {{



            padding: 8px;



            border: 2px solid {theme['border']};



            border-radius: 6px;



            background-color: {theme['input_bg']};



            color: {theme['input_text']};



            font-size: 13px;



        }}



        QPushButton {{ padding: 7px 12px; border-radius: 6px; font-weight: bold; font-size: 12px; color: {theme['button_text']}; border: 1px solid {theme['control_border']}; }}



        QPushButton#askAiBtn {{ background-color: transparent; border: 1px solid {theme['control_border']}; color: {theme['text']}; padding: 7px 12px; }}



        QPushButton#askAiBtn:hover {{ background-color: {theme['panel_bg']}; border-color: {theme['control_hover_border']}; }}



        QPushButton#askAiBtn:disabled {{ background-color: transparent; border-color: {theme['subtle_border']}; color: {theme['quiet_text']}; }}



        QPushButton#findRelatedBtn {{ background-color: transparent; border: 1px solid {theme['control_border']}; color: {theme['text']}; padding: 7px 12px; }}



        QPushButton#findRelatedBtn:hover {{ background-color: {theme['panel_bg']}; border-color: {theme['control_hover_border']}; }}



        QPushButton#findRelatedBtn:disabled {{ background-color: transparent; border-color: {theme['subtle_border']}; color: {theme['quiet_text']}; }}



        QPushButton#searchBtn {{ background-color: {primary_btn_bg}; border: 1px solid {primary_btn_border}; color: {primary_btn_text}; padding: 7px 14px; }}



        QPushButton#searchBtn:hover {{ background-color: {primary_btn_hover}; border-color: {primary_btn_bg}; }}



        QPushButton#searchBtn:disabled {{ background-color: {theme['muted_btn']}; border-color: {theme['subtle_border']}; color: {theme['quiet_text']}; }}



        QPushButton#clearChatBtn {{ background-color: transparent; border: 1px solid {theme['control_border']}; color: {theme['text']}; padding: 7px 12px; }}



        QPushButton#clearChatBtn:hover {{ background-color: {theme['panel_bg']}; border-color: {theme['control_hover_border']}; }}



        QPushButton#clearChatBtn:disabled {{ background-color: transparent; border-color: {theme['subtle_border']}; color: {theme['quiet_text']}; }}



        QPushButton#settingsBtn {{ background-color: transparent; border: 1px solid {theme['subtle_border']}; color: {theme['text']}; padding: 6px 12px; }}



        QPushButton#settingsBtn:hover {{ background-color: {theme['panel_bg']}; border-color: {theme['control_hover_border']}; }}



        QPushButton#viewBtn {{ background-color: transparent; border: 1px solid {theme['control_border']}; color: {theme['text']}; }}



        QPushButton#viewBtn:hover {{ background-color: {theme['panel_bg']}; border-color: {theme['control_hover_border']}; }}



        QPushButton#viewBtn:disabled {{ background-color: transparent; border-color: {theme['subtle_border']}; color: {theme['quiet_text']}; }}



        QPushButton#viewAllBtn {{ background-color: transparent; border: 1px solid {theme['control_border']}; color: {theme['text']}; }}



        QPushButton#viewAllBtn:hover {{ background-color: {theme['panel_bg']}; border-color: {theme['control_hover_border']}; }}



        QPushButton#viewAllBtn:disabled {{ background-color: transparent; border-color: {theme['subtle_border']}; color: {theme['quiet_text']}; }}



        QPushButton#showAllDynamicResultsBtn {{ background-color: transparent; border: 1px solid {theme['subtle_border']}; color: {theme['subtext']}; padding: 6px 10px; font-weight: 600; }}



        QPushButton#showAllDynamicResultsBtn:hover {{ background-color: {theme['panel_bg']}; border-color: {theme['control_hover_border']}; color: {theme['text']}; }}



        QPushButton#showAllDynamicResultsBtn:disabled {{ background-color: transparent; border-color: {theme['subtle_border']}; color: {theme['quiet_text']}; }}



        QPushButton#closeBtn {{ background-color: {theme['danger']}; border: 2px solid {theme['danger_border']}; color: {theme['button_text']}; }}



        QPushButton#closeBtn:hover {{ background-color: {theme['danger_hover']}; border-color: {theme['danger']}; }}



        QPushButton#toggleSelectBtn {{ background-color: transparent; border: 1px solid {theme['control_border']}; color: {theme['text']}; padding: 4px 8px; font-size: 11px; }}



        QPushButton#toggleSelectBtn:hover {{ background-color: {theme['panel_bg']}; border-color: {theme['control_hover_border']}; }}



        QPushButton#toggleSelectBtn:disabled {{ background-color: transparent; border-color: {theme['subtle_border']}; color: {theme['quiet_text']}; }}



        QTableWidget {{



            border: 1px solid {theme['subtle_border']};



            border-radius: 6px;



            background-color: {theme['input_bg']};



            color: {theme['input_text']};



            gridline-color: {theme['field_row_border']};



            alternate-background-color: {theme['panel_bg']};



        }}



        QTableWidget::item {{ padding: 5px 6px; border: none; }}



        QTableWidget::item:hover {{ background-color: {theme['panel_bg']}; }}



        QTableWidget::item:selected {{ background-color: {theme['section_header_checked']}; color: {theme['text']}; font-weight: 700; }}



        QTableWidget::item:selected:hover {{ background-color: {theme['panel_bg']}; }}



        QTableWidget::indicator {{

            width: 16px;

            height: 16px;

            border-radius: 3px;

            border: 1px solid {theme['control_hover_border']};

            background-color: {theme['input_bg']};

        }}



        QTableWidget::indicator:hover {{

            border: 2px solid {theme['accent']};

        }}



        QTableWidget::indicator:checked {{

            border: 2px solid {theme['selected_text']};

            background-color: {theme['accent']};

        }}



        QTableWidget::indicator:checked:selected {{

            border: 2px solid {theme['selected_text']};

            background-color: {theme['accent']};

        }}



        QHeaderView::section {{



            background-color: {theme['section_header_bg']};



            color: {theme['text']};



            padding: 6px 8px;



            border: 1px solid {theme['field_row_border']};



            font-weight: bold;



        }}



        QSlider::groove:horizontal {{ border: 1px solid {theme['subtle_border']}; height: 6px; background: {theme['input_bg']}; border-radius: 3px; }}



        QSlider::handle:horizontal {{ background: {theme['accent']}; border: 1px solid {theme['accent_border']}; width: 16px; margin: -5px 0; border-radius: 8px; }}



        QProgressBar {{ text-align: center; color: palette(window-text); font-weight: bold; }}



        QProgressBar::chunk {{ background-color: {theme['accent']}; border-radius: 3px; }}



        QToolTip {{

            background-color: {theme['panel_bg']};

            color: {theme['text']};

            border: 1px solid {theme['subtle_border']};

            border-radius: 4px;

            padding: 6px 8px;

        }}



        """



    )



    dialog.setup_ui()
