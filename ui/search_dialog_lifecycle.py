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



        QPushButton {{ padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 12px; color: {theme['button_text']}; }}



        QPushButton#searchBtn {{ background-color: {theme['accent']}; border: none; }}



        QPushButton#searchBtn:hover {{ background-color: {theme['accent_hover']}; }}



        QPushButton#settingsBtn {{ background-color: {theme['muted_btn']}; border: none; padding: 6px 12px; }}



        QPushButton#settingsBtn:hover {{ background-color: {theme['muted_btn_hover']}; }}



        QPushButton#viewBtn {{ background-color: {theme['success']}; border: 2px solid {theme['success_border']}; color: {theme['button_text']}; }}



        QPushButton#viewBtn:hover {{ background-color: {theme['success_hover']}; border-color: {theme['success']}; }}



        QPushButton#viewBtn:disabled {{ background-color: {theme['muted_btn']}; border-color: {theme['panel_border']}; color: {theme['subtext']}; }}



        QPushButton#viewAllBtn {{ background-color: {theme['teal']}; border: 2px solid {theme['teal_border']}; color: {theme['button_text']}; }}



        QPushButton#viewAllBtn:hover {{ background-color: {theme['teal_hover']}; border-color: {theme['teal']}; }}



        QPushButton#viewAllBtn:disabled {{ background-color: {theme['muted_btn']}; border-color: {theme['panel_border']}; color: {theme['subtext']}; }}



        QPushButton#closeBtn {{ background-color: {theme['danger']}; border: 2px solid {theme['danger_border']}; color: {theme['button_text']}; }}



        QPushButton#closeBtn:hover {{ background-color: {theme['danger_hover']}; border-color: {theme['danger']}; }}



        QPushButton#toggleSelectBtn {{ background-color: {theme['accent']}; border: 2px solid {theme['accent_border']}; color: {theme['button_text']}; padding: 4px 8px; font-size: 11px; }}



        QPushButton#toggleSelectBtn:hover {{ background-color: {theme['accent_hover']}; border-color: {theme['accent']}; }}



        QPushButton#toggleSelectBtn:disabled {{ background-color: {theme['muted_btn']}; border-color: {theme['panel_border']}; color: {theme['subtext']}; }}



        QTableWidget {{



            border: 2px solid {theme['border']};



            border-radius: 6px;



            background-color: {theme['input_bg']};



            color: {theme['input_text']};



            gridline-color: {theme['panel_border']};



            alternate-background-color: {theme['panel_bg']};



        }}



        QTableWidget::item {{ padding: 8px; border: none; }}



        QTableWidget::item:hover {{ background-color: {theme['panel_bg']}; }}



        QTableWidget::item:selected {{ background-color: {theme['accent']}; color: {theme['selected_text']}; }}



        QTableWidget::item:selected:hover {{ background-color: {theme['accent_hover']}; }}



        QHeaderView::section {{



            background-color: {theme['header_bg']};



            color: {theme['text']};



            padding: 8px;



            border: 1px solid {theme['panel_border']};



            font-weight: bold;



        }}



        QSlider::groove:horizontal {{ border: 1px solid {theme['panel_border']}; height: 8px; background: {theme['input_bg']}; border-radius: 4px; }}



        QSlider::handle:horizontal {{ background: {theme['accent']}; border: 1px solid {theme['accent_hover']}; width: 18px; margin: -5px 0; border-radius: 9px; }}



        QProgressBar {{ text-align: center; color: palette(window-text); font-weight: bold; }}



        QProgressBar::chunk {{ background-color: {theme['accent']}; border-radius: 3px; }}



        """



    )



    dialog.setup_ui()
