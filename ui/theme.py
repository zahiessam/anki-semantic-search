from aqt.qt import QApplication, QPalette

def get_addon_theme(is_dark=None):
    """Shared visual tokens for Settings/Search dialogs and sidebar."""
    if is_dark is None:
        palette = QApplication.palette()
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128

    if is_dark:
        return {
            "bg": "#1e1e1e",
            "text": "#e0e0e0",
            "subtext": "#95a5a6",
            "input_bg": "#2d2d2d",
            "input_text": "#e0e0e0",
            "border": "#3498db",
            "accent": "#3498db",
            "accent_hover": "#5dade2",
            "success": "#27ae60",
            "success_hover": "#2ecc71",
            "muted_btn": "#555555",
            "muted_btn_hover": "#777777",
            "panel_bg": "#252525",
            "panel_border": "#3d3d3d",
            "header_bg": "#1e1e1e",
            "warn_bg": "rgba(230, 126, 34, 0.1)",
            "warn_text": "#e67e22",
        }
    return {
        "bg": "#f5f5f5",
        "text": "#2c3e50",
        "subtext": "#7f8c8d",
        "input_bg": "#ffffff",
        "input_text": "#1a1a1a",
        "border": "#3498db",
        "accent": "#3498db",
        "accent_hover": "#5dade2",
        "success": "#27ae60",
        "success_hover": "#229954",
        "muted_btn": "#95a5a6",
        "muted_btn_hover": "#7f8c8d",
        "panel_bg": "#e9eef3",
        "panel_border": "#c8d2dc",
        "header_bg": "#ffffff",
        "warn_bg": "rgba(243, 156, 18, 0.1)",
        "warn_text": "#d35400",
    }
