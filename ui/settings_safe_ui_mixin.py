"""Safe Qt widget access helpers for Settings."""


class SettingsSafeUiMixin:
    """Owns defensive widget reads/writes used during Settings load/save."""

    def _safe_set_checked(self, widget, value):
        """Set checked state only when the Qt wrapper still points to a live object."""
        try:
            try:
                import sip
            except ImportError:
                from PyQt6 import sip

            if widget is not None and not sip.isdeleted(widget):
                widget.setChecked(bool(value))
        except (RuntimeError, ImportError, AttributeError):
            pass

    def _safe_get_ui_value(self, attr_name, default_value):
        """Return a widget value only when the Qt wrapper is alive and readable."""
        try:
            try:
                import sip
            except ImportError:
                from PyQt6 import sip

            widget = getattr(self, attr_name, None)
            if widget is None or sip.isdeleted(widget):
                return default_value

            if hasattr(widget, "currentData") and hasattr(widget, "currentText"):
                data = widget.currentData()
                return data if data is not None else widget.currentText().strip()
            if hasattr(widget, "isChecked"):
                return bool(widget.isChecked())
            if hasattr(widget, "value"):
                return widget.value()
            if hasattr(widget, "toPlainText"):
                return widget.toPlainText().strip()
            if hasattr(widget, "text"):
                return widget.text().strip()
            return default_value
        except Exception:
            return default_value
