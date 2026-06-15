"""Settings dialog wheel-event and control-sizing helpers."""

from aqt.qt import QAbstractSpinBox, QCheckBox, QComboBox, QEvent, QScrollArea, QSlider


class SettingsWheelMixin:
    """Owns wheel forwarding and guarded Settings control sizing."""

    def eventFilter(self, obj, event):
        """Forward mouse wheel gestures to scrolling instead of mutating controls."""
        if event.type() != QEvent.Type.Wheel:
            return super().eventFilter(obj, event)

        if self._is_wheel_guarded_widget(obj):
            guarded_scroll_area = self._nearest_scroll_area(obj) or getattr(self, "_settings_scroll_area", None)
            if guarded_scroll_area and self._scroll_area_by_wheel(guarded_scroll_area, event):
                return True

        scroll_area = getattr(self, "_settings_scroll_area", None)
        tabs = getattr(self, "_settings_tabs", None)
        scroll_content = getattr(self, "_settings_scroll_content", None)
        if not scroll_area or not tabs or scroll_content is None:
            return super().eventFilter(obj, event)

        target = obj
        while target:
            if target == scroll_content or target == tabs or target == scroll_area.viewport():
                break
            target = target.parentWidget() if hasattr(target, "parentWidget") else None
        else:
            return super().eventFilter(obj, event)

        if self._scroll_area_by_wheel(scroll_area, event):
            return True

        return super().eventFilter(obj, event)

    def _install_wheel_scroll_guard(self, root):
        """Prevent touchpad/wheel gestures from changing settings controls."""
        guarded_types = (QAbstractSpinBox, QComboBox, QSlider, QCheckBox)
        self._wheel_guarded_widgets = set()
        for widget in root.findChildren(guarded_types):
            widget.installEventFilter(self)
            self._wheel_guarded_widgets.add(widget)

    def _apply_settings_control_sizing(self, root):
        """Keep Settings controls readable without letting numeric fields stretch too far."""
        for spin in root.findChildren(QAbstractSpinBox):
            if spin.maximumWidth() >= 16777215:
                spin.setMaximumWidth(170)
            spin.setMinimumWidth(90)
        for combo in root.findChildren(QComboBox):
            combo.setMinimumWidth(180)

    def _is_wheel_guarded_widget(self, obj):
        guarded = getattr(self, "_wheel_guarded_widgets", set())
        target = obj
        while target:
            if target in guarded:
                return True
            target = target.parentWidget() if hasattr(target, "parentWidget") else None
        return False

    def _nearest_scroll_area(self, widget):
        target = widget
        while target:
            if isinstance(target, QScrollArea):
                return target
            target = target.parentWidget() if hasattr(target, "parentWidget") else None
        return None

    def _scroll_area_by_wheel(self, scroll_area, event):
        if not scroll_area or not scroll_area.verticalScrollBar().isVisible():
            return False

        sb = scroll_area.verticalScrollBar()
        delta = event.pixelDelta().y() if hasattr(event, "pixelDelta") and not event.pixelDelta().isNull() else 0
        if not delta:
            delta = event.angleDelta().y() if hasattr(event, "angleDelta") else getattr(event, "delta", 0)
        if not delta:
            return False

        sb.setValue(sb.value() - delta)
        return True
