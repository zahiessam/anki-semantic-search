"""Focused Settings dialog layout builder mixin."""

from aqt.qt import *

from .widgets import settings_inline_row, settings_page, settings_section


class SettingsStylingTabMixin:
    """Builds the Appearance settings tab."""

    def _build_styling_tab(self, theme):
        style_tab, style_layout = settings_page(
            theme,
            "\U0001F3A8 Appearance",
            "Font sizes, window size, and layout. Optional; defaults work for most users.",
            max_width=980,
        )

        font_group = settings_section(theme, "Font sizes")
        self.font_group = font_group
        font_layout = font_group.content_layout

        self.question_font_spin = QSpinBox()
        self.question_font_spin.setRange(10, 20)
        self.question_font_spin.setValue(13)
        self.question_font_spin.setSuffix(" px")
        font_layout.addWidget(settings_inline_row(theme, "Question input font size", self.question_font_spin, 120))

        self.answer_font_spin = QSpinBox()
        self.answer_font_spin.setRange(10, 20)
        self.answer_font_spin.setValue(13)
        self.answer_font_spin.setSuffix(" px")
        font_layout.addWidget(settings_inline_row(theme, "Ask AI answer font size", self.answer_font_spin, 120))

        self.notes_font_spin = QSpinBox()
        self.notes_font_spin.setRange(10, 18)
        self.notes_font_spin.setValue(12)
        self.notes_font_spin.setSuffix(" px")
        font_layout.addWidget(settings_inline_row(theme, "Notes list font size", self.notes_font_spin, 120))

        self.label_font_spin = QSpinBox()
        self.label_font_spin.setRange(11, 18)
        self.label_font_spin.setValue(14)
        self.label_font_spin.setSuffix(" px")
        font_layout.addWidget(settings_inline_row(theme, "Label font size", self.label_font_spin, 120))

        style_layout.addWidget(font_group)

        window_group = settings_section(theme, "Window size")
        self.window_group = window_group
        window_layout = window_group.content_layout

        self.width_spin = QSpinBox()
        self.width_spin.setRange(800, 1600)
        self.width_spin.setValue(1100)
        self.width_spin.setSuffix(" px")
        window_layout.addWidget(settings_inline_row(theme, "Default window width", self.width_spin, 130))

        self.height_spin = QSpinBox()
        self.height_spin.setRange(600, 1200)
        self.height_spin.setValue(800)
        self.height_spin.setSuffix(" px")
        window_layout.addWidget(settings_inline_row(theme, "Default window height", self.height_spin, 130))

        style_layout.addWidget(window_group)

        layout_group = settings_section(theme, "Layout")
        self.layout_group = layout_group
        layout_layout = layout_group.content_layout

        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Side-by-side (answer | notes)", "side_by_side")
        self.layout_combo.addItem("Stacked (answer above notes)", "stacked")
        layout_layout.addWidget(settings_inline_row(theme, "Answer & notes", self.layout_combo, 240))

        style_layout.addWidget(layout_group)

        spacing_group = settings_section(theme, "Spacing and padding")
        self.spacing_group = spacing_group
        spacing_layout = spacing_group.content_layout

        self.section_spacing_spin = QSpinBox()
        self.section_spacing_spin.setRange(5, 20)
        self.section_spacing_spin.setValue(12)
        self.section_spacing_spin.setSuffix(" px")
        spacing_layout.addWidget(settings_inline_row(theme, "Section spacing", self.section_spacing_spin, 120))

        self.answer_spacing_combo = QComboBox()
        self.answer_spacing_combo.addItem("Compact", "compact")
        self.answer_spacing_combo.addItem("Normal", "normal")
        self.answer_spacing_combo.addItem("Comfortable", "comfortable")
        spacing_layout.addWidget(settings_inline_row(theme, "Answer line spacing", self.answer_spacing_combo, 170))

        style_layout.addWidget(spacing_group)
        style_layout.addStretch()

        return style_tab
