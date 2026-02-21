"""Search dialog UI components: table delegates for results display."""

from aqt.qt import (
    QStyledItemDelegate,
    Qt,
    QStyleOptionViewItem,
    QColor,
)


class RelevanceBarDelegate(QStyledItemDelegate):
    """Delegate to show Relevance as a progress bar + percentage."""

    def paint(self, painter, option, index):
        if not index.isValid():
            super().paint(painter, option, index)
            return
        value = index.data(Qt.ItemDataRole.DisplayRole)
        try:
            pct = max(0, min(100, int(float(value))))
        except (TypeError, ValueError):
            pct = 0
        rect = option.rect
        margin = 2
        bar_h = max(4, rect.height() // 3)
        bar_y_off = (rect.height() - bar_h) // 2
        bar_rect = rect.adjusted(margin, bar_y_off, -margin, -(rect.height() - bar_y_off - bar_h))
        if bar_rect.width() > 4:
            painter.save()
            painter.fillRect(bar_rect, QColor("#ecf0f1"))
            fill_w = int(bar_rect.width() * pct / 100)
            if fill_w > 0:
                fill_color = QColor("#27ae60") if pct >= 80 else (QColor("#f39c12") if pct >= 50 else QColor("#e74c3c"))
                painter.fillRect(bar_rect.x(), bar_rect.y(), fill_w, bar_rect.height(), fill_color)
            painter.restore()
        opt = QStyleOptionViewItem(option)
        opt.text = f"{pct}%"
        opt.displayAlignment = Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        super().paint(painter, opt, index)

    def displayText(self, value, locale):
        if value is not None:
            try:
                return f"{int(float(value))}%"
            except (TypeError, ValueError):
                pass
        return super().displayText(value, locale)
