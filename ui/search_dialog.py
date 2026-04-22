"""Search dialog UI components: table delegates for results display."""

from aqt.qt import (
    QStyledItemDelegate,
    Qt,
    QStyleOptionViewItem,
    QColor,
    QStyle,
    QRect,
    QPainter,
)


class RelevanceBarDelegate(QStyledItemDelegate):
    """Delegate to show Relevance as a progress bar, or percentage on hover."""

    def paint(self, painter, option, index):
        if not index.isValid():
            super().paint(painter, option, index)
            return

        # Get relevance value (0-100)
        value = index.data(Qt.ItemDataRole.DisplayRole)
        try:
            pct = max(0, min(100, int(float(value))))
        except (TypeError, ValueError):
            pct = 0

        painter.save()
        rect = option.rect

        # Check if the mouse is hovering over this specific item
        if hasattr(QStyle, 'StateFlag'):
            is_hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        else:
            is_hovered = bool(option.state & QStyle.State_MouseOver)

        if is_hovered:
            # Show percentage text alone when hovering
            painter.setPen(option.palette.text().color())
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            text = f"{pct}%"
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        else:
            # Show progress bar alone normally
            margin = 4
            bar_h = 6
            # Center vertically
            bar_y = rect.y() + (rect.height() - bar_h) // 2
            bar_rect = QRect(rect.x() + margin, bar_y, rect.width() - (2 * margin), bar_h)

            # Background
            if hasattr(QPainter, 'RenderHint'):
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            else:
                try:
                    painter.setRenderHint(QPainter.Antialiasing)
                except:
                    pass
            painter.setBrush(QColor("#2c3e50" if pct > 0 else "#34495e")) # Darker bg
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bar_rect, 3, 3)

            # Fill
            fill_w = int(bar_rect.width() * pct / 100)
            if fill_w > 5:
                # Color logic: Green (>80), Orange (>50), Red (<50)
                if pct >= 80:
                    color = QColor("#27ae60")
                elif pct >= 50:
                    color = QColor("#f39c12")
                else:
                    color = QColor("#e74c3c")

                fill_rect = QRect(bar_rect.x(), bar_rect.y(), fill_w, bar_rect.height())
                painter.setBrush(color)
                painter.drawRoundedRect(fill_rect, 3, 3)

        painter.restore()

    def helpEvent(self, event, view, option, index):
        # Ensure we catch hover events to trigger repaint
        return super().helpEvent(event, view, option, index)

    def displayText(self, value, locale):
        if value is not None:
            try:
                return f"{int(float(value))}%"
            except (TypeError, ValueError):
                pass
        return super().displayText(value, locale)


class ContentDelegate(QStyledItemDelegate):
    """Delegate to show only the first field in the row, clean and simple."""

    def paint(self, painter, option, index):
        if not index.isValid():
            super().paint(painter, option, index)
            return

        # Get full content from UserRole + 2 (display_content)
        full_content = index.data(Qt.ItemDataRole.UserRole + 2)
        if not full_content:
            full_content = index.data(Qt.ItemDataRole.DisplayRole) or ""

        painter.save()

        # Draw background if selected
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
            painter.setPen(option.palette.highlightedText().color())
        else:
            painter.setPen(option.palette.text().color())

        rect = option.rect.adjusted(4, 0, -4, 0)

        # Always show only the first field in the row
        display_text = full_content.split(" | ")[0].strip()
        metrics = painter.fontMetrics()
        elided = metrics.elidedText(display_text, Qt.TextElideMode.ElideRight, rect.width())
        painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, elided)

        painter.restore()
