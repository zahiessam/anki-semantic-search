"""Composer image attachment behavior for the search dialog."""

import base64
import traceback

from aqt.qt import QFileDialog, QPixmap, Qt
from aqt.utils import tooltip

from .image_attachments import (
    ATTACHMENT_THUMBNAIL_SIZE,
    payload_from_image_file,
    payload_from_qimage,
    snapshot_image_payloads,
)
from ..utils import log_debug


class SearchImageAttachmentMixin:
    def _get_composer_image_payloads(self):
        return list(getattr(self, "_composer_image_payloads", []) or [])

    def _snapshot_composer_image_payloads(self):
        return snapshot_image_payloads(self._get_composer_image_payloads())

    def _set_composer_image_attachment(self, payload):
        self._composer_image_payloads = [dict(payload)] if payload else []
        self._refresh_composer_image_chip()
        self._refresh_image_related_button()

    def _clear_composer_image_attachment(self):
        self._composer_image_payloads = []
        self._refresh_composer_image_chip()
        self._refresh_image_related_button()

    def _refresh_image_related_button(self):
        related_btn = getattr(self, "find_related_btn", None)
        if related_btn is None:
            return
        has_review = bool(getattr(self, "_review_note_id", None) or getattr(self, "_review_context_text", ""))
        has_image = bool(self._get_composer_image_payloads())
        related_btn.setVisible(has_review or has_image)
        related_btn.setEnabled(has_review or has_image)
        if has_image and not has_review:
            related_btn.setToolTip("Search for notes related to the attached image.")
        else:
            related_btn.setToolTip("Search for notes related to the current review note.")

    def _choose_composer_image_attachment(self):
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Attach image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if not path:
            return
        try:
            self._set_composer_image_attachment(payload_from_image_file(path))
        except Exception as exc:
            tooltip(str(exc))
            log_debug(f"Could not attach image file: {exc}\n{traceback.format_exc()}")

    def _attach_composer_image_from_qimage(self, image):
        try:
            self._set_composer_image_attachment(payload_from_qimage(image, filename="clipboard.png"))
            tooltip("Image attached")
        except Exception as exc:
            tooltip(str(exc))
            log_debug(f"Could not attach pasted image: {exc}\n{traceback.format_exc()}")

    def _refresh_composer_image_chip(self):
        chip = getattr(self, "composer_image_chip", None)
        if chip is None:
            return
        payloads = self._get_composer_image_payloads()
        if not payloads:
            chip.hide()
            name_label = getattr(self, "composer_image_name_label", None)
            if name_label is not None:
                name_label.setText("")
            thumb_label = getattr(self, "composer_image_thumb_label", None)
            if thumb_label is not None:
                thumb_label.clear()
            return

        payload = payloads[0]
        name = payload.get("filename") or "Attached image"
        name_label = getattr(self, "composer_image_name_label", None)
        if name_label is not None:
            name_label.setText(name)
            name_label.setToolTip(name)
        thumb_label = getattr(self, "composer_image_thumb_label", None)
        if thumb_label is not None:
            pixmap = None
            try:
                pixmap = QPixmap()
                raw = base64.b64decode(payload.get("base64") or "")
                pixmap.loadFromData(raw)
            except Exception as exc:
                log_debug(f"Could not load attachment thumbnail: {exc}")
                pixmap = None
            if pixmap is not None and not pixmap.isNull():
                try:
                    thumb_label.setPixmap(
                        pixmap.scaled(
                            ATTACHMENT_THUMBNAIL_SIZE,
                            ATTACHMENT_THUMBNAIL_SIZE,
                            getattr(getattr(Qt, "AspectRatioMode", Qt), "KeepAspectRatio"),
                            getattr(getattr(Qt, "TransformationMode", Qt), "SmoothTransformation"),
                        )
                    )
                except Exception as exc:
                    log_debug(f"Could not render attachment thumbnail: {exc}")
                    thumb_label.clear()
            else:
                thumb_label.clear()
                thumb_label.setText("IMG")
                thumb_label.setAlignment(getattr(getattr(Qt, "AlignmentFlag", Qt), "AlignCenter"))
        chip.show()
