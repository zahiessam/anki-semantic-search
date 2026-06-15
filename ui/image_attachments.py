"""Composer image attachment helpers."""

import base64
import os
import re
import tempfile

from aqt.qt import QImage, Qt

try:
    from ..utils import log_debug
except ImportError:  # test imports may load ui as a top-level package
    from utils import log_debug


MAX_IMAGE_FILE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1024
ATTACHMENT_THUMBNAIL_SIZE = 64
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
IMAGE_SUPPORT_ERROR = (
    "This model does not support image input. Choose a vision-capable model in Settings, "
    "or remove the image and try again."
)
IMAGE_REFUSAL_PHRASES = (
    "i cannot",
    "i'm unable",
    "i can't",
    "as an ai",
    "i don't",
    "i am not able",
)


def _qt_keep_aspect_ratio():
    return getattr(getattr(Qt, "AspectRatioMode", Qt), "KeepAspectRatio")


def _qt_smooth_transformation():
    return getattr(getattr(Qt, "TransformationMode", Qt), "SmoothTransformation")


def image_mime_for_path(path):
    ext = os.path.splitext(path or "")[1].lower()
    return SUPPORTED_IMAGE_MIME.get(ext)


def validate_image_file_path(path):
    if not path or not os.path.isfile(path):
        raise ValueError("Image file was not found.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        if ext == ".gif":
            raise ValueError("GIF images are not supported yet. Please attach a PNG, JPG, or WebP image.")
        raise ValueError("Please attach a PNG, JPG, or WebP image.")
    size = os.path.getsize(path)
    if size > MAX_IMAGE_FILE_BYTES:
        raise ValueError("Image is larger than 5 MB. Please attach a smaller image.")
    return image_mime_for_path(path)


def _encode_qimage(image, mime_type):
    fmt = "JPG" if mime_type == "image/jpeg" else "PNG"
    suffix = ".jpg" if mime_type == "image/jpeg" else ".png"
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        if not image.save(temp_path, fmt):
            raise ValueError("Could not encode image attachment.")
        with open(temp_path, "rb") as image_file:
            raw = image_file.read()
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    return base64.b64encode(raw).decode("ascii")


def _scaled_image_if_needed(image):
    if image.width() <= MAX_IMAGE_DIMENSION and image.height() <= MAX_IMAGE_DIMENSION:
        return image
    return image.scaled(
        MAX_IMAGE_DIMENSION,
        MAX_IMAGE_DIMENSION,
        _qt_keep_aspect_ratio(),
        _qt_smooth_transformation(),
    )


def _payload_from_raw_image_file(path, mime_type):
    with open(path, "rb") as image_file:
        raw = image_file.read()
    return {
        "filename": os.path.basename(path),
        "mime_type": mime_type,
        "base64": base64.b64encode(raw).decode("ascii"),
        "source": "user_upload",
    }


def payload_from_image_file(path):
    input_mime_type = validate_image_file_path(path)
    try:
        image = QImage(path)
        if image.isNull():
            raise ValueError("Qt could not read image.")
        image = _scaled_image_if_needed(image)
        mime_type = "image/jpeg" if input_mime_type == "image/jpeg" else "image/png"
        return {
            "filename": os.path.basename(path),
            "mime_type": mime_type,
            "base64": _encode_qimage(image, mime_type),
            "source": "user_upload",
        }
    except Exception as exc:
        log_debug(f"Qt image scaling/encoding failed; attaching raw image bytes instead: {exc}")
        return _payload_from_raw_image_file(path, input_mime_type)


def image_payload_from_path_for_tests(path):
    return payload_from_image_file(path)


def payload_from_qimage(image, filename="clipboard.png", mime_type="image/png"):
    if image is None or image.isNull():
        raise ValueError("Clipboard image could not be read.")
    image = _scaled_image_if_needed(image)
    return {
        "filename": filename,
        "mime_type": mime_type,
        "base64": _encode_qimage(image, mime_type),
        "source": "user_upload",
    }


def snapshot_image_payloads(payloads):
    return [dict(payload) for payload in (payloads or []) if payload]


def normalize_review_image_payloads(payloads):
    normalized = []
    for payload in payloads or []:
        item = dict(payload)
        item.setdefault("source", "anki_field")
        normalized.append(item)
    return normalized


def has_user_upload(payloads):
    return any((payload or {}).get("source") == "user_upload" for payload in (payloads or []))


def is_image_support_error(error):
    text = str(error or "").lower()
    return any(pattern in text for pattern in (
        "does not support image input",
        "image input not supported",
        "does not support images",
        "images not supported",
        "multimodal",
        "vision",
        "invalid image",
    ))


def clean_visual_terms(raw_text):
    raw = str(raw_text or "").strip()
    lower = raw.lower()
    if not raw:
        log_debug("Image visual terms response was empty.")
        return ""
    if any(phrase in lower for phrase in IMAGE_REFUSAL_PHRASES):
        log_debug(f"Image visual terms refusal-like response: {raw[:500]}")
        return ""
    text = raw
    if text.startswith("```"):
        text = text.strip("`").strip()
    text = re.sub(r"^\s*(visual findings|findings|search terms|terms|description)\s*:\s*", "", text, flags=re.IGNORECASE)
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*[-*•\d.)]+\s*", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    text = " ".join(lines) if lines else text
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip("'")
    if not text:
        log_debug(f"Image visual terms cleaned to empty from response: {raw[:500]}")
        return ""
    if len(text) > 200:
        text = text[:200].rstrip()
    return text


def merge_visual_findings_query(*segments, image_terms=""):
    parts = [re.sub(r"\s+", " ", str(segment or "")).strip().rstrip(".") for segment in segments]
    parts = [part for part in parts if part]
    if image_terms:
        parts.append(f"Visual findings: {image_terms.strip()}")
    return ". ".join(parts).strip()
