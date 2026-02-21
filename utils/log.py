"""Debug logging with rotation. API keys and other sensitive values are redacted."""
import os
import re
import datetime


def _scrub_sensitive(message):
    """Replace api_key / secret values in log message with ***."""
    if not isinstance(message, str):
        return str(message)
    # Redact values after keys containing 'api_key', 'secret', 'key' (when followed by a value that looks like a key)
    message = re.sub(
        r'(["\']?(?:api_key|apikey|api-key|secret|password|token)["\']?\s*[:=]\s*["\']?)([^"\'}\s]{8,})(["\']?)',
        r'\1***\3',
        message,
        flags=re.IGNORECASE
    )
    return message


def log_debug(message):
    """Write debug messages to a log file with automatic rotation. Sensitive keys are redacted."""
    try:
        message = _scrub_sensitive(message)
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_file = os.path.join(addon_dir, "debug_log.txt")
        MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
        if os.path.exists(log_file) and os.path.getsize(log_file) > MAX_LOG_SIZE:
            old_log = log_file + ".old"
            if os.path.exists(old_log):
                os.remove(old_log)
            os.rename(log_file, old_log)
            log_debug("Log file rotated due to size limit")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"Logging error: {e}")
