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


def _log_file_path():
    """Returns the path to the isolated user_files log."""
    addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    user_dir = os.path.join(addon_dir, "user_files")
    if not os.path.exists(user_dir):
        try:
            os.makedirs(user_dir, exist_ok=True)
        except Exception:
            return os.path.join(addon_dir, "debug_log.txt")
    return os.path.join(user_dir, "debug_log.txt")

def clear_log():
    """Wipe the debug log file to start fresh (usually on startup)."""
    try:
        log_file = _log_file_path()
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"--- Log Cleared on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    except Exception:
        pass

def log_debug(message, is_error=False):
    """Write debug messages with optional error highlighting."""
    try:
        message = _scrub_sensitive(str(message))

        # Highlight errors for easier manual inspection
        if is_error or "Exception" in message or "Error" in message:
            message = f"\n\n[!!! ERROR !!!] {message}\n[!!! ERROR !!!]\n"

        log_file = _log_file_path()
        MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
        if os.path.exists(log_file) and os.path.getsize(log_file) > MAX_LOG_SIZE:
            old_log = log_file + ".old"
            if os.path.exists(old_log):
                try:
                    os.remove(old_log)
                except Exception:
                    pass
            try:
                os.rename(log_file, old_log)
            except Exception:
                pass
            # Start a new log file with rotation note
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Log file rotated due to size limit\n")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"Logging error: {e}")
