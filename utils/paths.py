"""Paths for embeddings cache and checkpoint."""
import os
from .log import log_debug


def _addon_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _profile_hash():
    """Collection-specific hash for storage paths (main thread only).

    Prefer the collection creation timestamp over the profile name/path so
    moving the Anki2 folder or renaming the profile does not orphan an
    existing embeddings database.
    """
    try:
        from aqt import mw
        import hashlib

        col = getattr(mw, "col", None) if mw else None
        if col:
            crt = getattr(col, "crt", None)
            if callable(crt):
                crt = crt()
            if not crt and getattr(col, "db", None):
                crt = col.db.scalar("select crt from col")
            if crt:
                return hashlib.md5(f"collection:{crt}".encode("utf-8")).hexdigest()[:12]

        pm = getattr(mw, "pm", None) if mw else None
        profile_name = getattr(pm, "name", None)
        if callable(profile_name):
            profile_name = profile_name()
        if profile_name:
            return hashlib.md5(str(profile_name).encode("utf-8")).hexdigest()[:12]

        if mw and getattr(mw, "col", None) and getattr(mw.col, "path", None):
            col_path = str(mw.col.path)
            parent_name = os.path.basename(os.path.dirname(col_path))
            profile_key = parent_name or col_path
            return hashlib.md5(profile_key.encode("utf-8")).hexdigest()[:12]
    except Exception:
        pass
    return None


def _user_files_dir():
    """Returns the path to the user-specific data directory, creating it if needed."""
    addon_dir = _addon_dir()
    user_dir = os.path.join(addon_dir, "user_files")
    if not os.path.exists(user_dir):
        try:
            os.makedirs(user_dir, exist_ok=True)
        except Exception:
            return addon_dir
    return user_dir


def get_embeddings_storage_path():
    """Get path to legacy JSON embeddings file (profile-specific)."""
    user_dir = _user_files_dir()
    h = _profile_hash()
    if h:
        return os.path.join(user_dir, f"embeddings_cache_{h}.json")
    return os.path.join(user_dir, "embeddings_cache.json")


def get_embeddings_db_path():
    """Get path to SQLite embeddings database (collection-specific)."""
    user_dir = _user_files_dir()
    h = _profile_hash()
    if h:
        return os.path.join(user_dir, f"embeddings_{h}.db")
    return os.path.join(user_dir, "embeddings.db")


def get_embeddings_db_path_for_read():
    """Path for loading embeddings; falls back to the largest existing DB."""
    path = get_embeddings_db_path()
    if os.path.exists(path):
        return path
    try:
        import glob
        candidates = glob.glob(os.path.join(_user_files_dir(), "embeddings_*.db"))
        candidates = [p for p in candidates if os.path.isfile(p)]
        if candidates:
            candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
            log_debug(f"Using existing embeddings DB (profile path not found): {os.path.basename(candidates[0])}")
            return candidates[0]
    except Exception as e:
        log_debug(f"Fallback DB lookup failed: {e}")
    return path


def get_embeddings_storage_path_for_read():
    """Path for loading embeddings; falls back to any existing cache if profile path missing."""
    path = get_embeddings_storage_path()
    if os.path.exists(path):
        return path
    addon_dir = _addon_dir()
    try:
        import glob
        candidates = glob.glob(os.path.join(_user_files_dir(), "embeddings_cache_*.json"))
        candidates = [p for p in candidates if not p.endswith(".tmp") and ".recovered" not in p]
        if candidates:
            if len(candidates) == 1:
                log_debug(f"Using existing embeddings cache (profile path not found): {os.path.basename(candidates[0])}")
                return candidates[0]
            base = os.path.basename(path)
            for p in candidates:
                if os.path.basename(p) == base:
                    return p
            log_debug(f"Multiple embedding caches found; using first: {os.path.basename(candidates[0])}")
            return candidates[0]
    except Exception as e:
        log_debug(f"Fallback cache lookup failed: {e}")
    return path


def get_checkpoint_path():
    """Get path to checkpoint file."""
    return os.path.join(_user_files_dir(), "embeddings_checkpoint.json")
