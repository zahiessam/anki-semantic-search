#!/usr/bin/env python3
"""Build a clean .ankiaddon release zip. Excludes user artifacts and secrets.
Run from the add-on directory. Output: anki_semantic_search.ankiaddon in parent folder."""

import os
import sys
import json
import zipfile
import fnmatch

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
EXCLUDE_PATTERNS = [
    "debug_log.txt",
    "debug_log.txt.old",
    "search_history.json",
    "*.tmp",
    "*.pyc",
    "__pycache__",
    ".git",
    ".gitignore",
    "build_release.py",
    "REFACTOR_PLAN.md",
    "NOTEBOOKLM_COMPARISON.md",
    "ENTRY_AND_ACTIONS.md",
    "UI_IMPROVEMENTS.md",
]
EXCLUDE_GLOBS = [
    "embeddings_cache*.json",
    "embeddings_*.db",
    "embeddings_checkpoint.json",
]

INCLUDE_FILES = [
    "__init__.py",
    "config.json",
    "meta.json",
    "manifest.json",
    "config.md",
    "README.md",
    "LICENSE",
    "requirements-optional.txt",
    "rerank_helper.py",
]
INCLUDE_DIRS = ["utils", "core", "ui"]


def should_exclude(path: str, base: str) -> bool:
    rel = os.path.relpath(path, base)
    parts = rel.split(os.sep)
    for pattern in EXCLUDE_PATTERNS:
        if pattern in parts or path.endswith(pattern):
            return True
    for g in EXCLUDE_GLOBS:
        if fnmatch.fnmatch(os.path.basename(path), g):
            return True
    return False


def build_zip():
    os.chdir(ADDON_DIR)
    out_path = os.path.join(os.path.dirname(ADDON_DIR), "anki_semantic_search.ankiaddon")

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in INCLUDE_FILES:
            p = os.path.join(ADDON_DIR, f)
            if f == "config.json":
                # Always use sanitized defaults (no secrets, no user data)
                sys.path.insert(0, ADDON_DIR)
                from utils.config import DEFAULT_CONFIG
                zf.writestr(f, json.dumps(DEFAULT_CONFIG, indent=2))
            elif f == "meta.json":
                # Sanitized meta (no API keys)
                zf.writestr(f, json.dumps({"config": {"api_key": "", "provider": "ollama"}}, indent=2))
            elif os.path.isfile(p):
                zf.write(p, f)
            else:
                print(f"Warning: {f} not found")

        for d in INCLUDE_DIRS:
            dir_path = os.path.join(ADDON_DIR, d)
            if not os.path.isdir(dir_path):
                print(f"Warning: {d} not found")
                continue
            for root, dirs, files in os.walk(dir_path):
                if "__pycache__" in root:
                    continue
                for f in files:
                    if f.endswith(".pyc"):
                        continue
                    full = os.path.join(root, f)
                    if should_exclude(full, ADDON_DIR):
                        continue
                    arcname = os.path.relpath(full, ADDON_DIR)
                    zf.write(full, arcname)

    print(f"Created: {out_path}")
    return out_path


if __name__ == "__main__":
    build_zip()
