# ============================================================================
# Imports
# ============================================================================

import glob
import os
import sqlite3
import subprocess
import time

try:
    import sip
except ImportError:
    try:
        from PyQt6 import sip
    except ImportError:
        try:
            from PyQt5 import sip
        except ImportError:
            sip = None

from aqt import dialogs, mw
from aqt.qt import *
from aqt.utils import showInfo, showText, tooltip

from .dependency_install import _resolve_external_python_exe, install_dependencies
from .settings_constants import (
    ANSWER_CLOUD_PROVIDERS,
    ANSWER_KEY_PROVIDER_PREFIXES,
    EMBEDDING_CLOUD_PROVIDERS,
    EMBEDDING_KEY_PROVIDER_PREFIXES,
)
from .settings_rerank_workers import RerankModelDownloadWorker, RerankModelVerifyWorker
from .theme import (
    get_addon_theme,
    settings_button_style,
    settings_panel_style,
    settings_status_label_style,
    settings_text_style,
)
from .widgets import CollapsibleSection, apply_setting_row_tooltip, settings_field_row, sync_setting_row_tooltips
from ..core.cloud_diagnostics import (
    classify_provider_error,
    provider_status_message,
    test_cloud_answer_connection,
)
from ..core.engine import (
    analyze_note_eligibility,
    clear_checkpoint,
    count_notes_matching_config,
    get_deck_names,
    get_embedding_engine_id,
    get_embedding_for_query,
    get_models_with_fields,
    get_notes_count_per_deck,
    get_notes_count_per_model,
    get_ollama_models,
    load_checkpoint,
    load_embedding_engine_counts,
    load_embeddings_bulk,
    make_embedding_scope_id,
    _normalize_ollama_base_url,
)
from ..core.workers import EmbeddingWorker, RerankCheckWorker
from ..utils import (
    EmbeddingsTabMessages,
    format_partial_failure_completion,
    format_partial_failure_progress,
    get_effective_embedding_config,
    get_embeddings_db_path,
    get_embeddings_storage_path_for_read,
    get_retrieval_config,
    load_config,
    log_debug,
    save_config,
    validate_embedding_config,
)
from ..utils.config import (
    DEFAULT_RERANK_MODEL,
    RERANK_TIMEOUT_SECONDS_DEFAULT,
    RERANK_TIMEOUT_SECONDS_MAX,
    RERANK_TIMEOUT_SECONDS_MIN,
    RERANK_TOP_K_DEFAULT,
    get_rerank_config,
)

_addon_theme = get_addon_theme


def count_indexed_eligible_notes_for_engine(engine_id, eligible_ids):
    """Count eligible notes that already have an embedding for this engine."""
    if not eligible_ids:
        return 0
    db_path = get_embeddings_db_path()
    if not os.path.exists(db_path):
        return 0
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            indexed = 0
            eligible_ids = list(eligible_ids)
            for start in range(0, len(eligible_ids), 900):
                chunk = eligible_ids[start:start + 900]
                placeholders = ",".join("?" for _ in chunk)
                params = [engine_id] + chunk
                indexed += conn.execute(
                    f"SELECT COUNT(DISTINCT note_id) FROM embeddings "
                    f"WHERE engine_id = ? AND note_id IN ({placeholders})",
                    params,
                ).fetchone()[0]
            return indexed
        finally:
            conn.close()
    except Exception as exc:
        log_debug(f"Could not count indexed eligible notes: {exc}")
        return 0

class SettingsEmbeddingConnectionMixin:
    def _test_embedding_connection(self):



        """Test connection to the selected embedding engine (Voyage, OpenAI, Cohere, or Ollama). Shows pass/fail with latency."""



        import time



        test_text = "Test connection"



        # explanation: test the effective embedding config derived from the new API-tab controls.
        config = load_config()

        sc = dict(config.get("search_config") or {})

        sc.update(self._save_embedding_settings())

        config["search_config"] = sc

        config = self._config_with_current_answer_provider(config)

        valid, validation_message = validate_embedding_config(config)

        if not valid:

            showInfo(validation_message)

            return

        effective_config = get_effective_embedding_config(config)

        sc = effective_config.get("search_config") or {}

        engine = sc.get("embedding_engine") or "voyage"

        resolved_url = ""
        resolved_model = ""
        if engine == "ollama":
            resolved_url = (sc.get("ollama_base_url") or "http://localhost:11434").strip()
            resolved_model = (sc.get("ollama_embed_model") or "nomic-embed-text").strip()
        elif engine == "local_openai":
            resolved_url = (sc.get("local_llm_url") or "http://localhost:1234/v1").strip()
            resolved_model = (sc.get("local_llm_model") or "text-embedding-3-small").strip()

        config = effective_config



        if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



            self.embedding_status_label.setText("Testing connection...")



            QApplication.processEvents()



        try:



            t0 = time.perf_counter()



            embedding = get_embedding_for_query(test_text, config=config)



            elapsed_ms = int((time.perf_counter() - t0) * 1000)



            dim = len(embedding) if embedding else 0



            if embedding and dim > 0:



                engine_names = {"ollama": "Ollama", "voyage": "Voyage AI", "openai": "OpenAI", "cohere": "Cohere"}



                engine_name = engine_names.get(engine, engine)



                if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                    status_label = self.embedding_status_label

                else:

                    status_label = None

                self._show_connection_test_result(
                    success=True,
                    title=f"Embedding connection OK - {engine_name}",
                    details=f"Dimension: {dim} | Latency: {elapsed_ms} ms",
                    status_label=status_label,
                    status_text=(
                        f"{engine_name} OK ({elapsed_ms} ms)\n\n"
                        f"URL: {resolved_url}\n"
                        f"Model: {resolved_model}"
                        if resolved_model
                        else f"{engine_name} OK ({elapsed_ms} ms)"
                    ),
                )



            else:



                if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                    status_label = self.embedding_status_label

                else:

                    status_label = None

                self._show_connection_test_result(
                    success=False,
                    title="Embedding connection returned an empty embedding",
                    details="Check your engine settings (URL/model or API key) and try again.",
                    status_label=status_label,
                    status_text="Empty embedding",
                )



        except Exception as e:



            if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                status_label = self.embedding_status_label

            else:

                status_label = None



            if engine == "ollama":



                hint = "Make sure Ollama is running (ollama serve) and the model is pulled (e.g. ollama pull nomic-embed-text)."



            elif engine == "openai":



                hint = "Enter your OpenAI API key above (or set OPENAI_API_KEY) and check internet access."



            elif engine == "cohere":



                hint = "Enter your Cohere API key above (or set COHERE_API_KEY) and check internet access."



            else:



                hint = "Enter your API key above (or set the provider's env var) and check internet access."



            diagnostic_status = classify_provider_error(None, str(e), str(e))
            diagnostic_message = provider_status_message(diagnostic_status)

            self._show_connection_test_result(
                success=False,
                title="Embedding connection failed",
                details=f"{diagnostic_message}\n\nError: {e}\n\n{hint}",
                status_label=status_label,
                status_text=diagnostic_message if diagnostic_status != "server_error" else "Test failed",
            )


    def _count_indexed_eligible_notes(self, engine_id, eligible_ids):
        return count_indexed_eligible_notes_for_engine(engine_id, eligible_ids)


    def _clean_embedding_storage(self):
        active_db = os.path.abspath(get_embeddings_db_path())
        user_dir = os.path.dirname(active_db)
        if not os.path.isdir(user_dir):
            showInfo("No embedding storage folder was found.")
            return

        files_to_remove = []
        file_bytes_to_remove = 0
        patterns = [
            os.path.join(user_dir, "embeddings_*.db"),
            os.path.join(user_dir, "embeddings_*.db.bak-*"),
            os.path.join(user_dir, "embeddings_cache*.json"),
            os.path.join(user_dir, "embeddings_cache*.json.bak-*"),
        ]
        active_storage = os.path.abspath(get_embeddings_storage_path_for_read())
        for pattern in patterns:
            for path in glob.glob(pattern):
                abs_path = os.path.abspath(path)
                if abs_path in (active_db, active_storage):
                    continue
                if not abs_path.startswith(os.path.abspath(user_dir) + os.sep):
                    continue
                try:
                    size = os.path.getsize(abs_path)
                    files_to_remove.append((abs_path, os.path.basename(abs_path), size))
                    file_bytes_to_remove += size
                except Exception as exc:
                    log_debug(f"Embedding storage cleanup could not inspect {abs_path}: {exc}")

        deleted_note_rows = 0
        exact_duplicate_rows = 0
        delete_rowids = []
        active_db_size_before = os.path.getsize(active_db) if os.path.exists(active_db) else 0
        active_db_size_after = active_db_size_before
        if os.path.exists(active_db):
            try:
                current_note_ids = set(int(nid) for (nid,) in mw.col.db.execute("select id from notes"))
                conn = sqlite3.connect(active_db, timeout=30)
                try:
                    rows = conn.execute(
                        "SELECT rowid, engine_id, note_id, content_hash, timestamp FROM embeddings"
                    ).fetchall()
                    newest_by_exact_embedding = {}
                    for rowid, engine_id, note_id, content_hash, timestamp in rows:
                        note_id = int(note_id)
                        if note_id not in current_note_ids:
                            delete_rowids.append((rowid,))
                            deleted_note_rows += 1
                            continue
                        key = (engine_id, note_id, content_hash)
                        previous = newest_by_exact_embedding.get(key)
                        current_sort = str(timestamp or "")
                        if previous is None or current_sort >= previous[0]:
                            if previous is not None:
                                delete_rowids.append((previous[1],))
                                exact_duplicate_rows += 1
                            newest_by_exact_embedding[key] = (current_sort, rowid)
                        else:
                            delete_rowids.append((rowid,))
                            exact_duplicate_rows += 1
                finally:
                    conn.close()
            except Exception as exc:
                log_debug(f"Embedding active DB cleanup preview failed: {exc}", is_error=True)
                showInfo(f"Embedding storage cleanup preview failed:\n\n{exc}")
                return

        preview = (
            "Clean embedding storage preview\n\n"
            f"Old separate files to remove: {len(files_to_remove)}\n"
            f"Rows for deleted notes to remove: {deleted_note_rows:,}\n"
            f"True duplicate exact rows to remove: {exact_duplicate_rows:,}\n"
            f"Estimated file space before DB vacuum: {file_bytes_to_remove / (1024 * 1024):.1f} MB\n\n"
            "Valid chunk rows for long notes will be kept.\n\n"
            "This will permanently remove only the files and rows listed above.\n"
            "Proceed with cleanup?"
        )
        if not files_to_remove and not delete_rowids:
            showInfo(
                "Embedding storage cleanup preview\n\n"
                "No old separate files, deleted-note rows, or exact duplicate rows were found."
            )
            return
        reply = QMessageBox.question(
            self,
            "Confirm embedding storage cleanup",
            preview,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        removed_files = []
        removed_file_bytes = 0
        for abs_path, basename, size in files_to_remove:
            try:
                os.remove(abs_path)
                removed_files.append(basename)
                removed_file_bytes += size
            except Exception as exc:
                log_debug(f"Embedding storage cleanup could not delete {abs_path}: {exc}")

        if os.path.exists(active_db) and delete_rowids:
            try:
                conn = sqlite3.connect(active_db, timeout=30)
                try:
                    if delete_rowids:
                        conn.executemany("DELETE FROM embeddings WHERE rowid = ?", delete_rowids)
                        conn.commit()
                        conn.execute("VACUUM")
                        conn.commit()
                finally:
                    conn.close()
                active_db_size_after = os.path.getsize(active_db)
            except Exception as exc:
                log_debug(f"Embedding active DB cleanup failed: {exc}", is_error=True)
                showInfo(f"Embedding storage cleanup partly failed:\n\n{exc}")
                return

        saved_mb = (removed_file_bytes + max(0, active_db_size_before - active_db_size_after)) / (1024 * 1024)
        summary = (
            "Embedding storage cleanup complete.\n\n"
            f"Old separate files removed: {len(removed_files)}\n"
            f"Rows for deleted notes removed: {deleted_note_rows:,}\n"
            f"True duplicate exact rows removed from active DB: {exact_duplicate_rows:,}\n"
            f"Approx. space freed: {saved_mb:.1f} MB\n\n"
            "All engines and valid chunk rows inside the active DB were kept."
        )
        log_debug(
            f"Embedding storage cleanup: files={len(removed_files)}, deleted_note_rows={deleted_note_rows}, "
            f"exact_duplicate_rows={exact_duplicate_rows}, saved_mb={saved_mb:.1f}"
        )
        showInfo(summary)


