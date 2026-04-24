# ============================================================================
# Imports
# ============================================================================

import time
import os
import sys
try:
    import sip
except ImportError:
    try:
        from PyQt6 import sip
    except ImportError:
        try:
            from PyQt5 import sip
        except ImportError:
            # Fallback for older versions or other environments
            sip = None
import re
import datetime
import json
import hashlib
import urllib.request
import urllib.error
import subprocess
from aqt.qt import *
from aqt import mw, gui_hooks, dialogs
from aqt.utils import askUser, showInfo, showText, tooltip
from anki.hooks import addHook
from . import dialog_entrypoints as _dialog_entrypoints
from . import answer_formatting
from . import answer_navigation
from . import note_content
from . import query_enhancement

# Local imports
from .theme import get_addon_theme
from .widgets import (
    SpellCheckPlainTextEdit,
    CollapsibleSection,
    SemanticSearchSideBar,
    _get_spell_checker
)
from .search_dialog import RelevanceBarDelegate, ContentDelegate
from ..utils import (
    log_debug,
    load_config,
    save_config,
    get_config_value,
    get_effective_embedding_config,
    validate_embedding_config,
    get_addon_name,
    get_embeddings_db_path,
    get_embeddings_storage_path_for_read,
    get_checkpoint_path,
    EmbeddingsTabMessages,
    ErrorAndEngineMessages,
    format_partial_failure_progress,
    format_partial_failure_completion,
    format_dimension_mismatch_hint,
    get_search_history_queries,
    load_search_history,
    save_search_history,
    clear_search_history,
)
from ..utils.text import unescape_string, clean_html, reveal_cloze, semantic_chunk_text
from ..utils.config import VOYAGE_EMBEDDING_MODELS
from ..core.engine import (
    get_ollama_models,
    estimate_tokens,
    get_notes_count_per_model,
    get_models_with_fields,
    get_deck_names,
    get_notes_count_per_deck,
    _build_deck_query,
    count_notes_matching_config,
    analyze_note_eligibility,
    get_embedding_engine_id,
    migrate_embeddings_json_to_db,
    get_embedding_for_query,
    load_checkpoint,
    clear_checkpoint,
    extract_keywords_improved,
    compute_tfidf_scores,
    aggregate_scored_notes_by_note_id,
    load_embedding,
    save_embedding,
    flush_embedding_batch,
    save_checkpoint as core_save_checkpoint
)
from ..core.errors import _is_embedding_dimension_mismatch

# ============================================================================
# Module Configuration And Shared Helpers
# ============================================================================

_addon_theme = get_addon_theme
ADDON_NAME = get_addon_name()
_semantic_chunk_text = semantic_chunk_text


def get_safe_config(value):
    """Return config-like data with secrets redacted for logging."""
    secret_markers = ("api_key", "token", "secret", "password")

    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            key_str = str(key).lower()
            if any(marker in key_str for marker in secret_markers):
                safe[key] = "***REDACTED***" if item else item
            else:
                safe[key] = get_safe_config(item)
        return safe

    if isinstance(value, list):
        return [get_safe_config(item) for item in value]

    return value


def get_notes_content_with_col(col, config):
    """Load searchable note content in a background QueryOp using the provided collection."""
    notes_data = []
    ntf = (config or {}).get('note_type_filter', {}) or {}

    if ntf.get('fields_to_search') and not ntf.get('note_type_fields'):
        global_flds = set(f.lower() for f in ntf['fields_to_search'])
        ntf = dict(ntf)
        ntf['note_type_fields'] = {}
        for model in col.models.all():
            field_names = [fld['name'] for fld in model.get('flds', [])]
            ntf['note_type_fields'][model['name']] = [
                field_name for field_name in field_names if field_name.lower() in global_flds
            ]

    legacy_fields = None
    if not ntf:
        enabled_set = None
        search_all = False
        ntf_fields = {}
        use_first = False
        legacy_fields = {'text', 'extra'}
        fields_description = "Text & Extra"
    else:
        enabled = ntf.get('enabled_note_types')
        enabled_set = set(enabled) if enabled else None
        search_all = bool(ntf.get('search_all_fields', False))
        ntf_fields = ntf.get('note_type_fields') or {}
        use_first = bool(ntf.get('use_first_field_fallback', True))
        fields_description = "all fields" if search_all else "per-type"

    deck_q = _build_deck_query(ntf.get('enabled_decks') if ntf else None)
    note_ids = col.find_notes(deck_q) if deck_q else col.find_notes("")
    total_notes = len(note_ids)
    cache_key = (
        deck_q or '',
        frozenset(enabled_set) if enabled_set is not None else None,
        search_all,
        tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in ntf_fields.items())),
        total_notes,
    )

    model_map = {}
    for model in col.models.all():
        mid = model['id']
        model_name = model['name']
        if enabled_set is not None and model_name not in enabled_set:
            continue
        flds = model.get('flds', [])
        if search_all:
            indices = list(range(len(flds)))
        else:
            if legacy_fields is not None:
                wanted = legacy_fields
            else:
                wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))
                if not wanted and use_first and flds:
                    wanted = {flds[0]['name'].lower()}
            indices = [i for i, field in enumerate(flds) if field['name'].lower() in wanted]
        if indices:
            model_map[mid] = (model_name, indices)

    if not note_ids:
        return notes_data, fields_description, cache_key

    chunk_target = 500
    id_list = ",".join(map(str, note_ids))
    for nid, mid, flds_str in col.db.execute(f"select id, mid, flds from notes where id in ({id_list})"):
        model_info = model_map.get(mid)
        if not model_info:
            continue
        model_name, indices = model_info
        fields = (flds_str or "").split("\x1f")
        content_parts = []
        for index in indices:
            if index < len(fields):
                value = fields[index].strip()
                if value:
                    content_parts.append(value)
        if not content_parts:
            continue

        content = clean_html(" | ".join(content_parts)).strip()
        if not content:
            continue

        chunks = _semantic_chunk_text(content, chunk_target)
        if len(chunks) <= 1:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            notes_data.append({
                'id': nid,
                'content': content,
                'content_hash': content_hash,
                'model': model_name,
                'display_content': content,
            })
            continue

        for chunk_idx, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            notes_data.append({
                'id': nid,
                'content': chunk,
                'content_hash': chunk_hash,
                'model': model_name,
                'display_content': chunk,
                'chunk_index': chunk_idx,
                '_full_content': content,
            })

    return notes_data, fields_description, cache_key

# --- Regular Expression Constants ---

# Constants from original __init__.py
HTML_TAG_RE = re.compile(r'<.*?>', re.DOTALL)
CLOZE_RE = re.compile(r'\{\{c\d+::(.*?)(?=\}\}|::)(?:::[^}]*)?\}\}')
SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?\n])\s+|\n+')
CITATION_RE = re.compile(r'\[([\d,\s]+)\]')
CITATION_N_RE = re.compile(r'\[N([\d,\sN]+)\]')
MD_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
MD_BOLD_ALT_RE = re.compile(r'__(.+?)__')
MD_HIGHLIGHT_RE = re.compile(r'~~(.+?)~~')
MD_HEADER_RE = re.compile(r'^(.{1,50}):(\s*)$', re.MULTILINE)
MD_UNTERMINATED_BOLD_RE = re.compile(r'\*\*([^*]+)$')
WORD_BOUNDARY_RE = re.compile(r'\b\w+\b')
DIGIT_RE = re.compile(r'\d+')

# --- Debug Logging Helpers ---

def _session_debug_log(hypothesis_id, location, message, data=None):
    """Write NDJSON to session log file for debug."""
    try:
        entry = {
            "sessionId": "85902e",
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
        }
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(addon_dir, "user_files", "debug-85902e.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _agent_debug_log(run_id, hypothesis_id, location, message, data=None):
    """Lightweight debug logger for agent-driven investigations."""
    try:
        entry = {
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
        }
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(addon_dir, "user_files", "debug.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ============================================================================
# AI Search Dialog
# ============================================================================

class AISearchDialog(QDialog):



    # --- Lifecycle And Window Setup ---

    def __init__(self, parent=None):



        super().__init__(parent)



        self.setWindowTitle("Anki Semantic Search")







        config = load_config()



        styling = config.get('styling', {})



        default_width = styling.get('window_width', 1100)



        default_height = styling.get('window_height', 800)







        self.setMinimumWidth(1000)



        self.setMinimumHeight(750)



        self.resize(default_width, default_height)







        # Behave like a normal window so minimize/maximize work (don't use dialog-only flags)



        self.setWindowFlags(



            Qt.WindowType.Window



            | Qt.WindowType.WindowMinimizeButtonHint



            | Qt.WindowType.WindowMaximizeButtonHint



            | Qt.WindowType.WindowCloseButtonHint



        )







        self.styling_config = styling



        self.sensitivity_slider = None







        palette = QApplication.palette()



        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128



        theme = _addon_theme(is_dark)



        self.setStyleSheet(



            f"""



            QDialog {{ background-color: {theme['bg']}; }}



            QLabel {{ color: {theme['text']}; }}



            QLineEdit, QTextEdit, QPlainTextEdit {{



                padding: 8px;



                border: 2px solid {theme['border']};



                border-radius: 6px;



                background-color: {theme['input_bg']};



                color: {theme['input_text']};



                font-size: 13px;



            }}



            QPushButton {{ padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 12px; color: white; }}



            QPushButton#searchBtn {{ background-color: {theme['accent']}; border: none; }}



            QPushButton#searchBtn:hover {{ background-color: {theme['accent_hover']}; }}



            QPushButton#settingsBtn {{ background-color: {theme['muted_btn']}; border: none; padding: 6px 12px; }}



            QPushButton#settingsBtn:hover {{ background-color: {theme['muted_btn_hover']}; }}



            QPushButton#viewBtn {{ background-color: {theme['success']}; border: 2px solid #1e8449; color: white; }}



            QPushButton#viewBtn:hover {{ background-color: {theme['success_hover']}; border-color: {theme['success']}; }}



            QPushButton#viewBtn:disabled {{ background-color: {theme['muted_btn']}; border-color: {theme['panel_border']}; color: {theme['subtext']}; }}



            QPushButton#viewAllBtn {{ background-color: #16a085; border: 2px solid #117a65; color: white; }}



            QPushButton#viewAllBtn:hover {{ background-color: #1abc9c; border-color: #16a085; }}



            QPushButton#viewAllBtn:disabled {{ background-color: {theme['muted_btn']}; border-color: {theme['panel_border']}; color: {theme['subtext']}; }}



            QPushButton#closeBtn {{ background-color: #c0392b; border: 2px solid #922b21; color: white; }}



            QPushButton#closeBtn:hover {{ background-color: #e74c3c; border-color: #c0392b; }}



            QPushButton#toggleSelectBtn {{ background-color: {theme['accent']}; border: 2px solid #2980b9; color: white; padding: 4px 8px; font-size: 11px; }}



            QPushButton#toggleSelectBtn:hover {{ background-color: {theme['accent_hover']}; border-color: {theme['accent']}; }}



            QPushButton#toggleSelectBtn:disabled {{ background-color: {theme['muted_btn']}; border-color: {theme['panel_border']}; color: {theme['subtext']}; }}



            QTableWidget {{



                border: 2px solid {theme['border']};



                border-radius: 6px;



                background-color: {theme['input_bg']};



                color: {theme['input_text']};



                gridline-color: {theme['panel_border']};



                alternate-background-color: {theme['panel_bg']};



            }}



            QTableWidget::item {{ padding: 8px; border: none; }}



            QTableWidget::item:hover {{ background-color: {theme['panel_bg']}; }}



            QTableWidget::item:selected {{ background-color: {theme['accent']}; color: white; }}



            QTableWidget::item:selected:hover {{ background-color: {theme['accent_hover']}; }}



            QHeaderView::section {{



                background-color: {theme['header_bg']};



                color: {theme['text']};



                padding: 8px;



                border: 1px solid {theme['panel_border']};



                font-weight: bold;



            }}



            QSlider::groove:horizontal {{ border: 1px solid {theme['panel_border']}; height: 8px; background: {theme['input_bg']}; border-radius: 4px; }}



            QSlider::handle:horizontal {{ background: {theme['accent']}; border: 1px solid {theme['accent_hover']}; width: 18px; margin: -5px 0; border-radius: 9px; }}



            QProgressBar {{ text-align: center; color: palette(window-text); font-weight: bold; }}



            QProgressBar::chunk {{ background-color: {theme['accent']}; border-radius: 3px; }}



            """



        )



        self.setup_ui()







    # --- Search Entry Points ---

    def perform_search(self):



        """



        Bridge method so AISearchDialog always has perform_search.



        Delegates to the implementation that was originally defined on EmbeddingSearchWorker.



        """



        return EmbeddingSearchWorker.perform_search(self)







    # --- Selection Controls ---

    def toggle_select_all(self):



        """Bridge to shared toggle_select_all implementation."""



        return EmbeddingSearchWorker.toggle_select_all(self)







    def select_all_notes(self):



        """Bridge to shared select_all_notes implementation."""



        return EmbeddingSearchWorker.select_all_notes(self)







    def deselect_all_notes(self):



        """Bridge to shared deselect_all_notes implementation."""



        return EmbeddingSearchWorker.deselect_all_notes(self)







    # --- Defaults And UI Construction ---

    def reset_to_medical_defaults(self):

        """Resets all settings to high-yield medical defaults as suggested by Glutamine."""

        from aqt.utils import askUser

        if not askUser("Reset all settings to high-yield clinical defaults (Resident-approved)?"):

            return



        # 1. API & Provider Defaults

        self.answer_provider_combo.setCurrentIndex(0) # Default to Ollama (Local)

        self.local_llm_url.setText("http://localhost:1234/v1")

        self.local_llm_model.setText("llama3.2")



        # 2. Search & Embedding Defaults (Medical High-Yield)

        self.search_method_combo.setCurrentIndex(2) # Hybrid (RRF) - Best for Med

        self.min_relevance_spin.setValue(55) # Filter out noise

        self.max_results_spin.setValue(50) # Comprehensive for medical use

        self.hybrid_weight_spin.setValue(40) # Slightly favor keywords for drug names/genes

        self.strict_relevance_cb.setChecked(True)

        self.enable_query_expansion_cb.setChecked(True) # AI synonyms are great for medicine

        self.use_ai_generic_term_detection_cb.setChecked(True)



        # 3. Styling Defaults (Dark mode friendly)

        self.question_font_spin.setValue(14)

        self.answer_font_spin.setValue(13)

        self.notes_font_spin.setValue(12)

        self.layout_combo.setCurrentIndex(0) # Side-by-side



        # 4. Note Types & Fields

        self.include_all_note_types_cb.setChecked(True)

        self.include_all_decks_cb.setChecked(True)

        self.use_first_field_cb.setChecked(True)



        from aqt.utils import showInfo

        showInfo("Settings reset to Clinical High-Yield defaults. Click 'Save' to apply.")



    def setup_ui(self):



        root_layout = QHBoxLayout()



        root_layout.setSpacing(8)



        root_layout.setContentsMargins(8, 8, 8, 8)







        main_container = QWidget()



        layout = QVBoxLayout(main_container)



        section_spacing = self.styling_config.get('section_spacing', 8)



        layout.setSpacing(section_spacing)



        layout.setContentsMargins(10, 8, 10, 8)







        palette = QApplication.palette()



        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128







        theme = _addon_theme(is_dark)







        # Compact header: title + hint + scope banner + settings (shrunk)



        header_layout = QHBoxLayout()



        title_label = QLabel("\U0001F50D Anki Semantic Search")



        title_label.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {'#ffffff' if is_dark else '#2c3e50'};")



        header_layout.addWidget(title_label)



        hint_label = QLabel("Ask a question \u2192 AI answer from your notes")



        hint_label.setStyleSheet(f"font-size: 10px; color: {'#95a5a6' if is_dark else '#7f8c8d'};")



        header_layout.addWidget(hint_label)



        self.scope_banner = QLabel("")



        self.scope_banner.setStyleSheet(f"font-size: 10px; color: {'#95a5a6' if is_dark else '#7f8c8d'}; padding: 2px 8px;")



        self.scope_banner.setToolTip("Search scope. Click Settings to change note types, fields, decks.")



        self.scope_banner.setTextFormat(Qt.TextFormat.RichText)



        self.scope_banner.setOpenExternalLinks(False)



        self.scope_banner.linkActivated.connect(lambda _: self.open_settings())



        header_layout.addWidget(self.scope_banner)



        header_layout.addStretch()



        settings_btn = QPushButton("\u2699 Settings")



        settings_btn.setObjectName("settingsBtn")



        settings_btn.setToolTip("Configure API key, note types, decks, and search behavior")



        settings_btn.clicked.connect(self.open_settings)



        header_layout.addWidget(settings_btn)



        layout.addLayout(header_layout)







        # Search input (shrunk)



        search_container = QWidget()



        search_container.setStyleSheet("background-color: rgba(52, 152, 219, 0.12); border-radius: 4px; padding: 4px 6px;")



        search_layout = QVBoxLayout(search_container)



        search_layout.setSpacing(2)



        search_layout.setContentsMargins(4, 4, 4, 4)







        label_font_size = max(11, self.styling_config.get('label_font_size', 13) - 2)



        search_label = QLabel("Search:")



        search_label.setToolTip("Type a question; matching notes will be found and the AI will answer using them. Ctrl+Enter to search.")



        search_label.setStyleSheet(f"font-weight: bold; font-size: {label_font_size}px; color: {'#ffffff' if is_dark else '#2c3e50'};")



        search_layout.addWidget(search_label)







        search_input_layout = QHBoxLayout()



        try:



            self.search_input = SpellCheckPlainTextEdit()



        except Exception:



            self.search_input = QPlainTextEdit()



        self.search_input.setPlaceholderText("e.g., hypertension  or  causes of heart failure \u2014 Ctrl+Enter to search")



        self.search_input.setMinimumHeight(44)



        self.search_input.setMaximumHeight(100)



        spell_hint = " Right-click misspelled words for corrections." if _get_spell_checker() else ""



        self.search_input.setToolTip(f"Type your question. Ctrl+Enter to search.{spell_hint}")



        question_font_size = max(11, self.styling_config.get('question_font_size', 13) - 1)



        self.search_input.setStyleSheet(f"font-size: {question_font_size}px;")

        # History dropdown for recent searches

        self._search_history_model = QStringListModel(get_search_history_queries())

        self.search_history_combo = QComboBox()

        self.search_history_combo.setModel(self._search_history_model)

        self.search_history_combo.setEditable(False)

        self.search_history_combo.setMinimumWidth(80)

        self.search_history_combo.setMaximumWidth(150)

        self.search_history_combo.setToolTip("Recent searches \u2014 select to fill the search box, then click Search.")

        self.search_history_combo.activated.connect(self._on_search_history_selected)



        clear_history_btn = QPushButton("Clear history")

        clear_history_btn.setToolTip("Delete all search history")

        clear_history_btn.setMaximumWidth(90)

        clear_history_btn.clicked.connect(self._on_clear_search_history)



        self.search_btn = QPushButton("\U0001F50D Search")

        self.search_btn.setObjectName("searchBtn")

        self.search_btn.setMinimumHeight(42)

        self.search_btn.setMinimumWidth(100)

        self.search_btn.clicked.connect(self.perform_search)



        search_input_layout.addWidget(self.search_input, 4) # Give much more stretch to input

        search_input_layout.addWidget(self.search_history_combo, 1) # Keep history small

        search_input_layout.addWidget(clear_history_btn)

        search_input_layout.addWidget(self.search_btn)

        search_layout.addLayout(search_input_layout)







        # Shortcuts and History moved from sidebar to below search input



        extra_info_layout = QHBoxLayout()



        shortcuts_hint = QLabel("<b>Shortcuts:</b> Ctrl+Enter search | Ctrl+A select | Ctrl+D deselect")



        shortcuts_hint.setStyleSheet(f"font-size: 10px; color: {theme['subtext']};")



        extra_info_layout.addWidget(shortcuts_hint)



        extra_info_layout.addStretch()



        search_layout.addLayout(extra_info_layout)







        layout.addWidget(search_container)







        # Ctrl+Enter to search



        search_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)



        search_shortcut.activated.connect(self.perform_search)







        # Keyboard shortcuts for select/deselect



        select_all_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)



        select_all_shortcut.activated.connect(self.select_all_notes)



        deselect_all_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)



        deselect_all_shortcut.activated.connect(self.deselect_all_notes)







        layout.addWidget(search_container)







        # Splitter for resizable sections (side-by-side or stacked)



        layout_mode = self.styling_config.get('layout_mode', 'side_by_side')



        self.use_side_by_side = layout_mode == 'side_by_side'



        split_orientation = Qt.Orientation.Horizontal if self.use_side_by_side else Qt.Orientation.Vertical



        main_splitter = QSplitter(split_orientation)







        # AI Answer section



        answer_container = QWidget()



        answer_container.setStyleSheet("background-color: rgba(46, 204, 113, 0.12); border-radius: 6px; padding: 8px;")



        answer_layout = QVBoxLayout(answer_container)







        answer_header = QHBoxLayout()



        answer_label = QLabel("\U0001F4A1 AI Answer:")



        answer_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #27ae60;")



        answer_header.addWidget(answer_label)



        answer_header.addStretch()



        self.copy_answer_btn = QPushButton("\U0001F4CB Copy")



        self.copy_answer_btn.setMaximumWidth(80)



        self.copy_answer_btn.setToolTip("Copy AI answer (paste into Word for bullets and formatting)")



        self.copy_answer_btn.clicked.connect(self.copy_answer_to_clipboard)



        self.copy_answer_btn.setEnabled(False)



        answer_header.addWidget(self.copy_answer_btn)



        answer_layout.addLayout(answer_header)







        try:



            self.answer_box = QTextBrowser()



        except NameError:



            self.answer_box = QTextEdit()



        self.answer_box.setReadOnly(True)



        self.answer_box.setOpenExternalLinks(False)



        # Ensure link clicks are emitted (not opened) so citation links work (Ollama and all providers)



        if hasattr(self.answer_box, 'setOpenLinks'):



            self.answer_box.setOpenLinks(False)



        if hasattr(self.answer_box, 'setOpenExternalLinks'):



            self.answer_box.setOpenExternalLinks(False)



        # Explicitly enable link interaction so [1], [2] citation links are clickable



        if hasattr(Qt, 'TextInteractionFlag') and hasattr(Qt.TextInteractionFlag, 'TextBrowserInteraction'):



            self.answer_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)



        elif hasattr(Qt, 'TextBrowserInteraction'):



            self.answer_box.setTextInteractionFlags(Qt.TextBrowserInteraction)



        if hasattr(self.answer_box, 'setPlaceholderText'):



            self.answer_box.setPlaceholderText("Enter a question above and click Search to see an AI answer based on your notes.")



        self.answer_box.setMinimumHeight(100)



        self.answer_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



        answer_font_size = self.styling_config.get('answer_font_size', 13)



        self.answer_box.setStyleSheet(



            f"background-color: {'#2d2d2d' if is_dark else '#ffffff'}; "



            f"border: 2px solid #27ae60; color: {'#ffffff' if is_dark else '#1a1a1a'}; "



            f"font-size: {answer_font_size}px; padding: 10px;"



            f"a {{ color: #3498db; text-decoration: underline; }} "



            f"a:hover {{ color: #5dade2; }} "



        )



        # Connect link click: anchorClicked (Qt6) or linkActivated (PyQt5) so citation links work with Ollama and all providers



        if hasattr(self.answer_box, 'anchorClicked'):



            self.answer_box.anchorClicked.connect(self._on_answer_link_clicked)



        elif hasattr(self.answer_box, 'linkActivated'):



            self.answer_box.linkActivated.connect(self._on_answer_link_clicked)



        answer_layout.addWidget(self.answer_box)







        # Hint: where the answer came from (API name or local model)



        self.answer_source_label = QLabel("")



        self.answer_source_label.setStyleSheet(f"font-size: 11px; color: {'#95a5a6' if is_dark else '#7f8c8d'}; font-style: italic; margin-top: 4px;")



        self.answer_source_label.setWordWrap(True)



        self.answer_source_label.setToolTip("Shows whether the answer came from an online API or a local model (Ollama).")



        answer_layout.addWidget(self.answer_source_label)







        main_splitter.addWidget(answer_container)







        # Results section



        results_container = QWidget()



        results_layout = QVBoxLayout(results_container)







        results_header = QHBoxLayout()



        results_label = QLabel("\U0001F4CB Matching notes:")



        results_label.setToolTip("Notes that match your question. Check the ones to send to the AI for the answer.")



        results_label.setStyleSheet(f"font-weight: bold; font-size: {label_font_size}px; color: {'#ffffff' if is_dark else '#2c3e50'};")



        results_header.addWidget(results_label)



        self.selected_count_label = QLabel("(0 selected)")



        self.selected_count_label.setStyleSheet(f"font-size: {label_font_size - 2}px; color: {'#95a5a6' if is_dark else '#7f8c8d'}; font-style: italic;")



        self.selected_count_label.setToolTip("Notes you checked (for \"View Selected\"). Not the same as \"cited\" in the AI answer\u2014use \"Show only cited notes\" for that.")



        results_header.addWidget(self.selected_count_label)



        results_header.addStretch()



        # Compact secondary controls



        preview_label = QLabel("Preview:")



        preview_label.setStyleSheet(f"font-size: 10px; color: {'#95a5a6' if is_dark else '#7f8c8d'};")



        preview_label.setToolTip("Preview length (characters)")



        results_header.addWidget(preview_label)



        self.preview_slider = QSlider(Qt.Orientation.Horizontal)



        self.preview_slider.setMinimum(50)



        self.preview_slider.setMaximum(500)



        self.preview_slider.setValue(150)



        self.preview_slider.setMaximumWidth(80)



        self.preview_slider.setToolTip("Preview length (characters)")



        self.preview_slider.valueChanged.connect(self.on_preview_length_changed)



        results_header.addWidget(self.preview_slider)



        self.preview_length_label = QLabel("150 chars")



        self.preview_length_label.setMinimumWidth(40)



        self.preview_length_label.setAlignment(Qt.AlignmentFlag.AlignCenter)



        self.preview_length_label.setStyleSheet(f"font-size: 10px; color: {'#95a5a6' if is_dark else '#7f8c8d'};")



        self.preview_length_label.setToolTip("Preview length in characters")



        results_header.addWidget(self.preview_length_label)



        self.toggle_select_btn = QPushButton("\u2713 Select All")



        self.toggle_select_btn.setObjectName("toggleSelectBtn")



        self.toggle_select_btn.setMaximumWidth(95)



        self.toggle_select_btn.setToolTip("Toggle select/deselect all (Ctrl+A / Ctrl+D)")



        self.toggle_select_btn.clicked.connect(self.toggle_select_all)



        self.toggle_select_btn.setEnabled(False)



        results_header.addWidget(self.toggle_select_btn)



        results_layout.addLayout(results_header)







        # Create table: Ref (citation [1],[2]...) | Content | Note ID | Relevance



        self.results_list = QTableWidget()



        self.results_list.setColumnCount(4)



        self.results_list.setHorizontalHeaderLabels(["Ref", "Content", "Note ID", "Relevance"])



        self.results_list.setMinimumHeight(120)



        self.results_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



        notes_font_size = self.styling_config.get('notes_font_size', 12)



        self.results_list.setStyleSheet(f"font-size: {notes_font_size}px;")



        self.results_list.setWordWrap(True)







        # Configure columns



        self.results_list.setColumnWidth(0, 42)   # Ref (citation number matching [1], [2] in answer)



        self.results_list.setColumnWidth(1, 400)  # Content



        self.results_list.setColumnWidth(2, 80)   # Note ID (hidden by default)



        self.results_list.setColumnWidth(3, 100)  # Relevance (bar + %)



        self.results_list.setColumnHidden(2, True)  # Hide Note ID column (right-click header to show)



        self.results_list.horizontalHeader().setStretchLastSection(False)



        self.results_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)



        self.results_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)



        self.results_list.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)



        self.results_list.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)



        # Make column headers readable on dark and light themes (was hard to see on dark)



        header_color = "#ecf0f1" if is_dark else "#2c3e50"



        self.results_list.horizontalHeader().setStyleSheet(



            f"color: {header_color}; font-weight: bold; font-size: {max(11, notes_font_size - 1)}px;"



        )







        self.results_list.setSortingEnabled(True)



        self.results_list.setItemDelegateForColumn(1, ContentDelegate(self.results_list)) # First field normally, both on hover

        self.results_list.setItemDelegateForColumn(3, RelevanceBarDelegate(self.results_list))  # Relevance bar + %



        self.results_list.sortItems(3, Qt.SortOrder.DescendingOrder)  # Sort by Relevance







        # Enable double-click on rows



        self.results_list.itemDoubleClicked.connect(self.open_in_browser)







        # Hide vertical header (row numbers)



        self.results_list.verticalHeader().setVisible(False)







        # Set selection behavior to select entire rows



        self.results_list.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)



        self.results_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)







        # Enable alternating row colors (zebra striping)



        self.results_list.setAlternatingRowColors(True)







        # Store preview length setting



        self.preview_length = 150  # Default preview length







        results_layout.addWidget(self.results_list, 1)







        # Label at bottom of notes area: which search mechanism yielded these results



        self.search_method_result_label = QLabel("")



        self.search_method_result_label.setStyleSheet(



            "font-size: 11px; color: #7f8c8d; padding: 4px 0; margin-top: 4px;"



        )



        self.search_method_result_label.setWordWrap(True)



        results_layout.addWidget(self.search_method_result_label)







        main_splitter.addWidget(results_container)



        main_splitter.setSizes([450, 550] if self.use_side_by_side else [350, 450])



        main_splitter.setChildrenCollapsible(False)



        main_splitter.setHandleWidth(8)







        layout.addWidget(main_splitter, 1)







        # Result tuning: relevance mode controls (Focused / Balanced / Broad); no slider



        sensitivity_container = QWidget()



        sensitivity_container.setStyleSheet("background-color: rgba(241, 196, 15, 0.1); border-radius: 6px; padding: 6px;")



        sensitivity_layout = QVBoxLayout(sensitivity_container)



        sensitivity_layout.setSpacing(4)



        sensitivity_layout.setContentsMargins(4, 2, 4, 2)



        self.sensitivity_slider = None  # Slider removed; mode alone controls how many notes are shown



        self.sensitivity_value_label = None







        # Relevance modes + "Show only cited notes" aligned on one line



        mode_and_filter_row = QHBoxLayout()



        mode_and_filter_row.setSpacing(8)



        mode_and_filter_row.setContentsMargins(0, 0, 0, 0)







        # Relevance mode selector (Focused / Balanced / Broad)



        mode_layout = QHBoxLayout()



        mode_layout.setSpacing(0)



        mode_layout.setContentsMargins(0, 0, 0, 0)



        self.relevance_mode_group = QButtonGroup(self)



        self.relevance_mode_group.setExclusive(True)







        # Determine initial mode from config (fallback to strict_relevance when missing)



        try:



            sc_mode = load_config().get("search_config", {}).get("relevance_mode", "")



        except Exception:



            sc_mode = ""



        if not sc_mode:



            try:



                sc = load_config().get("search_config", {})



                sc_mode = "focused" if sc.get("strict_relevance", True) else "balanced"



            except Exception:



                sc_mode = "balanced"



        sc_mode = (sc_mode or "balanced").lower()



        if sc_mode not in ("focused", "balanced", "broad"):



            sc_mode = "balanced"



        self.relevance_mode = sc_mode







        # Shared segmented-control style for relevance mode buttons



        mode_border_color = "#f1c40f"



        mode_text_color = "#f1c40f" if is_dark else "#7d6608"



        mode_checked_bg = "#f1c40f" if is_dark else "#f9e79f"



        mode_checked_text = "#1e1e1e" if is_dark else "#7d6608"



        mode_hover_bg = "rgba(241, 196, 15, 0.18)"



        mode_btn_style = (



            "QRadioButton {"



            "  font-size: 11px;"



            "  padding: 4px 10px;"



            "  border: 1px solid " + mode_border_color + ";"



            "  color: " + mode_text_color + ";"



            "  background-color: transparent;"



            "  margin: 0;"



            "}"



            "QRadioButton::indicator {"



            "  width: 0px;"



            "  height: 0px;"



            "}"



            "QRadioButton:hover {"



            "  background-color: " + mode_hover_bg + ";"



            "}"



            "QRadioButton:checked {"



            "  background-color: " + mode_checked_bg + ";"



            "  color: " + mode_checked_text + ";"



            "}"



        )







        def _add_mode_button(label, mode_key, tooltip):



            btn = QRadioButton(label)



            btn.setToolTip(tooltip)



            btn.setProperty("mode_key", mode_key)



            btn.setMinimumWidth(72)



            btn.setStyleSheet(mode_btn_style)



            self.relevance_mode_group.addButton(btn)



            mode_layout.addWidget(btn)



            if mode_key == self.relevance_mode:



                btn.setChecked(True)







        _add_mode_button(



            "Focused",



            "focused",



            "Fewer notes, most on-topic only. Same search results; only which notes are shown changes.",



        )



        _add_mode_button(



            "Balanced",



            "balanced",



            "Moderate set. Same search results; only which notes are shown changes.",



        )



        _add_mode_button(



            "Broad",



            "broad",



            "More notes, including tangential. Same search results; only which notes are shown changes.",



        )







        self.relevance_mode_group.buttonToggled.connect(self._on_relevance_mode_changed)



        mode_and_filter_row.addLayout(mode_layout)







        self.show_only_cited_cb = QCheckBox("Show only cited notes")



        self.show_only_cited_cb.setToolTip(



            "Show only notes that the AI explicitly cited in its answer."



        )



        try:



            sc = load_config().get('search_config', {})



            self.show_only_cited_cb.setChecked(bool(sc.get('show_only_cited', False)))



        except Exception:



            self.show_only_cited_cb.setChecked(False)



        self.show_only_cited_cb.stateChanged.connect(self._on_show_only_cited_changed)



        mode_and_filter_row.addWidget(self.show_only_cited_cb)



        mode_and_filter_row.addStretch()







        sensitivity_layout.addLayout(mode_and_filter_row)







        layout.addWidget(sensitivity_container)







        # Action buttons



        btn_container = QWidget()



        btn_container.setObjectName("actionBar")



        btn_bar_bg = "#2d2d2d" if is_dark else "#d5d8dc"



        btn_bar_border = "#555555" if is_dark else "#95a5a6"



        btn_container.setStyleSheet(f"""



            QWidget#actionBar {{



                background-color: {btn_bar_bg};



                border: 1px solid {btn_bar_border};



                border-radius: 6px;



                padding: 6px;



            }}



        """)



        btn_layout = QHBoxLayout(btn_container)







        self.view_btn = QPushButton("\U0001F441 View Selected")



        self.view_btn.setObjectName("viewBtn")



        self.view_btn.setToolTip("Open selected notes in the Anki browser")



        self.view_btn.setMinimumHeight(32)



        self.view_btn.clicked.connect(self.open_selected_in_browser)



        self.view_btn.setEnabled(False)







        self.view_all_btn = QPushButton("\U0001F4DA View All")



        self.view_all_btn.setObjectName("viewAllBtn")



        self.view_all_btn.setToolTip(



            "Open all visible notes in the Anki browser. "



            "No notes in the list \u2014 run a search first."



        )



        self.view_all_btn.setMinimumHeight(32)



        self.view_all_btn.clicked.connect(self.open_all_in_browser)



        self.view_all_btn.setEnabled(False)







        btn_layout.addWidget(self.view_btn)



        btn_layout.addWidget(self.view_all_btn)



        btn_layout.addStretch()







        close_btn = QPushButton("\u2716 Close")



        close_btn.setObjectName("closeBtn")



        close_btn.setMinimumHeight(32)



        close_btn.setMinimumWidth(80)



        close_btn.clicked.connect(self.close)



        btn_layout.addWidget(close_btn)







        layout.addWidget(btn_container)







        # Status bar



        status_container = QWidget()



        status_container.setStyleSheet("background-color: rgba(52, 152, 219, 0.15); border-radius: 4px; padding: 4px;")



        status_layout = QHBoxLayout(status_container)



        status_layout.setContentsMargins(8, 4, 8, 4)







        status_icon = QLabel("\u2139\ufe0f")



        self.status_label = QLabel("Ready")



        self.status_label.setStyleSheet(f"color: {'#ffffff' if is_dark else '#2c3e50'}; font-size: 12px; font-weight: 600;")



        self.status_label.setToolTip(



            "Showing X of Y: X = notes passing the Sensitivity filter, Y = notes in this result set (from Min relevance % and Max results in Settings). "



            "Move the Sensitivity slider to change X. Raise Min relevance % in Settings to get a smaller result set (smaller Y)."



        )







        status_layout.addWidget(status_icon)



        status_layout.addWidget(self.status_label)



        status_layout.addStretch()



        # Progress bar and % for embedding search (hidden when idle)



        self.search_progress_bar = QProgressBar()



        self.search_progress_bar.setMinimum(0)



        self.search_progress_bar.setMaximum(100)



        self.search_progress_bar.setValue(0)



        self.search_progress_bar.setMaximumWidth(120)



        self.search_progress_bar.setMinimumWidth(80)



        self.search_progress_bar.setTextVisible(True)



        self.search_progress_bar.setFormat("%p%")



        self.search_progress_bar.setStyleSheet("QProgressBar { text-align: center; color: palette(window-text); }")



        self.search_progress_bar.setVisible(False)



        status_layout.addWidget(self.search_progress_bar)



        self.search_progress_label = QLabel("")



        self.search_progress_label.setStyleSheet(f"font-size: 11px; color: {'#95a5a6' if is_dark else '#7f8c8d'};")



        self.search_progress_label.setVisible(False)



        status_layout.addWidget(self.search_progress_label)







        layout.addWidget(status_container)







        # Main content added directly to root layout



        root_layout.addWidget(main_container, 1)







        self.setLayout(root_layout)







        # Enable buttons based on selections



        self.results_list.itemSelectionChanged.connect(



            lambda: self.view_btn.setEnabled(bool(self.results_list.selectedItems()))



        )







        self.results_list.itemSelectionChanged.connect(self._update_view_all_button_state)







        # Track checkbox state changes to update count and button text



        # Only track changes in column 1 (content column with checkbox)



        self.results_list.itemChanged.connect(self.on_item_changed)







        # Store selected note IDs for persistence



        self.selected_note_ids = set()



        self._pinned_note_ids = set()  # note IDs from clicked [N] refs in AI answer



        self._cited_note_ids = set()   # note IDs cited in AI answer ([1], [2], ...) for "Show only cited" filter







        self._refresh_search_history()



        QTimer.singleShot(100, self._refresh_scope_banner)







    # --- Search History And Scope Banner ---

    def _update_view_all_button_state(self):



        """Update View All button enabled state and tooltip based on whether the results list has rows."""



        if not hasattr(self, 'view_all_btn') or not self.view_all_btn:



            return



        has_rows = self.results_list.rowCount() > 0 if hasattr(self, 'results_list') and self.results_list else False



        self.view_all_btn.setEnabled(has_rows)



        self.view_all_btn.setToolTip(



            "Open all visible notes in the Anki browser"



            if has_rows



            else "No notes in the list \u2014 run a search first."



        )







    def showEvent(self, event):



        """Refresh scope banner when dialog is shown (e.g. after Settings changed)."""



        super().showEvent(event)



        self._refresh_scope_banner()







    def _refresh_scope_banner(self):



        """Update scope banner: X note types, Y fields, Z decks with shortcut to Settings."""



        if not hasattr(self, 'scope_banner') or not self.scope_banner:



            return



        try:



            config = load_config()



            ntf = config.get('note_type_filter') or {}



            enabled_types = ntf.get('enabled_note_types') or []



            ntf_fields = ntf.get('note_type_fields') or {}



            enabled_decks = ntf.get('enabled_decks') or []



            search_all = bool(ntf.get('search_all_fields', False))



            n_types = len(enabled_types) if enabled_types else len(get_all_note_types())



            n_decks = len(enabled_decks) if enabled_decks else len(get_deck_names())



            fields_set = set()



            if search_all:



                fields_set = set(f['name'] for m in (mw.col.models.all() if mw and mw.col else []) for f in m.get('flds', []))



            else:



                for flist in ntf_fields.values():



                    fields_set.update(flist or [])



            n_fields = len(fields_set) if fields_set else 1



            txt = f"Searching: {n_types} note types, {n_fields} fields, {n_decks} decks \u2014 <a href='#settings' style='color:#3498db;'>Settings</a>"



            self.scope_banner.setText(txt)



            if hasattr(self, 'sidebar_scope_label') and self.sidebar_scope_label:



                self.sidebar_scope_label.setText(



                    f"Scope\n{n_types} note types\n{n_fields} fields\n{n_decks} decks"



                )



        except Exception as e:



            log_debug(f"Scope banner refresh error: {e}")



            self.scope_banner.setText("")



            if hasattr(self, 'sidebar_scope_label') and self.sidebar_scope_label:



                self.sidebar_scope_label.setText("Scope unavailable")







    def _on_search_history_selected(self, index):



        """When user selects a recent search from the dropdown, populate the input."""



        if index >= 0 and hasattr(self, 'search_input') and hasattr(self, 'search_history_combo'):



            text = self.search_history_combo.currentText()



            self._set_query_text(text)







    def _on_sidebar_history_selected(self, item):



        """Load query from left panel history list."""



        if not item:



            return



        self._set_query_text(item.text())







    def _set_query_text(self, text):



        """Set query text in the editor and show a short status hint."""



        if not text or not hasattr(self, 'search_input'):



            return



        self.search_input.setPlainText(text)



        self.search_input.setFocus()



        if hasattr(self, 'status_label') and self.status_label:



            self.status_label.setText("Query loaded - press Ctrl+Enter to search.")



            QTimer.singleShot(3000, self._clear_query_loaded_status)







    def _clear_query_loaded_status(self):



        """Clear the 'Query loaded' status after a delay if it wasn't replaced by search results."""



        if hasattr(self, 'status_label') and self.status_label:



            if self.status_label.text().startswith("Query loaded"):



                self.status_label.setText("Ready")







    def _on_clear_search_history(self):

        """Clear all search history with a confirmation dialog."""

        from aqt.utils import askUser

        if not askUser("Are you sure you want to permanently delete your entire search history?"):

            return



        if clear_search_history():

            self._refresh_search_history()







    def _refresh_search_history(self):



        """Reload the previous-searches list from search_history.json."""



        try:



            history = get_search_history_queries()



            if hasattr(self, '_search_history_model'):



                self._search_history_model.setStringList(history)



            if hasattr(self, 'sidebar_history_list') and self.sidebar_history_list:



                self.sidebar_history_list.blockSignals(True)



                self.sidebar_history_list.clear()



                self.sidebar_history_list.addItems(history)



                self.sidebar_history_list.blockSignals(False)



        except Exception:



            pass







    # --- Relevance And Display Options ---

    def on_item_changed(self, item):



        """Handle item changes - only update count if checkbox column changed"""



        if item.column() == 1:  # Only process changes in content column (checkbox)



            self.update_selection_count()







    def _on_relevance_mode_changed(self, _btn, checked):



        """Persist relevance_mode (Focused/Balanced/Broad) and refresh current view."""



        # #region agent log



        _session_debug_log(



            "H1",



            "__init__._on_relevance_mode_changed.entry",



            "mode change handler",



            data={"checked": checked, "relevance_mode_before": getattr(self, "relevance_mode", None), "_effective_relevance_mode": getattr(self, "_effective_relevance_mode", None)},



        )



        # #endregion



        if not checked:



            return



        try:



            btn = self.relevance_mode_group.checkedButton()



            mode_key = (btn.property("mode_key") or "balanced").lower()



        except Exception:



            mode_key = "balanced"



        if mode_key not in ("focused", "balanced", "broad"):



            mode_key = "balanced"



        self.relevance_mode = mode_key



        self._effective_relevance_mode = mode_key  # so status bar and "Results from" label show the selected mode



        # #region agent log



        _session_debug_log(



            "H1",



            "__init__._on_relevance_mode_changed.after_assign",



            "relevance_mode set",



            data={"mode_key": mode_key, "relevance_mode": self.relevance_mode, "has_all_scored_notes": hasattr(self, "all_scored_notes")},



        )



        # #endregion



        # Persist last-used mode and keep strict_relevance in sync for compatibility



        try:



            config = load_config()



            sc = dict(config.get("search_config") or {})



            sc["relevance_mode"] = mode_key



            sc["strict_relevance"] = True if mode_key == "focused" else False



            config["search_config"] = sc



            save_config(config)



        except Exception:



            pass



        # Update "Results from: ... \u00b7 Mode \u00b7 Embeddings: ..." so it shows the new mode



        if hasattr(self, "search_method_result_label") and getattr(self, "_last_search_method", None):



            try:



                sc = load_config().get("search_config") or {}



                mode_display = {"focused": "Focused", "balanced": "Balanced", "broad": "Broad"}.get(mode_key, "Balanced")



                engine = (sc.get("embedding_engine") or "ollama").strip().lower()



                engine_display = {"ollama": "Ollama (local)", "voyage": "Voyage AI", "openai": "OpenAI", "cohere": "Cohere"}.get(engine, engine or "unknown")



                self.search_method_result_label.setText(f"Results from: {self._last_search_method} \u00b7 {mode_display} \u00b7 Embeddings: {engine_display}")



                self.search_method_result_label.setVisible(True)



            except Exception:



                pass



        # Update any existing results with the new mode (will use effective flags)



        if hasattr(self, "all_scored_notes"):



            self.filter_and_display_notes()







    def on_sensitivity_changed(self, value):



        if getattr(self, 'sensitivity_value_label', None) is not None:



            self.sensitivity_value_label.setText(f"{value}%")



        # Persist so next time the add-on opens with the same choice (keep full search_config)



        try:



            config = load_config()



            sc = dict(config.get('search_config') or {})  # copy so we don't lose other keys



            sc['sensitivity_percent'] = value



            config['search_config'] = sc



            save_config(config)



        except Exception:



            pass



        if hasattr(self, 'all_scored_notes'):



            self.filter_and_display_notes()







    def _on_show_only_cited_changed(self, _state):



        """Persist 'Show only cited notes' and refresh the table."""



        try:



            config = load_config()



            sc = dict(config.get('search_config') or {})



            sc['show_only_cited'] = getattr(self, 'show_only_cited_cb', None) and self.show_only_cited_cb.isChecked()



            config['search_config'] = sc



            save_config(config)



        except Exception:



            pass



        if hasattr(self, 'all_scored_notes'):



            self.filter_and_display_notes()







    # --- Answer Formatting And Citation Navigation ---

    def _restore_answer_html(self, html):
        return answer_navigation._restore_answer_html(self, html)







    def _on_answer_link_clicked(self, url):
        return answer_navigation._on_answer_link_clicked(self, url)







    def _citation_timer_clear(self):
        return answer_navigation._citation_timer_clear(self)







    def _bring_browser_to_front(self, browser):
        return answer_navigation._bring_browser_to_front(self, browser)







    def _open_note_in_browser(self, note_id):
        return answer_navigation._open_note_in_browser(self, note_id)







    def _spacing_styles(self):
        return answer_formatting.spacing_styles(self.styling_config.get('answer_spacing', 'normal'))





    def format_answer(self, answer):
        return answer_formatting.format_answer_html(
            answer,
            getattr(self, '_context_note_ids', None) or [],
            self.styling_config.get('answer_spacing', 'normal'),
            {
                'citation_n': CITATION_N_RE,
                'citation': CITATION_RE,
                'md_bold': MD_BOLD_RE,
                'md_unterminated_bold': MD_UNTERMINATED_BOLD_RE,
                'md_bold_alt': MD_BOLD_ALT_RE,
                'md_header': MD_HEADER_RE,
                'md_highlight': MD_HIGHLIGHT_RE,
            },
        )







    # --- Clipboard, Settings, And Config Access ---

    def copy_answer_to_clipboard(self):
        return answer_navigation.copy_answer_to_clipboard(self)







    def open_settings(self):



        """Open the settings dialog in a non-modal window so Anki stays usable."""



        dialog = SettingsDialog(self)



        dialog.setWindowModality(Qt.WindowModality.NonModal)



        dialog.show()



        dialog.raise_()



        dialog.activateWindow()







    def get_config(self):



        config = load_config()



        if not config or 'api_key' not in config:



            return None



        return config







    # --- Note Collection And Text Preparation ---

    def get_all_notes_content(self):
        return note_content.get_all_notes_content(self)





    # --- Keyword And TF-IDF Helpers ---

    def _aggregate_scored_notes_by_note_id(self, scored_notes):
        return note_content.aggregate_scored_notes_by_note_id(scored_notes)





    def strip_html(self, text):
        return note_content.strip_html(text)





    def reveal_cloze_for_display(self, text):
        return note_content.reveal_cloze_for_display(text)





    def _simple_stem(self, word):
        return note_content._simple_stem(word)





    def _get_extended_stop_words(self):
        search_config = (load_config() or {}).get('search_config') or {}
        return note_content.get_extended_stop_words(search_config)





    def _extract_keywords_improved(self, query):
        return note_content.extract_keywords_for_dialog(self, query, _agent_debug_log)





    def _compute_tfidf_scores(self, notes, query_keywords):
        return note_content.compute_tfidf_scores_for_dialog(self, notes, query_keywords)







    # --- Embedding Search Helpers ---

    def _get_note_embedding(self, note_content, note_id=None):



        """Get embedding for a note (checks persistent storage first, then cache).







        IMPORTANT:



        - Embeddings are keyed by a hash of the *generation-time* note content,



          not necessarily the string we display in the search UI.



        - Originally we hashed the UI content, which differed (HTML stripped,



          different separators), so lookups always missed and embeddings were



          treated as "not available".







        This helper now reconstructs the same content that was used during the



        Create/Update Embeddings workflow so the hashes line up.



        """



        import hashlib







        # Lazily initialise in\xe2\u20ac\u2018memory cache



        if not hasattr(self, "_embedding_cache"):



            self._embedding_cache = {}







        # Try fast path: hash of the UI content (in case future versions align)



        ui_hash = hashlib.md5(note_content.encode()).hexdigest()



        if ui_hash in self._embedding_cache:



            return self._embedding_cache.get(ui_hash)







        # If we don't have a note id we can't reconstruct the generation



        # content reliably, so bail out early.



        if note_id is None:



            return None







        # Reconstruct the content string the embedding generator used so that



        # we derive the exact same MD5 and key.



        try:



            from aqt import mw  # Local import to avoid issues at import time



            note = mw.col.get_note(note_id)



            m = note.note_type()



            model_name = m["name"]



            flds = m["flds"]







            config = load_config()



            ntf = config.get("note_type_filter", {})



            enabled = set(ntf.get("enabled_note_types") or [])



            search_all = ntf.get("search_all_fields", False)



            ntf_fields = ntf.get("note_type_fields", {})



            use_first = ntf.get("use_first_field_fallback", True)







            # If embeddings were originally restricted to certain note types,



            # mirror that logic here. If the model wasn't enabled at generation



            # time we almost certainly won't have an embedding anyway.



            if enabled and model_name not in enabled:



                return None







            if search_all:



                indices = [



                    i for i in range(min(len(note.fields), len(flds)))



                    if note.fields[i].strip()



                ]



            else:



                wanted = set(



                    f.lower() for f in (ntf_fields.get(model_name) or [])



                )



                if not wanted and use_first and flds:



                    wanted = {flds[0]["name"].lower()}



                indices = [



                    i for i, f in enumerate(flds)



                    if i < len(note.fields)



                    and f["name"].lower() in wanted



                    and note.fields[i].strip()



                ]



            if not indices:



                return None



            content_parts = [note.fields[i] for i in indices]







            # This mirrors the generator: join with spaces and *do not* strip HTML.



            generation_content = " ".join(content_parts)



            content_hash = hashlib.md5(generation_content.encode()).hexdigest()



        except Exception:



            # Fall back to the UI hash; if that doesn't exist in storage,



            # load_embedding will just return None.



            content_hash = ui_hash







        # Check cache again with the (possibly different) generation hash



        if content_hash in self._embedding_cache:



            return self._embedding_cache.get(content_hash)







        # Then check persistent storage using the reconstructed hash



        persistent_embedding = load_embedding(note_id, content_hash)



        if persistent_embedding is not None:



            # Cache using the generation hash; we also alias under the UI hash



            # so repeated lookups for the same text are fast.



            self._embedding_cache[content_hash] = persistent_embedding



            self._embedding_cache[ui_hash] = persistent_embedding



            return persistent_embedding







        # If not found, we skip generating on-the-fly to avoid extra API usage.



        return None







    def _embedding_search(self, query, notes):



        """Semantic search using precomputed embeddings (Voyage).







        Uses the configured embedding engine (Voyage or Ollama) for the query



        embedding, and expects note embeddings to have been generated ahead of



        time via the Create/Update Embeddings workflow.



        """



        try:



            import numpy as np



        except ImportError:



            log_debug("numpy not available, embedding search disabled")



            return None







        try:



            # Get query embedding via configured engine (Voyage or Ollama)



            embedding_list = get_embedding_for_query(query)



            if not embedding_list:



                log_debug("Empty query embedding from embedding engine")



                return None



            query_embedding = np.array(embedding_list)







            # Compute similarities



            scored_notes = []







            # For very large collections, limit how many notes we run full



            # embedding search over to keep things responsive.



            max_notes_for_embedding = 5000



            total_notes = len(notes)



            if total_notes > max_notes_for_embedding:



                log_debug(f"Embedding search: limiting to first {max_notes_for_embedding} of {total_notes} notes for performance.")



                notes_iter = notes[:max_notes_for_embedding]



            else:



                notes_iter = notes







            # Keep UI responsive during long embedding loops



            try:



                from aqt.qt import QApplication



            except Exception:



                QApplication = None







            for idx, note in enumerate(notes_iter):



                if QApplication is not None and idx % 500 == 0:



                    QApplication.processEvents()



                    log_debug(f"Embedding search progress: processed {idx}/{len(notes_iter)} notes")







                note_embedding = self._get_note_embedding(note['content'], note.get('id'))



                if note_embedding is not None:



                    # Cosine similarity



                    similarity = np.dot(query_embedding, note_embedding) / (



                        np.linalg.norm(query_embedding) * np.linalg.norm(note_embedding)



                    )



                    # Convert to 0-100 score



                    score = (similarity + 1) * 50  # Normalize from [-1,1] to [0,100]



                    scored_notes.append((score, note))







            scored_notes.sort(reverse=True, key=lambda x: x[0])



            return scored_notes[:80]



        except Exception as e:



            log_debug(f"Error in embedding search: {e}")



            return None







    # --- Metadata And Context Boosting ---

    def _get_note_metadata(self, note_id):



        """Get metadata for context-aware ranking"""



        try:



            note = mw.col.get_note(note_id)



            card = note.cards()[0] if note.cards() else None







            metadata = {



                'deck_id': card.did if card else None,



                'note_type': note.note_type()['name'] if note else None,



                'mod_time': note.mod if hasattr(note, 'mod') else 0,



                'review_count': card.reps if card else 0,



                'last_review': card.rep if card else 0,



            }



            return metadata



        except:



            return {}







    def _context_aware_boost(self, note, base_score, selected_note_ids=None):



        """Apply context-aware ranking boosts"""



        if not hasattr(self, 'selected_note_ids') or not self.selected_note_ids:



            selected_note_ids = set()



        else:



            selected_note_ids = self.selected_note_ids







        boost = 1.0



        metadata = self._get_note_metadata(note['id'])







        # Boost notes from same deck/note type as previously selected



        if selected_note_ids:



            try:



                selected_metadata = [self._get_note_metadata(nid) for nid in selected_note_ids]



                selected_decks = {m.get('deck_id') for m in selected_metadata if m.get('deck_id')}



                selected_types = {m.get('note_type') for m in selected_metadata if m.get('note_type')}







                if metadata.get('deck_id') in selected_decks:



                    boost *= 1.2



                if metadata.get('note_type') in selected_types:



                    boost *= 1.15



            except:



                pass







        # Boost recent notes (last 30 days)



        if metadata.get('mod_time'):



            try:



                from datetime import datetime, timedelta



                mod_date = datetime.fromtimestamp(metadata['mod_time'] / 1000)



                days_ago = (datetime.now() - mod_date).days



                if days_ago < 30:



                    boost *= 1.1



            except:



                pass







        # Slight boost for well-reviewed notes



        if metadata.get('review_count', 0) > 10:



            boost *= 1.05







        return base_score * boost







    # --- AI Query Expansion And Relevance Filtering ---

    def _expand_query(self, query, config):
        return query_enhancement._expand_query(self, query, config)







    def _get_ai_excluded_terms(self, query, search_config):
        return query_enhancement._get_ai_excluded_terms(self, query, search_config)







    def _generate_hyde_document(self, query, config):
        return query_enhancement._generate_hyde_document(self, query, config)







    def _passes_focused_balanced_broad(self, final_score, emb_score, matched_keywords, keywords, search_method, max_emb_score):
        return query_enhancement._passes_focused_balanced_broad(
            self, final_score, emb_score, matched_keywords, keywords, search_method, max_emb_score
        )











# ============================================================================
# Search Worker Compatibility And Standalone Search Helpers
# ============================================================================

# --- Embedding Search Execution ---




# ============================================================================
# Search Worker Compatibility And Standalone Search Helpers
# ============================================================================

from .search_workers import (
    MAX_RERANK_COUNT,
    RRF_K,
    AskAIWorker,
    EmbeddingSearchWorker,
    KeywordFilterContinueWorker,
    KeywordFilterWorker,
    RelevanceRerankWorker,
    RerankCheckWorker,
    RerankWorker,
    _do_rerank,
    _run_embedding_search_sync,
)











# --- Anthropic Prompt Construction ---

# ============================================================================
# Anthropic Prompt Construction
# ============================================================================

from .answer_prompts import _build_anthropic_prompt_parts, _normalize_query_space











# --- Search Workflow Methods Copied Onto AISearchDialog ---

# ============================================================================
# Search Workflow Compatibility Import And Wiring
# ============================================================================

from .search_workflow import AnthropicStreamWorker, configure_search_workflow_globals, install_search_workflow_methods

configure_search_workflow_globals(
    DIGIT_RE=DIGIT_RE,
    _agent_debug_log=_agent_debug_log,
    _session_debug_log=_session_debug_log,
    get_notes_content_with_col=get_notes_content_with_col,
    get_safe_config=get_safe_config,
)
install_search_workflow_methods(AISearchDialog)

# ============================================================================
# Search Dialog Singleton And Sidebar Controls
# ============================================================================

def toggle_sidebar_visibility(visible: bool):
    return _dialog_entrypoints.toggle_sidebar_visibility(visible, dialogs_module=sys.modules[__name__])


# ============================================================================
# Settings Dialog Compatibility Import
# ============================================================================

from .settings_dialog import SettingsDialog







# END OF PART 1 - Continue to PART 2

# PART 2 OF 3 - Continue from PART 1



from aqt.qt import *

# Use pyqtSignal directly as it is the most stable across Anki versions

try:

    from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

except ImportError:

    from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot



# ============================================================================
# Public Dialog Entry Points
# ============================================================================

def show_search_dialog():
    return _dialog_entrypoints.show_search_dialog(dialogs_module=sys.modules[__name__])



def _on_search_dialog_closed():
    return _dialog_entrypoints._on_search_dialog_closed(dialogs_module=sys.modules[__name__])



def show_settings_dialog(open_to_embeddings=False, auto_start_indexing=False):
    return _dialog_entrypoints.show_settings_dialog(
        open_to_embeddings=open_to_embeddings,
        auto_start_indexing=auto_start_indexing,
        dialogs_module=sys.modules[__name__],
    )







def show_debug_log():
    return _dialog_entrypoints.show_debug_log()







# ============================================================================
# Dependency Detection And Installation Helpers
# ============================================================================

from .dependency_install import (
    _check_sentence_transformers_installed_subprocess,
    _patch_colorama_for_transformers,
    _resolve_external_python_exe,
    check_dependency_installed,
    check_vc_redistributables,
    fix_pytorch_dll_issue,
    get_pytorch_dll_error_guidance,
    install_dependencies,
    install_vc_redistributables,
    try_alternative_pytorch_install,
)







log_debug("=== Anki Semantic Search Add-on Loaded ===")



log_debug(f"Addon directory: {os.path.dirname(__file__)}")



log_debug(f"Addon folder name: {ADDON_NAME}")







# Note: colorama is already patched at module load time via _patch_colorama_early()







# ============================================================================
# Add-on Menu And Hook Registration
# ============================================================================

# Add menu items



ai_search_menu = QMenu("\U0001F50D Anki Semantic Search", mw)







# search_action = QAction("Search Notes", mw)



# search_action.triggered.connect(show_search_dialog)



# search_action.setToolTip("Open Anki Semantic Search window")



# search_action.setShortcut("Ctrl+Shift+S")



# mw.form.menuTools.addAction(search_action)







mw.addonManager.setConfigAction(ADDON_NAME, show_settings_dialog)







# Background indexer: re-enabled using QueryOp \u2014 collection access only on main thread via QueryOp; worker thread does API + save only (no mw.col).


