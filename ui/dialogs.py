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
from ..core.workers import (
    EmbeddingWorker,
    EmbeddingSearchWorker,
    AskAIWorker,
    RerankCheckWorker,
    KeywordFilterWorker,
    KeywordFilterContinueWorker,
    RerankWorker,
    RelevanceRerankWorker,
    AnthropicStreamWorker
)

_addon_theme = get_addon_theme
ADDON_NAME = get_addon_name()


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

class AISearchDialog(QDialog):



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







    def perform_search(self):



        """



        Bridge method so AISearchDialog always has perform_search.



        Delegates to the implementation that was originally defined on EmbeddingSearchWorker.



        """



        return EmbeddingSearchWorker.perform_search(self)







    def toggle_select_all(self):



        """Bridge to shared toggle_select_all implementation."""



        return EmbeddingSearchWorker.toggle_select_all(self)







    def select_all_notes(self):



        """Bridge to shared select_all_notes implementation."""



        return EmbeddingSearchWorker.select_all_notes(self)







    def deselect_all_notes(self):



        """Bridge to shared deselect_all_notes implementation."""



        return EmbeddingSearchWorker.deselect_all_notes(self)







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







    def _restore_answer_html(self, html):



        """Restore the answer box HTML (used after link click so the AI answer does not disappear)."""



        if html and hasattr(self, 'answer_box'):



            self.answer_box.setHtml(html)







    def _on_answer_link_clicked(self, url):



        """Citation links: single-click highlights note in Matching notes; double-click opens in Anki Browser (over add-on). Supports #cite-N, anki:goto_note:{note_id}, and legacy note:N."""



        import time



        saved_html = getattr(self, '_last_formatted_answer', None) or (self.answer_box.toHtml() if hasattr(self.answer_box, 'toHtml') else None)



        s = url.toString() if hasattr(url, 'toString') else str(url)



        if not s:



            if saved_html:



                self.answer_box.setHtml(saved_html)



            return



        ctx = getattr(self, '_context_note_ids', None) or []



        note_id = None



        num = None



        if s.startswith('#cite-'):



            try:



                num = int(s.replace('#cite-', '').strip())



                if 1 <= num <= len(ctx):



                    note_id = ctx[num - 1]



            except (ValueError, TypeError):



                pass



        elif s.startswith('anki:goto_note:'):



            try:



                note_id = int(s.split(':', 2)[2].strip())



                if ctx and note_id in ctx:



                    num = ctx.index(note_id) + 1



                else:



                    num = note_id



            except (ValueError, IndexError):



                pass



        elif s.startswith('note:'):



            try:



                num = int(s.split(':', 1)[1].strip())



                if 1 <= num <= len(ctx):



                    note_id = ctx[num - 1]



            except (ValueError, IndexError):



                pass



        if note_id is None:



            if saved_html:



                self.answer_box.setHtml(saved_html)



            return



        if num is None:



            num = (ctx.index(note_id) + 1) if note_id in ctx else note_id







        # Single vs double click: second click on same link within 400ms = open browser; else only highlight



        now = time.time()



        last = getattr(self, '_citation_last_click', None)



        is_double = last is not None and last[0] == s and (now - last[1]) < 0.4



        self._citation_last_click = None if is_double else (s, now)







        if is_double:



            self._open_note_in_browser(note_id, num)







        # Always highlight the corresponding row in the results list



        if hasattr(self, 'all_scored_notes') and self.all_scored_notes:



            self._pinned_note_ids.add(note_id)



            if hasattr(self, 'selected_note_ids'):



                self.selected_note_ids.add(note_id)



            max_score = self.all_scored_notes[0][0]



            thresh = self.sensitivity_slider.value() if self.sensitivity_slider else 0



            min_score = (thresh / 100.0) * max_score if max_score > 0 else 0



            id_to_score = {n['id']: s for s, n in self.all_scored_notes}



            pinned_orig_scores = [id_to_score.get(nid, 0) for nid in self._pinned_note_ids]



            any_filtered = any(orig < min_score for orig in pinned_orig_scores)



            if any_filtered and self.sensitivity_slider is not None:



                self.sensitivity_slider.blockSignals(True)



                self.sensitivity_slider.setValue(0)



                if self.sensitivity_value_label is not None:



                    self.sensitivity_value_label.setText("0%")



                self.sensitivity_slider.blockSignals(False)



            order = {nid: i for i, nid in enumerate(ctx)}



            pinned = []



            rest = []



            for score, note in self.all_scored_notes:



                if note['id'] in self._pinned_note_ids:



                    pinned.append((max_score, note))



                else:



                    rest.append((score, note))



            pinned.sort(key=lambda x: order.get(x[1]['id'], 999))



            self.all_scored_notes = pinned + rest



            self.filter_and_display_notes()







            # Scroll to and highlight the note row (match by Ref when available so chunk display scrolls to cited ref)



            for row in range(self.results_list.rowCount()):



                ref_item = self.results_list.item(row, 0)



                content_item = self.results_list.item(row, 1)



                if num is not None and ref_item and str(ref_item.text()) == str(num):



                    if content_item:



                        self.results_list.selectRow(row)



                        self.results_list.scrollToItem(content_item)



                    break



                if num is None and content_item and content_item.data(Qt.ItemDataRole.UserRole) == note_id:



                    self.results_list.selectRow(row)



                    self.results_list.scrollToItem(content_item)



                    break







        if saved_html:



            QTimer.singleShot(0, lambda h=saved_html: self._restore_answer_html(h))







    def _citation_timer_clear(self):



        self._citation_last_click = None







    def _bring_browser_to_front(self, browser):



        """Raise browser window after a short delay so it stays on top of the add-on dialog."""



        if browser and hasattr(browser, 'activateWindow'):



            browser.activateWindow()



            browser.raise_()







    def _open_note_in_browser(self, note_id, num):



        """Open note in Anki Browser (used when user double-clicks a citation link). Brings browser to front over add-on."""



        try:



            browser = dialogs.open("Browser", mw)



            if browser:



                browser.form.searchEdit.lineEdit().setText(f"nid:{note_id}")



                browser.onSearchActivated()



                QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))



                tooltip(f"Opened note [{num}] (ID: {note_id}) in browser")



        except Exception as e:



            log_debug(f"Error opening note in browser: {e}")



            tooltip(f"Could not open note [{num}] in browser")







    def _spacing_styles(self):



        mode = self.styling_config.get('answer_spacing', 'normal')



        if mode == 'compact':



            return {'lh': '1.2', 'p': '0.15em 0 0.3em 0', 'ul': '0.15em 0 0.3em 0', 'li': '0.08em 0'}



        if mode == 'comfortable':



            return {'lh': '1.5', 'p': '0.3em 0 0.5em 0', 'ul': '0.3em 0 0.5em 0', 'li': '0.15em 0'}



        return {'lh': '1.35', 'p': '0.2em 0 0.4em 0', 'ul': '0.2em 0 0.4em 0', 'li': '0.1em 0'}







    def format_answer(self, answer):



        import re



        import html







        if not answer:



            return ""







        s = self._spacing_styles()



        # Escape HTML first, then we'll allow <strong> etc. via placeholders



        def rich_escape(text):



            escaped = html.escape(text)



            # Bold: **...** (allow multiline) and **... at end of line (missing closing **)

            escaped = MD_BOLD_RE.sub(r'<strong style="font-weight: bold;">\1</strong>', escaped)

            escaped = MD_UNTERMINATED_BOLD_RE.sub(r'<strong style="font-weight: bold;">\1</strong>', escaped)

            # __...__ ΓåÆ <strong>

            escaped = MD_BOLD_ALT_RE.sub(r'<strong style="font-weight: bold;">\1</strong>', escaped)

            # Short "Label:" lines (e.g. Locations:, Risk Factors:) ΓåÆ bold the label (max 50 chars)

            escaped = MD_HEADER_RE.sub(r'<strong style="font-weight: bold;">\1</strong>:\2', escaped)

            # Highlight exam-style: ~~...~~ ΓåÆ yellow background

            escaped = MD_HIGHLIGHT_RE.sub(r'<span style="background-color: rgba(255,235,59,0.45); padding: 0 2px;">\1</span>', escaped)



            # Normalize bracket-like characters so [1], [47] etc. always match (some models output \xef\xbc\xbb\xef\xbc\xbd\xe3\u20ac\x90\xe3\u20ac\u2018)



            for _open, _close in (('\uFF3B', '\uFF3D'), ('\u3010', '\u3011'), ('\u301A', '\u301B')):



                escaped = escaped.replace(_open, '[').replace(_close, ']')



            # Convert [1], [2], [N40] to clickable links.

            ctx = getattr(self, '_context_note_ids', None) or []

            ctx_len = len(ctx)

            def cite_link(m):

                raw = m.group(1)

                pairs = []  # (note_id or None, display text, 1-based pos)

                for part in raw.split(','):

                    d = part.strip()

                    n = d.lstrip('N').strip()

                    if n.isdigit():

                        pos = int(n)

                        if 1 <= pos <= ctx_len:

                            note_id = ctx[pos - 1]

                            pairs.append((note_id, d, pos))

                        else:

                            pairs.append((None, d, 0))

                if not pairs:

                    return m.group(0)

                links = []

                for note_id, disp, pos in pairs:

                    if note_id is not None:

                        # Use #cite-N fragment so links are reliably clickable in QTextBrowser (anki: scheme can be blocked)

                        links.append(f'<a href="#cite-{pos}" style="color:#3498db;text-decoration:underline;cursor:pointer;" title="Single-click: highlight in list. Double-click: open in browser.">[{disp}]</a>')

                    else:

                        links.append(f'<span title="Citation out of range (max {ctx_len})" style="color:#95a5a6;">[{disp}]</span>')

                return '[' + ','.join(links) + ']'



            # [N2, N4, N8] or [N40]: allow N and digits inside brackets

            escaped = CITATION_N_RE.sub(cite_link, escaped)

            escaped = CITATION_RE.sub(cite_link, escaped)  # [1], [2], [38,43]

            return escaped







        lines = answer.split('\n')



        result_lines = []



        in_list = False



        list_depth = 0  # 0 = none, 1 = top-level ul, 2 = nested ul







        for raw in lines:



            line = raw.rstrip()



            if not line:



                if in_list:



                    if list_depth == 2:



                        result_lines.append('</ul></li></ul>')



                    elif list_depth == 1:



                        result_lines.append('</ul>')



                    in_list = False



                    list_depth = 0



                result_lines.append('<br>')



                continue







            # Detect indent for sub-bullets (2+ spaces or tab before bullet)



            stripped = line.lstrip()



            indent = len(line) - len(stripped)



            is_sub = indent >= 2 and (stripped.startswith('\u2022') or stripped.startswith('-') or stripped.startswith('*'))







            # Section header: ## Something \u2192 bold with ring (\u25cf\x8f), no hashes



            if stripped.startswith('##'):



                if in_list:



                    if list_depth == 2:



                        result_lines.append('</ul></li></ul>')



                    elif list_depth == 1:



                        result_lines.append('</ul>')



                    in_list = False



                    list_depth = 0



                title = stripped.lstrip('#').strip()



                title = rich_escape(title)



                result_lines.append(f'<p style="margin: 0.75em 0 0.4em 0; font-weight: bold; font-size: 1.22em;">\u25cf\x8f {title}</p>')



                continue







            # Single # heading (treat as section too)



            if stripped.startswith('#') and not stripped.startswith('##'):



                if in_list:



                    if list_depth == 2:



                        result_lines.append('</ul></li></ul>')



                    elif list_depth == 1:



                        result_lines.append('</ul>')



                    in_list = False



                    list_depth = 0



                title = stripped.lstrip('#').strip()



                title = rich_escape(title)



                result_lines.append(f'<p style="margin: 0.75em 0 0.4em 0; font-weight: bold; font-size: 1.22em;">\u25cf\x8f {title}</p>')



                continue







            # Bullet (\u2022, -, *)



            if stripped.startswith('\u2022') or stripped.startswith('-') or stripped.startswith('*'):



                content = stripped.lstrip('\u2022-*').strip()



                # Strip any remaining leading bullets (e.g. "\u2022 \u2022 Types..." or "\u00b7 Types...")



                bullet_chars = '\u2022-*\u00b7\u25cf\x8f\u25e6\u2219\u2022\u2023\u00b7'



                while content and content[0] in bullet_chars:



                    content = content[1:].lstrip()



                content = rich_escape(content)



                # Skip empty bullets (e.g. "- " or trailing bullet) so answer doesn't look incomplete



                if not content or not content.strip():



                    continue



                if is_sub and in_list and list_depth == 1:



                    # Start nested list inside the previous <li> (replace last </li> with <ul><li>...)



                    if result_lines and result_lines[-1].strip().endswith('</li>'):



                        result_lines[-1] = result_lines[-1].rstrip().rstrip('</li>').rstrip() + '<ul style="margin: 0.2em 0 0.2em 0.8em; padding-left: 1.2em; list-style-type: circle;">'



                    else:



                        result_lines.append('<ul style="margin: 0.2em 0 0.2em 0.8em; padding-left: 1.2em; list-style-type: circle;">')



                    result_lines.append(f'<li style="margin: {s["li"]};">{content}</li>')



                    list_depth = 2



                elif is_sub and in_list and list_depth == 2:



                    result_lines.append(f'<li style="margin: {s["li"]};">{content}</li>')



                elif not in_list:



                    result_lines.append(f'<ul style="margin: {s["ul"]}; padding-left: 1.3em; list-style-type: disc;">')



                    result_lines.append(f'<li style="margin: {s["li"]};">{content}</li>')



                    in_list = True



                    list_depth = 1



                else:



                    if list_depth == 2:



                        result_lines.append('</ul></li>')



                        list_depth = 1



                    result_lines.append(f'<li style="margin: {s["li"]};">{content}</li>')



                continue







            # Plain paragraph



            if in_list:



                if list_depth == 2:



                    result_lines.append('</ul></li></ul>')



                elif list_depth == 1:



                    result_lines.append('</ul>')



                in_list = False



                list_depth = 0



            result_lines.append(f'<p style="margin: {s["p"]};">{rich_escape(line)}</p>')







        if in_list:



            if list_depth == 2:



                result_lines.append('</ul></li></ul>')



            elif list_depth == 1:



                result_lines.append('</ul>')







        html_content = ''.join(result_lines)



        # Force link styling inside the document (QTextBrowser may not apply widget stylesheet to content)



        link_style = (



            "<style>a { color: #3498db !important; text-decoration: underline !important; } "



            "a:hover { color: #5dade2 !important; }</style>"



        )



        # Add invisible targets for #cite-N so QTextBrowser doesn't navigate away when link is clicked



        ctx_len = len(getattr(self, '_context_note_ids', None) or [])



        anchors = ''.join(f'<span id="cite-{i}" style="position:absolute;width:0;height:0;"></span>' for i in range(1, ctx_len + 1))



        return f'{link_style}<div style="line-height: {s["lh"]}; margin: 0;">{html_content}</div>{anchors}'







    def copy_answer_to_clipboard(self):



        html = getattr(self, '_last_formatted_answer', None) or ""



        plain = self.answer_box.toPlainText().strip()



        if html or plain:



            cb = QApplication.clipboard()



            if cb:



                mime = QMimeData()



                if html:



                    mime.setHtml(html)



                mime.setText(plain if plain else "")



                cb.setMimeData(mime)



                tooltip("Copied (paste into Word for bullets and formatting)")



        else:



            tooltip("No answer to copy")







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







    def get_all_notes_content(self):



        log_debug("Starting to load notes from collection...")



        notes_data = []



        config = load_config()



        ntf = config.get('note_type_filter', {})







        # Migrate: fields_to_search -> note_type_fields



        if ntf.get('fields_to_search') and not ntf.get('note_type_fields'):



            global_flds = set(f.lower() for f in ntf['fields_to_search'])



            ntf = dict(ntf)



            ntf['note_type_fields'] = {}



            for model_name, _c, field_names in get_models_with_fields():



                ntf['note_type_fields'][model_name] = [f for f in field_names if f.lower() in global_flds]







        # Backward compat: empty ntf -> legacy Text & Extra globally



        legacy_fields = None



        if not ntf:



            enabled_set = None



            search_all = False



            ntf_fields = {}



            use_first = False



            legacy_fields = {'text', 'extra'}



            self.fields_description = "Text & Extra"



        else:



            enabled = ntf.get('enabled_note_types')



            enabled_set = set(enabled) if (enabled and len(enabled) > 0) else None



            search_all = bool(ntf.get('search_all_fields', False))



            ntf_fields = ntf.get('note_type_fields') or {}



            use_first = bool(ntf.get('use_first_field_fallback', True))



            self.fields_description = "all fields" if search_all else "per-type"



        deck_q = _build_deck_query(ntf.get('enabled_decks') if ntf else None)



        note_ids = mw.col.find_notes(deck_q) if deck_q else mw.col.find_notes("")



        total_notes = len(note_ids)



        cache_key = (deck_q or '', frozenset(enabled_set) if enabled_set is not None else None, search_all, tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in (ntf_fields or {}).items())), total_notes)



        if getattr(self, '_cached_notes_key', None) == cache_key and getattr(self, '_cached_notes', None) is not None:



            log_debug("Using cached notes (singleton search engine)")



            return self._cached_notes



        log_debug(f"Found {total_notes} total notes (decks: {'all' if not deck_q else 'filtered'}, note types: {'all' if enabled_set is None else 'filtered'}, fields: {self.fields_description})")







        # Pre-calculate field indices for each model

        model_map = {}

        for m in mw.col.models.all():

            mid = m['id']

            m_name = m['name']

            if enabled_set is not None and m_name not in enabled_set:

                continue

            flds = m['flds']

            if search_all:

                indices = list(range(len(flds)))

            else:

                if legacy_fields is not None:

                    wanted = legacy_fields

                else:

                    wanted = set(f.lower() for f in (ntf_fields.get(m_name) or []))

                    if not wanted and use_first and flds:

                        wanted = {flds[0]['name'].lower()}

                indices = [i for i, f in enumerate(flds) if f['name'].lower() in wanted]

            if indices:

                model_map[mid] = (m_name, indices)



        import hashlib

        CHUNK_TARGET = 500

        id_list = ",".join(map(str, note_ids))

        for idx, (nid, mid, flds_str) in enumerate(mw.col.db.execute(f"select id, mid, flds from notes where id in ({id_list})")):

            try:

                model_info = model_map.get(mid)

                if not model_info:

                    continue

                model_name, indices = model_info

                fields = flds_str.split("\x1f")

                content_parts = []

                for i in indices:

                    if i < len(fields):

                        val = fields[i].strip()

                        if val:

                            content_parts.append(val)



                if not content_parts:

                    continue



                content = " | ".join(content_parts)

                content = self.strip_html(content)

                if not content.strip():

                    continue



                # Semantic chunking: split long notes at sentence boundaries (~500 chars), same Note ID per chunk for section citation

                chunks = _semantic_chunk_text(content, CHUNK_TARGET)

                if len(chunks) <= 1:

                    content_hash = hashlib.md5(content.encode()).hexdigest()

                    notes_data.append({'id': nid, 'content': content, 'content_hash': content_hash, 'model': model_name, 'display_content': content})

                else:

                    for chunk_idx, chunk in enumerate(chunks):

                        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()

                        notes_data.append({

                            'id': nid, 'content': chunk, 'content_hash': chunk_hash, 'model': model_name,

                            'display_content': chunk, 'chunk_index': chunk_idx, '_full_content': content,

                        })



                if (idx + 1) % 1000 == 0:

                    progress = ((idx + 1) / total_notes) * 100

                    log_debug(f"Processed {idx + 1}/{total_notes} notes ({progress:.1f}%)")

                    try:

                        if hasattr(self, 'status_label') and self.status_label:

                            self.status_label.setText(f"Loading notes... {idx + 1}/{total_notes} ({progress:.1f}%)")

                            QApplication.processEvents()

                    except RuntimeError:

                        pass

            except Exception as e:

                log_debug(f"Error processing note {nid}: {str(e)}")

                continue







        if not hasattr(self, 'fields_description'):



            self.fields_description = "Text & Extra"



        log_debug(f"Loaded {len(notes_data)} notes from collection")



        self._cached_notes = notes_data



        self._cached_notes_key = cache_key



        return notes_data







    def _aggregate_scored_notes_by_note_id(self, scored_notes):



        """After semantic chunking, collapse multiple chunks per note to one entry per note (best score, full content)."""



        if not scored_notes:



            return scored_notes



        by_id = {}



        for score, note in scored_notes:



            nid = note.get('id')



            if nid is None:



                by_id[id(note)] = (score, note)



                continue



            if nid not in by_id or score > by_id[nid][0]:



                rep = dict(note)



                if rep.get('_full_content'):



                    rep['display_content'] = rep['content'] = rep['_full_content']



                by_id[nid] = (score, rep)



        return sorted(by_id.values(), key=lambda x: -x[0])







    def strip_html(self, text):



        return _strip_html_plain(text)







    def reveal_cloze_for_display(self, text):

        """Reveal cloze deletions for display: {{cN::answer}} or {{cN::answer::hint}} -> answer (plain text with answer shown)."""

        if not text:

            return text

        # {{c1::answer}} or {{c1::answer::hint}} -> answer (capture until }} or :: so answer can contain colons)

        return CLOZE_RE.sub(r'\1', text)









    # ========== SEMANTIC SEARCH IMPROVEMENTS ==========







    def _simple_stem(self, word):



        """Simple stemming: remove common suffixes"""



        if len(word) <= 3:



            return word



        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's', 'es', 'tion', 'sion', 'ness', 'ment']



        word_lower = word.lower()



        for suffix in suffixes:



            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:



                return word_lower[:-len(suffix)]



        return word_lower







    def _get_extended_stop_words(self):



        """Return extended stop words, including optional domain-specific extras from config."""



        builtin_stop_words = {



            'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who',



            'does', 'do', 'can', 'could', 'would', 'should', 'tell', 'me', 'about', 'explain',



            'describe', 'define', 'list', 'show', 'give', 'provide', 'this', 'that', 'these',



            'those', 'with', 'from', 'for', 'and', 'or', 'but', 'not', 'have', 'has', 'had',



            'been', 'being', 'was', 'were', 'will', 'would', 'may', 'might', 'must', 'shall',



            'which', 'work', 'works', 'working', 'use', 'uses', 'used', 'using',



            'cause', 'causes',



            'overview', 'introduction', 'review', 'study', 'case', 'cases',



            'difference', 'between', 'compared', 'comparison', 'similar', 'similarity',



            'different', 'same', 'other', 'another', 'each', 'every', 'both', 'either', 'neither',



            'like', 'such', 'same', 'common', 'generally', 'usually', 'often', 'typically',



            'example', 'examples', 'including', 'involves', 'involve', 'related', 'association',



        }



        try:



            config = load_config()



            sc = (config or {}).get('search_config') or {}



            stop_words = set(builtin_stop_words)



            extra = sc.get('extra_stop_words') or []



            if isinstance(extra, str):



                extra = [extra]



            extra_set = {



                (w or '').strip().lower()



                for w in extra



                if isinstance(w, str) and (w or '').strip()



            }



            stop_words.update(extra_set)



        except Exception:



            stop_words = set(builtin_stop_words)



        return stop_words







    def _extract_keywords_improved(self, query):



        """Improved keyword extraction with stemming and better stop word handling"""



        import re







        query_lower = query.lower()



        query_words = re.findall(r'\b\w+\b', query_lower)







        # Extended stop words (generic question words + structural fillers + optional domain extras)



        stop_words = self._get_extended_stop_words()



        ai_excluded = getattr(self, '_query_ai_excluded_terms', None) or set()



        if not isinstance(ai_excluded, set):



            ai_excluded = set(ai_excluded) if ai_excluded else set()







        # Extract keywords with stemming



        keywords = []



        stems = {}



        for w in query_words:



            if w not in stop_words and w not in ai_excluded and len(w) > 2:



                stem = self._simple_stem(w)



                keywords.append(w)



                if stem != w:



                    stems[stem] = w







        # Generate n-grams (bigrams and trigrams)



        phrases = []



        if len(keywords) > 1:



            for i in range(len(keywords) - 1):



                phrases.append(keywords[i] + " " + keywords[i + 1])



        if len(keywords) > 2:



            for i in range(len(keywords) - 2):



                phrases.append(keywords[i] + " " + keywords[i + 1] + " " + keywords[i + 2])



        # region agent log



        try:



            if "trisom" in query_lower:



                _agent_debug_log(



                    run_id="pre-fix",



                    hypothesis_id="H1",



                    location="__init__._extract_keywords_improved",



                    message="keywords_extracted",



                    data={



                        "query": query,



                        "keywords": keywords,



                        "stems": stems,



                        "phrases": phrases[:10],



                    },



                )



        except Exception:



            pass



        # endregion



        return keywords, stems, phrases







    def _compute_tfidf_scores(self, notes, query_keywords):



        """Compute TF-IDF scores for keywords across notes"""



        import math







        # Handle trivial cases up front



        if not notes or not query_keywords:



            # Reset per-query high-frequency keywords cache



            try:



                self._query_high_freq_keywords = set()



            except Exception:



                pass



            return {}







        # Term frequency in each note



        note_tfs = {}



        # Document frequency (how many notes contain each keyword)



        doc_freq = {}







        for note in notes:



            content_lower = note['content'].lower()



            note_tfs[note['id']] = {}



            for keyword in query_keywords:



                count = content_lower.count(keyword)



                if count > 0:



                    note_tfs[note['id']][keyword] = count



                    doc_freq[keyword] = doc_freq.get(keyword, 0) + 1







        # Compute TF-IDF scores



        total_notes = max(1, len(notes))



        # Automatically down-weight very common keywords so you don't have to



        # manually add every generic word as a stop word. Any query keyword that



        # appears in a large fraction of notes for this search is treated as a



        # high-frequency term and ignored by the later keyword scorer.



        high_freq_threshold = 0.65  # e.g. appears in >=65% of candidate notes



        high_freq_keywords = {



            kw for kw, df in doc_freq.items()



            if df / total_notes >= high_freq_threshold



        }



        try:



            # Stash on the instance so scoring functions can see it for this query



            self._query_high_freq_keywords = high_freq_keywords



        except Exception:



            self._query_high_freq_keywords = high_freq_keywords







        tfidf_scores = {}



        for note_id, tfs in note_tfs.items():



            score = 0



            for keyword, tf in tfs.items():



                idf = math.log(total_notes / (doc_freq.get(keyword, 1) + 1))



                tfidf = tf * idf



                score += tfidf



            tfidf_scores[note_id] = score







        return tfidf_scores







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







    def _expand_query(self, query, config):



        """Expand query with synonyms using AI (optional). Supports Ollama and cloud providers."""



        search_config = load_config().get('search_config', {})







        # Always apply a small built\xe2\u20ac\u2018in synonym map for very common medical



        # variants so you don't have to remember which spelling your deck



        # used (e.g. adrenaline vs epinephrine). This runs even when the



        # AI-based expansion setting is disabled.



        try:



            q_lower = (query or "").lower()



            extra_terms = []



            # Pairs and small groups of common aliases / spelling variants



            synonym_groups = [



                # Catecholamines



                ["adrenaline", "epinephrine"],



                ["noradrenaline", "norepinephrine"],



                # Analgesics



                ["acetaminophen", "paracetamol"],



                # Hormones / vitamins (common exam phrasing variants)



                ["pth", "parathyroid hormone"],



                ["vitamin d", "cholecalciferol", "ergocalciferol"],



            ]



            for group in synonym_groups:



                present = [term for term in group if term in q_lower]



                if present:



                    for term in group:



                        if term not in q_lower:



                            extra_terms.append(term)



            # Config-driven synonym overrides (same logic; no UI, edit config.json if needed)



            overrides = search_config.get("synonym_overrides") or []



            for item in overrides:



                if isinstance(item, (list, tuple)) and len(item) >= 2:



                    group = [str(t).strip().lower() for t in item if t and str(t).strip()]



                    if len(group) < 2:



                        continue



                    present = [term for term in group if term in q_lower]



                    if present:



                        for term in group:



                            if term not in q_lower:



                                extra_terms.append(term)



            if extra_terms:



                query = f"{query} " + " ".join(extra_terms)



        except Exception:



            # If anything goes wrong here, just fall back to the original query



            pass







        # Optional AI-based expansion (controlled from settings)



        if not search_config.get('enable_query_expansion', False):



            return query







        try:



            # Use a simple prompt to get synonyms / closely related terms



            prompt = (



                "Given this search query, list 2\xe2\u20ac\u201c4 key synonyms or closely related medical terms "



                "that would help find the same content. Return only the terms, comma-separated, "



                "no explanations or labels.\n\n"



                f"Query: {query}\n\n"



                "Synonyms:"



            )







            provider = config.get('provider', 'openai')



            import urllib.request



            import json







            # Ollama: fully local HTTP API



            if provider == 'ollama':



                sc = config.get('search_config') or search_config



                base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()



                # Allow a dedicated expansion model, falling back to chat model



                model = (



                    sc.get('ollama_query_expansion_model')



                    or sc.get('ollama_chat_model')



                    or 'llama3.2'



                )



                model = str(model).strip()



                url = base_url.rstrip("/") + "/api/generate"



                data = {



                    "model": model,



                    "prompt": prompt,



                    "stream": False,



                    "options": {"num_predict": 64},



                }



                req = urllib.request.Request(



                    url,



                    data=json.dumps(data).encode(),



                    headers={"Content-Type": "application/json"},



                    method="POST",



                )



                resp = urllib.request.urlopen(req, timeout=8)



                result = json.loads(resp.read())



                expanded = (result.get("response") or "").strip()







            else:



                api_key = config.get('api_key', '')



                if not api_key:



                    return query







                model = self.get_best_model(provider)







                # Quick API call for expansion (cloud providers)



                if provider == "openai":



                    url = "https://api.openai.com/v1/chat/completions"



                    data = {



                        "model": model,



                        "messages": [{"role": "user", "content": prompt}],



                        "max_tokens": 50,



                        "temperature": 0.3,



                    }



                    headers = {



                        "Authorization": f"Bearer {api_key}",



                        "Content-Type": "application/json",



                    }



                elif provider == "anthropic":



                    url = "https://api.anthropic.com/v1/messages"



                    data = {



                        "model": model,



                        "max_tokens": 50,



                        "messages": [{"role": "user", "content": prompt}],



                    }



                    headers = {



                        "x-api-key": api_key,



                        "anthropic-version": "2023-06-01",



                        "Content-Type": "application/json",



                    }



                elif provider == "google":



                    url = (



                        "https://generativelanguage.googleapis.com/v1beta/models/"



                        f"{model}:generateContent?key={api_key}"



                    )



                    data = {



                        "contents": [{"parts": [{"text": prompt}]}],



                        "generationConfig": {"maxOutputTokens": 50, "temperature": 0.3},



                    }



                    headers = {"Content-Type": "application/json"}



                else:



                    # Skip expansion for unsupported providers



                    return query







                req = urllib.request.Request(



                    url,



                    data=json.dumps(data).encode(),



                    headers=headers,



                )



                resp = urllib.request.urlopen(req, timeout=5)



                result = json.loads(resp.read())







                if provider == "openai":



                    expanded = result["choices"][0]["message"]["content"].strip()



                elif provider == "anthropic":



                    expanded = result["content"][0]["text"].strip()



                elif provider == "google":



                    expanded = result["candidates"][0]["content"]["parts"][0]["text"].strip()



                else:



                    return query







            # Clean up and combine



            expanded = (expanded or "").replace("Synonyms:", "").strip()



            if not expanded:



                return query







            # Parse comma-separated terms, trim, and drop empties



            terms = [t.strip() for t in expanded.split(",") if t.strip()]



            if not terms:



                return query







            # Append terms to the original query so keyword extraction sees them



            return f"{query} " + " ".join(terms)



        except Exception as e:



            log_debug(f"Query expansion failed: {e}")







        return query







    def _get_ai_excluded_terms(self, query, config):



        """One short LLM call to detect generic query terms to exclude. Returns set of lowercased terms, or empty set on failure."""



        prompt = (



            "List only words that are too generic to help find specific study notes "



            "(e.g. 'difference', 'between', 'what', 'the', 'mechanism', 'treatment'). "



            "Include question filler and words that appear in most notes. "



            "Return comma-separated words only, or exactly 'none' if all words are useful. "



            f"Query: {query}"



        )



        try:



            import urllib.request



            import json



            provider = config.get('provider', 'openai')



            search_config = config.get('search_config') or {}



            response_text = ""



            if provider == 'ollama':



                sc = config.get('search_config') or search_config



                base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()



                model = (sc.get('ollama_chat_model') or 'llama3.2').strip()



                url = base_url.rstrip("/") + "/api/generate"



                data = {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 50}}



                req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}, method="POST")



                resp = urllib.request.urlopen(req, timeout=8)



                result = json.loads(resp.read())



                response_text = (result.get("response") or "").strip()



            else:



                api_key = config.get('api_key', '')



                if not api_key:



                    return set()



                model = self.get_best_model(provider)



                if provider == "openai":



                    url = "https://api.openai.com/v1/chat/completions"



                    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 50, "temperature": 0.1}



                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}



                    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")



                    resp = urllib.request.urlopen(req, timeout=8)



                    result = json.loads(resp.read())



                    response_text = (result.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()



                elif provider == "anthropic":



                    url = "https://api.anthropic.com/v1/messages"



                    data = {"model": model, "max_tokens": 50, "messages": [{"role": "user", "content": prompt}]}



                    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}



                    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")



                    resp = urllib.request.urlopen(req, timeout=8)



                    result = json.loads(resp.read())



                    response_text = (result.get("content") or [{}])[0].get("text", "").strip()



                elif provider == "google":



                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"



                    data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 50, "temperature": 0.1}}



                    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}, method="POST")



                    resp = urllib.request.urlopen(req, timeout=8)



                    result = json.loads(resp.read())



                    response_text = (result.get("candidates") or [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()



                else:



                    return set()



            raw = (response_text or "").strip().lower()



            if not raw:



                return set()



            if raw in ("none", "n/a", "no", "nil"):



                return set()



            terms = []



            for part in raw.replace(";", ",").split(","):



                t = part.strip()



                if t and t not in ("none", "n/a", "no", "nil"):



                    terms.append(t)



            result = set(terms)



            if search_config.get('verbose_search_debug') and result:



                log_debug(f"AI generic term detection excluded for this query: {result}")



            return result



        except Exception as e:



            log_debug(f"AI generic term detection failed: {e}")



            return set()







    def _generate_hyde_document(self, query, config):



        """Generate a brief hypothetical answer (HyDE) for retrieval: AI 'hallucinates' an answer, then we search on it."""



        HYDE_MAX_TOKENS = 60



        prompt = (



            "Write a brief 1\xe2\u20ac\u201c2 sentence hypothetical answer, as if from your study notes. "



            "Plain text only, no markdown.\n\nQuestion: " + query



        )



        try:



            provider = config.get('provider', 'openai')



            api_key = config.get('api_key', '')



            if provider == 'ollama':



                sc = config.get('search_config') or {}



                base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()



                model = (sc.get('ollama_chat_model') or 'llama3.2').strip()



                url = base_url.rstrip("/") + "/api/generate"



                data = {



                    "model": model,



                    "prompt": prompt,



                    "stream": False,



                    "options": {"num_predict": HYDE_MAX_TOKENS}



                }



                req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}, method="POST")



                resp = urllib.request.urlopen(req, timeout=15)



                result = json.loads(resp.read())



                return (result.get("response") or "").strip()



            model = self.get_best_model(provider)



            if provider == "openai":



                url = "https://api.openai.com/v1/chat/completions"



                data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": HYDE_MAX_TOKENS, "temperature": 0.3}



                req = urllib.request.Request(url, data=json.dumps(data).encode(),



                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")



                resp = urllib.request.urlopen(req, timeout=60)



                result = json.loads(resp.read())



                return result['choices'][0]['message']['content'].strip()



            elif provider == "anthropic":



                url = "https://api.anthropic.com/v1/messages"



                data = {"model": model, "max_tokens": HYDE_MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]}



                req = urllib.request.Request(url, data=json.dumps(data).encode(),



                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, method="POST")



                resp = urllib.request.urlopen(req, timeout=60)



                result = json.loads(resp.read())



                return result['content'][0]['text'].strip()



            elif provider == "google":



                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"



                data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": HYDE_MAX_TOKENS, "temperature": 0.3}}



                req = urllib.request.Request(url, data=json.dumps(data).encode(),



                    headers={"Content-Type": "application/json"}, method="POST")



                resp = urllib.request.urlopen(req, timeout=60)



                result = json.loads(resp.read())



                return result['candidates'][0]['content']['parts'][0]['text'].strip()



            elif provider == "openrouter":



                url = "https://openrouter.ai/api/v1/chat/completions"



                data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": HYDE_MAX_TOKENS, "temperature": 0.3}



                req = urllib.request.Request(url, data=json.dumps(data).encode(),



                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")



                resp = urllib.request.urlopen(req, timeout=60)



                result = json.loads(resp.read())



                return result['choices'][0]['message']['content'].strip()



            else:



                # Custom endpoint (OpenAI-compatible shape)



                api_url = (config.get('api_url') or '').strip()



                if not api_url:



                    return None



                data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": HYDE_MAX_TOKENS, "temperature": 0.3}



                req = urllib.request.Request(api_url, data=json.dumps(data).encode(),



                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")



                resp = urllib.request.urlopen(req, timeout=60)



                result = json.loads(resp.read())



                if 'choices' in result:



                    return result['choices'][0]['message']['content'].strip()



                if 'content' in result and result['content']:



                    return result['content'][0].get('text', '').strip()



                return None



        except Exception as e:



            log_debug(f"HyDE generation failed: {e}")



        return None







    def _passes_focused_balanced_broad(



        self,



        matched_keywords,



        final_score,



        emb_score,



        max_emb_score,



        keywords,



        search_method,



        embeddings_available,



        min_emb_frac=0.25,



        very_high_emb_frac=0.9,



    ):



        """Compute whether a note would pass Focused, Balanced, or Broad inclusion. Returns (passes_focused, passes_balanced, passes_broad)."""



        n_kw = len(keywords) if keywords else 0



        # Focused



        min_kw_focused = max(2, int(n_kw * 0.4)) if n_kw else 1



        if n_kw <= 2:



            min_kw_focused = 1



        min_score_focused = 18



        # Balanced



        min_kw_balanced = max(1, int(n_kw * 0.25)) if n_kw else 1



        min_score_balanced = 10



        # Broad



        min_kw_broad = max(1, int(n_kw * 0.2)) if n_kw else 1



        min_score_broad = 8







        if search_method == "embedding" and embeddings_available:



            if emb_score > 0:



                return (True, True, True)



            return (False, False, False)







        if search_method == "hybrid" and embeddings_available and max_emb_score > 0:



            very_high = emb_score >= very_high_emb_frac * max_emb_score



            decent = emb_score >= min_emb_frac * max_emb_score



            pf = (



                (decent and (matched_keywords >= min_kw_focused or final_score > min_score_focused)) or very_high



            ) and (matched_keywords > 0 or very_high)



            pb_al = (decent and (matched_keywords >= min_kw_balanced or final_score > min_score_balanced)) or very_high



            pb_br = (decent and (matched_keywords >= min_kw_broad or final_score > min_score_broad)) or very_high



            return (pf, pb_al, pb_br)







        # Keyword-only or fallback



        pf = matched_keywords >= min_kw_focused



        if n_kw <= 2:



            pb_al = matched_keywords >= 1



            pb_br = matched_keywords >= 1



        else:



            pb_al = matched_keywords >= min_kw_balanced or final_score > min_score_balanced



            pb_br = matched_keywords >= min_kw_broad or final_score > min_score_broad



        return (pf, pb_al, pb_br)











def _is_embedding_dimension_mismatch(exc):



    """Detect NumPy shape mismatch (e.g. 768 vs 1024) from embedding engine switch."""



    s = str(exc)



    return "not aligned" in s or ("shapes" in s and "dim" in s)











def _run_embedding_search_sync(embedding_query, notes, config, db_path=None):



    """Run embedding search in a background thread (for taskman). Returns scored_notes or None.



    db_path: from main thread so profile-specific path is correct in background."""



    import hashlib



    try:



        import numpy as np



    except ImportError:



        return None



    try:



        embedding_list = get_embedding_for_query(embedding_query, config)



        if not embedding_list:



            return None



        query_embedding = np.array(embedding_list)



        query_dim = len(query_embedding)



        total = len(notes)



        scored_notes = []



        for idx, note in enumerate(notes):



            content_hash = note.get('content_hash')



            from_note = content_hash is not None



            if content_hash is None:



                content_hash = hashlib.md5(note['content'].encode()).hexdigest()



            note_embedding = load_embedding(note.get('id'), content_hash, db_path=db_path)



            if note_embedding is not None:



                emb = np.array(note_embedding)



                if len(emb) != query_dim:



                    log_debug(f"Embedding search: skipping note {note.get('id')} (dimension mismatch query={query_dim} stored={len(emb)})")



                    continue



                similarity = np.dot(query_embedding, emb) / (



                    max(np.linalg.norm(query_embedding) * np.linalg.norm(emb), 1e-9)



                )



                score = (similarity + 1) * 50



                scored_notes.append((score, note))



        scored_notes.sort(reverse=True, key=lambda x: x[0])



        return scored_notes[:80]



    except Exception as e:



        log_debug(f"Embedding search error: {e}")



        if _is_embedding_dimension_mismatch(e):



            return {"embedding_results": None, "error": "dimension_mismatch"}



        return None











class EmbeddingSearchWorker(QThread):



    """Worker thread for embedding search (fallback when taskman not used)."""



    progress_signal = pyqtSignal(int, int, str)



    finished_signal = pyqtSignal(object)



    error_signal = pyqtSignal(str)







    def __init__(self, embedding_query, notes, config, db_path=None):



        super().__init__()



        self.embedding_query = embedding_query



        self.notes = notes



        self.config = config



        self.db_path = db_path







    def run(self):



        import hashlib



        try:



            import numpy as np



        except ImportError:



            self.error_signal.emit("numpy not available")



            self.finished_signal.emit(None)



            return



        try:



            embedding_list = get_embedding_for_query(self.embedding_query, self.config)



            if not embedding_list:



                self.finished_signal.emit(None)



                return



            query_embedding = np.array(embedding_list)



            total = len(self.notes)



            scored_notes = []



            progress_interval = max(50, total // 40)



            for idx, note in enumerate(self.notes):



                if self.isInterruptionRequested():



                    self.finished_signal.emit(None)



                    return



                if idx % progress_interval == 0 or idx == total - 1:



                    pct = int(100 * (idx + 1) / total) if total else 0



                    self.progress_signal.emit(idx + 1, total, f"Embedding search: {idx + 1}/{total} ({pct}%)")



                content_hash = note.get('content_hash')



                if content_hash is None:



                    content_hash = hashlib.md5(note['content'].encode()).hexdigest()



                note_embedding = load_embedding(note.get('id'), content_hash, db_path=getattr(self, 'db_path', None))



                if note_embedding is not None:



                    emb = np.array(note_embedding)



                    if len(emb) != len(query_embedding):



                        log_debug(f"EmbeddingSearchWorker: skipping note {note.get('id')} (dimension mismatch query={len(query_embedding)} stored={len(emb)})")



                        continue



                    similarity = np.dot(query_embedding, emb) / (



                        max(np.linalg.norm(query_embedding) * np.linalg.norm(emb), 1e-9)



                    )



                    score = (similarity + 1) * 50



                    scored_notes.append((score, note))



            scored_notes.sort(reverse=True, key=lambda x: x[0])



            self.finished_signal.emit(scored_notes[:80])



        except Exception as e:



            log_debug(f"EmbeddingSearchWorker error: {e}")



            if _is_embedding_dimension_mismatch(e):



                msg = (



                    "Embedding dimension mismatch: your notes were embedded with a different engine "



                    "(e.g. Voyage). Run Create/Update Embeddings with your current engine (Ollama) to enable hybrid search."



                )



                self.error_signal.emit(msg)



            else:



                self.error_signal.emit(str(e)[:200])



            self.finished_signal.emit(None)











MAX_RERANK_COUNT = 15  # Limit rerank to top 15 to avoid CPU bottleneck (~2s with 50 notes)



RRF_K = 60  # Reciprocal Rank Fusion constant (standard in retrieval literature; 1/(k+rank) per list)











def _semantic_chunk_text(text, target_size=500):

    """Split text into chunks of ~target_size chars at sentence boundaries. Returns list of non-empty strings."""

    text = (text or "").strip()

    if not text or len(text) <= target_size:

        return [text] if text else []

    chunks = []

    # Use pre-compiled sentence boundary pattern

    start = 0

    while start < len(text):



        end = min(start + target_size, len(text))



        if end >= len(text):



            chunk = text[start:].strip()



            if chunk:



                chunks.append(chunk)



            break



        # Find last sentence boundary in this window



        segment = text[start:end]



        last_dot = segment.rfind('.')



        last_excl = segment.rfind('!')



        last_q = segment.rfind('?')



        last_nl = segment.rfind('\n')



        best = max(last_dot, last_excl, last_q, last_nl)



        if best >= target_size // 2:



            end = start + best + 1



            chunk = text[start:end].strip()



        else:



            chunk = text[start:end].strip()



        if chunk:



            chunks.append(chunk)



        start = end



    return chunks if chunks else [text]











def _do_rerank(query, scored_notes, top_k, search_config):



    """



    Re-rank top results using a cross-encoder (gold standard for NotebookLM-style accuracy).



    Uses top_k=15 by default to avoid CPU bottleneck. Blends cross-encoder scores with pre-rerank.



    Returns (scored_notes, success).



    """



    import json



    import os



    import subprocess



    top_k = min(top_k, MAX_RERANK_COUNT)



    top_notes = scored_notes[:top_k]



    if not top_notes:



        return scored_notes, False



    pre_scores = {note['id']: score for score, note in top_notes}



    contents = [note.get('content', '')[:512] for _, note in top_notes]



    rerank_python = (search_config.get('rerank_python_path') or '').strip()



    if rerank_python:



        python_exe = rerank_python



        if os.path.isdir(rerank_python):



            python_exe = os.path.join(rerank_python, "python.exe")



            if not os.path.isfile(python_exe):



                python_exe = os.path.join(rerank_python, "python")



        if os.path.isfile(python_exe):



            addon_dir = os.path.dirname(os.path.abspath(__file__))



            helper_path = os.path.join(addon_dir, "rerank_helper.py")



            if os.path.isfile(helper_path):



                try:



                    payload = json.dumps({"query": query, "contents": contents})



                    creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0



                    proc = subprocess.Popen(



                        [python_exe, helper_path],



                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,



                        text=True, creationflags=creationflags



                    )



                    out, err = proc.communicate(input=payload, timeout=60)



                    if proc.returncode != 0:



                        log_debug(f"Rerank helper failed: {err or out}")



                        return scored_notes, False



                    data = json.loads(out)



                    if "error" in data:



                        log_debug(f"Rerank helper error: {data['error']}")



                        return scored_notes, False



                    scores = data.get("scores", [])



                    if len(scores) != len(top_notes):



                        return scored_notes, False



                    reranked = list(zip(scores, [note for _, note in top_notes]))



                    reranked.sort(reverse=True, key=lambda x: x[0])



                    min_s, max_s = min(s[0] for s in reranked), max(s[0] for s in reranked)



                    span = max_s - min_s if max_s > min_s else 1



                    normalized = [(50 + 50 * (s - min_s) / span, note) for s, note in reranked]



                    # Blend for ordering; display % = cross-encoder score so "answers the question" gets high, tangential gets low



                    blend = []



                    for rn, note in normalized:



                        pre = pre_scores.get(note['id'], 50)



                        blended = 0.5 * rn + 0.5 * pre



                        blend.append((blended, note))



                        # rn is 50-100; show as 0-100 so relevance reflects "answers this question" not just topic similarity



                        note['_display_relevance'] = max(0, min(100, round((rn - 50) * 2)))



                    blend.sort(reverse=True, key=lambda x: x[0])



                    max_b = blend[0][0] if blend else 1



                    scaled = [(b / max_b * 100.0, note) for b, note in blend]



                    # Soft floor: notes that were top-15 by pre-rerank don't show below 55%



                    top_pre_ids = set(nid for _, nid in sorted(pre_scores.items(), key=lambda x: -x[1])[:15])



                    scaled = [(max(pct, 55.0) if note['id'] in top_pre_ids else pct, note) for pct, note in scaled]



                    scaled.sort(reverse=True, key=lambda x: x[0])



                    # Renormalize _display_relevance so top note(s) show 100% and rest spread below



                    max_d = max((note.get('_display_relevance') or 0) for _, note in scaled)



                    if max_d > 0:



                        for _, note in scaled:



                            p = note.get('_display_relevance')



                            if p is not None:



                                note['_display_relevance'] = max(0, min(100, round(100 * p / max_d)))



                    # Rest notes weren't reranked; no _display_relevance (UI will use score)



                    rest = [(0.0, note) for _, note in scored_notes[top_k:]]



                    return scaled + rest, True



                except subprocess.TimeoutExpired:



                    proc.kill()



                    log_debug("Rerank helper timed out")



                    return scored_notes, False



                except Exception as e:



                    log_debug(f"Rerank subprocess failed: {e}")



                    return scored_notes, False



    try:



        _patch_colorama_early()



        _ensure_stderr_patched()



        from sentence_transformers import CrossEncoder



    except ImportError:



        log_debug("Cross-encoder re-ranking skipped: sentence-transformers not installed")



        try:



            from aqt.utils import showInfo



            showInfo(



                "sentence-transformers is not installed.\n\n"



                "Cross-Encoder re-ranking is disabled. To enable it, click "



                "'Install Dependencies' in the AI Search menu (or in Settings \u2192 Search & Embeddings)."



            )



        except Exception:



            pass



        return scored_notes, False



    except OSError as e:



        log_debug(f"Cross-encoder re-ranking skipped: library load failed ({e})")



        # Do not use showInfo in background thread. Return success=False and handle in UI.



        return scored_notes, "LIBRARY_LOAD_FAILED"



    except Exception as e:



        log_debug(f"Cross-encoder re-ranking failed with unexpected error: {e}")



        return scored_notes, False



    try:



        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



        pairs = [(query, note['content'][:512]) for _, note in top_notes]



        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)



        reranked = list(zip(scores, [note for _, note in top_notes]))



        reranked.sort(reverse=True, key=lambda x: x[0])



        min_s, max_s = min(s[0] for s in reranked), max(s[0] for s in reranked)



        span = max_s - min_s if max_s > min_s else 1



        normalized = [(50 + 50 * (s - min_s) / span, note) for s, note in reranked]



        # Blend for ordering; display % = cross-encoder score so "answers the question" gets high, tangential gets low



        blend = []



        for rn, note in normalized:



            pre = pre_scores.get(note['id'], 50)



            blended = 0.5 * rn + 0.5 * pre



            blend.append((blended, note))



            # rn is 50-100; show as 0-100 so relevance reflects "answers this question" not just topic similarity



            note['_display_relevance'] = max(0, min(100, round((rn - 50) * 2)))



        blend.sort(reverse=True, key=lambda x: x[0])



        max_b = blend[0][0] if blend else 1



        scaled = [(b / max_b * 100.0, note) for b, note in blend]



        # Soft floor: notes that were top-15 by pre-rerank don't show below 55%



        top_pre_ids = set(nid for _, nid in sorted(pre_scores.items(), key=lambda x: -x[1])[:15])



        scaled = [(max(pct, 55.0) if note['id'] in top_pre_ids else pct, note) for pct, note in scaled]



        scaled.sort(reverse=True, key=lambda x: x[0])



        # Renormalize _display_relevance so top note(s) show 100% and rest spread below



        max_d = max((note.get('_display_relevance') or 0) for _, note in scaled)



        if max_d > 0:



            for _, note in scaled:



                p = note.get('_display_relevance')



                if p is not None:



                    note['_display_relevance'] = max(0, min(100, round(100 * p / max_d)))



        # Rest notes weren't reranked; no _display_relevance (UI will use score)



        rest = [(0.0, note) for _, note in scored_notes[top_k:]]



        return scaled + rest, True



    except Exception as e:



        log_debug(f"Cross-encoder re-ranking failed: {e}")



        return scored_notes, False











class RerankCheckWorker(QThread):



    """Worker thread for checking sentence-transformers availability so Settings doesn't freeze."""



    finished_signal = pyqtSignal(bool)  # available







    def __init__(self, dialog, python_path=None):



        super().__init__()



        self._dialog = dialog



        self._python_path = python_path







    def run(self):



        try:



            result = self._dialog._check_rerank_available(python_path=self._python_path)



            self.finished_signal.emit(result)



        except Exception:



            self.finished_signal.emit(False)











class KeywordFilterWorker(QThread):



    """Worker thread for keyword_filter so search doesn't freeze the main thread."""



    finished_signal = pyqtSignal(object)  # result from keyword_filter







    def __init__(self, dialog, query, notes):



        super().__init__()



        self._dialog = dialog



        self._query = query



        self._notes = notes







    def run(self):



        try:



            result = self._dialog.keyword_filter(self._query, self._notes)



            self.finished_signal.emit(result)



        except Exception as e:



            log_debug(f"KeywordFilterWorker error: {e}")



            self.finished_signal.emit(None)











class KeywordFilterContinueWorker(QThread):



    """Worker thread for keyword_filter_continue so combining results doesn't freeze the UI. Emits progress."""



    progress_signal = pyqtSignal(int, int, str)  # current, total, message



    finished_signal = pyqtSignal(object)  # result







    def __init__(self, dialog, state, embedding_results):



        super().__init__()



        self._dialog = dialog



        self._state = state



        self._embedding_results = embedding_results







    def run(self):



        try:



            def progress_callback(idx, total):



                self.progress_signal.emit(idx, total, f"Combining results... {idx}/{total}")







            result = self._dialog.keyword_filter_continue(



                self._state, self._embedding_results, progress_callback=progress_callback



            )



            self.finished_signal.emit(result)



        except Exception as e:



            log_debug(f"KeywordFilterContinueWorker error: {e}")



            self.finished_signal.emit(None)











class RerankWorker(QThread):



    """Worker thread for cross-encoder reranking so the UI stays responsive."""



    finished_signal = pyqtSignal(object, bool)  # (scored_notes, success)







    def __init__(self, query, scored_notes, top_k, search_config):



        super().__init__()



        self.query = query



        self.scored_notes = scored_notes



        self.top_k = top_k



        self.search_config = search_config







    def run(self):



        try:



            scored_notes, success = _do_rerank(self.query, self.scored_notes, self.top_k, self.search_config)



            self.finished_signal.emit(scored_notes, success)



        except Exception as e:



            log_debug(f"RerankWorker error: {e}")



            self.finished_signal.emit(self.scored_notes, False)











class RelevanceRerankWorker(QThread):



    """Worker for re-ranking notes by similarity to the AI answer (relevance-from-answer). Runs embeddings off the main thread to avoid lag."""



    progress_signal = pyqtSignal(int, str)   # percent, message



    finished_signal = pyqtSignal(object)     # new all_scored_notes or None on failure







    def __init__(self, answer_text, note_texts, all_scored_notes, config):



        super().__init__()



        self.answer_text = answer_text



        self.note_texts = note_texts



        self.all_scored_notes = all_scored_notes



        self.config = config







    def run(self):



        try:



            import numpy as np



            self.progress_signal.emit(5, "Re-ranking by relevance... (embedding answer)")



            answer_emb = get_embedding_for_query(self.answer_text, self.config)



            if not answer_emb:



                self.finished_signal.emit(None)



                return



            self.progress_signal.emit(20, "Re-ranking by relevance... (embedding notes)")



            note_embs = get_embeddings_batch(self.note_texts, input_type="document", config=self.config)



            if not note_embs or len(note_embs) != len(self.all_scored_notes):



                self.finished_signal.emit(None)



                return



            self.progress_signal.emit(70, "Re-ranking by relevance... (scoring)")



            answer_vec = np.array(answer_emb, dtype=float)



            norm_a = max(np.linalg.norm(answer_vec), 1e-9)



            new_scores = []



            for i, (_, note) in enumerate(self.all_scored_notes):



                ne = np.array(note_embs[i], dtype=float)



                norm_n = max(np.linalg.norm(ne), 1e-9)



                sim = float(np.dot(answer_vec, ne) / (norm_a * norm_n))



                pct = max(0, min(100, round((sim + 1) * 50)))



                note['_display_relevance'] = pct



                new_scores.append((float(pct), note))



            new_scores.sort(reverse=True, key=lambda x: x[0])



            if new_scores:



                max_pct = new_scores[0][0]



                if max_pct > 0:



                    for score, note in new_scores:



                        note['_display_relevance'] = max(0, min(100, round(100 * (note['_display_relevance'] or 0) / max_pct)))



                    new_scores = [(100.0 if i == 0 else (note['_display_relevance'] or 0), note) for i, (_, note) in enumerate(new_scores)]



                    new_scores.sort(reverse=True, key=lambda x: x[0])



            self.progress_signal.emit(100, "Re-ranking by relevance... (done)")



            self.finished_signal.emit(new_scores)



        except Exception as e:



            log_debug(f"RelevanceRerankWorker error: {e}")



            self.finished_signal.emit(None)











class AskAIWorker(QThread):



    """Run ask_ai in a background thread so the main thread stays responsive (no 'Not Responding')."""



    success_signal = pyqtSignal(object, object)  # (answer, relevant_indices)



    error_signal = pyqtSignal(str)







    def __init__(self, dialog, query, context_notes, context, config):



        super().__init__()



        self._dialog = dialog



        self._query = query



        self._context_notes = context_notes



        self._context = context



        self._config = config







    def run(self):



        try:



            answer, relevant_indices = self._dialog.ask_ai(



                self._query, self._context_notes, self._context, self._config



            )



            self.success_signal.emit(answer, relevant_indices)



        except Exception as e:



            log_debug(f"AskAIWorker error: {e}")



            self.error_signal.emit(str(e))











def _normalize_query_space(text):
    return re.sub(r"\s+", " ", (text or "")).strip()











def _build_anthropic_prompt_parts(query, context, focus_instruction=None, answer_style_instruction=None, constraint_instruction=None):



    """Build (system_blocks, user_content) for Anthropic with prompt caching. System + context use cache_control."""



    system_instruction = """You are an assistant for question-answering over provided notes. Use ONLY the numbered notes below as your factual source (you may add brief connecting logic, but no outside facts).



If the notes contain at least some relevant information, give the **best partial answer you can** based only on these notes and then briefly mention what is missing.



Only if the notes are essentially unrelated to the question, say exactly: "The provided notes do not contain enough information to answer this."







Rules:



- Base every claim strictly on these notes. One sentence or bullet per idea is fine.



- Write in a clear, exam-oriented style: use bullet points (\u2022) for key points; use 2-space indented bullets for sub-points. Use **double asterisks** around important terms (diagnoses, drugs, criteria). Do not use ## for headings\u2014use a single bold line with \u25cf\x8f then bullets underneath.



- NUMBERED LISTS: When the question or notes refer to a numbered list (e.g. six childhood exanthems, 1st\xe2\u20ac\u201c6th disease, steps 1\xe2\u20ac\u201c6 of ventricular partitioning), **always** present in **strict numerical order**: 1st, then 2nd, then 3rd, then 4th, then 5th, then 6th. **Include every step or number that any note describes**\u2014if Note X explicitly mentions "3rd step" or "third step", you MUST include it in your answer and cite it [X]. Do not skip steps; do not reorder by relevance.



- INLINE CITATIONS: Cite the supporting note(s) using [N] or [N,M] where N is between 1 and the number of notes below only. Do not use citation numbers outside that range.



- At the end, on one line, list all note numbers you cited. Format: RELEVANT_NOTES: 1,3,5"""



    num_notes = context.count("Note ")  # approximate; we pass explicit N in caller if needed



    context_block = f"""Context information is below. There are notes numbered Note 1, Note 2, ... (cite only using numbers 1 to the number of notes below).



---------------------



{context}



---------------------"""



    system_blocks = [



        {"type": "text", "text": system_instruction},



        {"type": "text", "text": context_block, "cache_control": {"type": "ephemeral"}},



    ]



    user_content = f"""Given the context information and not prior knowledge, answer the question.







Question: {query}"""



    extra_instructions = [focus_instruction, answer_style_instruction, constraint_instruction]
    for instruction in extra_instructions:
        if instruction:
            user_content += "\n" + instruction



    return system_blocks, user_content











class AnthropicStreamWorker(QThread):



    """Worker thread for Anthropic streaming. Emits text chunks for real-time UI updates."""



    chunk_signal = pyqtSignal(str)



    done_signal = pyqtSignal(str)



    error_signal = pyqtSignal(str)







    def __init__(self, api_key, model, system_blocks, user_content, notes):



        super().__init__()



        self.api_key = api_key



        self.model = model



        self.system_blocks = system_blocks



        self.user_content = user_content



        self.notes = notes







    def run(self):



        try:



            import anthropic



            client = anthropic.Anthropic(api_key=self.api_key)



            full_text = ""



            with client.messages.stream(



                model=self.model,



                max_tokens=4096,



                system=self.system_blocks,



                messages=[{"role": "user", "content": self.user_content}],



            ) as stream:



                for text in stream.text_stream:



                    full_text += text



                    self.chunk_signal.emit(text)



            self.done_signal.emit(full_text)



        except Exception as e:



            log_debug(f"AnthropicStreamWorker error: {e}")



            self.error_signal.emit(str(e))











# END OF PART 2 - PART 3: Methods below are indented under EmbeddingSearchWorker



# but are copied to AISearchDialog at module load (see _aisearch_methods_from_worker).







    def _get_answer_source_text(self, config):



        """Return a short hint: where the answer came from (online API name or local model)."""



        if not config:



            return ""



        provider = config.get("provider", "openai")



        if provider == "ollama":



            sc = config.get("search_config") or {}



            model = (sc.get("ollama_chat_model") or "llama3.2").strip()



            return f"Ollama (local) \u2014 {model}"



        names = {



            "anthropic": "Anthropic (Claude)",



            "openai": "OpenAI (GPT)",



            "google": "Google (Gemini)",



            "openrouter": "OpenRouter",



            "custom": "Custom API",



        }



        name = names.get(provider, "API")



        model = self.get_best_model(provider)



        return f"{name} \u2014 {model}"







    def _on_embedding_search_progress(self, current, total, message):



        """Update status and progress bar while embedding search runs in background."""



        try:



            if hasattr(self, 'status_label') and self.status_label:



                self.status_label.setText(message)



            if hasattr(self, 'search_progress_bar') and self.search_progress_bar and total > 0:



                self.search_progress_bar.setRange(0, total)



                self.search_progress_bar.setValue(current)



                self.search_progress_bar.setVisible(True)



            if hasattr(self, 'search_progress_label') and self.search_progress_label:



                self.search_progress_label.setText(f"{current}/{total}")



                self.search_progress_label.setVisible(True)



        except Exception:



            pass







    def _show_busy_progress(self, message=""):



        """Show indeterminate progress bar and optional label during long operations (re-rank, AI call, load)."""



        self._show_centile_progress(message, 0)







    def _show_centile_progress(self, message="", percent=0):



        """Show 0\xe2\u20ac\u201c100% progress bar and label. Use for estimated or real progress during long operations."""



        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:



            self.search_progress_bar.setRange(0, 100)



            self.search_progress_bar.setValue(max(0, min(100, round(percent))))



            self.search_progress_bar.setVisible(True)



        if hasattr(self, 'search_progress_label') and self.search_progress_label:



            self.search_progress_label.setText(message)



            self.search_progress_label.setVisible(True)



        self._last_progress_message = message







    def _start_estimated_progress_timer(self, duration_sec, start_pct=5, end_pct=95):



        """Advance progress bar from start_pct to end_pct over duration_sec (est. wait). Call _stop_estimated_progress_timer when done."""



        import time



        self._stop_estimated_progress_timer()



        self._progress_estimate_active = True



        self._progress_estimate_start = time.time()



        self._progress_estimate_duration = max(1, duration_sec)



        self._progress_estimate_start_pct = start_pct



        self._progress_estimate_end_pct = end_pct







        def _tick():



            if not getattr(self, '_progress_estimate_active', False):



                return



            elapsed = time.time() - getattr(self, '_progress_estimate_start', 0)



            dur = getattr(self, '_progress_estimate_duration', 30)



            s = getattr(self, '_progress_estimate_start_pct', 5)



            e = getattr(self, '_progress_estimate_end_pct', 95)



            pct = s + (elapsed / dur) * (e - s)



            pct = max(s, min(e, pct))



            msg = getattr(self, '_last_progress_message', '')



            self._show_centile_progress(msg, pct)



            if elapsed < dur:



                QTimer.singleShot(500, _tick)







        QTimer.singleShot(300, _tick)







    def _stop_estimated_progress_timer(self):



        """Stop the estimated progress timer (e.g. when the long operation finishes)."""



        self._progress_estimate_active = False







    def _hide_busy_progress(self):



        """Hide progress bar and label; reset bar to deterministic range for next use."""



        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:



            self.search_progress_bar.setRange(0, 100)



            self.search_progress_bar.setValue(0)



            self.search_progress_bar.setVisible(False)



        if hasattr(self, 'search_progress_label') and self.search_progress_label:



            self.search_progress_label.setText("")



            self.search_progress_label.setVisible(False)







    def _on_embedding_search_finished(self, embedding_results):



        """Embedding search worker finished; continue with scoring and display."""



        try:



            if hasattr(self, 'search_progress_bar') and self.search_progress_bar:



                self.search_progress_bar.setVisible(False)



            if hasattr(self, 'search_progress_label') and self.search_progress_label:



                self.search_progress_label.setVisible(False)



            state = getattr(self, '_search_pending_state', None)



            notes = getattr(self, '_search_pending_notes', None)



            if state is None or notes is None:



                self.status_label.setText("Ready")



                self.search_btn.setEnabled(True)



                return



            combine_in_background = False



            # Handle dimension-mismatch dict from _run_embedding_search_sync



            if isinstance(embedding_results, dict) and "error" in embedding_results:



                setattr(self, "_last_embedding_error", embedding_results.get("error"))



                embedding_results = embedding_results.get("embedding_results")



            else:



                setattr(self, "_last_embedding_error", None)



            # Run keyword_filter_continue in a worker so the UI shows progress and does not freeze.



            if embedding_results:



                if hasattr(self, 'status_label') and self.status_label:



                    self.status_label.setText("Combining results...")



                if hasattr(self, 'search_progress_bar') and self.search_progress_bar:



                    self.search_progress_bar.setVisible(True)



                    self.search_progress_bar.setRange(0, 100)



                    self.search_progress_bar.setValue(0)



                if hasattr(self, 'search_progress_label') and self.search_progress_label:



                    self.search_progress_label.setVisible(True)



                QApplication.processEvents()



                self._keyword_filter_continue_worker = KeywordFilterContinueWorker(self, state, embedding_results)



                self._keyword_filter_continue_worker.progress_signal.connect(self._on_embedding_search_progress)



                self._keyword_filter_continue_worker.finished_signal.connect(



                    lambda result: self._on_keyword_filter_continue_done(result, state)



                )



                self._keyword_filter_continue_worker.start()



                combine_in_background = True



            else:



                result = self.keyword_filter_continue(state, embedding_results)



                self._on_keyword_filter_continue_done(result, state)



        except Exception as e:



            log_debug(f"Embedding search finish error: {e}")



            if hasattr(self, 'status_label') and self.status_label:



                self.status_label.setText("Error occurred")



            self.answer_box.setText(f"\xe2\x9d\u0152 Error after embedding search:\n{str(e)}")



        finally:



            if not combine_in_background and not getattr(self, '_pending_rerank', False):



                self.search_btn.setEnabled(True)



            QApplication.processEvents()







    def _on_keyword_filter_continue_done(self, result, state):



        """Called when keyword_filter_continue finishes (worker or inline). Starts rerank or continues to display."""



        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:



            self.search_progress_bar.setVisible(False)



        if hasattr(self, 'search_progress_label') and self.search_progress_label:



            self.search_progress_label.setVisible(False)



        notes = state.get("notes") or getattr(self, '_search_pending_notes', None)



        if result is None or notes is None:



            self.status_label.setText("Ready")



            self.search_btn.setEnabled(True)



            return



        if isinstance(result, tuple) and result[0] == "PENDING_RERANK":



            _, notes, scored_notes, effective_method, _ = result



            self._pending_rerank = True



            search_config = state.get("search_config") or {}



            self._rerank_continue = (notes, effective_method, search_config)



            if hasattr(self, 'status_label') and self.status_label:



                self.status_label.setText("Re-ranking results... (may take 10-30 s)")



            self._show_centile_progress("Re-ranking...", 0)



            self._start_estimated_progress_timer(25, 5, 95)



            query = state.get("query", "")



            try:



                from aqt.operations import QueryOp



                op = QueryOp(



                    parent=mw,



                    op=lambda col: _do_rerank(query, scored_notes, MAX_RERANK_COUNT, search_config),



                    success=lambda pair: self._on_rerank_done(pair[0], pair[1]),



                )



                op.run_in_background()



            except Exception:



                self._rerank_worker = RerankWorker(query, scored_notes, MAX_RERANK_COUNT, search_config)



                self._rerank_worker.finished_signal.connect(self._on_rerank_done)



                self._rerank_worker.start()



            return



        scored_notes, effective_method, total_above_threshold = result



        self._perform_search_continue(notes, scored_notes, effective_method, total_above_threshold)



        self.search_btn.setEnabled(True)







    def _on_embedding_search_error(self, error_msg):



        """Handle embedding search worker error."""



        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:



            self.search_progress_bar.setVisible(False)



        if hasattr(self, 'search_progress_label') and self.search_progress_label:



            self.search_progress_label.setVisible(False)



        self.status_label.setText("Error occurred")



        self.answer_box.setText(f"\xe2\x9d\u0152 Embedding search failed:\n{error_msg}")



        self.search_btn.setEnabled(True)







    def _perform_search_continue(self, notes, scored_notes, effective_method, total_above_threshold):



        """Continue search after keyword_filter (or after embedding worker): display results, call AI, etc."""



        config = load_config()



        query = getattr(self, 'current_query', '')



        log_debug(f"Filtered to {len(scored_notes)} potentially relevant notes (method: {effective_method}, total above threshold: {total_above_threshold})")



        self.all_scored_notes = scored_notes



        self._total_above_threshold = total_above_threshold



        self._last_search_method = effective_method



        if hasattr(self, 'search_method_result_label'):



            search_config = config.get("search_config") or {}



            mode = (search_config.get("relevance_mode") or "").strip().lower()



            if mode == "focused":



                mode_display = "Focused"



            elif mode == "broad":



                mode_display = "Broad"



            elif mode:



                mode_display = mode.capitalize()



            else:



                mode_display = "Balanced"



            engine = (search_config.get("embedding_engine") or "ollama").strip().lower()



            engine_display = {



                "ollama": "Ollama (local)",



                "voyage": "Voyage AI",



                "openai": "OpenAI",



                "cohere": "Cohere",



            }.get(engine, engine or "unknown")



            label_text = f"Results from: {effective_method} \u00b7 {mode_display} \u00b7 Embeddings: {engine_display}"



            self.search_method_result_label.setText(label_text)



            self.search_method_result_label.setVisible(True)



            # When user chose embedding/hybrid but we fell back to keyword, add a hint



            if "Keyword only" in effective_method and getattr(self, '_last_requested_search_method', None) in ('embedding', 'hybrid'):



                err = getattr(self, "_last_embedding_error", None)



                if err == "dimension_mismatch":



                    engine = (config.get("search_config") or {}).get("embedding_engine") or "ollama"



                    engine_display = {"ollama": "Ollama", "voyage": "Voyage AI", "openai": "OpenAI", "cohere": "Cohere"}.get(engine, engine)



                    hint = format_dimension_mismatch_hint(engine_display)



                else:



                    hint = "embedding unavailable \u2014 run Create/Update Embeddings in Settings \u2192 Search & Embeddings, or check API key"



                self.search_method_result_label.setText(f"Results from: {effective_method} ({hint})")



        if not scored_notes:



            n_searched = getattr(self, 'total_notes_searched', None) or len(set(n['id'] for n in notes))



            self.answer_box.setText(f"No notes found matching keywords from your query. Searched {n_searched} notes ({getattr(self, 'fields_description', 'Text & Extra')}).")



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.status_label.setText("No matches found")



            return



        # Use un-aggregated list for AI context when we have chunks so the AI can cite specific sections



        raw_for_context = getattr(self, '_scored_notes_for_context', None)



        if raw_for_context:



            relevant_notes = [note for _, note in raw_for_context]



        else:



            relevant_notes = [note for _, note in scored_notes]



        history_result = load_search_history(query)



        used_history = False



        if history_result:



            log_debug("Using cached search result from history")



            if hasattr(self, 'search_method_result_label'):



                if getattr(self, '_last_search_method', None) == "Hybrid":



                    self.search_method_result_label.setText("Results from: cache (same query as before)")



                    self.search_method_result_label.setVisible(True)



                else:



                    self.search_method_result_label.setVisible(False)



            if not hasattr(self, '_total_above_threshold'):



                self._total_above_threshold = len(self.all_scored_notes)



            self.status_label.setText("\U0001F4DA Loading from cache... (saved AI API call)")



            QApplication.processEvents()



            answer = history_result.get('answer', '')



            relevant_indices = []



            used_history = True



            self._context_note_ids = history_result.get('context_note_ids') or []



            self._context_note_id_and_chunk = None  # History has no chunk info; use ctx order for Ref



            self._display_scored_notes = None  # History uses aggregated list



            if 'scored_notes' in history_result:



                history_scored = []



                note_id_map = {note['id']: note for _, note in scored_notes}



                for score, hist_note in history_result['scored_notes']:



                    note_id = hist_note.get('id')



                    if note_id in note_id_map:



                        history_scored.append((score, note_id_map[note_id]))



                if history_scored:



                    history_note_ids = {note['id'] for _, note in history_scored}



                    for score, note in scored_notes:



                        if note['id'] not in history_note_ids:



                            history_scored.append((score, note))



                    self.all_scored_notes = sorted(history_scored, reverse=True, key=lambda x: x[0])



                relevant_note_ids = set(history_result.get('relevant_note_ids', []))



                relevant_notes = [note for _, note in self.all_scored_notes]



                for idx, note in enumerate(relevant_notes):



                    if note['id'] in relevant_note_ids:



                        relevant_indices.append(idx)



                if not self._context_note_ids and self.all_scored_notes:



                    self._context_note_ids = [n['id'] for _, n in self.all_scored_notes]



            else:



                self._context_note_ids = [n['id'] for _, n in self.all_scored_notes] if self.all_scored_notes else []



        else:



            # Cap notes/chunks sent to the AI (avoids rate limits and token overflow; chunked results can be huge)



            search_config = config.get('search_config') or {}



            max_context = max(5, min(50, search_config.get('max_results', 12)))



            selected_ids = set(getattr(self, 'selected_note_ids', set()) or [])



            pinned_ids = set(getattr(self, '_pinned_note_ids', set()) or [])



            priority_ids = selected_ids | pinned_ids



            if priority_ids:



                prioritized = [n for n in relevant_notes if n['id'] in priority_ids]



                remaining = [n for n in relevant_notes if n['id'] not in priority_ids]



                context_notes = (prioritized + remaining)[:max_context]



            else:



                context_notes = list(relevant_notes)[:max_context]



            context_note_ids = [n['id'] for n in context_notes]



            # Store (note_id, chunk_index) in context order so Ref column and citation [N] match



            self._context_note_id_and_chunk = [(n['id'], n.get('chunk_index')) for n in context_notes]



            # Store (score, note) for each context item so display can show all refs the AI can cite



            if raw_for_context:



                note_to_score = {}



                for s, n in raw_for_context:



                    key = (n['id'], n.get('chunk_index'))



                    note_to_score[key] = s



                display_pairs = []



                for n in context_notes:



                    k = (n['id'], n.get('chunk_index'))



                    display_pairs.append((note_to_score.get(k) or 0, n))



                self._display_scored_notes = display_pairs



            else:



                self._display_scored_notes = None



            # Per-note limit: 0 = full content; >0 = truncate (Settings \u2192 Search & Embeddings \u2192 Max chars per note)



            context_chars_per_note = max(0, search_config.get('context_chars_per_note', 0))



            # When we have chunks, label sections so the AI can cite specific sections (e.g. [1], [2] for Note 1 section 2)



            def _context_line(i, n):



                chunk_idx = n.get('chunk_index')



                text = self.reveal_cloze_for_display(n['content'])



                if context_chars_per_note:



                    text = text[:context_chars_per_note]



                if chunk_idx is not None:



                    return f"Note {i+1} (section {chunk_idx + 1} of note ID {n['id']}): {text}"



                return f"Note {i+1}: {text}"



            context = "\n\n".join([_context_line(i, n) for i, n in enumerate(context_notes)])



            n_notes = len(context_notes)



            self.status_label.setText(f"Asking AI... (sending top {n_notes} notes, 10-30 s)")



            self._show_centile_progress(f"Asking AI... ({n_notes} notes)", 0)



            self._start_estimated_progress_timer(30, 5, 95)



            self.answer_box.setPlainText("Thinking...")



            QApplication.processEvents()



            provider = config.get('provider', 'openai')



            if provider == "anthropic":



                try:



                    self._start_anthropic_stream(query, context_notes, context, config, relevant_notes, scored_notes, context_note_ids)



                    return



                except Exception as e:



                    log_debug(f"Anthropic streaming not available, falling back to non-streaming: {e}")



            # Run ask_ai in background so UI stays responsive (avoids "Not Responding" during 10\xe2\u20ac\u201c300 s request)



            self._ask_ai_relevant_notes = relevant_notes



            self._ask_ai_scored_notes = scored_notes



            self._ask_ai_context_note_ids = context_note_ids



            self._ask_ai_used_history = used_history



            self._ask_ai_notes = notes



            self._ask_ai_config = config



            self._ask_ai_worker = AskAIWorker(self, query, context_notes, context, config)



            self._ask_ai_worker.success_signal.connect(self._on_ask_ai_success)



            self._ask_ai_worker.error_signal.connect(self._on_ask_ai_error)



            self._ask_ai_worker.finished.connect(self._on_ask_ai_worker_finished)



            self._ask_ai_worker.start()



            return







    def _on_ask_ai_worker_finished(self):



        """Clear worker reference after thread finishes."""



        self._ask_ai_worker = None







    def _on_ask_ai_error(self, error_msg):



        """Handle AI request failure (runs on main thread)."""



        log_debug(f"Error calling AI API: {error_msg}")



        config = getattr(self, '_ask_ai_config', None) or {}



        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.answer_box.setText(



                "\xe2\x8f\xb1\ufe0f Request timed out. This could mean:\n"



                "\u2022 The API service is slow or overloaded\n"



                "\u2022 Your internet connection is unstable\n\n"



                "Try again or reduce the number of notes in your collection."



            )



        elif "401" in error_msg or "403" in error_msg or "authentication" in error_msg.lower():



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.answer_box.setText("\u26a0\ufe0f Authentication Error:\nYour API key appears to be invalid or expired.\n\nPlease check your API key in Settings.")



        elif "429" in error_msg or "rate limit" in error_msg.lower():



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.answer_box.setText("\xe2\u0161\xa0\ufe0f Rate Limit Exceeded:\nYou've made too many requests.\n\nPlease wait a few minutes and try again.")



        elif config.get('provider') == 'ollama' and any(x in error_msg.lower() for x in ('connection', 'refused', 'connect', 'unreachable')):



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.answer_box.setText(



                "\xe2\x9d\u0152 Cannot reach Ollama.\n\n"



                "Make sure Ollama is running (ollama serve) and the URL in Settings \u2192 Search & Embeddings (Ollama URL) is correct."



            )



        else:



            self.answer_box.setText(f"\xe2\x9d\u0152 Error calling AI API:\n{error_msg}\n\nPlease check your API key and internet connection.")



        if hasattr(self, 'answer_source_label'):



            self.answer_source_label.setText("")



        self._stop_estimated_progress_timer()



        self._hide_busy_progress()



        self.status_label.setText("Error occurred")







    def _on_ask_ai_success(self, answer, relevant_indices):



        """Handle AI answer from worker (runs on main thread)."""



        self._stop_estimated_progress_timer()



        log_debug(f"AI returned answer (length: {len(answer) if answer else 0})")



        relevant_notes = getattr(self, '_ask_ai_relevant_notes', [])



        scored_notes = getattr(self, '_ask_ai_scored_notes', [])



        context_note_ids = getattr(self, '_ask_ai_context_note_ids', [])



        used_history = getattr(self, '_ask_ai_used_history', False)



        notes = getattr(self, '_ask_ai_notes', [])



        config = getattr(self, '_ask_ai_config', {})



        relevant_note_ids = [relevant_notes[idx]['id'] for idx in relevant_indices if 0 <= idx < len(relevant_notes)]



        save_search_history(getattr(self, 'current_query', ''), answer, relevant_note_ids, scored_notes, context_note_ids)



        self._context_note_ids = context_note_ids



        self.current_answer = answer



        ai_relevant_note_ids = set()



        for idx in relevant_indices:



            if 0 <= idx < len(relevant_notes):



                ai_relevant_note_ids.add(relevant_notes[idx]['id'])



        self._cited_note_ids = ai_relevant_note_ids



        improved_scored_notes = []



        for score, note in self.all_scored_notes:



            if note['id'] in ai_relevant_note_ids:



                improved_score = score * 2



            else:



                improved_score = score



            improved_scored_notes.append((improved_score, note))



        improved_scored_notes.sort(reverse=True, key=lambda x: x[0])



        if improved_scored_notes:



            max_boosted = improved_scored_notes[0][0]



            if max_boosted > 0:



                self.all_scored_notes = [(score / max_boosted * 100.0, note) for score, note in improved_scored_notes]



            else:



                self.all_scored_notes = improved_scored_notes



        else:



            self.all_scored_notes = improved_scored_notes



        search_config = config.get('search_config') or {}



        if search_config.get('relevance_from_answer'):



            answer_text = (answer[:8000]).strip()



            note_texts = []



            for _, note in self.all_scored_notes:



                raw = note.get('display_content') or note.get('content', '')



                text = self.reveal_cloze_for_display(raw) if hasattr(self, 'reveal_cloze_for_display') else raw



                note_texts.append((text[:2000]) if text else "")



            if answer_text and note_texts:



                self.status_label.setText("Re-ranking by relevance to answer...")



                self._show_centile_progress("Re-ranking by relevance...", 0)



                self._relevance_rerank_worker = RelevanceRerankWorker(answer_text, note_texts, self.all_scored_notes, config)



                self._relevance_rerank_worker.progress_signal.connect(lambda p, m: self._show_centile_progress(m, p))



                self._relevance_rerank_worker.finished_signal.connect(



                    lambda res: self._on_relevance_rerank_done(res, answer, config, used_history, notes)



                )



                self._relevance_rerank_worker.start()



                return



            try:



                self._rerank_by_relevance_to_answer(answer, config)



            except Exception as e:



                log_debug(f"Relevance-from-answer rerank failed: {e}")



        self._display_answer_and_notes_after_rerank(answer, config, used_history, notes)







    def _on_relevance_rerank_done(self, result, answer, config, used_history, notes):



        """Called when RelevanceRerankWorker finishes. Apply result and display answer/notes."""



        self._hide_busy_progress()



        if getattr(self, '_relevance_rerank_worker', None):



            self._relevance_rerank_worker = None



        if result is not None:



            self.all_scored_notes = result



        self._display_answer_and_notes_after_rerank(answer, config, used_history, notes)







    def _display_answer_and_notes_after_rerank(self, answer, config, used_history, notes):



        """Display formatted answer, filter notes, and update status (shared after rerank or when no rerank)."""



        log_debug("Displaying answer and filtering notes...")



        formatted_answer = self.format_answer(answer)



        self._last_formatted_answer = formatted_answer



        self.answer_box.setHtml(formatted_answer)



        if hasattr(self, 'answer_source_label'):



            src = self._get_answer_source_text(config)



            self.answer_source_label.setText(f"Answer from: {src}" if src else "")



        for attr in ('copy_answer_btn',):



            if hasattr(self, attr):



                getattr(self, attr).setEnabled(True)



        self.filter_and_display_notes()



        threshold = self.sensitivity_slider.value() if getattr(self, 'sensitivity_slider', None) else 0



        cache_indicator = " (\U0001F4DA from cache)" if used_history else ""



        if self.all_scored_notes:



            n_searched = getattr(self, 'total_notes_searched', None) or len(set(n['id'] for n in notes))



            base_text = self.status_label.text() or ""



            suffix = f" (searched {n_searched} in {getattr(self, 'fields_description', 'Text & Extra')}){cache_indicator}"



            if " (searched " in base_text:



                base_text = base_text.split(" (searched ")[0]



            self.status_label.setText(base_text + suffix)



        else:



            self.status_label.setText(f"Found {len(self.all_scored_notes)} relevant notes{cache_indicator}")



        self._refresh_search_history()



        log_debug("Search completed successfully")







    def perform_search(self):



        """Perform search with proper error handling and UI updates"""



        log_debug("=== Perform Search Called ===")







        # Check for config



        config = self.get_config()



        log_debug(f"Retrieved config for search: {get_safe_config(config)}")







        if not config:



            log_debug("ERROR: No config found")



            self.answer_box.setText("Please configure your API key first. Click the \u2699 button.")



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            tooltip("API not configured")



            return







        query = self.search_input.toPlainText().strip()



        if not query:



            tooltip("Please enter a search query")



            return







        # Disable search button to prevent multiple clicks



        self.search_btn.setEnabled(False)



        self.status_label.setText("Searching notes... (this may take 10-30 seconds)")



        self.answer_box.clear()



        self.results_list.setRowCount(0)  # Clear table



        self._update_view_all_button_state()



        if hasattr(self, 'search_method_result_label'):



            self.search_method_result_label.setText("")



        self.total_notes_searched = None



        self._pinned_note_ids = set()



        self._cited_note_ids = set()  # clear until new answer has citations



        # Clear selected note IDs when starting new search



        if hasattr(self, 'selected_note_ids'):



            self.selected_note_ids.clear()



        for attr in ('copy_answer_btn',):



            if hasattr(self, attr):



                getattr(self, attr).setEnabled(False)



        # Disable toggle button when list is cleared



        if hasattr(self, 'toggle_select_btn'):



            self.toggle_select_btn.setEnabled(False)



        if hasattr(self, 'selected_count_label'):



            self.selected_count_label.setText("(0 selected)")



        self._last_formatted_answer = None



        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:



            self.search_progress_bar.setVisible(False)



        if hasattr(self, 'search_progress_label') and self.search_progress_label:



            self.search_progress_label.setVisible(False)



        QApplication.processEvents()







        try:



            search_config = config.get('search_config', {})



            self._last_requested_search_method = search_config.get('search_method', 'hybrid')



            self._search_pending_query = query



            self._search_pending_config = config



            self._search_pending_async = True



            self.status_label.setText("Loading notes...")



            self._show_centile_progress("Loading notes...", 0)



            self._start_estimated_progress_timer(30, 5, 95)



            log_debug("Starting to load notes in background...")



            from aqt.operations import QueryOp



            op = QueryOp(



                parent=mw,



                op=lambda col: get_notes_content_with_col(col, config),



                success=self._on_get_notes_done,



            )



            op.run_in_background()



            return



        except Exception as e:



            log_debug(f"Unexpected error in perform_search: {type(e).__name__}: {str(e)}")



            import traceback



            log_debug(f"Traceback: {traceback.format_exc()}")



            self.answer_box.setText(



                f"Γ¥î Unexpected Error:\n{str(e)}\n\n"



                "Please check the debug log for details."



            )



            self.status_label.setText("Error occurred")



            self._search_pending_async = False



        finally:



            if not getattr(self, '_search_pending_async', False) and not getattr(self, '_pending_rerank', False):



                self.search_btn.setEnabled(True)



            self._hide_busy_progress()



            QApplication.processEvents()







    def _on_get_notes_done(self, payload):



        """Called when background get_notes_content_with_col finishes. Starts keyword_filter in worker."""



        import time



        self._search_pending_async = False



        self._stop_estimated_progress_timer()



        self._hide_busy_progress()



        if payload is None or not isinstance(payload, (list, tuple)) or len(payload) != 3:



            self.search_btn.setEnabled(True)



            self.status_label.setText("Ready")



            return



        notes, fields_description, cache_key = payload



        self.fields_description = fields_description



        self._cached_notes = notes



        self._cached_notes_key = cache_key



        unique_note_count = len(set(n['id'] for n in notes)) if notes else 0



        if not notes:



            self.answer_box.setText(f"No notes with {fields_description} content found in your collection.")



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.status_label.setText("Ready")



            self.search_btn.setEnabled(True)



            return



        self.total_notes_searched = unique_note_count



        self._search_pending_notes = notes



        self.status_label.setText(f"Filtering {unique_note_count} notes...")



        QApplication.processEvents()



        self._search_pending_async = True



        self._keyword_filter_worker = KeywordFilterWorker(self, self._search_pending_query, notes)



        self._keyword_filter_worker.finished_signal.connect(self._on_keyword_filter_done)



        self._keyword_filter_worker.start()







    def _on_keyword_filter_done(self, result):



        """Called when KeywordFilterWorker finishes. Handles PENDING_EMBEDDING, PENDING_RERANK, or direct result."""



        import time



        self._search_pending_async = False



        notes = getattr(self, '_search_pending_notes', None)



        if notes is None and hasattr(self, '_cached_notes'):



            notes = self._cached_notes



        config = getattr(self, '_search_pending_config', None) or load_config()



        query = getattr(self, '_search_pending_query', '')



        if result is None:



            self.status_label.setText("Error during search")



            self.search_btn.setEnabled(True)



            return



        if isinstance(result, tuple) and result[0] == "PENDING_EMBEDDING":



            # Embedding search will run in background worker; show progress and return



            _, embedding_query, notes_for_embedding, state = result



            self._search_pending_state = state



            self._search_pending_notes = notes



            setattr(self, "_last_embedding_error", None)



            self.current_query = state["query"]



            config = state["config"]



            if hasattr(self, 'search_progress_bar') and self.search_progress_bar:



                self.search_progress_bar.setVisible(True)



                self.search_progress_bar.setRange(0, 100)



                self.search_progress_bar.setValue(0)



            if hasattr(self, 'search_progress_label') and self.search_progress_label:



                self.search_progress_label.setVisible(True)



                self.search_progress_label.setText("Starting embedding search...")



            self.status_label.setText("Embedding search: starting...")



            db_path = get_embeddings_db_path()



            self._embedding_search_worker = EmbeddingSearchWorker(embedding_query, notes_for_embedding, config, db_path=db_path)



            self._embedding_search_worker.progress_signal.connect(self._on_embedding_search_progress)



            self._embedding_search_worker.finished_signal.connect(self._on_embedding_search_finished)



            self._embedding_search_worker.error_signal.connect(self._on_embedding_search_error)



            # Use QThread worker so embedding search always runs off the main thread and does not freeze the UI.



            self._embedding_search_worker.start()



            return



        if isinstance(result, tuple) and result[0] == "PENDING_RERANK":



            # Rerank in background so UI stays responsive



            _, scored_notes, effective_method, _total_above, notes = result



            self.current_query = query



            self._pending_rerank = True



            search_config = config.get('search_config') or {}



            self._rerank_continue = (notes, effective_method, search_config)



            if hasattr(self, 'status_label') and self.status_label:



                self.status_label.setText("Re-ranking results... (may take 10-30 s)")



            self._show_centile_progress("Re-ranking...", 0)



            self._start_estimated_progress_timer(25, 5, 95)



            try:



                from aqt.operations import QueryOp



                op = QueryOp(



                    parent=mw,



                    op=lambda col: _do_rerank(query, scored_notes, MAX_RERANK_COUNT, search_config),



                    success=lambda pair: self._on_rerank_done(pair[0], pair[1]),



                )



                op.run_in_background()



            except Exception:



                self._rerank_worker = RerankWorker(query, scored_notes, MAX_RERANK_COUNT, search_config)



                self._rerank_worker.finished_signal.connect(self._on_rerank_done)



                self._rerank_worker.start()



            return



        scored_notes, effective_method, total_above_threshold = result



        self.current_query = query



        self._perform_search_continue(notes, scored_notes, effective_method, total_above_threshold)



        self.search_btn.setEnabled(True)







    def _on_rerank_done(self, scored_notes, success):



        """Called when RerankWorker finishes; apply min_relevance/max_results and continue search."""



        self._pending_rerank = False



        self._stop_estimated_progress_timer()



        self._hide_busy_progress()



        self.search_btn.setEnabled(True)







        if success == "LIBRARY_LOAD_FAILED":



            from aqt.utils import showInfo



            showInfo(



                "Reranking was skipped: sentence-transformers/torch could not be loaded (e.g. DLL error on Windows).\n\n"



                "This usually means the Visual C++ Redistributable is missing or corrupted.\n\n"



                "Search results are still shown using the initial ranking. To fix reranking, try:\n"



                "1. Tools \u2192 Anki Semantic Search \u2192 Install extra model (reinstalls dependencies)\n"



                "2. Check 'Python for Cross-Encoder' in Settings if using an external Python."



            )



            # Continue with original notes as success=False but success=="LIBRARY_LOAD_FAILED"



            # we already have scored_notes as the original ones.



            success = False







        notes, effective_method, search_config = getattr(self, '_rerank_continue', (None, '', {}))



        if notes is None:



            return



        MAX_STORED_FOR_MODES = 100



        min_relevance = max(15, min(75, search_config.get('min_relevance_percent', 55)))



        min_relevance_stored = min(20, min_relevance)



        all_above_stored = [x for x in scored_notes if x[0] >= min_relevance_stored]



        scored_notes = all_above_stored[:MAX_STORED_FOR_MODES] if all_above_stored else scored_notes[:5]



        total_above_threshold = sum(1 for x in scored_notes if x[0] >= min_relevance)



        self._perform_search_continue(notes, scored_notes, effective_method, total_above_threshold)







    def _rerank_with_cross_encoder(self, query, scored_notes, top_k=15):



        """



        Re-rank top results using a cross-encoder (delegates to _do_rerank).



        Limited to top 15 to avoid CPU bottleneck. Use RerankWorker for non-blocking UI.



        """



        config = load_config()



        sc = config.get('search_config') or {}



        top_k = min(top_k, MAX_RERANK_COUNT)



        return _do_rerank(query, scored_notes, top_k, sc)







    def _passes_focused_balanced_broad(



        self,



        matched_keywords,



        final_score,



        emb_score,



        max_emb_score,



        keywords,



        search_method,



        embeddings_available,



        min_emb_frac=0.25,



        very_high_emb_frac=0.9,



    ):



        """Compute whether a note would pass Focused, Balanced, or Broad inclusion. Returns (passes_focused, passes_balanced, passes_broad)."""



        n_kw = len(keywords) if keywords else 0



        # Focused



        min_kw_focused = max(2, int(n_kw * 0.4)) if n_kw else 1



        if n_kw <= 2:



            min_kw_focused = 1



        min_score_focused = 18



        # Balanced



        min_kw_balanced = max(1, int(n_kw * 0.25)) if n_kw else 1



        min_score_balanced = 10



        # Broad



        min_kw_broad = max(1, int(n_kw * 0.2)) if n_kw else 1



        min_score_broad = 8







        if search_method == "embedding" and embeddings_available:



            if emb_score > 0:



                return (True, True, True)



            return (False, False, False)







        if search_method == "hybrid" and embeddings_available and max_emb_score > 0:



            very_high = emb_score >= very_high_emb_frac * max_emb_score



            decent = emb_score >= min_emb_frac * max_emb_score



            # Focused: (decent + (kw or score)) or very_high; and (not strict or matched_kw>0 or very_high)



            pf = (



                (decent and (matched_keywords >= min_kw_focused or final_score > min_score_focused)) or very_high



            ) and (matched_keywords > 0 or very_high)



            # Balanced



            pb_al = (decent and (matched_keywords >= min_kw_balanced or final_score > min_score_balanced)) or very_high



            # Broad



            pb_br = (decent and (matched_keywords >= min_kw_broad or final_score > min_score_broad)) or very_high



            return (pf, pb_al, pb_br)







        # Keyword-only or fallback



        pf = matched_keywords >= min_kw_focused



        if n_kw <= 2:



            pb_al = matched_keywords >= 1



            pb_br = matched_keywords >= 1



        else:



            pb_al = matched_keywords >= min_kw_balanced or final_score > min_score_balanced



            pb_br = matched_keywords >= min_kw_broad or final_score > min_score_broad



        return (pf, pb_al, pb_br)







    def keyword_filter(self, query, notes):



        """



        Enhanced semantic search with multiple methods:



        - Improved keyword extraction (stemming, n-grams, TF-IDF)



        - Optional embedding-based search using cloud embeddings (Voyage)



        - Hybrid approach combining both methods



        - Context-aware ranking



        """



        import re







        # Get search configuration



        config = load_config()



        search_config = config.get('search_config', {}) or {}



        # Effective relevance mode for this search (Focused/Balanced/Broad)



        mode = getattr(self, 'relevance_mode', None) or search_config.get('relevance_mode') or ''



        mode = (mode or '').lower()



        if mode not in ('focused', 'balanced', 'broad'):



            # Backwards compatibility: infer from strict_relevance when mode missing



            mode = 'focused' if search_config.get('strict_relevance', True) else 'balanced'



        self._effective_relevance_mode = mode



        self._effective_strict_relevance = (mode == 'focused')



        original_search_method = search_config.get('search_method', 'hybrid')



        search_method = original_search_method  # 'keyword', 'keyword_rerank', 'embedding', 'hybrid'



        use_context_boost = search_config.get('use_context_boost', True)



        # keyword_rerank = keyword scoring then cross-encoder rerank (no embeddings)



        if search_method == 'keyword_rerank':



            search_method = 'keyword'  # use keyword path; effective_method will show "Keyword + Re-rank"







        # Always run synonym expansion (built-in medical aliases + config synonym_overrides).



        # Optional AI-based expansion runs inside _expand_query when enable_query_expansion is on.



        query = self._expand_query(query, config)







        # Optional: AI-mediated generic term exclusion (one short LLM call per search)



        if search_config.get('use_ai_generic_term_detection', False):



            try:



                self._query_ai_excluded_terms = self._get_ai_excluded_terms(query, config)



            except Exception:



                self._query_ai_excluded_terms = set()



        else:



            self._query_ai_excluded_terms = set()







        # Improved keyword extraction



        keywords, stems, phrases = self._extract_keywords_improved(query)







        if not keywords and not phrases:



            return ([(1, note) for note in notes[:50]], "Keyword only", min(50, len(notes)))







        # Compute TF-IDF scores



        tfidf_scores = self._compute_tfidf_scores(notes, keywords)







        # Get embedding scores if available and method requires it



        embedding_scores = None



        embeddings_available = False



        # HyDE: optional hypothetical document for better semantic retrieval



        embedding_query = query



        if search_method in ('embedding', 'hybrid') and search_config.get('enable_hyde', False):



            try:



                if hasattr(self, 'status_label') and self.status_label:



                    self.status_label.setText("Generating HyDE... (one short API call, usually 5-30 s)")



                    QApplication.processEvents()



            except Exception:



                pass



            hyde_doc = self._generate_hyde_document(query, config)



            try:



                if hasattr(self, 'status_label') and self.status_label and hyde_doc:



                    self.status_label.setText("Searching notes... (embedding search)")



                    QApplication.processEvents()



            except Exception:



                pass



            if hyde_doc:



                embedding_query = hyde_doc



                log_debug("Using HyDE hypothetical document for embedding search")







        if search_method in ('embedding', 'hybrid'):



            # For speed and better relevance, only run the slower



            # embedding-based search on the top N TF-IDF candidates.



            max_notes_for_embedding = 2000



            notes_sorted = None



            if len(notes) > max_notes_for_embedding:



                notes_sorted = sorted(



                    notes,



                    key=lambda n: tfidf_scores.get(n['id'], 0),



                    reverse=True,



                )



                notes_for_embedding = notes_sorted[:max_notes_for_embedding]



            else:



                notes_for_embedding = notes



            # Run embedding search in background worker so UI stays responsive



            state = dict(



                notes=notes, query=query, keywords=keywords, stems=stems, phrases=phrases,



                tfidf_scores=tfidf_scores, search_method=search_method,



                original_search_method=original_search_method, search_config=search_config,



                use_context_boost=use_context_boost, config=config,



                notes_sorted=notes_sorted, max_notes_for_embedding=max_notes_for_embedding,



            )



            return ("PENDING_EMBEDDING", embedding_query, notes_for_embedding, state)







        # Score notes using selected method (keyword-only path)



        scored_notes = []



        keyword_scored_list = []  # For RRF: (keyword_score, note) for hybrid



        max_score = 0



        max_emb_score = max(embedding_scores.values()) if embedding_scores else 0.0



        min_emb_frac = 0.25  # hybrid weighted fallback



        very_high_emb_frac = 0.9  # near-best semantic match gets through even w/ weak keywords



        use_rrf = (search_method == 'hybrid' and embedding_scores and embeddings_available)



        high_freq_keywords = getattr(self, "_query_high_freq_keywords", set()) or set()



        if not isinstance(high_freq_keywords, set):



            try:



                high_freq_keywords = set(high_freq_keywords)



            except Exception:



                high_freq_keywords = set()



        # Search cascade: when all query keywords are high-freq (generic), down-weight keyword-only score



        query_all_high_freq = bool(keywords and all(k in high_freq_keywords for k in keywords))







        try:



            from aqt.qt import QApplication



        except Exception:



            QApplication = None







        for idx, note in enumerate(notes):



            # Let the UI breathe every few hundred notes so Anki doesn't show "Not Responding"



            if QApplication is not None and idx % 500 == 0:



                QApplication.processEvents()



            content_lower = note['content'].lower()



            keyword_score = 0



            matched_keywords = 0







            # Improved keyword matching with stemming



            for keyword in keywords:



                # Skip auto-detected high-frequency filler terms for this query



                if keyword in high_freq_keywords:



                    continue



                # Exact whole word match

                whole_word = bool(re.search(r'\b' + re.escape(keyword) + r'\b', content_lower))



                if whole_word:



                    matched_keywords += 1



                    count = content_lower.count(keyword)



                    keyword_score += min(count * 12, 45) + 10



                elif keyword in content_lower:



                    matched_keywords += 1



                    keyword_score += 4







                # Check stemmed versions



                stem = self._simple_stem(keyword)



                if stem != keyword and stem in content_lower:



                    keyword_score += 3







            # Phrase matching (bigrams and trigrams)



            for phrase in phrases:



                # Don't give phrase bonus if all its tokens are high-frequency fillers



                tokens = phrase.split()



                if tokens and all(t in high_freq_keywords for t in tokens):



                    continue



                if phrase in content_lower:



                    keyword_score += 18







            # Add TF-IDF component



            tfidf_score = tfidf_scores.get(note['id'], 0) * 2  # Weight TF-IDF



            keyword_score += tfidf_score







            # Combine with embedding score if available



            if search_method == 'embedding':



                if embedding_scores and embeddings_available:



                    # Use only embedding score



                    final_score = embedding_scores.get(note['id'], 0)



                else:



                    # Fallback to keyword if embeddings not available



                    # This ensures the search still works even without embeddings



                    final_score = keyword_score



            elif search_method == 'hybrid':



                final_score = keyword_score



            else:



                final_score = keyword_score







            # Cascade: don't treat generic-keyword-only matches as highly relevant



            if query_all_high_freq and final_score == keyword_score:



                final_score = final_score * 0.3







            # Apply context-aware boost



            if use_context_boost:



                final_score = self._context_aware_boost(note, final_score)







            # For RRF hybrid: collect keyword scores; attach Focused/Balanced/Broad flags for UI filtering.



            if use_rrf:



                emb_score = embedding_scores.get(note['id'], 0) if embedding_scores else 0



                passes_focused, passes_balanced, passes_broad = self._passes_focused_balanced_broad(



                    matched_keywords, final_score, emb_score, max_emb_score, keywords,



                    search_method, embeddings_available, min_emb_frac, very_high_emb_frac,



                )



                note['_passes_focused'] = passes_focused



                note['_passes_balanced'] = passes_balanced



                note['_passes_broad'] = passes_broad



                keyword_scored_list.append((keyword_score, final_score, note))  # final_score has context boost



                continue







            # Inclusion criteria for non-RRF: include if passes Broad (superset); attach all three flags for UI filtering.



            emb_score = embedding_scores.get(note['id'], 0) if embedding_scores else 0



            passes_focused, passes_balanced, passes_broad = self._passes_focused_balanced_broad(



                matched_keywords, final_score, emb_score, max_emb_score, keywords,



                search_method, embeddings_available, min_emb_frac, very_high_emb_frac,



            )



            note['_passes_focused'] = passes_focused



            note['_passes_balanced'] = passes_balanced



            note['_passes_broad'] = passes_broad







            # region agent log



            try:



                q_lower = (getattr(self, "_search_pending_query", None) or query or "").lower()



                if "trisom" in q_lower:



                    _agent_debug_log(



                        run_id="post-fix_candidate",



                        hypothesis_id="H4",



                        location="__init__.keyword_filter",



                        message="candidate_inclusion_decision",



                        data={



                            "note_id": note.get("id"),



                            "matched_keywords": matched_keywords,



                            "final_score": final_score,



                            "emb_score": emb_score,



                            "search_method": search_method,



                            "len_keywords": len(keywords),



                            "passes_focused": passes_focused,



                            "passes_balanced": passes_balanced,



                            "passes_broad": passes_broad,



                        },



                    )



            except Exception:



                pass



            # endregion







            if passes_broad:



                scored_notes.append((final_score, note))



                max_score = max(max_score, final_score)







        # RRF (Reciprocal Rank Fusion): combine keyword and vector rankings. Standard formula 1/(k+rank); more effective than weighted averaging.



        # With chunks, use best rank per note id (min rank across chunks).



        if use_rrf and embedding_results:



            k = RRF_K



            emb_weight = max(0, min(1, (search_config.get('hybrid_embedding_weight', 50) or 50) / 100.0))



            kw_weight = 1.0 - emb_weight



            keyword_ranked = sorted(keyword_scored_list, key=lambda x: (x[0], x[1]), reverse=True)



            kw_rank = {}



            for rank, (_, _, note) in enumerate(keyword_ranked, start=1):



                nid = note['id']



                kw_rank[nid] = min(kw_rank.get(nid, rank), rank)



            emb_rank = {}



            for rank, (_, note) in enumerate(embedding_results, start=1):



                nid = note['id']



                emb_rank[nid] = min(emb_rank.get(nid, rank), rank)



            all_ids = set(kw_rank) | set(emb_rank)



            rrf_scores = []



            for nid in all_ids:



                rrf = 0



                if nid in kw_rank:



                    rrf += kw_weight * (1.0 / (k + kw_rank[nid]))



                if nid in emb_rank:



                    rrf += emb_weight * (1.0 / (k + emb_rank[nid]))



                if rrf > 0:



                    note = next((n for _, _, n in keyword_ranked if n['id'] == nid), None) or next((n for _, n in embedding_results if n['id'] == nid), None)



                    if note:



                        rrf_scores.append((rrf, note))



            scored_notes = sorted(rrf_scores, reverse=True, key=lambda x: x[0])



            # Notes that came only from embedding_results may lack _passes_*; show them in Broad/Balanced.



            for _s, note in scored_notes:



                if '_passes_broad' not in note:



                    note['_passes_broad'] = True



                    note['_passes_balanced'] = True



                    note['_passes_focused'] = False



            max_score = scored_notes[0][0] if scored_notes else 1







        # Normalize scores to 0-100 range if needed



        if max_score > 0 and max_score != 100:



            scored_notes = [(score / max_score * 100, note) for score, note in scored_notes]







        scored_notes.sort(reverse=True, key=lambda x: x[0])



        # Keep un-aggregated list for AI context so it can cite specific sections (chunks)



        has_chunks = any(n.get('chunk_index') is not None for _, n in scored_notes)



        if has_chunks:



            self._scored_notes_for_context = list(scored_notes)



        else:



            self._scored_notes_for_context = None



        # Aggregate chunks by note id: one entry per note (best score), display full content



        scored_notes = self._aggregate_scored_notes_by_note_id(scored_notes)







        # Effective method shown to user (may differ from config if embeddings unavailable)



        if original_search_method == "keyword_rerank":



            effective_method = "Keyword + Re-rank"



        elif search_method == "embedding" and embeddings_available:



            effective_method = "Embedding only"



        elif search_method == "hybrid" and embeddings_available:



            effective_method = "Hybrid"



        else:



            effective_method = "Keyword only"







        # Optional: Cross-encoder re-ranking in background (avoids UI freeze)



        if scored_notes and (search_config.get('enable_rerank', False) or original_search_method == 'keyword_rerank'):



            return ("PENDING_RERANK", scored_notes, effective_method + " + Re-ranked", 0, notes)







        # Stored superset for mode switching: keep notes above a low bar, cap size (filter_and_display_notes applies mode + sensitivity).



        MAX_STORED_FOR_MODES = 100



        min_relevance = max(15, min(75, search_config.get('min_relevance_percent', 55)))



        min_relevance_stored = min(20, min_relevance)



        all_above_stored = [x for x in scored_notes if x[0] >= min_relevance_stored]



        scored_notes = all_above_stored[:MAX_STORED_FOR_MODES] if all_above_stored else scored_notes[:5]



        total_above_threshold = sum(1 for x in scored_notes if x[0] >= min_relevance)



        return scored_notes, effective_method, total_above_threshold







    def keyword_filter_continue(self, state, embedding_results, progress_callback=None):



        """Continue keyword_filter after embedding search worker finishes. Uses state from keyword_filter.



        progress_callback(idx, total) is called every 500 notes when provided (e.g. from a worker thread).



        When notes count is large and we have embedding results, only score a subset (top by TF-IDF + embedding note ids) to avoid long freezes."""



        import re



        notes = state["notes"]



        query = state["query"]



        keywords = state["keywords"]



        stems = state["stems"]



        phrases = state["phrases"]



        tfidf_scores = state["tfidf_scores"]



        search_method = state["search_method"]



        original_search_method = state["original_search_method"]



        search_config = state["search_config"]



        use_context_boost = state["use_context_boost"]



        config = state["config"]



        embedding_scores = None



        embeddings_available = False



        if embedding_results:



            embedding_scores = {note['id']: score for score, note in embedding_results}



            embeddings_available = True



            # Limit to subset when very large so "Combining results" does not take minutes



            COMBINE_MAX_NOTES = 6000



            if len(notes) > COMBINE_MAX_NOTES:



                emb_ids = {note['id'] for _, note in embedding_results}



                top_by_tfidf = sorted(notes, key=lambda n: tfidf_scores.get(n['id'], 0), reverse=True)[:COMBINE_MAX_NOTES]



                top_ids = {n['id'] for n in top_by_tfidf}



                subset_ids = top_ids | emb_ids



                notes = [n for n in notes if n['id'] in subset_ids]



                log_debug(f"keyword_filter_continue: limited to {len(notes)} notes (top {COMBINE_MAX_NOTES} by TF-IDF + embedding results)")



        else:



            if search_method == 'embedding' and not hasattr(self, '_embedding_warning_shown'):



                tooltip(



                    "No embeddings found for this search. Using keyword search.\n\n"



                    "If you already ran Create/Update Embeddings: the selected decks/note types may not match.",



                    period=5000,



                )



                self._embedding_warning_shown = True



            elif search_method == 'hybrid' and not hasattr(self, '_hybrid_warning_shown'):



                tooltip(



                    "No embeddings for these notes. Using keyword-only search.\n\n"



                    "Run Create/Update Embeddings (Settings \u2192 Search & Embeddings) for the selected decks/note types.",



                    period=4000,



                )



                self._hybrid_warning_shown = True



        scored_notes = []



        keyword_scored_list = []



        max_score = 0



        max_emb_score = max(embedding_scores.values()) if embedding_scores else 0.0



        min_emb_frac = 0.25



        very_high_emb_frac = 0.9



        use_rrf = (search_method == 'hybrid' and embedding_scores and embeddings_available)



        total_notes = len(notes)



        high_freq_keywords = getattr(self, "_query_high_freq_keywords", set()) or set()



        if not isinstance(high_freq_keywords, set):



            try:



                high_freq_keywords = set(high_freq_keywords)



            except Exception:



                high_freq_keywords = set()



        for idx, note in enumerate(notes):



            if progress_callback and idx > 0 and idx % 500 == 0:



                try:



                    progress_callback(idx + 1, total_notes)



                except Exception:



                    pass



            content_lower = note['content'].lower()



            keyword_score = 0



            matched_keywords = 0



            for keyword in keywords:



                if keyword in high_freq_keywords:



                    continue



                whole_word = bool(re.search(r'\b' + re.escape(keyword) + r'\b', content_lower))



                if whole_word:



                    matched_keywords += 1



                    count = content_lower.count(keyword)



                    keyword_score += min(count * 12, 45) + 10



                elif keyword in content_lower:



                    matched_keywords += 1



                    keyword_score += 4



                stem = self._simple_stem(keyword)



                if stem != keyword and stem in content_lower:



                    keyword_score += 3



            for phrase in phrases:



                tokens = phrase.split()



                if tokens and all(t in high_freq_keywords for t in tokens):



                    continue



                if phrase in content_lower:



                    keyword_score += 18



            keyword_score += tfidf_scores.get(note['id'], 0) * 2



            if search_method == 'embedding':



                final_score = embedding_scores.get(note['id'], 0) if (embedding_scores and embeddings_available) else keyword_score



            else:



                final_score = keyword_score



            if use_context_boost:



                final_score = self._context_aware_boost(note, final_score)



            if use_rrf:



                emb_score = embedding_scores.get(note['id'], 0) if embedding_scores else 0



                passes_focused, passes_balanced, passes_broad = self._passes_focused_balanced_broad(



                    matched_keywords, final_score, emb_score, max_emb_score, keywords,



                    search_method, embeddings_available, min_emb_frac, very_high_emb_frac,



                )



                note['_passes_focused'] = passes_focused



                note['_passes_balanced'] = passes_balanced



                note['_passes_broad'] = passes_broad



                keyword_scored_list.append((keyword_score, final_score, note))



                continue



            # Inclusion: include if passes Broad; attach flags for UI mode filtering.



            emb_score = embedding_scores.get(note['id'], 0) if embedding_scores else 0



            passes_focused, passes_balanced, passes_broad = self._passes_focused_balanced_broad(



                matched_keywords, final_score, emb_score, max_emb_score, keywords,



                search_method, embeddings_available, min_emb_frac, very_high_emb_frac,



            )



            note['_passes_focused'] = passes_focused



            note['_passes_balanced'] = passes_balanced



            note['_passes_broad'] = passes_broad



            if passes_broad:



                scored_notes.append((final_score, note))



                max_score = max(max_score, final_score)



        if progress_callback and total_notes > 0:



            try:



                progress_callback(total_notes, total_notes)



            except Exception:



                pass



        # Optional verbose debug logging for search behavior tuning



        if search_config.get("verbose_search_debug", False):



            try:



                high_freq = getattr(self, "_query_high_freq_keywords", set()) or set()



                if not isinstance(high_freq, set):



                    high_freq = set(high_freq)



            except Exception:



                high_freq = set()



            top_notes = scored_notes[:5]



            debug_rows = []



            for score, note in top_notes:



                nid = note.get("id")



                emb_score = embedding_scores.get(nid, 0) if embedding_scores else 0



                tfidf = tfidf_scores.get(nid, 0)



                snippet = (note.get("content", "") or "")[:160].replace("\n", " ")



                debug_rows.append(



                    {



                        "note_id": nid,



                        "score": round(score, 2),



                        "embedding_score": round(emb_score, 4) if isinstance(emb_score, (int, float)) else emb_score,



                        "tfidf": round(tfidf, 4) if isinstance(tfidf, (int, float)) else tfidf,



                        "snippet": snippet,



                    }



                )



            log_debug(



                f"verbose_search_debug: query={query!r}, keywords={keywords}, phrases={phrases[:6]}, "



                f"high_freq_keywords={sorted(list(high_freq))[:12]}, top_notes={debug_rows}"



            )



        if use_rrf and embedding_results:



            k = RRF_K



            emb_weight = max(0, min(1, (search_config.get('hybrid_embedding_weight', 50) or 50) / 100.0))



            kw_weight = 1.0 - emb_weight



            keyword_ranked = sorted(keyword_scored_list, key=lambda x: (x[0], x[1]), reverse=True)



            kw_rank = {}



            for rank, (_, _, note) in enumerate(keyword_ranked, start=1):



                nid = note['id']



                kw_rank[nid] = min(kw_rank.get(nid, rank), rank)



            emb_rank = {}



            for rank, (_, note) in enumerate(embedding_results, start=1):



                nid = note['id']



                emb_rank[nid] = min(emb_rank.get(nid, rank), rank)



            all_ids = set(kw_rank) | set(emb_rank)



            rrf_scores = []



            for nid in all_ids:



                rrf = 0



                if nid in kw_rank:



                    rrf += kw_weight * (1.0 / (k + kw_rank[nid]))



                if nid in emb_rank:



                    rrf += emb_weight * (1.0 / (k + emb_rank[nid]))



                if rrf > 0:



                    note = next((n for _, _, n in keyword_ranked if n['id'] == nid), None) or next((n for _, n in embedding_results if n['id'] == nid), None)



                    if note:



                        rrf_scores.append((rrf, note))



            scored_notes = sorted(rrf_scores, reverse=True, key=lambda x: x[0])



            for _s, note in scored_notes:



                if '_passes_broad' not in note:



                    note['_passes_broad'] = True



                    note['_passes_balanced'] = True



                    note['_passes_focused'] = False



            max_score = scored_notes[0][0] if scored_notes else 1



        if max_score > 0 and max_score != 100:



            scored_notes = [(score / max_score * 100, note) for score, note in scored_notes]



        scored_notes.sort(reverse=True, key=lambda x: x[0])



        has_chunks_cont = any(n.get('chunk_index') is not None for _, n in scored_notes)



        if has_chunks_cont:



            self._scored_notes_for_context = list(scored_notes)



        else:



            self._scored_notes_for_context = None



        scored_notes = self._aggregate_scored_notes_by_note_id(scored_notes)



        if original_search_method == "keyword_rerank":



            effective_method = "Keyword + Re-rank"



        elif search_method == "embedding" and embeddings_available:



            effective_method = "Embedding only"



        elif search_method == "hybrid" and embeddings_available:



            effective_method = "Hybrid"



        else:



            effective_method = "Keyword only"



        if scored_notes and (search_config.get('enable_rerank', False) or original_search_method == 'keyword_rerank'):



            return ("PENDING_RERANK", notes, scored_notes, effective_method + " + Re-ranked", 0)



        MAX_STORED_FOR_MODES = 100



        min_relevance = max(15, min(75, search_config.get('min_relevance_percent', 55)))



        min_relevance_stored = min(20, min_relevance)



        all_above_stored = [x for x in scored_notes if x[0] >= min_relevance_stored]



        scored_notes = all_above_stored[:MAX_STORED_FOR_MODES] if all_above_stored else scored_notes[:5]



        total_above_threshold = sum(1 for x in scored_notes if x[0] >= min_relevance)



        return scored_notes, effective_method, total_above_threshold







    def get_best_model(self, provider):



        models = {



            'anthropic': 'claude-sonnet-4-20250514',



            'openai': 'gpt-4o-mini',



            'google': 'gemini-1.5-flash',



            'openrouter': 'google/gemini-flash-1.5',



            'ollama': 'llama3.2'



        }



        return models.get(provider, 'gpt-4o-mini')







    def ask_ai(self, query, notes, context, config):



        provider = config.get('provider', 'openai')



        api_key = config.get('api_key', '')



        model = self.get_best_model(provider)







        num_notes = len(notes)



        focus_block = ""



        prompt = f"""You are an assistant for question-answering over provided notes. Use ONLY the numbered notes below as your factual source (you may add brief connecting logic, but no outside facts or external guidelines).



If the notes contain at least some relevant information, give the **best partial answer you can** based only on these notes and then briefly mention what is missing.



Only if the notes are essentially unrelated to the question, say exactly: "The provided notes do not contain enough information to answer this."







Context information is below. There are exactly {num_notes} notes: Note 1 = highest relevance, Note 2 = second, ... Note {num_notes} = last. Cite ONLY using numbers from 1 to {num_notes} (e.g. [1], [2], [1,3]). Do not use numbers outside 1\xe2\u20ac\u201c{num_notes}.



---------------------



{context}



---------------------



Given the context information and not prior knowledge, answer the question.







Question: {query}{focus_block}







Rules:



- Base every claim strictly on these notes. Do **not** invent mechanisms, receptor types, dosages, diagnostic criteria, or risk factors that are not supported by the notes. One sentence or bullet per idea is fine.



- Write in a clear, exam-oriented style: use bullet points (\u2022) for key points; use 2-space indented bullets for sub-points. Use **double asterisks** around important terms (diagnoses, drugs, criteria). Do not use ## for headings\u2014use a single bold line with \u25cf\x8f then bullets underneath.



- When the question asks about **receptors, mechanisms, pathways, or numbered lists (e.g. 1st\xe2\u20ac\u201c6th diseases, steps 1\xe2\u20ac\u201c6)**, present them in a clean ordered list and attach citations for each item.



- NUMBERED LISTS: When the question or notes refer to a numbered list (e.g. six childhood exanthems, 1st\xe2\u20ac\u201c6th disease, steps 1\xe2\u20ac\u201c6 of ventricular partitioning), **always** present in **strict numerical order**: 1st, then 2nd, then 3rd, then 4th, then 5th, then 6th. **Include every step or number that any note describes**\u2014if Note X explicitly mentions "3rd step" or "third step", you MUST include it in your answer and cite it [X]. Do not skip steps; do not reorder by relevance.



- INLINE CITATIONS: Cite the supporting note(s) using [N] or [N,M] where N is between 1 and {num_notes} only. Example: "Hypertension increases stroke risk [1,3]." Do not use citation numbers outside 1\xe2\u20ac\u201c{num_notes}.



- At the end, on one line, list all note numbers you cited. Format: RELEVANT_NOTES: 1,3,5"""







        # Estimate input tokens



        input_tokens = estimate_tokens(prompt)







        if provider == "ollama":



            sc = config.get('search_config') or {}



            base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()



            model = (sc.get('ollama_chat_model') or 'llama3.2').strip()



            answer, relevant_indices = self.call_ollama(prompt, base_url, model, notes)



        elif provider == "anthropic":



            system_blocks, user_content = _build_anthropic_prompt_parts(query, context)



            answer, relevant_indices = self.call_anthropic(



                api_key=api_key, model=model, notes=notes,



                system_blocks=system_blocks, user_content=user_content



            )



        elif provider == "openai":



            answer, relevant_indices = self.call_openai(prompt, api_key, model, notes)



        elif provider == "google":



            answer, relevant_indices = self.call_google(prompt, api_key, model, notes)



        elif provider == "openrouter":



            answer, relevant_indices = self.call_openrouter(prompt, api_key, model, notes)



        else:



            api_url = config.get('api_url', '')



            answer, relevant_indices = self.call_custom(prompt, api_key, model, api_url, notes)







        log_debug(f"AI answer length: input ~{input_tokens} tokens, output ~{estimate_tokens(answer)} tokens")



        return answer, relevant_indices







    def call_ollama(self, prompt, base_url, model, notes):



        """Call Ollama /api/generate for AI answers (no API key)."""



        import json



        import urllib.request



        import urllib.error



        log_debug(f"Calling Ollama API: {base_url}, model={model}")



        url = base_url.rstrip("/") + "/api/generate"



        data = {



            "model": model,



            "prompt": prompt,



            "stream": False,



            "options": {"num_predict": 4096}



        }



        req = urllib.request.Request(



            url,



            data=json.dumps(data).encode("utf-8"),



            headers={"Content-Type": "application/json"},



            method="POST"



        )



        try:



            # Reasoning models (e.g. deepseek-r1) can take several minutes; use 5 min timeout



            with urllib.request.urlopen(req, timeout=300) as resp:



                result = json.loads(resp.read().decode("utf-8"))



            # /api/generate returns "response"; /api/chat returns message.content; some models use "thinking"



            full_response = (



                result.get("response")



                or (result.get("message") or {}).get("content")



                or result.get("thinking")



                or ""



            )



            if isinstance(full_response, list):



                # Some models return content as list of parts



                full_response = "".join(



                    p.get("text", p) if isinstance(p, dict) else str(p)



                    for p in full_response



                )



            full_response = (full_response or "").strip()



            return self.parse_response(full_response, notes)



        except urllib.error.HTTPError as e:



            try:



                err_body = e.read().decode("utf-8")



            except Exception:



                err_body = str(e)



            log_debug(f"Ollama HTTP error: {e.code} {err_body}")



            raise Exception(f"Ollama error ({e.code}): {err_body[:200]}")



        except urllib.error.URLError as e:



            msg = str(getattr(e, "reason", e))



            if "timed out" in msg.lower():



                raise Exception("Ollama request timed out. Try a smaller model or more notes.")



            raise Exception(f"Cannot reach Ollama: {msg}. Is Ollama running (ollama serve)?")



        except Exception as e:



            log_debug(f"Ollama error: {e}")



            raise Exception(f"Ollama error: {e}")







    def call_anthropic(self, prompt=None, api_key=None, model=None, notes=None, system_blocks=None, user_content=None):



        """Call Anthropic API. Use system_blocks+user_content for prompt caching (recommended); else single prompt."""



        log_debug(f"Calling Anthropic API with model: {model}")



        url = "https://api.anthropic.com/v1/messages"



        headers = {



            "Content-Type": "application/json",



            "x-api-key": api_key,



            "anthropic-version": "2023-06-01"



        }



        if system_blocks is not None and user_content is not None:



            data = {



                "model": model,



                "max_tokens": 4096,



                "system": system_blocks,



                "messages": [{"role": "user", "content": user_content}]



            }



        else:



            data = {



                "model": model,



                "max_tokens": 4096,



                "messages": [{"role": "user", "content": prompt}]



            }



        response_text = self.make_request(url, headers, data)



        result = json.loads(response_text)



        full_response = result['content'][0]['text']



        return self.parse_response(full_response, notes)







    def call_openai(self, prompt, api_key, model, notes):



        url = "https://api.openai.com/v1/chat/completions"



        headers = {



            "Content-Type": "application/json",



            "Authorization": f"Bearer {api_key}"



        }



        data = {



            "model": model,



            "messages": [{"role": "user", "content": prompt}],



            "max_tokens": 4096



        }







        response_text = self.make_request(url, headers, data)



        result = json.loads(response_text)



        full_response = result['choices'][0]['message']['content']



        return self.parse_response(full_response, notes)







    def call_google(self, prompt, api_key, model, notes):



        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"



        headers = {"Content-Type": "application/json"}



        data = {



            "contents": [{"parts": [{"text": prompt}]}],



            "generationConfig": {



                "maxOutputTokens": 4096



            }



        }







        response_text = self.make_request(url, headers, data)



        result = json.loads(response_text)



        full_response = result['candidates'][0]['content']['parts'][0]['text']



        return self.parse_response(full_response, notes)







    def call_openrouter(self, prompt, api_key, model, notes):



        url = "https://openrouter.ai/api/v1/chat/completions"



        headers = {



            "Content-Type": "application/json",



            "Authorization": f"Bearer {api_key}"



        }



        data = {



            "model": model,



            "messages": [{"role": "user", "content": prompt}],



            "max_tokens": 4096



        }







        response_text = self.make_request(url, headers, data)



        result = json.loads(response_text)



        full_response = result['choices'][0]['message']['content']



        return self.parse_response(full_response, notes)







    def call_custom(self, prompt, api_key, model, api_url, notes):



        headers = {



            "Content-Type": "application/json",



            "Authorization": f"Bearer {api_key}"



        }



        data = {



            "model": model,



            "messages": [{"role": "user", "content": prompt}],



            "max_tokens": 4096



        }







        response_text = self.make_request(api_url, headers, data)



        result = json.loads(response_text)







        if 'choices' in result:



            full_response = result['choices'][0]['message']['content']



        elif 'content' in result:



            full_response = result['content'][0]['text']



        else:



            full_response = str(result)







        return self.parse_response(full_response, notes)







    def make_request(self, url, headers, data):



        """Make HTTP request with proper timeout and error handling"""



        log_debug(f"Making API request to: {url}")



        log_debug(f"Request data keys: {list(data.keys())}")







        req = urllib.request.Request(



            url,



            data=json.dumps(data).encode('utf-8'),



            headers=headers,



            method='POST'



        )







        try:



            # Use 30 second timeout (60 was too long)



            log_debug("Opening URL connection (timeout: 30 seconds)...")



            with urllib.request.urlopen(req, timeout=30) as response:



                log_debug(f"Received response, status: {response.status}")



                response_data = response.read().decode('utf-8')



                log_debug(f"Response length: {len(response_data)} characters")



                return response_data







        except urllib.error.HTTPError as e:



            try:



                error_msg = e.read().decode('utf-8')



            except:



                error_msg = str(e)



            log_debug(f"HTTP Error {e.code}: {error_msg}")



            raise Exception(f"API Error ({e.code}): {error_msg}")







        except urllib.error.URLError as e:



            # Handle common "no internet / host unreachable" cases more clearly



            reason = getattr(e, "reason", e)



            msg = str(reason)



            log_debug(f"URL Error: {msg}")



            lower = msg.lower()







            # Windows / general "no internet or cannot resolve host" patterns



            if (



                "getaddrinfo failed" in lower



                or "name or service not known" in lower



                or "temporary failure in name resolution" in lower



                or "nodename nor servname provided" in lower



                or "winerror 11001" in lower  # host not found



                or "winerror 10051" in lower  # network unreachable



                or "winerror 10065" in lower  # no route to host



            ):



                raise Exception("No internet connection or the API host cannot be reached.")







            if "timed out" in lower:



                raise Exception("Request timed out after 30 seconds. The API may be slow or overloaded.")







            raise Exception(f"Network error: {msg}")







        except Exception as e:



            log_debug(f"Unexpected error: {type(e).__name__}: {str(e)}")



            raise Exception(f"Request error: {str(e)}")







    def parse_response(self, full_response, notes):



        import re



        answer_part = ""



        relevant_notes = []



        if "RELEVANT_NOTES:" in full_response:

            parts = full_response.split("RELEVANT_NOTES:")

            answer_part = parts[0].strip()



            if len(parts) > 1:

                notes_str = parts[1].strip()

                numbers = DIGIT_RE.findall(notes_str)

                relevant_notes = [int(n) - 1 for n in numbers if n.isdigit() and 0 <= int(n) - 1 < len(notes)]

        else:

            answer_part = full_response

            relevant_notes = list(range(min(3, len(notes))))







        log_debug(f"Parsed {len(relevant_notes)} relevant notes from AI response")



        return answer_part, relevant_notes







    def _start_anthropic_stream(self, query, context_notes, context, config, relevant_notes, scored_notes, context_note_ids):



        """Start Anthropic streaming worker; UI updates in real-time via chunk_signal."""



        import anthropic  # raise if not installed



        api_key = config.get('api_key', '')



        model = self.get_best_model('anthropic')



        system_blocks, user_content = _build_anthropic_prompt_parts(query, context)



        self._streamed_answer = ""



        self._stream_context_notes = context_notes



        self._stream_relevant_notes = relevant_notes



        self._stream_scored_notes = scored_notes



        self._stream_context_note_ids = context_note_ids



        self._stream_config = config



        self._stream_query = query



        self.answer_box.setPlainText("Thinking...")



        worker = AnthropicStreamWorker(api_key, model, system_blocks, user_content, context_notes)



        worker.chunk_signal.connect(self._append_stream_chunk)



        worker.done_signal.connect(self._on_anthropic_stream_done)



        worker.error_signal.connect(self._on_anthropic_stream_error)



        self._anthropic_stream_worker = worker



        worker.start()







    def _append_stream_chunk(self, chunk):



        """Append a streamed text chunk to the answer box; first chunk replaces 'Thinking...' placeholder."""



        self._streamed_answer = getattr(self, '_streamed_answer', '') + chunk



        self.answer_box.setPlainText(self._streamed_answer)







    def _on_anthropic_stream_done(self, full_text):



        """Handle stream completion: parse response, update cited notes, format and display."""



        worker = getattr(self, '_anthropic_stream_worker', None)



        if worker:



            worker.chunk_signal.disconnect()



            worker.done_signal.disconnect()



            worker.error_signal.disconnect()



            self._anthropic_stream_worker = None



        context_notes = getattr(self, '_stream_context_notes', [])



        relevant_notes = getattr(self, '_stream_relevant_notes', [])



        scored_notes = getattr(self, '_stream_scored_notes', [])



        context_note_ids = getattr(self, '_stream_context_note_ids', [])



        config = getattr(self, '_stream_config', {})



        query = getattr(self, '_stream_query', '')



        answer, relevant_indices = self.parse_response(full_text, context_notes)



        relevant_note_ids = [relevant_notes[idx]['id'] for idx in relevant_indices if 0 <= idx < len(relevant_notes)]



        save_search_history(query, answer, relevant_note_ids, scored_notes, context_note_ids)



        self._context_note_ids = context_note_ids



        self._context_note_id_and_chunk = [(n['id'], n.get('chunk_index')) for n in context_notes]



        self.current_answer = answer



        ai_relevant_note_ids = set()



        for idx in relevant_indices:



            if 0 <= idx < len(relevant_notes):



                ai_relevant_note_ids.add(relevant_notes[idx]['id'])



        self._cited_note_ids = ai_relevant_note_ids



        improved_scored_notes = []



        for score, note in self.all_scored_notes:



            improved_score = score * 2 if note['id'] in ai_relevant_note_ids else score



            improved_scored_notes.append((improved_score, note))



        improved_scored_notes.sort(reverse=True, key=lambda x: x[0])



        if improved_scored_notes:



            max_boosted = improved_scored_notes[0][0]



            self.all_scored_notes = [(score / max_boosted * 100.0, note) for score, note in improved_scored_notes] if max_boosted > 0 else improved_scored_notes



        else:



            self.all_scored_notes = improved_scored_notes



        search_config = config.get('search_config') or {}



        if search_config.get('relevance_from_answer'):



            answer_text = (answer[:8000]).strip()



            note_texts = []



            for _, note in self.all_scored_notes:



                raw = note.get('display_content') or note.get('content', '')



                text = self.reveal_cloze_for_display(raw) if hasattr(self, 'reveal_cloze_for_display') else raw



                note_texts.append((text[:2000]) if text else "")



            if answer_text and note_texts:



                self.status_label.setText("Re-ranking by relevance to answer...")



                self._show_centile_progress("Re-ranking by relevance...", 0)



                self._relevance_rerank_worker = RelevanceRerankWorker(answer_text, note_texts, self.all_scored_notes, config)



                self._relevance_rerank_worker.progress_signal.connect(lambda p, m: self._show_centile_progress(m, p))



                self._relevance_rerank_worker.finished_signal.connect(



                    lambda res: self._on_relevance_rerank_done_stream(res, answer, config)



                )



                self._relevance_rerank_worker.start()



                return



            try:



                self._rerank_by_relevance_to_answer(answer, config)



            except Exception as e:



                log_debug(f"Relevance-from-answer rerank failed: {e}")



        self._finish_anthropic_stream_display(answer, config)







    def _on_relevance_rerank_done_stream(self, result, answer, config):



        """Called when RelevanceRerankWorker finishes (streaming path). Apply result and finish display."""



        self._hide_busy_progress()



        if getattr(self, '_relevance_rerank_worker', None):



            self._relevance_rerank_worker = None



        if result is not None:



            self.all_scored_notes = result



        self._finish_anthropic_stream_display(answer, config)







    def _finish_anthropic_stream_display(self, answer, config):



        """Format answer, update table, and set status (streaming path)."""



        formatted_answer = self.format_answer(answer)



        self._last_formatted_answer = formatted_answer



        self.answer_box.setHtml(formatted_answer)



        if hasattr(self, 'answer_source_label'):



            src = self._get_answer_source_text(config)



            self.answer_source_label.setText(f"Answer from: {src}" if src else "")



        for attr in ('copy_answer_btn',):



            if hasattr(self, attr):



                getattr(self, attr).setEnabled(True)



        self.filter_and_display_notes()



        if self.all_scored_notes:



            threshold = self.sensitivity_slider.value() if getattr(self, 'sensitivity_slider', None) else 0



            max_score = self.all_scored_notes[0][0]



            min_score = (threshold / 100.0) * max_score if max_score > 0 else 0



            effective_pct = round(100 * min_score / max_score) if (max_score > 0 and threshold > 0) else None



            sensitivity_text = f" (score \xe2\u2030\xa5 {effective_pct}%)" if effective_pct is not None else " (sensitivity filter)"



            filtered_count = sum(1 for score, _ in self.all_scored_notes if score >= min_score)



            total_in_result = len(self.all_scored_notes)



            mode = getattr(self, "_effective_relevance_mode", getattr(self, "relevance_mode", "balanced")) or "balanced"



            mode = (mode or "").lower()



            mode_label = {"focused": "Focused", "balanced": "Balanced", "broad": "Broad"}.get(mode, "Balanced")



            mode_suffix = f" | Mode: {mode_label}"



            self.status_label.setText(



                f"Showing {filtered_count} of {total_in_result}{sensitivity_text}{mode_suffix} "



                f"| Answer from: {self._get_answer_source_text(config) or 'Anthropic'}"



            )



        else:



            self.status_label.setText("Answer from: Anthropic (streaming)")



    def _on_anthropic_stream_error(self, error_msg):



        """Show streaming error in answer box."""



        if getattr(self, '_anthropic_stream_worker', None):



            self._anthropic_stream_worker = None



        if hasattr(self, 'answer_source_label'):



            self.answer_source_label.setText("")



        self.answer_box.setText(f"\xe2\x9d\u0152 Error calling Anthropic API:\n{error_msg}\n\nCheck your API key and internet connection.")



        self.status_label.setText("Error occurred")







    def _rerank_by_relevance_to_answer(self, answer, config):



        """Re-rank all_scored_notes by similarity of each note to the AI answer text.



        Sets _display_relevance on each note and replaces all_scored_notes with (score, note) sorted by this.



        Uses the configured embedding engine (Voyage/Ollama)."""



        if not answer or not getattr(self, 'all_scored_notes', None):



            return



        import numpy as np



        sc = (config or load_config()).get('search_config') or {}



        try:



            answer_text = (answer[:8000]).strip()



            if not answer_text:



                return



            answer_emb = get_embedding_for_query(answer_text, config)



            if not answer_emb:



                return



            answer_vec = np.array(answer_emb, dtype=float)



            note_texts = []



            for _, note in self.all_scored_notes:



                raw = note.get('display_content') or note.get('content', '')



                text = self.reveal_cloze_for_display(raw) if hasattr(self, 'reveal_cloze_for_display') else raw



                note_texts.append((text[:2000]) if text else "")



            if not note_texts:



                return



            note_embs = get_embeddings_batch(note_texts, input_type="document", config=config)



            if not note_embs or len(note_embs) != len(self.all_scored_notes):



                return



            norm_a = max(np.linalg.norm(answer_vec), 1e-9)



            new_scores = []



            for i, (_, note) in enumerate(self.all_scored_notes):



                ne = np.array(note_embs[i], dtype=float)



                norm_n = max(np.linalg.norm(ne), 1e-9)



                sim = float(np.dot(answer_vec, ne) / (norm_a * norm_n))



                pct = max(0, min(100, round((sim + 1) * 50)))



                note['_display_relevance'] = pct



                new_scores.append((float(pct), note))



            new_scores.sort(reverse=True, key=lambda x: x[0])



            # Renormalize so top note(s) show 100% and rest spread below



            if new_scores:



                max_pct = new_scores[0][0]



                if max_pct > 0:



                    for score, note in new_scores:



                        note['_display_relevance'] = max(0, min(100, round(100 * (note['_display_relevance'] or 0) / max_pct)))



                    new_scores = [(100.0 if i == 0 else (note['_display_relevance'] or 0), note) for i, (_, note) in enumerate(new_scores)]



                    new_scores.sort(reverse=True, key=lambda x: x[0])



            self.all_scored_notes = new_scores



        except Exception as e:



            log_debug(f"Relevance-from-answer rerank failed: {e}")







    def _get_matching_terms_for_note(self, note, query):



        """Return list of query terms that appear in the note (for 'Why this result?' explainability)."""



        if not query or not hasattr(self, '_extract_keywords_improved'):



            return []



        try:



            keywords, stems, phrases = self._extract_keywords_improved(query)



            content_lower = (note.get('content') or note.get('display_content') or '').lower()



            if not content_lower:



                return []



            matched = set()



            # Phrase matches first \xe2\u20ac\u201c these are usually the most informative



            for p in phrases:



                if p and p.lower() in content_lower:



                    matched.add(p)



            # Exact keyword matches



            for w in keywords:



                wl = (w or '').lower()



                if wl and wl in content_lower:



                    matched.add(w)



            # Stem-based matches: map stem -> representative original query word



            stem_to_display = dict(stems or {})



            for w in keywords:



                stem = self._simple_stem(w)



                if stem:



                    stem_to_display.setdefault(stem, w)



            for stem, display in stem_to_display.items():



                # Skip extremely short stems to avoid over-matching



                if not stem or len(stem) <= 3:



                    continue



                if stem in content_lower:



                    matched.add(display)



            if not matched:



                return []



            # Filter out generic/low-information terms and very short tokens.



            # Also exclude per-query high-frequency terms (set during search) and AI-detected



            # generic terms so we don't show uninformative words in "Why this result?".



            stop_words = self._get_extended_stop_words()



            high_freq = getattr(self, '_query_high_freq_keywords', None) or set()



            if not isinstance(high_freq, set):



                try:



                    high_freq = set(high_freq)



                except Exception:



                    high_freq = set()



            ai_excluded = getattr(self, '_query_ai_excluded_terms', None) or set()



            if not isinstance(ai_excluded, set):



                try:



                    ai_excluded = set(ai_excluded)



                except Exception:



                    ai_excluded = set()







            def _is_meaningful(term: str) -> bool:



                t = (term or '').strip().lower()



                if not t:



                    return False



                if t in stop_words:



                    return False



                if t in high_freq:



                    return False



                if t in ai_excluded:



                    return False



                # Drop very short tokens by default (can whitelist later if needed)



                if len(t) <= 3:



                    return False



                return True



            filtered = [t for t in matched if _is_meaningful(t)]



            if not filtered:



                return []



            # Prefer more specific/phrase-like terms: phrases first, then by length



            filtered.sort(key=lambda t: (0 if ' ' in t else 1, -len(t), t.lower()))



            # region agent log



            try:



                if "trisom" in (query or "").lower():



                    _agent_debug_log(



                        run_id="pre-fix",



                        hypothesis_id="H2",



                        location="__init__._get_matching_terms_for_note",



                        message="matching_terms_for_note",



                        data={



                            "note_id": note.get("id"),



                            "query": query,



                            "matched_all": sorted(matched),



                            "filtered": filtered[:12],



                        },



                    )



            except Exception:



                pass



            # endregion



            return filtered[:12]



        except Exception as ex:



            return []







    def filter_and_display_notes(self):



        # Use chunk-level display list when AI received more items than aggregated display (fixes Ref 35 vs 32)



        display_source = getattr(self, '_display_scored_notes', None)



        ctx = getattr(self, '_context_note_ids', None) or []



        if display_source and ctx and len(ctx) > len(getattr(self, 'all_scored_notes', None) or []):



            notes_to_display = display_source



        elif hasattr(self, 'all_scored_notes') and self.all_scored_notes:



            notes_to_display = self.all_scored_notes



        else:



            return



        if not notes_to_display:



            return







        # Store current row count before clearing



        old_row_count = self.results_list.rowCount()







        # Clear table



        self.results_list.setRowCount(0)







        threshold = self.sensitivity_slider.value() if self.sensitivity_slider else 0



        sensitivity_threshold = threshold  # save before IDF block may overwrite threshold



        max_score = notes_to_display[0][0] if notes_to_display else 1



        min_score = (threshold / 100.0) * max_score if max_score > 0 else 0







        filtered_notes = [(score, note) for score, note in notes_to_display if score >= min_score]







        # Additional strict gating in Focused mode: IDF-based "specific keyword" filter.



        # Notes that match only generic words (appearing in >50% of results) are excluded.



        try:



            config = load_config()



            sc = config.get('search_config') or {}



        except Exception:



            sc = {}



        # Prefer per-search effective strictness when available



        strict = bool(getattr(self, '_effective_strict_relevance', sc.get('strict_relevance', False)))



        # Skip IDF filter when "Relevance from answer" is enabled: ranking is by similarity to answer,



        # not query keywords, so requiring query keywords in note text can empty the list incorrectly.



        relevance_from_answer_enabled = bool(sc.get('relevance_from_answer', False))



        if strict and filtered_notes and not relevance_from_answer_enabled:



            import re



            cq = getattr(self, 'current_query', '') or ''



            try:



                kw, _stems, _phrases = self._extract_keywords_improved(cq)



            except Exception:



                kw = []



            if kw:



                n_notes = len(filtered_notes)



                # Document frequency: how many notes contain each keyword



                doc_freq = {}



                for _score, note in filtered_notes:



                    content_lower = (note.get('content') or note.get('display_content') or '').lower()



                    for k in kw:



                        k_lower = (k or '').lower()



                        if not k_lower:



                            continue



                        if re.search(r'\b' + re.escape(k_lower) + r'\b', content_lower) or k_lower in content_lower:



                            doc_freq[k] = doc_freq.get(k, 0) + 1



                # Specific = appears in fewer than 50% of notes (discriminative)



                threshold = 0.5



                specific_kw = {k for k in kw if doc_freq.get(k, 0) / max(1, n_notes) < threshold}



                if specific_kw:



                    kept = []



                    for score, note in filtered_notes:



                        content_lower = (note.get('content') or note.get('display_content') or '').lower()



                        if any(



                            (k and (re.search(r'\b' + re.escape(k.lower()) + r'\b', content_lower) or k.lower() in content_lower))



                            for k in specific_kw



                        ):



                            kept.append((score, note))



                    filtered_notes = kept



                # else: all keywords generic, skip filter (keep all notes)







        # Optionally restrict to notes cited in the AI answer ([1], [2], ...)



        if getattr(self, 'show_only_cited_cb', None) and self.show_only_cited_cb.isChecked():



            cited = getattr(self, '_cited_note_ids', None)



            if cited:



                filtered_notes = [(score, note) for score, note in filtered_notes if note['id'] in cited]







        # Filter by current relevance mode (Focused / Balanced / Broad) using precomputed flags; no extra search/API.



        mode = (getattr(self, 'relevance_mode', None) or 'balanced').lower()



        # When "Relevance from answer" is on, ranking is by similarity to answer; _passes_* often all True.



        # Differentiate modes by score percentile so Focused = fewer, Broad = all.



        if relevance_from_answer_enabled and filtered_notes:



            n_total = len(filtered_notes)



            if mode == 'focused':



                cap = max(1, int(n_total * 0.4))



                filtered_notes = filtered_notes[:cap]



            elif mode == 'balanced':



                cap = max(1, int(n_total * 0.7))



                filtered_notes = filtered_notes[:cap]



            # else broad: keep all



        else:



            if mode == 'focused':



                filtered_notes = [(s, n) for s, n in filtered_notes if n.get('_passes_focused')]



            elif mode == 'balanced':



                filtered_notes = [(s, n) for s, n in filtered_notes if n.get('_passes_balanced')]



            else:



                filtered_notes = [(s, n) for s, n in filtered_notes if n.get('_passes_broad', True)]







        # Set row count



        self.results_list.setRowCount(len(filtered_notes))



        # Disable sorting while populating so rows stay 0..N and every row gets the correct content (fixes empty rows when toggling "Show only cited notes")



        self.results_list.setSortingEnabled(False)







        # 1-based position in context (order sent to AI) so [1], [2], [19] in answer match this #



        cited_ids = getattr(self, '_cited_note_ids', set()) or set()



        # Build Ref = context position so citation [N] matches row labeled N (works after re-rank and with chunks)



        context_id_and_chunk = getattr(self, '_context_note_id_and_chunk', None)



        if context_id_and_chunk:



            ref_from_context = {(nid, cidx): i + 1 for i, (nid, cidx) in enumerate(context_id_and_chunk)}



            def get_ref(note, row):



                return ref_from_context.get((note['id'], note.get('chunk_index')), row + 1)



        else:



            order_for_note = {nid: i + 1 for i, nid in enumerate(ctx)} if ctx else {}



            def get_ref(note, row):



                return order_for_note.get(note['id'], row + 1)







        for row, (score, note) in enumerate(filtered_notes):



            # Content-based relevance when set (HyDE/query similarity); else rank-based



            percentage = note.get('_display_relevance')



            if percentage is None:



                percentage = int((score / max_score) * 100) if max_score > 0 else 0



            else:



                percentage = max(0, min(100, int(percentage)))



            # Steepen display so 100 stands out and lower scores spread (100\u2192100, 99\u219297, 95\u219292, 80\u219272)



            display_pct = max(0, min(100, round(100 * (percentage / 100) ** 0.6))) if percentage else 0







            # Column 0 (Ref): Citation number from context (matches [1], [2] in AI answer). Cited notes: blue + bold.



            order_num = get_ref(note, row)



            order_item = QTableWidgetItem()



            order_item.setData(Qt.ItemDataRole.DisplayRole, order_num)



            order_item.setData(Qt.ItemDataRole.UserRole, note['id'])



            order_item.setFlags(order_item.flags() & ~Qt.ItemFlag.ItemIsEditable)



            order_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)



            if note['id'] in cited_ids:



                order_item.setForeground(QColor('#3498db'))



                font = order_item.font()



                font.setBold(True)



                order_item.setFont(font)



            why_ref = "Cited in answer: Yes" if note['id'] in cited_ids else "Cited in answer: No"



            order_item.setToolTip(f"Why this result?\n{why_ref}\nCitation [N] in answer matches Ref. Note ID: {note['id']}")



            self.results_list.setItem(row, 0, order_item)







            # Column 1: Content (with checkbox)



            preview_len = getattr(self, 'preview_length', 150)



            # Use full note content when available (chunk-level display); so preview shows start of full note, not chunk



            raw_for_display = note.get('_full_content') or note.get('display_content') or note['content']



            display_content = self.reveal_cloze_for_display(raw_for_display)



            content_preview = display_content[:preview_len] + "..." if len(display_content) > preview_len else display_content



            content_item = QTableWidgetItem(content_preview)



            content_item.setCheckState(Qt.CheckState.Unchecked)



            content_item.setData(Qt.ItemDataRole.UserRole, note['id'])



            content_item.setData(Qt.ItemDataRole.UserRole + 1, display_pct)



            content_item.setData(Qt.ItemDataRole.UserRole + 2, display_content)



            matching_terms = self._get_matching_terms_for_note(note, getattr(self, 'current_query', ''))



            import html

            formatted_content = html.escape(display_content).replace(" | ", "<br><hr style='border: 0; border-top: 1px solid #555;'><br>")



            why_html = f"<b>Why this result?</b><br>Relevance: {display_pct}%<br>{why_ref}"

            if matching_terms:

                why_html += f"<br>Matching terms: {', '.join(matching_terms[:8])}{'...' if len(matching_terms) > 8 else ''}"



            tooltip_html = f"""

            <div style='font-family: sans-serif; min-width: 300px;'>

                {why_html}

                <br><br>

                <b>Note ID:</b> {note['id']}

                <br><br>

                <b>Full Content:</b><br>

                {formatted_content}

            </div>

            """



            content_item.setToolTip(tooltip_html.strip())



            content_item.setFlags(content_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)



            self.results_list.setItem(row, 1, content_item)







            # Column 2: Note ID



            note_id_item = QTableWidgetItem(str(note['id']))



            note_id_item.setData(Qt.ItemDataRole.UserRole, note['id'])



            note_id_item.setFlags(note_id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)



            note_id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)



            note_id_item.setToolTip(f"Note ID: {note['id']}\nDouble-click to open in browser")



            self.results_list.setItem(row, 2, note_id_item)







            # Column 3: Relevance (steepened display %)



            percentage_item = QTableWidgetItem()



            percentage_item.setData(Qt.ItemDataRole.DisplayRole, display_pct)



            percentage_item.setData(Qt.ItemDataRole.UserRole, display_pct)



            percentage_item.setFlags(percentage_item.flags() & ~Qt.ItemFlag.ItemIsEditable)



            percentage_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)



            if display_pct >= 80:



                percentage_item.setForeground(QColor("#27ae60"))



                relevance_desc = "High relevance"



            elif display_pct >= 50:



                percentage_item.setForeground(QColor("#f39c12"))



                relevance_desc = "Medium relevance"



            else:



                percentage_item.setForeground(QColor("#e74c3c"))



                relevance_desc = "Low relevance"



            why_pct = f"Why this result?\nRelevance: {display_pct}% ({relevance_desc})\n{why_ref}"



            if matching_terms:



                why_pct += f"\nMatching terms: {', '.join(matching_terms[:6])}{'...' if len(matching_terms) > 6 else ''}"



            # region agent log



            try:



                cq = getattr(self, "current_query", "") or ""



                if "trisom" in cq.lower():



                    _agent_debug_log(



                        run_id="pre-fix",



                        hypothesis_id="H3",



                        location="__init__.filter_and_display_notes",



                        message="note_row_display",



                        data={



                            "note_id": note.get("id"),



                            "row": row,



                            "raw_score": score,



                            "display_relevance": display_pct,



                            "matching_terms": matching_terms[:6] if matching_terms else [],



                        },



                    )



            except Exception:



                pass



            # endregion



            percentage_item.setToolTip(why_pct)



            self.results_list.setItem(row, 3, percentage_item)







        # Update status to match table: "Showing X of Y" so it always matches Matching notes row count



        filtered_count = len(filtered_notes)



        total_in_result = len(notes_to_display)



        # No slider: omit score % from status; with slider would show " (score \xe2\u2030\xa5 X%)"



        sensitivity_text = ""



        if self.sensitivity_slider is not None:



            effective_pct = round(100 * min_score / max_score) if (max_score > 0 and sensitivity_threshold > 0) else None



            sensitivity_text = f" (score \xe2\u2030\xa5 {effective_pct}%)" if effective_pct is not None else " (sensitivity filter)"



            if self.sensitivity_value_label is not None:



                if sensitivity_threshold == 0:



                    self.sensitivity_value_label.setText("0%")



                elif sensitivity_threshold > 0 and max_score > 0:



                    self.sensitivity_value_label.setText(f"ΓëÑ{effective_pct}%")



        searched_suffix = ""



        if hasattr(self, 'total_notes_searched') and self.total_notes_searched is not None:



            searched_suffix = f" (searched {self.total_notes_searched} in {getattr(self, 'fields_description', 'Text & Extra')})"



        mode = getattr(self, "_effective_relevance_mode", getattr(self, "relevance_mode", "balanced")) or "balanced"



        mode = (mode or "").lower()



        mode_label = {"focused": "Focused", "balanced": "Balanced", "broad": "Broad"}.get(mode, "Balanced")



        # #region agent log



        _session_debug_log(



            "H1",



            "filter_and_display_notes.status_mode",



            "status bar mode",



            data={"relevance_mode": getattr(self, "relevance_mode", None), "_effective_relevance_mode": getattr(self, "_effective_relevance_mode", None), "mode_used": mode, "mode_label": mode_label},



        )



        # #endregion



        mode_suffix = f" | Mode: {mode_label}"



        self.status_label.setText(



            f"Showing {filtered_count} of {total_in_result}{sensitivity_text}{mode_suffix}{searched_suffix}"



        )







        # Enable/disable toggle button based on list content



        has_items = self.results_list.rowCount() > 0



        if hasattr(self, 'toggle_select_btn'):



            self.toggle_select_btn.setEnabled(has_items)







        # Restore selections from persistence



        if hasattr(self, 'selected_note_ids') and self.selected_note_ids:



            self.restore_selections()







        # Update selection count and button text



        self.update_selection_count()







        # Update View All button state and tooltip (enabled when list has rows)



        self._update_view_all_button_state()







        # Re-enable sorting and apply default sort (fixes empty rows when toggling "Show only cited notes")



        self.results_list.setSortingEnabled(True)



        if display_source is not None and display_source is notes_to_display:



            self.results_list.sortItems(0, Qt.SortOrder.AscendingOrder)



        else:



            self.results_list.sortItems(3, Qt.SortOrder.DescendingOrder)  # Sort by Relevance







    def update_selection_count(self):



        """Update the selected count display and toggle button text"""



        if not hasattr(self, 'results_list'):



            return







        checked_count = 0



        total_count = self.results_list.rowCount()







        # Initialize selected_note_ids if it doesn't exist



        if not hasattr(self, 'selected_note_ids'):



            self.selected_note_ids = set()







        # Update persistence set and count (check column 1 = Content which has the checkbox)



        for row in range(total_count):



            item = self.results_list.item(row, 1)  # Content column has checkbox



            if item:



                note_id = item.data(Qt.ItemDataRole.UserRole)



                if item.checkState() == Qt.CheckState.Checked:



                    checked_count += 1



                    if note_id:



                        self.selected_note_ids.add(note_id)



                else:



                    if note_id:



                        self.selected_note_ids.discard(note_id)







        # Update count label



        if hasattr(self, 'selected_count_label'):



            if total_count > 0:



                self.selected_count_label.setText(f"({checked_count} of {total_count} selected)")



            else:



                self.selected_count_label.setText("(0 selected)")







        # Update toggle button text



        if hasattr(self, 'toggle_select_btn'):



            if checked_count == total_count and total_count > 0:



                self.toggle_select_btn.setText("\xe2\u0153\u2014 Deselect All")



            else:



                self.toggle_select_btn.setText("\u2713 Select All")







    def on_preview_length_changed(self, value):



        """Update preview length and refresh display"""



        self.preview_length = value



        if hasattr(self, 'preview_length_label'):



            self.preview_length_label.setText(f"{value} chars")



        # Refresh the display if we have notes



        if hasattr(self, 'all_scored_notes') and self.all_scored_notes:



            self.filter_and_display_notes()







    def toggle_select_all(self):



        """Toggle between selecting all and deselecting all notes"""



        if not hasattr(self, 'results_list') or self.results_list.rowCount() == 0:



            return







        # Check if all are selected (check column 1 = Content which has the checkbox)



        all_selected = True



        for row in range(self.results_list.rowCount()):



            item = self.results_list.item(row, 1)



            if item and item.checkState() != Qt.CheckState.Checked:



                all_selected = False



                break







        # Toggle state



        if all_selected:



            self.deselect_all_notes()



        else:



            self.select_all_notes()







    def select_all_notes(self):



        """Select all notes in the results list"""



        if not hasattr(self, 'results_list'):



            return







        # Block signals to prevent multiple updates



        self.results_list.blockSignals(True)



        for row in range(self.results_list.rowCount()):



            item = self.results_list.item(row, 1)  # Content column has checkbox



            if item:



                item.setCheckState(Qt.CheckState.Checked)



                # Store in persistence



                note_id = item.data(Qt.ItemDataRole.UserRole)



                if note_id:



                    self.selected_note_ids.add(note_id)



        self.results_list.blockSignals(False)







        self.update_selection_count()



        tooltip(f"\u2713 Selected all {self.results_list.rowCount()} notes")







    def deselect_all_notes(self):



        """Deselect all notes in the results list"""



        if not hasattr(self, 'results_list'):



            return







        # Block signals to prevent multiple updates



        self.results_list.blockSignals(True)



        for row in range(self.results_list.rowCount()):



            item = self.results_list.item(row, 1)  # Content column has checkbox



            if item:



                item.setCheckState(Qt.CheckState.Unchecked)



                # Remove from persistence



                note_id = item.data(Qt.ItemDataRole.UserRole)



                if note_id:



                    self.selected_note_ids.discard(note_id)



        self.results_list.blockSignals(False)







        self.update_selection_count()



        tooltip(f"\xe2\u0153\u2014 Deselected all notes")







    def restore_selections(self):



        """Restore selections from stored note IDs"""



        if not hasattr(self, 'selected_note_ids') or not self.selected_note_ids:



            return







        # Block signals to prevent multiple updates



        self.results_list.blockSignals(True)



        for row in range(self.results_list.rowCount()):



            item = self.results_list.item(row, 1)  # Content column has checkbox



            if item:



                note_id = item.data(Qt.ItemDataRole.UserRole)



                if note_id in self.selected_note_ids:



                    item.setCheckState(Qt.CheckState.Checked)



        self.results_list.blockSignals(False)







    def open_selected_in_browser(self):



        checked_ids = []



        for row in range(self.results_list.rowCount()):



            item = self.results_list.item(row, 1)  # Content column has checkbox



            if item and item.checkState() == Qt.CheckState.Checked:



                note_id = item.data(Qt.ItemDataRole.UserRole)



                checked_ids.append(str(note_id))







        if not checked_ids:



            tooltip("Please check at least one note to view")



            return







        browser = dialogs.open("Browser", mw)



        search_query = "nid:" + ",".join(checked_ids)



        browser.form.searchEdit.lineEdit().setText(search_query)



        browser.onSearchActivated()



        QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))



        tooltip(f"\u2713 Opened {len(checked_ids)} selected notes in browser")







    def open_all_in_browser(self):



        if self.results_list.rowCount() == 0:



            tooltip("No notes to view")



            return







        note_ids = []



        for row in range(self.results_list.rowCount()):



            item = self.results_list.item(row, 1)  # Content column has note ID in UserRole



            if item:



                note_id = item.data(Qt.ItemDataRole.UserRole)



                note_ids.append(str(note_id))







        browser = dialogs.open("Browser", mw)



        search_query = "nid:" + ",".join(note_ids)



        browser.form.searchEdit.lineEdit().setText(search_query)



        browser.onSearchActivated()



        QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))



        tooltip(f"\u2713 Opened {len(note_ids)} notes in browser")







    def open_in_browser(self, item):



        """Open note in browser when double-clicked"""



        # Get the row of the clicked item



        row = item.row()



        # Get note ID from content column (column 1 = Content)



        content_item = self.results_list.item(row, 1)



        if content_item:



            note_id = content_item.data(Qt.ItemDataRole.UserRole)



            browser = dialogs.open("Browser", mw)



            browser.form.searchEdit.lineEdit().setText(f"nid:{note_id}")



            browser.onSearchActivated()



            QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))



            tooltip("\u2713 Note opened in browser")











# Copy methods that were defined on EmbeddingSearchWorker or RerankWorker to AISearchDialog



# (they are indented under those worker classes but are used on AISearchDialog instances)



_aisearch_methods_from_worker = (



    '_get_answer_source_text', '_on_embedding_search_progress', '_show_busy_progress', '_show_centile_progress',



    '_start_estimated_progress_timer', '_stop_estimated_progress_timer', '_hide_busy_progress', '_on_embedding_search_finished',



    '_on_keyword_filter_continue_done', '_on_embedding_search_error', '_on_rerank_done', '_on_get_notes_done', '_on_keyword_filter_done',



    '_perform_search_continue', 'perform_search',



    '_on_relevance_rerank_done', '_display_answer_and_notes_after_rerank', '_on_relevance_rerank_done_stream', '_finish_anthropic_stream_display',



    '_on_ask_ai_success', '_on_ask_ai_error', '_on_ask_ai_worker_finished',



    '_rerank_with_cross_encoder', 'keyword_filter', 'keyword_filter_continue', 'get_best_model',



    'ask_ai', 'call_ollama', 'call_anthropic', 'call_openai', 'call_google', 'call_openrouter',



    'call_custom', 'make_request', 'parse_response', '_rerank_by_relevance_to_answer', 'filter_and_display_notes', '_get_matching_terms_for_note', 'update_selection_count',



    '_start_anthropic_stream', '_append_stream_chunk', '_on_anthropic_stream_done', '_on_anthropic_stream_error',



    'on_preview_length_changed', 'toggle_select_all', 'select_all_notes', 'deselect_all_notes',



    'restore_selections', '_bring_browser_to_front', 'open_selected_in_browser', 'open_all_in_browser', 'open_in_browser',



)



for _name in _aisearch_methods_from_worker:



    _method = (



        getattr(EmbeddingSearchWorker, _name, None)



        or getattr(RerankWorker, _name, None)



        or getattr(AnthropicStreamWorker, _name, None)



    )



    if _method is not None:



        setattr(AISearchDialog, _name, _method)











# Singleton: only one search dialog instance so data is not loaded multiple times



_ai_search_dialog_instance = None







def toggle_sidebar_visibility(visible: bool):

    """Safely show or hide the sidebar drawer."""

    sidebar = globals().get("_sidebar_instance")
    updater = globals().get("update_sidebar_position")

    if visible:

        if callable(updater):

            updater()

        return

    if sidebar and hasattr(sidebar, "hide"):

        sidebar.hide()




class SettingsDialog(QDialog):



    def __init__(self, parent=None, open_to_embeddings=False):



        import time



        _t0 = time.time()



        super().__init__(parent)



        self.open_to_embeddings = open_to_embeddings



        self.setWindowTitle("Anki Semantic Search \u2014 Settings")



        self._rerank_check_done = False  # defer rerank check until after show



        # Size: allow small minimum, no max so user can maximize/resize to expand and reduce cramming



        self.setMinimumWidth(750)



        self.setMinimumHeight(550)



        # Open large by default so Search Settings content is less crammed; user can maximize or resize



        screen = QApplication.primaryScreen().geometry()



        w = min(1200, int(screen.width() * 0.96))



        h = min(960, int(screen.height() * 0.92))



        self.resize(w, h)



        # Behave like a normal top-level window so minimize/maximize work



        self.setWindowFlags(



            Qt.WindowType.Window



            | Qt.WindowType.WindowMinimizeButtonHint



            | Qt.WindowType.WindowMaximizeButtonHint



            | Qt.WindowType.WindowCloseButtonHint



        )







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



                font-size: 12px;



            }}



            QPushButton {{ padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 12px; color: white; }}



            QPushButton#saveBtn {{ background-color: {theme['success']}; border: none; }}



            QPushButton#saveBtn:hover {{ background-color: {theme['success_hover']}; }}



            QPushButton#cancelBtn {{ background-color: {theme['muted_btn']}; border: none; }}



            QPushButton#cancelBtn:hover {{ background-color: {theme['muted_btn_hover']}; }}



            QComboBox {{



                padding: 8px;



                border: 2px solid {theme['border']};



                border-radius: 6px;



                background-color: {theme['input_bg']};



                color: {theme['input_text']};



            }}



            QGroupBox {{



                font-weight: bold;



                border: 2px solid {theme['border']};



                border-radius: 5px;



                margin-top: 10px;



                padding-top: 10px;



                color: {theme['text']};



            }}



            QGroupBox:disabled {{ color: {theme['subtext']}; border-color: {theme['panel_border']}; }}



            QSpinBox {{



                padding: 5px;



                border: 2px solid {theme['border']};



                border-radius: 4px;



                background-color: {theme['input_bg']};



                color: {theme['input_text']};



            }}



            QTabWidget::pane {{ border: 1px solid {theme['panel_border']}; background-color: {theme['bg']}; }}



            QTabBar::tab {{



                background-color: {theme['input_bg']};



                color: {theme['text']};



                padding: 8px 16px;



                border-top-left-radius: 4px;



                border-top-right-radius: 4px;



            }}



            QTabBar::tab:selected {{ background-color: {theme['accent']}; color: #ffffff; }}



            """



        )



        log_debug("=== Settings Dialog Opened ===")



        # Store reference to service process



        self.service_process = None



        self.setup_ui()







    def showEvent(self, event):



        """Defer rerank availability check until after window is shown so opening Settings doesn't freeze."""



        super().showEvent(event)



        if not getattr(self, "_rerank_check_scheduled", False):



            self._rerank_check_scheduled = True



            from aqt.qt import QTimer



            QTimer.singleShot(80, self._deferred_check_rerank)







    def _deferred_check_rerank(self):



        """Run _check_rerank_available in a worker thread so Settings never freezes."""



        import time



        config = load_config()



        sc = (config or {}).get("search_config") or {}



        rerank_python = (sc.get("rerank_python_path") or "").strip() or None



        self._rerank_check_worker = RerankCheckWorker(self, rerank_python)



        self._rerank_check_start = time.time()







        def _on_rerank_check_done(available):

            try:

                try:
                    try:
                        import sip
                    except ImportError:
                        from PyQt6 import sip

                except ImportError:
                    from PyQt6 import sip



                self._rerank_available = available

                cb = getattr(self, "enable_rerank_cb", None)

                if cb is not None and not sip.isdeleted(cb):

                    cb.setEnabled(available)



                if hasattr(self, "_update_rerank_tooltip") and not sip.isdeleted(self):

                    self._update_rerank_tooltip()

                self._rerank_check_worker = None

            except (RuntimeError, AttributeError, ImportError):

                pass







        self._rerank_check_worker.finished_signal.connect(_on_rerank_check_done)



        self._rerank_check_worker.start()







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

        # --- RADICAL STABILIZATION: Guard against re-initialization ---

        if hasattr(self, "_ui_initialized") and self._ui_initialized:

            return

        self._ui_initialized = True



        current_config = load_config()

        self.current_config = current_config # Store for access in other methods



        import time

        start_time = time.time()



        log_debug("=== Settings Dialog UI Setup Started ===")



        palette = QApplication.palette()



        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128



        theme = _addon_theme(is_dark)







        # Main layout with proper spacing - fix layout issues



        main_layout = QVBoxLayout(self)



        main_layout.setSpacing(10)



        main_layout.setContentsMargins(15, 15, 15, 15)







        elapsed = time.time() - start_time



        log_debug(f"  [Timing] Layout setup: {elapsed:.3f}s")







        tabs = QTabWidget()



        # Size to content; scroll_content minimum and scroll area give enough height for scrolling when needed



        tabs.setMinimumHeight(0)







        # API Settings Tab



        api_tab = QWidget()



        api_layout = QVBoxLayout(api_tab)



        api_layout.setSpacing(15)



        api_layout.setContentsMargins(20, 20, 20, 20)







        info = QLabel("\U0001F511 API & AI provider")



        info.setWordWrap(True)



        info.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {theme['text']}; margin-bottom: 4px;")



        api_layout.addWidget(info)







        subtitle = QLabel(

            "Choose a Local Server (no key required) or a Cloud API key. Provider is detected automatically from the key. "

            "Works with OpenAI, Anthropic, Google, OpenRouter, or custom OpenAI-compatible endpoints."

        )



        subtitle.setWordWrap(True)



        subtitle.setStyleSheet(f"font-size: 11px; color: {theme['subtext']}; margin-bottom: 10px;")



        api_layout.addWidget(subtitle)







        privacy_note = QLabel(

            "\u26a0\ufe0f AI answers may send selected note content to the chosen provider. "

            "Use a Local Server (Ollama, LM Studio, Jan) to keep everything on your machine. "

            "Cloud providers require an API key and an internet connection \u2014 your note content will be sent to external APIs."

        )



        privacy_note.setWordWrap(True)



        privacy_note.setStyleSheet(



            f"font-size: 11px; color: {theme['warn_text']}; margin-bottom: 10px; "



            f"padding: 8px; background-color: {theme['warn_bg']}; border-radius: 4px;"



        )



        api_layout.addWidget(privacy_note)







        # Define sections first (initialize as None or create them)

        self.api_key_section = QWidget()

        self.local_server_section = QWidget()



        # Answer with: Local vs Cloud

        answer_provider_row = QFormLayout()

        self.answer_provider_combo = QComboBox()

        self.answer_provider_combo.blockSignals(True) # Silence during setup

        self.answer_provider_combo.addItem("\U0001f4bb Local Server (Ollama, LM Studio, Jan)", "local_server")

        self.answer_provider_combo.addItem("\u2601\ufe0f Cloud API (OpenAI, Anthropic, Gemini)", "api_key")



        self.answer_provider_combo.setToolTip(

            "Local Server: Best for privacy. Supports Ollama, LM Studio, Jan, etc.\n"

            "Cloud API: Best quality, requires internet and API key."

        )

        answer_provider_row.addRow("Answer with:", self.answer_provider_combo)



        # Set initial value based on config

        current_provider = current_config.get('provider', 'openai')

        if current_provider in ['ollama', 'local_openai', 'local_server']:

            self.answer_provider_combo.setCurrentIndex(0) # Local

        else:

            self.answer_provider_combo.setCurrentIndex(1) # Cloud



        self.answer_provider_combo.blockSignals(False) # Re-enable

        self.answer_provider_combo.currentIndexChanged.connect(self._on_answer_provider_changed)



        api_embed_note = QLabel("Embeddings (for semantic search) are configured in the Search & Embeddings tab.")



        api_embed_note.setWordWrap(True)



        api_embed_note.setStyleSheet(f"font-size: 10px; color: {theme['subtext']}; margin-top: 2px;")



        answer_provider_row.addRow("", api_embed_note)



        api_layout.addLayout(answer_provider_row)







        # API Key section (hidden when Ollama is selected)



        self.api_key_section = QWidget()



        api_key_section_layout = QVBoxLayout(self.api_key_section)



        api_key_section_layout.setContentsMargins(0, 0, 0, 0)



        key_layout = QHBoxLayout()



        key_label = QLabel("API Key:")



        self.api_key_input = QLineEdit()



        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)



        self.api_key_input.setPlaceholderText("Paste your API key here...")



        self.api_key_input.textChanged.connect(self.detect_provider)







        key_layout.addWidget(key_label)



        key_layout.addWidget(self.api_key_input)







        self.show_key_btn = QPushButton("Show")



        self.show_key_btn.setMaximumWidth(80)



        self.show_key_btn.clicked.connect(self.toggle_key_visibility)



        key_layout.addWidget(self.show_key_btn)



        api_key_section_layout.addLayout(key_layout)







        self.provider_label = QLabel()



        self.provider_label.setStyleSheet(



            f"background-color: {theme['header_bg']}; color: {theme['accent']}; padding: 10px; border-radius: 5px; font-weight: bold;"



        )



        self.provider_label.hide()



        api_key_section_layout.addWidget(self.provider_label)







        url_layout = QHBoxLayout()



        url_label = QLabel("API URL:")



        self.api_url_input = QLineEdit()



        self.api_url_input.setPlaceholderText("https://api.example.com/v1/chat/completions")



        url_layout.addWidget(url_label)



        url_layout.addWidget(self.api_url_input)



        self.url_widget = QWidget()



        self.url_widget.setLayout(url_layout)



        self.url_widget.hide()



        api_key_section_layout.addWidget(self.url_widget)



        api_layout.addWidget(self.api_key_section)







        # Unified Local Server Section

        self.local_server_section = QWidget()

        local_server_layout = QFormLayout(self.local_server_section)



        self.local_llm_url = QLineEdit()

        self.local_llm_url.setPlaceholderText("http://localhost:11434 (Ollama) or http://localhost:1234/v1 (LM Studio)")

        local_server_layout.addRow("Server URL:", self.local_llm_url)



        model_row = QHBoxLayout()

        self.local_llm_model = QLineEdit()

        self.local_llm_model.setPlaceholderText("e.g. llama3.2 or gemma2")

        model_row.addWidget(self.local_llm_model)



        self.ollama_chat_refresh_btn = QPushButton("\U0001F504 Refresh")

        self.ollama_chat_refresh_btn.setToolTip("Try to fetch models from the local server.")

        self.ollama_chat_refresh_btn.clicked.connect(self._refresh_local_models)

        model_row.addWidget(self.ollama_chat_refresh_btn)



        local_server_layout.addRow("Model Name:", model_row)



        self.local_server_test_btn = QPushButton("\U0001F50C Test Connection")

        self.local_server_test_btn.setToolTip("Test connection to your local server. Shows latency and availability.")

        self.local_server_test_btn.clicked.connect(self._test_local_server_connection)

        local_server_layout.addRow("", self.local_server_test_btn)



        local_guide = QLabel(

            "ΓÇó <b>Ollama:</b> http://localhost:11434<br>"

            "ΓÇó <b>LM Studio:</b> http://localhost:1234/v1<br>"

            "ΓÇó <b>Jan:</b> http://localhost:1337/v1"

        )

        local_guide.setStyleSheet("font-size: 10px; color: #7f8c8d;")

        local_server_layout.addRow("", local_guide)



        self.local_server_section.hide()

        api_layout.addWidget(self.local_server_section)







        # Collapsible help: "Need help?" toggles visibility



        self._api_help_visible = False



        api_help_btn = QPushButton("Need help? (providers, free options)")



        api_help_btn.setToolTip("Click to show or hide provider links and free options")



        self.info_text = QLabel()



        self.info_text.setWordWrap(True)



        self.info_text.setStyleSheet("background-color: #f0f0f0; color: #333333; padding: 10px; border-radius: 5px; font-size: 11px;")



        self.info_text.setText(



            "Answer with Ollama (local): no API key \u2014 choose 'Ollama (local)' above and set Chat model. Uses Ollama URL from Search & Embeddings tab.\n\n"



            "API Key (cloud):\n"



            "\u2022 Anthropic (Claude): sk-ant-... \u2192 console.anthropic.com\n"



            "\u2022 OpenAI (GPT): sk-... \u2192 platform.openai.com/api-keys\n"



            "\u2022 Google (Gemini): AI... \u2192 aistudio.google.com/app/apikey (FREE!)\n"



            "\u2022 OpenRouter: sk-or-... \u2192 openrouter.ai/keys\n\n"



            "\U0001F4A1 Free options: Google Gemini or Ollama (local)"



        )



        self.info_text.setVisible(False)



        def _toggle_api_help():



            self._api_help_visible = not self._api_help_visible



            self.info_text.setVisible(self._api_help_visible)



            api_help_btn.setText("\u25b2 Hide help" if self._api_help_visible else "Need help? (providers, free options)")



        api_help_btn.clicked.connect(_toggle_api_help)



        api_layout.addWidget(api_help_btn)



        api_layout.addWidget(self.info_text)



        api_layout.addStretch()







        tabs.addTab(api_tab, "\U0001F511 API Settings")







        # Styling Tab



        style_tab = QWidget()



        style_layout = QVBoxLayout(style_tab)



        style_layout.setSpacing(15)



        style_layout.setContentsMargins(20, 20, 20, 20)







        style_info = QLabel("\U0001F3A8 Appearance")



        style_info.setStyleSheet("font-size: 15px; font-weight: bold; color: #2c3e50; margin-bottom: 4px;")



        style_layout.addWidget(style_info)



        style_sub = QLabel("Font sizes, window size, and layout. Optional; defaults work for most users.")



        style_sub.setWordWrap(True)



        style_sub.setStyleSheet("font-size: 11px; color: #7f8c8d; margin-bottom: 10px;")



        style_layout.addWidget(style_sub)







        # Font Size Settings



        font_group = QGroupBox("Font Sizes")



        font_layout = QVBoxLayout(font_group)







        question_font_layout = QHBoxLayout()



        question_font_label = QLabel("Question Input Font Size:")



        self.question_font_spin = QSpinBox()



        self.question_font_spin.setRange(10, 20)



        self.question_font_spin.setValue(13)



        self.question_font_spin.setSuffix(" px")



        question_font_layout.addWidget(question_font_label)



        question_font_layout.addStretch()



        question_font_layout.addWidget(self.question_font_spin)



        font_layout.addLayout(question_font_layout)







        answer_font_layout = QHBoxLayout()



        answer_font_label = QLabel("AI Answer Font Size:")



        self.answer_font_spin = QSpinBox()



        self.answer_font_spin.setRange(10, 20)



        self.answer_font_spin.setValue(13)



        self.answer_font_spin.setSuffix(" px")



        answer_font_layout.addWidget(answer_font_label)



        answer_font_layout.addStretch()



        answer_font_layout.addWidget(self.answer_font_spin)



        font_layout.addLayout(answer_font_layout)







        notes_font_layout = QHBoxLayout()



        notes_font_label = QLabel("Notes List Font Size:")



        self.notes_font_spin = QSpinBox()



        self.notes_font_spin.setRange(10, 18)



        self.notes_font_spin.setValue(12)



        self.notes_font_spin.setSuffix(" px")



        notes_font_layout.addWidget(notes_font_label)



        notes_font_layout.addStretch()



        notes_font_layout.addWidget(self.notes_font_spin)



        font_layout.addLayout(notes_font_layout)







        label_font_layout = QHBoxLayout()



        label_font_label = QLabel("Label Font Size:")



        self.label_font_spin = QSpinBox()



        self.label_font_spin.setRange(11, 18)



        self.label_font_spin.setValue(14)



        self.label_font_spin.setSuffix(" px")



        label_font_layout.addWidget(label_font_label)



        label_font_layout.addStretch()



        label_font_layout.addWidget(self.label_font_spin)



        font_layout.addLayout(label_font_layout)







        style_layout.addWidget(font_group)







        # Window Size Settings



        window_group = QGroupBox("Window Size")



        window_layout = QVBoxLayout(window_group)







        width_layout = QHBoxLayout()



        width_label = QLabel("Default Window Width:")



        self.width_spin = QSpinBox()



        self.width_spin.setRange(800, 1600)



        self.width_spin.setValue(1100)



        self.width_spin.setSuffix(" px")



        width_layout.addWidget(width_label)



        width_layout.addStretch()



        width_layout.addWidget(self.width_spin)



        window_layout.addLayout(width_layout)







        height_layout = QHBoxLayout()



        height_label = QLabel("Default Window Height:")



        self.height_spin = QSpinBox()



        self.height_spin.setRange(600, 1200)



        self.height_spin.setValue(800)



        self.height_spin.setSuffix(" px")



        height_layout.addWidget(height_label)



        height_layout.addStretch()



        height_layout.addWidget(self.height_spin)



        window_layout.addLayout(height_layout)







        style_layout.addWidget(window_group)







        # Layout



        layout_group = QGroupBox("Layout")



        layout_layout = QVBoxLayout(layout_group)



        layout_row = QHBoxLayout()



        layout_label = QLabel("Answer & Notes:")



        self.layout_combo = QComboBox()



        self.layout_combo.addItem("Side-by-side (answer | notes)", "side_by_side")



        self.layout_combo.addItem("Stacked (answer above notes)", "stacked")



        layout_row.addWidget(layout_label)



        layout_row.addStretch()



        layout_row.addWidget(self.layout_combo)



        layout_layout.addLayout(layout_row)



        style_layout.addWidget(layout_group)







        # Spacing Settings



        spacing_group = QGroupBox("Spacing & Padding")



        spacing_layout = QVBoxLayout(spacing_group)







        section_spacing_layout = QHBoxLayout()



        section_spacing_label = QLabel("Section Spacing:")



        self.section_spacing_spin = QSpinBox()



        self.section_spacing_spin.setRange(5, 20)



        self.section_spacing_spin.setValue(12)



        self.section_spacing_spin.setSuffix(" px")



        section_spacing_layout.addWidget(section_spacing_label)



        section_spacing_layout.addStretch()



        section_spacing_layout.addWidget(self.section_spacing_spin)



        spacing_layout.addLayout(section_spacing_layout)







        answer_spacing_layout = QHBoxLayout()



        answer_spacing_label = QLabel("Answer line spacing:")



        self.answer_spacing_combo = QComboBox()



        self.answer_spacing_combo.addItem("Compact", "compact")



        self.answer_spacing_combo.addItem("Normal", "normal")



        self.answer_spacing_combo.addItem("Comfortable", "comfortable")



        answer_spacing_layout.addWidget(answer_spacing_label)



        answer_spacing_layout.addStretch()



        answer_spacing_layout.addWidget(self.answer_spacing_combo)



        spacing_layout.addLayout(answer_spacing_layout)







        style_layout.addWidget(spacing_group)



        style_layout.addStretch()







        tabs.addTab(style_tab, "\U0001F3A8 Styling")







        # --- Note Types & Fields Tab ---



        nt_tab = QWidget()



        nt_main = QVBoxLayout(nt_tab)



        nt_main.setSpacing(10)



        nt_main.setContentsMargins(20, 20, 20, 20)







        nt_info = QLabel(



            "\U0001F4CB Choose which note types and decks to search. Select fields to search per note type. "



            "Tip: For shared or public decks, select the note types and decks you want; you can leave all selected to search everything."



        )



        nt_info.setWordWrap(True)



        nt_info.setStyleSheet("font-size: 11px; color: #7f8c8d; margin-bottom: 6px;")



        nt_info.setStyleSheet("font-size: 15px; font-weight: bold; color: #2c3e50; margin-bottom: 6px;")



        nt_info.setWordWrap(True)



        nt_main.addWidget(nt_info)







        # Side-by-side: left = Note types + Decks (stacked), right = Fields by note type



        main_h_split = QSplitter(Qt.Orientation.Horizontal)



        left_v_split = QSplitter(Qt.Orientation.Vertical)







        # ---- Left column: Note types (top) ----



        nt_group = QGroupBox("Note types to include")



        nt_gl = QVBoxLayout(nt_group)



        nt_btn_row = QHBoxLayout()



        nt_select_btn = QPushButton("Select All")



        nt_select_btn.clicked.connect(lambda: self._set_note_types_checked(True))



        nt_deselect_btn = QPushButton("Deselect All")



        nt_deselect_btn.clicked.connect(lambda: self._set_note_types_checked(False))



        nt_btn_row.addWidget(nt_select_btn)



        nt_btn_row.addWidget(nt_deselect_btn)



        nt_btn_row.addStretch()



        # Sort options



        sort_label = QLabel("Sort by:")



        self.sort_combo = QComboBox()



        self.sort_combo.addItem("Note Count (Desc)", "count_desc")



        self.sort_combo.addItem("Note Count (Asc)", "count_asc")



        self.sort_combo.addItem("Name (A-Z)", "name_asc")



        self.sort_combo.addItem("Name (Z-A)", "name_desc")



        self.sort_combo.currentIndexChanged.connect(self._on_sort_note_types_changed)



        nt_btn_row.addWidget(sort_label)



        nt_btn_row.addWidget(self.sort_combo)



        nt_gl.addLayout(nt_btn_row)



        self.include_all_note_types_cb = QCheckBox("Include all note types")



        self.include_all_note_types_cb.setChecked(True)



        self.include_all_note_types_cb.stateChanged.connect(self._on_include_all_note_types_toggled)



        nt_gl.addWidget(self.include_all_note_types_cb)



        self.note_types_table = QTableWidget()



        self.note_types_table.setColumnCount(2)



        self.note_types_table.setHorizontalHeaderLabels(["Note Type", "Note Count"])



        self.note_types_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)



        self.note_types_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)



        self.note_types_table.setColumnWidth(1, 120)  # Set minimum width for count column



        self.note_types_table.setMinimumHeight(80)



        self.note_types_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



        self.note_types_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)



        self.note_types_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)



        self.note_types_table.itemChanged.connect(self._update_field_groups_enabled)



        self.note_types_table.setSortingEnabled(True)



        self.note_types_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)  # Sort by count descending by default



        # Add spacing between columns



        self.note_types_table.setColumnWidth(0, 200)



        nt_gl.addWidget(self.note_types_table)



        left_v_split.addWidget(nt_group)







        # ---- Left column: Decks (bottom) ----



        deck_group = QGroupBox("Decks to search")



        deck_gl = QVBoxLayout(deck_group)



        self.include_all_decks_cb = QCheckBox("Include all decks")



        self.include_all_decks_cb.setChecked(True)



        self.include_all_decks_cb.stateChanged.connect(self._on_include_all_decks_toggled)



        deck_gl.addWidget(self.include_all_decks_cb)



        deck_btn_row = QHBoxLayout()



        deck_select_btn = QPushButton("Select All")



        deck_select_btn.clicked.connect(lambda: self._set_decks_checked(True))



        deck_deselect_btn = QPushButton("Deselect All")



        deck_deselect_btn.clicked.connect(lambda: self._set_decks_checked(False))



        deck_btn_row.addWidget(deck_select_btn)



        deck_btn_row.addWidget(deck_deselect_btn)



        deck_btn_row.addStretch()



        deck_gl.addLayout(deck_btn_row)



        # Use QTreeWidget for hierarchical deck display (like main Anki interface)



        self.decks_list = QTreeWidget()



        self.decks_list.setMinimumHeight(80)



        self.decks_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



        # Simplified header: only show deck name and total notes



        self.decks_list.setHeaderLabels(["Deck", "Notes"])



        self.decks_list.setRootIsDecorated(True)  # Show expand/collapse arrows



        self.decks_list.setAlternatingRowColors(True)



        # Set column widths: deck name and notes



        self.decks_list.setColumnWidth(0, 260)  # Deck name



        self.decks_list.setColumnWidth(1, 70)   # Notes



        self.decks_list.itemChanged.connect(self._on_deck_item_changed)



        deck_gl.addWidget(self.decks_list)



        left_v_split.addWidget(deck_group)







        left_v_split.setSizes([220, 180])



        left_v_split.setChildrenCollapsible(False)



        left_v_split.setHandleWidth(6)



        main_h_split.addWidget(left_v_split)







        # ---- Right column: Fields by note type ----



        fld_outer = QGroupBox("Fields to search per note type (greyed if note type unchecked)")



        fld_outer_l = QVBoxLayout(fld_outer)



        self.search_all_fields_cb = QCheckBox("Search in all fields (ignore selections below)")



        self.search_all_fields_cb.setChecked(False)



        self.search_all_fields_cb.stateChanged.connect(self._on_search_all_fields_toggled)



        fld_outer_l.addWidget(self.search_all_fields_cb)



        self.use_first_field_cb = QCheckBox("Use first field when no fields selected for a note type")



        self.use_first_field_cb.setChecked(True)



        self.use_first_field_cb.setToolTip("If a note type has no checked fields, use its first field instead of skipping.")



        fld_outer_l.addWidget(self.use_first_field_cb)



        self.fields_by_note_type_scroll = QScrollArea()



        self.fields_by_note_type_scroll.setMinimumHeight(120)



        self.fields_by_note_type_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



        self.fields_by_note_type_scroll.setWidgetResizable(True)



        self.fields_by_note_type_inner = QWidget()



        self.fields_by_note_type_layout = QVBoxLayout(self.fields_by_note_type_inner)



        self.fields_by_note_type_layout.setAlignment(Qt.AlignmentFlag.AlignTop)



        self.fields_by_note_type_scroll.setWidget(self.fields_by_note_type_inner)



        fld_outer_l.addWidget(self.fields_by_note_type_scroll)



        main_h_split.addWidget(fld_outer)



        self._field_cbs = {}  # model_name -> { field_name: QCheckBox }



        self._field_groupboxes = {}  # model_name -> QGroupBox (for greying when note type unchecked)







        main_h_split.setSizes([380, 420])



        main_h_split.setChildrenCollapsible(False)



        main_h_split.setHandleWidth(6)



        nt_main.addWidget(main_h_split)







        # ---- Count notes, Save/Load/Delete, Refresh ----



        action_row = QHBoxLayout()



        self.count_notes_btn = QPushButton("\U0001F4CA Count notes (with current settings)")



        self.count_notes_btn.clicked.connect(self._on_count_notes)



        action_row.addWidget(self.count_notes_btn)



        action_row.addStretch()



        nt_main.addLayout(action_row)







        save_pl = QHBoxLayout()



        self.preset_name_edit = QLineEdit()



        self.preset_name_edit.setPlaceholderText("e.g. Default, Shared deck")



        self.preset_name_edit.setMaximumWidth(200)



        self.preset_name_edit.setToolTip("Name for this set of note types and decks (for quick switching)")



        save_pl.addWidget(QLabel("Save as preset:"))



        save_pl.addWidget(self.preset_name_edit)



        save_preset_btn = QPushButton("\U0001F4BE Save preset")



        save_preset_btn.clicked.connect(self._on_save_preset)



        save_pl.addWidget(save_preset_btn)



        save_pl.addStretch()



        save_pl.addWidget(QLabel("Load preset:"))



        self.load_preset_combo = QComboBox()



        self.load_preset_combo.setMaximumWidth(180)



        self.load_preset_combo.setEditable(False)



        save_pl.addWidget(self.load_preset_combo)



        load_preset_btn = QPushButton("Load")



        load_preset_btn.clicked.connect(self._on_load_preset)



        save_pl.addWidget(load_preset_btn)



        self.delete_preset_combo = QComboBox()



        self.delete_preset_combo.setMaximumWidth(160)



        save_pl.addWidget(QLabel("Delete:"))



        save_pl.addWidget(self.delete_preset_combo)



        delete_preset_btn = QPushButton("Delete preset")



        delete_preset_btn.clicked.connect(self._on_delete_preset)



        save_pl.addWidget(delete_preset_btn)



        nt_main.addLayout(save_pl)







        refresh_btn = QPushButton("\U0001F504 Refresh lists (detect new note types/fields/decks)")



        refresh_btn.clicked.connect(self._refresh_note_type_lists)



        nt_main.addWidget(refresh_btn)







        # Defer heavy operations to avoid lag (with timing logs)



        QTimer.singleShot(50, lambda: self._populate_note_type_lists_with_timing())



        QTimer.singleShot(100, lambda: self._populate_fields_by_note_type_with_timing())



        QTimer.singleShot(150, lambda: self._populate_decks_list_with_timing())



        QTimer.singleShot(200, lambda: self._refresh_preset_combos_with_timing())







        nt_main.addStretch()







        tabs.addTab(nt_tab, "\U0001F4CB Note Types & Fields")







        # --- Search Settings Tab (Reorganized by Glutamine) ---

        search_tab = QWidget()

        search_scroll = QScrollArea()

        search_scroll.setWidgetResizable(True)

        search_scroll.setFrameShape(QFrame.Shape.NoFrame)

        search_tab_inner = QWidget()

        search_layout = QVBoxLayout(search_tab_inner)

        search_layout.setSpacing(15)

        search_layout.setContentsMargins(24, 16, 24, 16)

        search_scroll.setWidget(search_tab_inner)



        main_search_layout = QVBoxLayout(search_tab)

        main_search_layout.setContentsMargins(0, 0, 0, 0)

        main_search_layout.addWidget(search_scroll)



        search_info = QLabel("\U0001F50D Search & embeddings")

        search_info.setStyleSheet("font-size: 15px; font-weight: bold; color: #2c3e50; margin-bottom: 2px;")

        search_layout.addWidget(search_info)



        search_sub = QLabel("Choose how notes are matched (keyword, hybrid, or embedding). Optional: tune result count and relevance. Works with any deck.")

        search_sub.setWordWrap(True)

        search_sub.setStyleSheet("font-size: 11px; color: #7f8c8d; margin-bottom: 5px;")

        search_layout.addWidget(search_sub)



        # Glutamine's Resident Reset Button

        reset_btn_layout = QHBoxLayout()

        self.medical_reset_btn = QPushButton("\u2695\ufe0f Reset to Clinical Defaults")

        self.medical_reset_btn.setToolTip("Sets all search, relevance, and AI settings to high-yield medical defaults (Resident-approved).")

        self.medical_reset_btn.setStyleSheet("background-color: #2c3e50; color: #ecf0f1; border: 1px solid #34495e; padding: 10px; font-weight: bold;")

        self.medical_reset_btn.clicked.connect(self.reset_to_medical_defaults)

        reset_btn_layout.addWidget(self.medical_reset_btn)

        search_layout.addLayout(reset_btn_layout)



        # --- High-Yield Action Zone (New persistent section for indexing) ---

        index_zone = QFrame()

        index_zone.setStyleSheet("QFrame { background-color: #2c3e50; border-radius: 6px; border: 1px solid #3498db; }")

        index_layout = QVBoxLayout(index_zone)



        self.embedding_status_label = QLabel("Ready to index...")

        self.embedding_status_label.setStyleSheet("color: #ecf0f1; font-weight: bold; border: none;")

        self.embedding_status_label.setWordWrap(True)

        index_layout.addWidget(self.embedding_status_label)



        index_btns = QHBoxLayout()

        self.create_embedding_btn = QPushButton("Create/Update Embeddings")

        self.create_embedding_btn.setToolTip(EmbeddingsTabMessages.CREATE_UPDATE_TOOLTIP)

        self.create_embedding_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 8px; font-weight: bold; border: none; } QPushButton:hover { background-color: #2980b9; }")

        self.create_embedding_btn.clicked.connect(self._create_or_update_embeddings)

        index_btns.addWidget(self.create_embedding_btn)

        self.review_ineligible_btn = QPushButton("Review Ineligible Notes")

        self.review_ineligible_btn.setToolTip("Open notes excluded from embeddings by the current deck, note type, and field filters.")

        self.review_ineligible_btn.setStyleSheet("QPushButton { background-color: #5d6d7e; color: white; padding: 8px; border: none; }")

        self.review_ineligible_btn.clicked.connect(self._review_ineligible_notes)

        index_btns.addWidget(self.review_ineligible_btn)



        self.test_connection_btn = QPushButton("Test Connection")

        self.test_connection_btn.setToolTip(EmbeddingsTabMessages.TEST_CONNECTION_TOOLTIP)

        self.test_connection_btn.setStyleSheet("QPushButton { background-color: #7f8c8d; color: white; padding: 8px; border: none; }")

        self.test_connection_btn.clicked.connect(self._test_embedding_connection)

        index_btns.addWidget(self.test_connection_btn)



        index_layout.addLayout(index_btns)

        search_layout.addWidget(index_zone)



        # --- 1. EMBEDDINGS SECTION (Critical Setup - Top Priority) ---

        self.embedding_section = CollapsibleSection("Embeddings (for semantic search)", is_expanded=False)

        engine_row = QFormLayout()

        self.embedding_engine_combo = QComboBox()

        self.embedding_engine_combo.addItem("Cloud AI (Voyage, OpenAI, Cohere)", "cloud")

        self.embedding_engine_combo.addItem("Local AI (Ollama, LM Studio)", "local")

        self.embedding_engine_combo.currentIndexChanged.connect(self._on_embedding_engine_changed)

        engine_row.addRow("AI Strategy:", self.embedding_engine_combo)

        self.embedding_section.addLayout(engine_row)



        self.cloud_provider_widget = QWidget()

        cloud_p_layout = QFormLayout(self.cloud_provider_widget)

        cloud_p_layout.setContentsMargins(0, 0, 0, 0)

        self.cloud_provider_combo = QComboBox()

        self.cloud_provider_combo.addItem("Voyage AI (Recommended)", "voyage")

        self.cloud_provider_combo.addItem("OpenAI", "openai")

        self.cloud_provider_combo.addItem("Cohere", "cohere")

        self.cloud_provider_combo.currentIndexChanged.connect(self._on_embedding_engine_changed)

        cloud_p_layout.addRow("Cloud Provider:", self.cloud_provider_combo)

        self.embedding_section.addWidget(self.cloud_provider_widget)



        # --- Cloud Specific Extras (RAG Hint/Apply) ---

        self.cloud_extras_widget = QWidget()

        cloud_extras_layout = QVBoxLayout(self.cloud_extras_widget)

        cloud_extras_layout.setContentsMargins(0, 0, 0, 0)



        self.embedding_hybrid_hint = QLabel("RAG split: Retrieval (here) = AI embedding for best quality. Answer (API tab) = Local AI.")

        self.embedding_hybrid_hint.setStyleSheet("font-size: 11px; color: #7f8c8d;")

        cloud_extras_layout.addWidget(self.embedding_hybrid_hint)



        self.apply_hybrid_btn = QPushButton("Apply: RAG-optimized (Cloud retrieval + Local answer)")

        self.apply_hybrid_btn.clicked.connect(self._on_apply_hybrid_retrieval)

        cloud_extras_layout.addWidget(self.apply_hybrid_btn)



        self.embedding_section.addWidget(self.cloud_extras_widget)

        search_layout.addWidget(self.embedding_section)



        # --- 2. SEARCH STRATEGY ---

        strategy_section = CollapsibleSection("Search Strategy", is_expanded=False)

        self.search_method_combo = QComboBox()

        self.search_method_combo.addItem("Keyword Only", "keyword")

        self.search_method_combo.addItem("Keyword + Re-rank", "keyword_rerank")

        self.search_method_combo.addItem("Hybrid (RRF)", "hybrid")

        self.search_method_combo.addItem("Embedding Only", "embedding")

        self.search_method_combo.currentIndexChanged.connect(self._on_search_method_changed)

        strategy_section.addWidget(self.search_method_combo)

        search_layout.addWidget(strategy_section)



        # --- 3. AI-ASSISTED RETRIEVAL ---

        ai_retrieval_section = CollapsibleSection("AI-Assisted Retrieval", is_expanded=False)

        self.enable_query_expansion_cb = QCheckBox("Query Expansion (AI adds medical synonyms)")

        ai_retrieval_section.addWidget(self.enable_query_expansion_cb)

        self.use_ai_generic_term_detection_cb = QCheckBox("Filter Filler Words (AI detects generic terms)")

        ai_retrieval_section.addWidget(self.use_ai_generic_term_detection_cb)

        self.enable_hyde_cb = QCheckBox("HyDE (AI generates hypothetical document first)")

        ai_retrieval_section.addWidget(self.enable_hyde_cb)

        search_layout.addWidget(ai_retrieval_section)



        # --- 4. CLINICAL ACCURACY TUNING ---

        accuracy_section = CollapsibleSection("Clinical Accuracy Tuning", is_expanded=False)

        accuracy_form = QFormLayout()

        self.min_relevance_spin = QSpinBox()

        self.min_relevance_spin.setRange(15, 75)

        accuracy_form.addRow("Minimum Relevance Threshold:", self.min_relevance_spin)

        self.max_results_spin = QSpinBox()

        self.max_results_spin.setRange(5, 50)

        accuracy_form.addRow("Max Results Pool:", self.max_results_spin)

        self.hybrid_weight_spin = QSpinBox()

        self.hybrid_weight_spin.setRange(0, 100)

        self.hybrid_weight_label = QLabel("Embedding Weight:")

        accuracy_form.addRow(self.hybrid_weight_label, self.hybrid_weight_spin)

        self.relevance_from_answer_cb = QCheckBox("Relevance from answer (Rerank by AI output)")

        accuracy_form.addRow("", self.relevance_from_answer_cb)

        self.strict_relevance_cb = QCheckBox("Strict Filter (Reduces tangential cards)")

        accuracy_form.addRow("", self.strict_relevance_cb)

        accuracy_section.addLayout(accuracy_form)

        search_layout.addWidget(accuracy_section)



        # --- 5. RE-RANKING SECTION ---

        rerank_section = CollapsibleSection("Re-Ranking (Advanced Accuracy)", is_expanded=False)

        self.enable_rerank_cb = QCheckBox("Enable Cross-Encoder Re-Ranking")

        self.enable_rerank_cb.setEnabled(False)

        rerank_section.addWidget(self.enable_rerank_cb)

        rerank_btn_row = QHBoxLayout()

        check_rerank_btn = QPushButton("\U0001F504 Refresh Status")

        check_rerank_btn.clicked.connect(self._on_check_rerank_again)

        rerank_btn_row.addWidget(check_rerank_btn)

        install_deps_btn = QPushButton("\U0001F4E5 Install into Anki")

        install_deps_btn.clicked.connect(lambda: install_dependencies(python_exe=None))

        rerank_btn_row.addWidget(install_deps_btn)

        self.install_external_btn = QPushButton("\U0001F4E5 Install into External")

        self.install_external_btn.clicked.connect(self._on_install_into_external_python)

        rerank_btn_row.addWidget(self.install_external_btn)

        rerank_section.addLayout(rerank_btn_row)

        self.python_path_widget = QWidget()

        python_path_layout = QVBoxLayout(self.python_path_widget)

        path_row = QHBoxLayout()

        self.rerank_python_path_input = QLineEdit()

        path_row.addWidget(self.rerank_python_path_input)

        self.autodetect_python_btn = QPushButton("\U0001F50D Autodetect")

        self.autodetect_python_btn.clicked.connect(self._on_autodetect_python)

        path_row.addWidget(self.autodetect_python_btn)

        python_path_layout.addLayout(path_row)

        rerank_section.addWidget(self.python_path_widget)

        self.python_path_widget.setVisible(False)

        self.use_context_boost_cb = QCheckBox("Context-Aware Ranking")

        rerank_section.addWidget(self.use_context_boost_cb)

        search_layout.addWidget(rerank_section)



        # --- 6. TECHNICAL DIAGNOSTICS (LAST PLACE) ---

        tech_section = CollapsibleSection("Technical Diagnostics (Expert Only)", is_expanded=False)

        tech_form = QFormLayout()

        self.verbose_search_debug_cb = QCheckBox("Verbose Search Debug")

        tech_form.addRow("", self.verbose_search_debug_cb)

        self.extra_stop_words_input = QLineEdit()

        tech_form.addRow("Extra Stop-words:", self.extra_stop_words_input)

        self.context_chars_per_note_spin = QSpinBox()

        self.context_chars_per_note_spin.setRange(0, 5000)

        tech_form.addRow("Max chars/note:", self.context_chars_per_note_spin)

        tech_section.addLayout(tech_form)

        search_layout.addWidget(tech_section)







        # --- Standardized Cloud Provider Panels ---

        self.voyage_options = self._create_api_config_group(

            "voyage", "Paste Voyage API key...",

            "Voyage AI API key from voyageai.com", VOYAGE_EMBEDDING_MODELS

        )

        self.embedding_section.addWidget(self.voyage_options)



        self.openai_options = self._create_api_config_group(

            "openai", "Paste OpenAI API key...",

            "OpenAI API key for embeddings", None, key_suffix="_embedding"

        )

        self.embedding_section.addWidget(self.openai_options)



        self.cohere_options = self._create_api_config_group(

            "cohere", "Paste Cohere API key...",

            "Cohere API key for embeddings", None

        )

        self.embedding_section.addWidget(self.cohere_options)







        self.cloud_batch_widget = QWidget()



        cloud_batch_layout = QFormLayout(self.cloud_batch_widget)



        self.voyage_batch_size_spin = QSpinBox()



        self.voyage_batch_size_spin.setRange(8, 256)



        self.voyage_batch_size_spin.setValue(64)



        self.voyage_batch_size_spin.setSuffix(" notes/batch")



        self.voyage_batch_size_spin.setToolTip("Batch size for cloud APIs (Voyage, OpenAI, Cohere). With dynamic batch size on, this adapts from response time.")



        cloud_batch_layout.addRow("Batch size:", self.voyage_batch_size_spin)



        self.embedding_section.addWidget(self.cloud_batch_widget)







        self.ollama_options = QWidget()

        ollama_form = QFormLayout(self.ollama_options)



        # Move "Zombie" Local Widgets here

        self.local_ai_status_label = QLabel("Scanning for local AI...")

        self.local_ai_status_label.setStyleSheet("font-weight: bold; color: #f39c12;")

        ollama_form.addRow("Detected Provider:", self.local_ai_status_label)



        self.scan_local_btn = QPushButton("Scan & Pull Models")

        self.scan_local_btn.clicked.connect(self._on_scan_and_pull)

        ollama_form.addRow("", self.scan_local_btn)



        self.ollama_base_url_input = QLineEdit()



        self.ollama_base_url_input.setMinimumWidth(380)



        self.ollama_base_url_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)



        self.ollama_base_url_input.setPlaceholderText("http://localhost:11434")



        self.ollama_base_url_input.setToolTip("Ollama server URL. Default: http://localhost:11434")



        ollama_form.addRow("Ollama URL:", self.ollama_base_url_input)



        # Model: editable combo so user can pick from detected list or type a custom model



        model_row = QHBoxLayout()



        self.ollama_embed_model_combo = QComboBox()



        self.ollama_embed_model_combo.setEditable(True)



        self.ollama_embed_model_combo.setMinimumWidth(280)



        self.ollama_embed_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)



        self.ollama_embed_model_combo.setToolTip("Embedding model. Click 'Refresh models' to load from Ollama, or type a model name (e.g. nomic-embed-text).")



        model_row.addWidget(self.ollama_embed_model_combo)



        self.ollama_refresh_models_btn = QPushButton("\U0001F504 Refresh models")



        self.ollama_refresh_models_btn.setToolTip("Fetch available models from Ollama (requires Ollama to be running)")



        self.ollama_refresh_models_btn.clicked.connect(self._refresh_ollama_models)



        model_row.addWidget(self.ollama_refresh_models_btn)



        ollama_form.addRow("Embed model:", model_row)



        self.ollama_batch_size_spin = QSpinBox()



        self.ollama_batch_size_spin.setRange(8, 256)



        self.ollama_batch_size_spin.setValue(64)



        self.ollama_batch_size_spin.setSuffix(" notes/batch")



        self.ollama_batch_size_spin.setToolTip("Starting batch size. With 'Use dynamic batch size' on, this adapts automatically from response time and notes/sec for best speed.")



        ollama_form.addRow("Batch size:", self.ollama_batch_size_spin)



        self.embedding_section.addWidget(self.ollama_options)







        self.use_dynamic_batch_size_cb = QCheckBox("Use dynamic batch size (adapt to response time for best speed)")



        self.use_dynamic_batch_size_cb.setChecked(True)



        self.use_dynamic_batch_size_cb.setToolTip("When enabled, batch size adapts both ways from response time: decrease if a batch is slow (>15s), increase if fast (<6s), to balance total time and responsiveness.")



        self.embedding_section.addWidget(self.use_dynamic_batch_size_cb)







        # Buttons for embedding operations

        embedding_btn_layout = QHBoxLayout()



        legacy_json_path = get_embeddings_storage_path_for_read()

        has_legacy_json = bool(

            legacy_json_path

            and isinstance(legacy_json_path, str)

            and legacy_json_path.endswith(".json")

            and os.path.exists(legacy_json_path)

        )



        if has_legacy_json:

            migrate_json_btn = QPushButton("\U0001F4E6 Legacy migration: JSON \u2192 DB")

            migrate_json_btn.setToolTip(

                "One-time legacy migration for users upgrading from older versions that stored embeddings in a JSON "

                "file. Copies existing embeddings from the old JSON cache into the SQLite database so you don't need "

                "to re-embed. Most users can ignore this."

            )

            migrate_json_btn.clicked.connect(self._migrate_json_to_db)

            embedding_btn_layout.addWidget(migrate_json_btn)



        search_layout.addWidget(self.embedding_section)



        self._tabs = tabs







        tabs.addTab(search_tab, "\U0001F50D Search & Embeddings")



        # Tab order: API \u2192 Search & Embeddings \u2192 Note Types & Fields \u2192 Styling



        tabs.removeTab(tabs.indexOf(search_tab))



        tabs.insertTab(1, search_tab, "\U0001F50D Search & Embeddings")



        tabs.removeTab(tabs.indexOf(nt_tab))



        tabs.insertTab(2, nt_tab, "\U0001F4CB Note Types & Fields")







        if self.open_to_embeddings:

            tabs.setCurrentWidget(search_tab)

            # Scroll to the embedding section (top of the search tab)

            QTimer.singleShot(200, lambda: self.embedding_section.toggle_button.setFocus())







        tabs_elapsed = time.time() - start_time



        log_debug(f"  [Timing] All tabs created: {tabs_elapsed:.3f}s")







        # Initialize embedding status (lazy load to avoid blocking)



        QTimer.singleShot(100, self._refresh_embedding_status)  # Slight delay to avoid race conditions







        # Scroll area: wrap tabs; compact min height to avoid excessive empty space



        scroll_content = QWidget()



        scroll_content.setMinimumHeight(380)



        scroll_content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)



        scroll_content_layout = QVBoxLayout(scroll_content)



        scroll_content_layout.setContentsMargins(0, 0, 0, 0)



        scroll_content_layout.addWidget(tabs)







        scroll_area = QScrollArea()



        scroll_area.setWidgetResizable(True)



        scroll_area.setWidget(scroll_content)



        scroll_area.setFrameShape(QFrame.Shape.NoFrame)



        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)



        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)



        scroll_area.setMinimumHeight(450)



        scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)



        scroll_area.setFocusPolicy(Qt.FocusPolicy.WheelFocus)



        self._settings_scroll_area = scroll_area



        self._settings_scroll_content = scroll_content



        self._settings_tabs = tabs



        scroll_content.installEventFilter(self)



        tabs.installEventFilter(self)



        scroll_area.viewport().installEventFilter(self)



        main_layout.addWidget(scroll_area, 1)







        # Final timing



        total_elapsed = time.time() - start_time



        log_debug(f"=== Settings Dialog UI Setup Completed: {total_elapsed:.3f}s total ===")







        # Buttons at bottom (always visible, not inside scroll)



        btn_layout = QHBoxLayout()



        btn_layout.addStretch()  # Push buttons to the right



        save_btn = QPushButton("Save Settings")

        save_btn.setObjectName("saveBtn")

        save_btn.clicked.connect(self.save_settings)



        cancel_btn = QPushButton("Cancel")

        cancel_btn.setObjectName("cancelBtn")



        cancel_btn.clicked.connect(self.close)







        btn_layout.addWidget(save_btn)



        btn_layout.addWidget(cancel_btn)



        main_layout.addLayout(btn_layout)







        # Load existing config



        config = load_config()



        if 'api_key' in config:



            self.api_key_input.setText(config['api_key'])



        # Answer provider: API key vs Ollama vs Local OpenAI

        provider = config.get('provider', 'ollama')

        idx = self.answer_provider_combo.findData(provider)

        if idx >= 0:

            self.answer_provider_combo.setCurrentIndex(idx)



        # Load Local LLM settings

        self.local_llm_url.setText(config.get('local_llm_url', 'http://localhost:1234/v1'))

        self.local_llm_model.setText(config.get('local_llm_model', 'model-identifier'))



        if provider == 'ollama':



            sc = config.get('search_config') or {}



            self.ollama_chat_model_combo.setCurrentText((sc.get('ollama_chat_model') or 'llama3.2').strip())



        self._on_answer_provider_changed()







        if 'api_url' in config:



            self.api_url_input.setText(config['api_url'])







        # Apply config to UI (this might involve slow operations)



        apply_start = time.time()



        if 'styling' in config:



            styling = config['styling']



            self.question_font_spin.setValue(styling.get('question_font_size', 13))



            self.answer_font_spin.setValue(styling.get('answer_font_size', 13))



            self.notes_font_spin.setValue(styling.get('notes_font_size', 12))



            self.label_font_spin.setValue(styling.get('label_font_size', 14))



            self.width_spin.setValue(styling.get('window_width', 1100))



            self.height_spin.setValue(styling.get('window_height', 800))



            self.section_spacing_spin.setValue(styling.get('section_spacing', 12))



            mode = styling.get('layout_mode', 'side_by_side')



            idx = self.layout_combo.findData(mode)



            if idx >= 0:



                self.layout_combo.setCurrentIndex(idx)



            spacing_mode = styling.get('answer_spacing', 'normal')



            idx = self.answer_spacing_combo.findData(spacing_mode)



            if idx >= 0:



                self.answer_spacing_combo.setCurrentIndex(idx)



        ntf = config.get('note_type_filter', {})



        ntf_start = time.time()



        self._apply_note_type_config(ntf)



        ntf_elapsed = time.time() - ntf_start



        log_debug(f"  [Timing] _apply_note_type_config() in __init__: {ntf_elapsed:.3f}s")







        # --- Load Configuration (Guarded) ---

        self._apply_config_to_ui()



    def _safe_set_checked(self, widget, value):

        """Radical safety: Check if C++ object exists before calling methods."""

        try:

            try:
                    try:
                        import sip
                    except ImportError:
                        from PyQt6 import sip

            except ImportError:
                    from PyQt6 import sip

            if widget is not None and not sip.isdeleted(widget):

                widget.setChecked(bool(value))

        except (RuntimeError, ImportError, AttributeError):

            pass



    def _load_config_into_ui(self):

        """Atomic configuration loading with strict safety checks."""

        config = load_config()

        search_config = config.get('search_config', {})



        # Guarded setters

        self._safe_set_checked(getattr(self, "enable_query_expansion_cb", None), search_config.get('enable_query_expansion', False))

        self._safe_set_checked(getattr(self, "use_ai_generic_term_detection_cb", None), search_config.get('use_ai_generic_term_detection', False))

        self._safe_set_checked(getattr(self, "enable_hyde_cb", None), search_config.get('enable_hyde', False))

        self._safe_set_checked(getattr(self, "enable_rerank_cb", None), search_config.get('enable_rerank', False))

        self._safe_set_checked(getattr(self, "use_context_boost_cb", None), search_config.get('use_context_boost', True))

        self._safe_set_checked(getattr(self, "strict_relevance_cb", None), search_config.get('strict_relevance', True))

        self._safe_set_checked(getattr(self, "relevance_from_answer_cb", None), search_config.get('relevance_from_answer', False))

        self._safe_set_checked(getattr(self, "verbose_search_debug_cb", None), search_config.get('verbose_search_debug', False))

        self._safe_set_checked(getattr(self, "use_dynamic_batch_size_cb", None), search_config.get('use_dynamic_batch_size', True))



        # Value setters (with try/except guards)

        try:

            if hasattr(self, "min_relevance_spin"):

                self.min_relevance_spin.setValue(max(15, min(75, search_config.get('min_relevance_percent', 55))))

            if hasattr(self, "max_results_spin"):

                self.max_results_spin.setValue(max(5, min(50, search_config.get('max_results', 50))))

            if hasattr(self, "hybrid_weight_spin"):

                self.hybrid_weight_spin.setValue(max(0, min(100, search_config.get('hybrid_embedding_weight', 40))))

            if hasattr(self, "rerank_python_path_input"):

                 path = (search_config.get('rerank_python_path') or '').strip()

                 self.rerank_python_path_input.setText(path)

                 if path and hasattr(self, "python_path_widget"):

                     self.python_path_widget.setVisible(True)

        except Exception:

            pass



        # ... rest of the existing loading logic if needed ...



        # Embedding engine: Voyage, OpenAI, Cohere, or Ollama (load keys, models, batch size)



        engine = search_config.get('embedding_engine') or 'voyage'



        self.voyage_api_key_input.setText((search_config.get('voyage_api_key') or '').strip())



        voyage_model = (search_config.get('voyage_embedding_model') or 'voyage-3.5-lite').strip()



        idx_v = self.voyage_embedding_model_combo.findData(voyage_model)



        if idx_v >= 0:



            self.voyage_embedding_model_combo.setCurrentIndex(idx_v)



        self.openai_embedding_api_key_input.setText((search_config.get('openai_embedding_api_key') or '').strip())



        self.openai_embedding_model_input.setText((search_config.get('openai_embedding_model') or 'text-embedding-3-small').strip())



        self.cohere_api_key_input.setText((search_config.get('cohere_api_key') or '').strip())



        self.cohere_embedding_model_input.setText((search_config.get('cohere_embedding_model') or 'embed-english-v3.0').strip())



        try:



            vb = int(search_config.get('voyage_batch_size', 64))



            self.voyage_batch_size_spin.setValue(max(8, min(256, vb)))



        except (TypeError, ValueError):



            self.voyage_batch_size_spin.setValue(64)



        idx = self.embedding_engine_combo.findData(engine)



        if idx >= 0:



            self.embedding_engine_combo.setCurrentIndex(idx)



        self.ollama_base_url_input.setText((search_config.get('ollama_base_url') or "http://localhost:11434").strip())



        self.ollama_embed_model_combo.setCurrentText((search_config.get('ollama_embed_model') or "nomic-embed-text").strip())



        try:



            ob = int(search_config.get('ollama_batch_size', 64))



            self.ollama_batch_size_spin.setValue(max(8, min(256, ob)))



        except (TypeError, ValueError):



            self.ollama_batch_size_spin.setValue(64)



        self.use_dynamic_batch_size_cb.setChecked(bool(search_config.get('use_dynamic_batch_size', True)))



        self._on_embedding_engine_changed()







    def _on_apply_hybrid_retrieval(self):



        """One-click RAG-optimized: AI embedding for retrieval, Ollama for answer; Hybrid + Re-rank for best quality with minimal cloud tokens."""



        idx_emb = self.embedding_engine_combo.findData("voyage")



        if idx_emb >= 0:



            self.embedding_engine_combo.setCurrentIndex(idx_emb)



        idx_ans = self.answer_provider_combo.findData("ollama")



        if idx_ans >= 0:



            self.answer_provider_combo.setCurrentIndex(idx_ans)



        idx_hybrid = self.search_method_combo.findData("hybrid")



        if idx_hybrid >= 0:



            self.search_method_combo.setCurrentIndex(idx_hybrid)



        if hasattr(self, 'enable_rerank_cb') and self.enable_rerank_cb is not None:



            self.enable_rerank_cb.setChecked(True)



        self._on_embedding_engine_changed()



        self._on_answer_provider_changed()



        if hasattr(self, '_on_search_method_changed'):



            self._on_search_method_changed()



        showInfo("RAG-optimized applied: Embeddings = Voyage, Answer = Ollama, Hybrid, Re-rank on. Click Save to apply and run Create/Update Embeddings if needed.")







    def _on_scan_and_pull(self):

        """Autodetect running local AI servers and update UI."""

        from aqt.utils import tooltip

        import requests



        # 1. Probe Ollama (11434)

        ollama_url = "http://localhost:11434"

        try:

            resp = requests.get(f"{ollama_url}/api/tags", timeout=2)

            if resp.status_code == 200:

                models = [m['name'] for m in resp.json().get('models', [])]

                self.local_ai_status_label.setText("Detected: Ollama (Running)")

                self.local_ai_status_label.setStyleSheet("font-weight: bold; color: #27ae60;")

                self.ollama_embed_model_combo.clear()

                self.ollama_embed_model_combo.addItems(models)

                self.ollama_base_url_input.setText(ollama_url)

                tooltip("Found Ollama! Models updated.")

                return

        except: pass



        # 2. Probe LM Studio / OpenAI-Compatible (1234)

        lm_url = "http://localhost:1234"

        try:

            resp = requests.get(f"{lm_url}/v1/models", timeout=2)

            if resp.status_code == 200:

                models = [m['id'] for m in resp.json().get('data', [])]

                self.local_ai_status_label.setText("Detected: LM Studio / Local Server")

                self.local_ai_status_label.setStyleSheet("font-weight: bold; color: #27ae60;")

                self.ollama_embed_model_combo.clear()

                self.ollama_embed_model_combo.addItems(models)

                self.ollama_base_url_input.setText(lm_url)

                tooltip("Found LM Studio! Models updated.")

                return

        except: pass



        self.local_ai_status_label.setText("No local AI found. Defaulting to Cloud or Manual.")

        self.local_ai_status_label.setStyleSheet("font-weight: bold; color: #e74c3c;")

        tooltip("No local AI detected. Ensure Ollama or LM Studio is running.")



    def _on_embedding_engine_changed(self):

        """Binary UI: Show only the relevant settings for Cloud or Local AI."""

        strategy = self.embedding_engine_combo.currentData() or "cloud"

        is_cloud = (strategy == "cloud")

        is_local = (strategy == "local")



        if hasattr(self, "cloud_provider_widget"):

            self.cloud_provider_widget.setVisible(is_cloud)



        if hasattr(self, "cloud_extras_widget"):

            self.cloud_extras_widget.setVisible(is_cloud)



        cloud_engine = self.cloud_provider_combo.currentData() or "voyage"

        self.voyage_options.setVisible(is_cloud and cloud_engine == "voyage")

        self.openai_options.setVisible(is_cloud and cloud_engine == "openai")

        self.cohere_options.setVisible(is_cloud and cloud_engine == "cohere")

        self.cloud_batch_widget.setVisible(is_cloud)



        if hasattr(self, "ollama_options"):

            self.ollama_options.setVisible(is_local)



        # Update status immediately based on selection

        self._refresh_embedding_status()



    def _create_api_config_group(self, provider_id, placeholder, tooltip_key, models=None, key_suffix=""):

        """Standardized UI group for API providers. key_suffix handles cases like 'openai_embedding_api_key'."""

        widget = QWidget()

        form = QFormLayout(widget)

        form.setContentsMargins(0, 5, 0, 5)



        # 1. API Key Row

        key_row = QHBoxLayout()

        key_input = QLineEdit()

        key_input.setEchoMode(QLineEdit.EchoMode.Password)

        key_input.setPlaceholderText(placeholder)

        key_input.setMinimumWidth(280)

        key_input.setToolTip(tooltip_key)



        show_btn = QPushButton("Show")

        show_btn.setMaximumWidth(60)

        show_btn.clicked.connect(lambda: key_input.setEchoMode(

            QLineEdit.EchoMode.Normal if key_input.echoMode() == QLineEdit.EchoMode.Password

            else QLineEdit.EchoMode.Password

        ))



        key_row.addWidget(key_input)

        key_row.addWidget(show_btn)

        form.addRow(f"{provider_id.capitalize()} API Key:", key_row)



        # 2. Model Input/Combo

        model_input = None

        if models:

            model_input = QComboBox()

            for m in models:

                model_input.addItem(m, m)

        else:

            model_input = QLineEdit()

            model_input.setPlaceholderText("e.g. text-embedding-3-small")



        form.addRow("Model:", model_input)



        # Mapping to existing names to prevent "Data Loss"

        key_attr = f"{provider_id}{key_suffix}_api_key_input"

        model_attr = f"{provider_id}_embedding_model_{'combo' if models else 'input'}"



        setattr(self, key_attr, key_input)

        setattr(self, model_attr, model_input)



        return widget

        """Binary UI: Show only the relevant settings for Cloud or Local AI."""

        strategy = self.embedding_engine_combo.currentData() or "cloud"



        is_cloud = (strategy == "cloud")

        is_local = (strategy == "local")



        # 1. Cloud Options (Provider selector + API keys)

        if hasattr(self, "cloud_provider_widget"):

            self.cloud_provider_widget.setVisible(is_cloud)



        cloud_engine = self.cloud_provider_combo.currentData() or "voyage"



        self.voyage_options.setVisible(is_cloud and cloud_engine == "voyage")

        self.openai_options.setVisible(is_cloud and cloud_engine == "openai")

        self.cohere_options.setVisible(is_cloud and cloud_engine == "cohere")

        self.cloud_batch_widget.setVisible(is_cloud)



        # 2. Local Options (Ollama/LM Studio)

        if hasattr(self, "ollama_options"):

            self.ollama_options.setVisible(is_local)



        # 3. Instruction Hint

        if hasattr(self, "embedding_hybrid_hint"):

            self.embedding_hybrid_hint.setVisible(is_cloud)

        if hasattr(self, "apply_hybrid_btn"):

            self.apply_hybrid_btn.setVisible(is_cloud)







    def eventFilter(self, obj, event):



        """Forward mouse wheel over tab content (and any child widget) to scroll area."""



        if event.type() != QEvent.Type.Wheel:



            return super().eventFilter(obj, event)



        scroll_area = getattr(self, "_settings_scroll_area", None)



        tabs = getattr(self, "_settings_tabs", None)



        scroll_content = getattr(self, "_settings_scroll_content", None)



        if not scroll_area or not tabs or scroll_content is None:



            return super().eventFilter(obj, event)



        # Allow scroll when wheel is over scroll content, tabs, viewport, or any descendant



        target = obj



        while target:



            if target == scroll_content or target == tabs or target == scroll_area.viewport():



                break



            target = target.parentWidget() if hasattr(target, "parentWidget") else None



        else:



            return super().eventFilter(obj, event)



        if scroll_area.verticalScrollBar().isVisible():



            delta = event.angleDelta().y() if hasattr(event, "angleDelta") else getattr(event, "delta", 0)



            sb = scroll_area.verticalScrollBar()



            sb.setValue(sb.value() - delta)



            return True



        return super().eventFilter(obj, event)







    def _refresh_ollama_models(self):



        """Fetch model list from Ollama and populate the embed model combo."""



        base_url = (self.ollama_base_url_input.text() or "http://localhost:11434").strip()



        current = self.ollama_embed_model_combo.currentText().strip() or "nomic-embed-text"



        try:



            names = get_ollama_models(base_url)



            self.ollama_embed_model_combo.clear()



            self.ollama_embed_model_combo.addItems(names)



            if current and current not in names:



                self.ollama_embed_model_combo.insertItem(0, current)



                self.ollama_embed_model_combo.setCurrentIndex(0)



            elif current in names:



                idx = self.ollama_embed_model_combo.findText(current)



                if idx >= 0:



                    self.ollama_embed_model_combo.setCurrentIndex(idx)



            if not names:



                self.ollama_embed_model_combo.setCurrentText(current or "nomic-embed-text")



            if names:



                showInfo(f"Found {len(names)} model(s) at {base_url}. Choose an embedding model (e.g. nomic-embed-text).")



            else:



                showInfo(



                    "No models returned from Ollama. Make sure Ollama is running (ollama serve) and you have pulled at least one model.\n\n"



                    "You can still type an embedding model name manually (e.g. nomic-embed-text)."



                )



        except Exception as e:



            showInfo(



                f"Could not fetch models from {base_url}.\n\n"



                f"Error: {e}\n\n"



                "Check that Ollama is running (ollama serve). You can type a model name manually (e.g. nomic-embed-text)."



            )







    def _populate_note_type_lists_with_timing(self):



        """Fill note types table with name and count columns (with timing)."""



        # Check if widget still exists (dialog might have been closed)



        try:



            if not hasattr(self, 'note_types_table') or self.note_types_table is None:



                return



            # Check if the C++ object is still valid



            if not sip.isdeleted(self.note_types_table) if hasattr(sip, 'isdeleted') else True:



                import time



                start = time.time()



                self._populate_note_type_lists()



                elapsed = time.time() - start



                log_debug(f"  [Timing] _populate_note_type_lists(): {elapsed:.3f}s")



                # After table is populated, apply saved note/deck/field config



                try:



                    cfg = load_config()



                    ntf = cfg.get('note_type_filter', {})



                    self._apply_note_type_config(ntf)



                except Exception as e:



                    log_debug(f"Error re-applying note_type_filter after note type populate: {e}")



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_note_type_lists_with_timing: {e}")







    def _populate_note_type_lists(self):



        """Fill note types table with name and count columns."""



        # Check if widget still exists



        if not hasattr(self, 'note_types_table') or self.note_types_table is None:



            return



        try:



            self.note_types_table.setRowCount(0)



            counts = get_notes_count_per_model()



            for name in sorted(counts.keys()):



                c = counts.get(name, 0)



                row = self.note_types_table.rowCount()



                self.note_types_table.insertRow(row)







                # Name column with checkbox



                name_item = QTableWidgetItem(name)



                name_item.setData(Qt.ItemDataRole.UserRole, name)



                name_item.setFlags(name_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)



                name_item.setCheckState(Qt.CheckState.Unchecked)



                self.note_types_table.setItem(row, 0, name_item)







                # Count column



                count_item = QTableWidgetItem()



                count_item.setData(Qt.ItemDataRole.DisplayRole, c)  # Store numeric value for proper sorting



                count_item.setText(str(c))



                count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)



                count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make read-only



                self.note_types_table.setItem(row, 1, count_item)







            # Sort by count descending by default



            self.note_types_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_note_type_lists: {e}")







    def _populate_fields_by_note_type(self):



        """Build collapsible field sections per note type and repopulate them."""



        # Check if widget still exists



        if not hasattr(self, 'fields_by_note_type_layout') or self.fields_by_note_type_layout is None:



            return



        try:



            while self.fields_by_note_type_layout.count() > 0:



                it = self.fields_by_note_type_layout.takeAt(0)



                if it and it.widget():



                    it.widget().deleteLater()



            self._field_cbs.clear()



            self._field_groupboxes.clear()



            included = None



            if not self.include_all_note_types_cb.isChecked():



                included = set()



                for i in range(self.note_types_table.rowCount()):



                    it = self.note_types_table.item(i, 0)



                    if it and it.checkState() == Qt.CheckState.Checked:



                        included_name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()



                        included.add(included_name)



            for model_name, count, field_names in get_models_with_fields():



                is_included = included is None or model_name in included



                gb = CollapsibleSection(f"{model_name}  ({count} notes)", is_expanded=is_included)



                cbs = {}



                for fn in field_names:



                    cb = QCheckBox(fn)



                    cbs[fn] = cb



                    gb.addWidget(cb)



                self._field_cbs[model_name] = cbs



                self._field_groupboxes[model_name] = gb



                self.fields_by_note_type_layout.addWidget(gb)



            self._update_field_groups_enabled()



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_fields_by_note_type: {e}")







    def _populate_fields_by_note_type_with_timing(self):



        """Populate fields by note type (with timing)."""



        import time



        start = time.time()



        self._populate_fields_by_note_type()



        elapsed = time.time() - start



        if elapsed > 0.1:  # Only log if it takes significant time



            log_debug(f"  [Timing] _populate_fields_by_note_type(): {elapsed:.3f}s")



        # Ensure field checkboxes match saved configuration



        try:



            cfg = load_config()



            ntf = cfg.get('note_type_filter', {})



            self._apply_note_type_config(ntf)



        except Exception as e:



            log_debug(f"Error re-applying note_type_filter after field populate: {e}")







    def _populate_decks_list_with_timing(self):



        """Populate decks list (with timing)."""



        # Check if widget still exists (dialog might have been closed)



        try:



            if not hasattr(self, 'decks_list') or self.decks_list is None:



                return



            import time



            start = time.time()



            self._populate_decks_list()



            elapsed = time.time() - start



            log_debug(f"  [Timing] _populate_decks_list(): {elapsed:.3f}s")



            # Ensure deck checkboxes match saved configuration



            try:



                cfg = load_config()



                ntf = cfg.get('note_type_filter', {})



                self._apply_note_type_config(ntf)



            except Exception as e:



                log_debug(f"Error re-applying note_type_filter after deck populate: {e}")



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_decks_list_with_timing: {e}")







    def _populate_decks_list(self):



        import time



        deck_start = time.time()







        # Check if widget still exists



        if not hasattr(self, 'decks_list') or self.decks_list is None:



            return



        try:



            self.decks_list.clear()







            if not mw or not mw.col:



                return







            counts_start = time.time()



            counts = get_notes_count_per_deck()



            counts_elapsed = time.time() - counts_start



            log_debug(f"  [Timing] get_notes_count_per_deck(): {counts_elapsed:.3f}s")







            # Get deck hierarchy and card counts



            deck_names_start = time.time()



            deck_names = get_deck_names()



            deck_names_elapsed = time.time() - deck_names_start



            log_debug(f"  [Timing] get_deck_names(): {deck_names_elapsed:.3f}s")







            # Build hierarchical deck structure



            deck_tree = {}  # parent_name -> [child_decks]



            top_level_decks = []







            for name in deck_names:



                if '::' in name:



                    # Sub-deck



                    parts = name.split('::')



                    parent = '::'.join(parts[:-1])



                    if parent not in deck_tree:



                        deck_tree[parent] = []



                    deck_tree[parent].append(name)



                else:



                    # Top-level deck



                    top_level_decks.append(name)







            # Get card counts for each deck (new, learn, due)

            card_counts = {}

            try:

                # In modern Anki, we don't need to import Scheduler; mw.col.sched is ready.

                # Use the safer get_deck_stats if available, or fallback.

                deck_ids = {name: mw.col.decks.id(name) for name in deck_names if mw.col.decks.by_name(name)}



                for name, deck_id in deck_ids.items():

                    try:

                        # Modern scheduler fallback

                        if hasattr(mw.col.sched, "counts"):

                            counts_info = mw.col.sched.counts(deck_id)

                            card_counts[name] = {

                                'new': getattr(counts_info, 'new', 0),

                                'learn': getattr(counts_info, 'learn', 0),

                                'due': getattr(counts_info, 'review', 0)

                            }

                        else:

                            # If counts() is missing, just use 0s for now

                            card_counts[name] = {'new': 0, 'learn': 0, 'due': 0}

                    except Exception:

                        card_counts[name] = {'new': 0, 'learn': 0, 'due': 0}



            except Exception as e:

                log_debug(f"Error getting card counts: {e}")

                # Fallback: set all to 0

                for name in deck_names:

                    card_counts[name] = {'new': 0, 'learn': 0, 'due': 0}







            # Sort decks



            top_level_decks.sort()



            for parent in deck_tree:



                deck_tree[parent].sort()







            # Create tree items



            def create_deck_item(name, is_parent=False):



                """Create a tree item for a deck (showing only name + total notes)."""



                note_count = counts.get(name, 0)



                # Hide the built-in empty 'Default' deck, which many users don't use



                if name == "Default" and note_count == 0:



                    return None







                # Extract display name (without parent prefix for sub-decks)



                display_name = name.split('::')[-1] if '::' in name else name







                item = QTreeWidgetItem([display_name, str(note_count)])







                # Store full deck name in item data for later retrieval



                item.setData(0, Qt.ItemDataRole.UserRole, name)







                # Make checkable



                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)



                item.setCheckState(0, Qt.CheckState.Unchecked)







                # Style parent decks (bold)



                if is_parent:



                    font = item.font(0)



                    font.setBold(True)



                    item.setFont(0, font)







                return item







            # Add top-level decks



            for deck_name in top_level_decks:



                parent_item = create_deck_item(deck_name, is_parent=(deck_name in deck_tree))



                if parent_item is None:



                    continue



                self.decks_list.addTopLevelItem(parent_item)







                # Add children if any



                if deck_name in deck_tree:



                    for child_name in deck_tree[deck_name]:



                        child_item = create_deck_item(child_name)



                        parent_item.addChild(child_item)







                    # Collapsed by default; user expands on click



                    parent_item.setExpanded(False)







            # Also add decks that are parents but not top-level (nested hierarchies)



            for parent_name in sorted(deck_tree.keys()):



                if parent_name not in top_level_decks:



                    # This is a nested parent, find its position in hierarchy



                    parts = parent_name.split('::')



                    if len(parts) > 1:



                        # Find parent item



                        grandparent_name = '::'.join(parts[:-1])



                        # Search for grandparent in tree



                        for i in range(self.decks_list.topLevelItemCount()):



                            parent_item = self._find_deck_item_recursive(self.decks_list.topLevelItem(i), grandparent_name)



                            if parent_item:



                                child_item = create_deck_item(parent_name, is_parent=True)



                                if child_item is None:



                                    break



                                parent_item.addChild(child_item)



                                # Add its children



                                for child_name in deck_tree[parent_name]:



                                    grandchild_item = create_deck_item(child_name)



                                    child_item.addChild(grandchild_item)



                                child_item.setExpanded(False)



                                break







            deck_elapsed = time.time() - deck_start



            log_debug(f"  [Timing] _populate_decks_list(): {deck_elapsed:.3f}s")







        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _populate_decks_list: {e}")



            import traceback



            log_debug(traceback.format_exc())







    def _find_deck_item_recursive(self, item, deck_name):



        """Recursively find a deck item by its full name"""



        if item is None:



            return None



        if item.data(0, Qt.ItemDataRole.UserRole) == deck_name:



            return item



        for i in range(item.childCount()):



            found = self._find_deck_item_recursive(item.child(i), deck_name)



            if found:



                return found



        return None







    def _iterate_all_deck_items(self):



        """Generator that yields all deck items (top-level and children)"""



        for i in range(self.decks_list.topLevelItemCount()):



            item = self.decks_list.topLevelItem(i)



            yield item



            # Recursively yield children



            for j in range(item.childCount()):



                yield from self._iterate_all_deck_items_recursive(item.child(j))







    def _iterate_all_deck_items_recursive(self, item):



        """Recursively yield item and all its children"""



        if item:



            yield item



            for i in range(item.childCount()):



                yield from self._iterate_all_deck_items_recursive(item.child(i))







    def _refresh_preset_combos_with_timing(self):



        """Refresh preset combos (with timing)."""



        # Check if widget still exists (dialog might have been closed)



        try:



            if not hasattr(self, 'load_preset_combo') or self.load_preset_combo is None:



                return



            import time



            start = time.time()



            self._refresh_preset_combos()



            elapsed = time.time() - start



            log_debug(f"  [Timing] _refresh_preset_combos(): {elapsed:.3f}s")



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _refresh_preset_combos_with_timing: {e}")







    def _refresh_preset_combos(self):



        # Check if widgets still exist



        if not hasattr(self, 'load_preset_combo') or self.load_preset_combo is None:



            return



        try:



            config = load_config()



            presets = config.get('saved_presets') or {}



            current_name = config.get('current_preset_name')



            names = sorted(presets.keys())



            self.load_preset_combo.clear()



            self.load_preset_combo.addItem("-- Select --", None)



            for n in names:



                self.load_preset_combo.addItem(n, n)



            selected_index = self.load_preset_combo.findData(current_name)



            if selected_index >= 0:



                self.load_preset_combo.setCurrentIndex(selected_index)



            if hasattr(self, 'delete_preset_combo') and self.delete_preset_combo is not None:



                try:



                    self.delete_preset_combo.clear()



                    self.delete_preset_combo.addItem("-- Select --", None)



                    for n in names:



                        self.delete_preset_combo.addItem(n, n)



                    delete_index = self.delete_preset_combo.findData(current_name)



                    if delete_index >= 0:



                        self.delete_preset_combo.setCurrentIndex(delete_index)



                except RuntimeError:



                    # Widget was deleted, ignore



                    pass



        except RuntimeError:



            # Widget was deleted, ignore



            pass



        except Exception as e:



            log_debug(f"Error in _refresh_preset_combos: {e}")







    def _apply_note_type_config(self, ntf):



        """Apply note_type_filter config. Migrate fields_to_search -> note_type_fields if needed."""



        self._applying_note_type_config = True



        try:



            self._apply_note_type_config_impl(ntf)



        finally:



            self._applying_note_type_config = False







    def _apply_note_type_config_impl(self, ntf):



        # Migrate: if fields_to_search exists but not note_type_fields, build note_type_fields



        ntf = dict(ntf)



        if ntf.get('fields_to_search') and not ntf.get('note_type_fields'):



            global_flds = set(f.lower() for f in ntf['fields_to_search'])



            ntf['note_type_fields'] = {}



            for model_name, _c, field_names in get_models_with_fields():



                ntf['note_type_fields'][model_name] = [f for f in field_names if f.lower() in global_flds]



        # Note types



        enabled = ntf.get('enabled_note_types')



        # Interpretation:



        #   None        -> include all note types



        #   [] (empty)  -> user has not chosen any specific types yet



        #                  (start with none selected to reduce workload)



        include_all_nt = (enabled is None)



        self.include_all_note_types_cb.setChecked(include_all_nt)



        self._on_include_all_note_types_toggled()



        if not include_all_nt and enabled:



            enabled_set = set(enabled)



            for i in range(self.note_types_table.rowCount()):



                it = self.note_types_table.item(i, 0)



                if it:



                    name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()



                    it.setCheckState(Qt.CheckState.Checked if (name in enabled_set) else Qt.CheckState.Unchecked)



        else:



            self._set_note_types_checked(True)



        # Search all / use first field



        self.search_all_fields_cb.setChecked(bool(ntf.get('search_all_fields', False)))



        self._on_search_all_fields_toggled()



        self.use_first_field_cb.setChecked(bool(ntf.get('use_first_field_fallback', True)))



        # Fields by note type (default to Text+Extra when neither note_type_fields nor fields_to_search)



        ntf_fields = ntf.get('note_type_fields') or {}



        default_flds = None



        if not ntf_fields and not ntf.get('fields_to_search'):



            default_flds = {'text', 'extra'}



        for model_name, cbs in self._field_cbs.items():



            wanted = set(f.lower() for f in (ntf_fields.get(model_name) or []))



            if not wanted and default_flds:



                wanted = default_flds



            for fn, cb in cbs.items():



                cb.setChecked(fn.lower() in wanted)



        # Decks (block signals so programmatic setCheckState doesn't trigger persist)



        deck_list = ntf.get('enabled_decks')



        # Interpretation:



        #   None        -> include all decks



        #   [] (empty)  -> no decks selected (all unchecked)



        #   [names]     -> only these decks checked



        include_all_d = (deck_list is None)



        self.include_all_decks_cb.blockSignals(True)



        self.include_all_decks_cb.setChecked(include_all_d)



        self.include_all_decks_cb.blockSignals(False)



        self._on_include_all_decks_toggled()



        if hasattr(self, 'decks_list') and self.decks_list:



            self.decks_list.blockSignals(True)



        try:



            if include_all_d:



                self._set_decks_checked(True)



            elif deck_list:



                ds = set(deck_list)



                for it in self._iterate_all_deck_items():



                    deck_name = it.data(0, Qt.ItemDataRole.UserRole) or it.text(0)



                    it.setCheckState(0, Qt.CheckState.Checked if (deck_name in ds) else Qt.CheckState.Unchecked)



            else:



                # Empty list: user chose no decks (all unchecked)



                self._set_decks_checked(False)



        finally:



            if hasattr(self, 'decks_list') and self.decks_list:



                self.decks_list.blockSignals(False)



        self._update_field_groups_enabled()







    def _update_field_groups_enabled(self):



        """Grey out and collapse field sections whose note type is unchecked."""



        if not getattr(self, '_field_groupboxes', None):



            return



        include_all = self.include_all_note_types_cb.isChecked()



        if include_all:



            included = None



        else:



            included = set()



            for i in range(self.note_types_table.rowCount()):



                it = self.note_types_table.item(i, 0)



                if it and it.checkState() == Qt.CheckState.Checked:



                    name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()



                    included.add(name)



        for model_name, gb in self._field_groupboxes.items():



            is_included = included is None or model_name in included

            gb.setEnabled(is_included)

            if hasattr(gb, 'setExpanded'):

                gb.setExpanded(is_included)







    def _on_include_all_note_types_toggled(self):



        self.note_types_table.setEnabled(not self.include_all_note_types_cb.isChecked())



        self._update_field_groups_enabled()







    def _on_sort_note_types_changed(self, index):



        """Handle sort combo box change."""



        data = self.sort_combo.itemData(index)



        if data == "count_desc":



            self.note_types_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)



        elif data == "count_asc":



            self.note_types_table.sortByColumn(1, Qt.SortOrder.AscendingOrder)



        elif data == "name_asc":



            self.note_types_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)



        elif data == "name_desc":



            self.note_types_table.sortByColumn(0, Qt.SortOrder.DescendingOrder)







    def _on_search_all_fields_toggled(self):



        en = not self.search_all_fields_cb.isChecked()



        self.fields_by_note_type_scroll.setEnabled(en)



        for cbs in self._field_cbs.values():



            for cb in cbs.values():



                cb.setEnabled(en)



        self._update_field_groups_enabled()







    def _on_include_all_decks_toggled(self):



        self.decks_list.setEnabled(not self.include_all_decks_cb.isChecked())



        # Also disable/enable header if needed



        header = self.decks_list.header()



        if header:



            header.setEnabled(not self.include_all_decks_cb.isChecked())



        self._persist_note_type_filter()







    def _persist_note_type_filter(self):



        """Save current Note Types & Fields (decks, note types, fields) to config so changes persist without clicking Save."""



        if getattr(self, '_applying_note_type_config', False):



            return



        try:



            config = load_config()



            config['note_type_filter'] = self._build_ntf_from_ui()



            save_config(config)



        except Exception as e:



            log_debug(f"Error persisting note_type_filter: {e}")







    def _on_deck_item_changed(self, item, column):



        """When user toggles a deck checkbox, persist so settings are saved."""



        if column == 0 and item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:



            self._persist_note_type_filter()







    def _set_note_types_checked(self, checked):



        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked



        for i in range(self.note_types_table.rowCount()):



            it = self.note_types_table.item(i, 0)



            if it:



                it.setCheckState(state)







    def _set_decks_checked(self, checked):



        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked



        for item in self._iterate_all_deck_items():



            item.setCheckState(0, state)







    def _get_note_type_fields_from_ui(self):



        out = {}



        for model_name, cbs in self._field_cbs.items():



            sel = [fn for fn, cb in cbs.items() if cb.isChecked()]



            if sel:



                out[model_name] = sel



        return out







    def _get_decks_from_ui(self):



        if self.include_all_decks_cb.isChecked():



            return None



        # Get checked deck names from tree widget



        checked_decks = []



        for item in self._iterate_all_deck_items():



            if item.checkState(0) == Qt.CheckState.Checked:



                # Get full deck name from item data



                deck_name = item.data(0, Qt.ItemDataRole.UserRole)



                if deck_name:



                    checked_decks.append(deck_name)



        return checked_decks







    def _build_ntf_from_ui(self):



        include_all_nt = self.include_all_note_types_cb.isChecked()



        enabled_nt = None if include_all_nt else [self.note_types_table.item(i, 0).data(Qt.ItemDataRole.UserRole) or self.note_types_table.item(i, 0).text().strip() for i in range(self.note_types_table.rowCount()) if self.note_types_table.item(i, 0) and self.note_types_table.item(i, 0).checkState() == Qt.CheckState.Checked]



        # Preserve enabled_decks from config if deck list not yet populated (async load at 150ms)



        if hasattr(self, 'decks_list') and self.decks_list and self.decks_list.topLevelItemCount() > 0:



            enabled_decks = self._get_decks_from_ui()



        else:



            enabled_decks = load_config().get('note_type_filter', {}).get('enabled_decks')



        return {



            'enabled_note_types': enabled_nt,



            'search_all_fields': self.search_all_fields_cb.isChecked(),



            'note_type_fields': self._get_note_type_fields_from_ui() if not self.search_all_fields_cb.isChecked() else {},



            'use_first_field_fallback': self.use_first_field_cb.isChecked(),



            'enabled_decks': enabled_decks,



        }







    def _on_count_notes(self):



        ntf = self._build_ntf_from_ui()



        c = count_notes_matching_config(ntf)



        showInfo(f"With current settings, about {c} notes would be searched.")







    def _on_save_preset(self):



        name = self.preset_name_edit.text().strip()



        if not name:



            showInfo("Enter a preset name.")



            return



        config = load_config()



        presets = config.get('saved_presets') or {}



        presets[name] = self._build_ntf_from_ui()



        config['saved_presets'] = presets



        config['current_preset_name'] = name



        if save_config(config):



            self.preset_name_edit.clear()



            self._refresh_preset_combos()



            selected_index = self.load_preset_combo.findData(name)



            if selected_index >= 0:



                self.load_preset_combo.setCurrentIndex(selected_index)



            showInfo(f"Preset '{name}' saved.")







    def _on_load_preset(self):



        name = self.load_preset_combo.currentData()



        if not name:



            showInfo("Select a preset to load.")



            return



        config = load_config()



        presets = config.get('saved_presets') or {}



        if name not in presets:



            showInfo("Preset not found.")



            return



        self._apply_note_type_config(presets[name])



        config['current_preset_name'] = name



        save_config(config)



        selected_index = self.load_preset_combo.findData(name)



        if selected_index >= 0:



            self.load_preset_combo.setCurrentIndex(selected_index)



        if hasattr(self, 'delete_preset_combo') and self.delete_preset_combo is not None:



            delete_index = self.delete_preset_combo.findData(name)



            if delete_index >= 0:



                self.delete_preset_combo.setCurrentIndex(delete_index)



        showInfo(f"Loaded preset '{name}'.")







    def _on_delete_preset(self):



        name = self.delete_preset_combo.currentData()



        if not name:



            showInfo("Select a preset to delete.")



            return



        config = load_config()



        presets = config.get('saved_presets') or {}



        if name in presets:



            del presets[name]



            config['saved_presets'] = presets



            if config.get('current_preset_name') == name:



                config['current_preset_name'] = None



            save_config(config)



            self._refresh_preset_combos()



            showInfo(f"Preset '{name}' deleted.")







    def _refresh_note_type_lists(self):



        """Repopulate all lists and preserve checked state where possible."""



        checked_nt = [self.note_types_table.item(i, 0).data(Qt.ItemDataRole.UserRole) or self.note_types_table.item(i, 0).text().strip() for i in range(self.note_types_table.rowCount()) if self.note_types_table.item(i, 0) and self.note_types_table.item(i, 0).checkState() == Qt.CheckState.Checked]



        ntf_prev = self._get_note_type_fields_from_ui()



        checked_decks = self._get_decks_from_ui()



        self._populate_note_type_lists()



        for i in range(self.note_types_table.rowCount()):



            it = self.note_types_table.item(i, 0)



            if it:



                name = it.data(Qt.ItemDataRole.UserRole) or it.text().strip()



                if name in checked_nt:



                    it.setCheckState(Qt.CheckState.Checked)



        self._populate_fields_by_note_type()



        for model_name, cbs in self._field_cbs.items():



            wanted = set(f.lower() for f in (ntf_prev.get(model_name) or []))



            for fn, cb in cbs.items():



                if fn.lower() in wanted:



                    cb.setChecked(True)



        self._populate_decks_list()



        if checked_decks:



            ds = set(checked_decks)



            for it in self._iterate_all_deck_items():



                deck_name = it.data(0, Qt.ItemDataRole.UserRole) or it.text(0)



                it.setCheckState(0, Qt.CheckState.Checked if (deck_name in ds) else Qt.CheckState.Unchecked)



        else:



            self._set_decks_checked(True)



        self._refresh_preset_combos()



        showInfo("Lists refreshed.")







    def detect_provider(self):



        api_key = self.api_key_input.text().strip()



        if not api_key:



            self.provider_label.hide()



            self.url_widget.hide()



            return ""







        if api_key.startswith("sk-ant-"):



            provider = "Anthropic (Claude)"



            self.url_widget.hide()



        elif api_key.startswith("sk-or-"):



            provider = "OpenRouter"



            self.url_widget.hide()



        elif api_key.startswith("sk-"):



            provider = "OpenAI (GPT)"



            self.url_widget.hide()



        elif api_key.startswith("AI"):



            provider = "Google (Gemini)"



            self.url_widget.hide()



        else:



            provider = "Custom/Unknown Provider"



            self.url_widget.show()







        self.provider_label.setText(f"\u2713 Detected: {provider}")



        self.provider_label.show()



        return provider







    def toggle_key_visibility(self):



        if self.api_key_input.echoMode() == QLineEdit.EchoMode.Password:



            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)



            self.show_key_btn.setText("Hide")



        else:



            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)



            self.show_key_btn.setText("Show")







    def _toggle_voyage_key_visibility(self):



        if self.voyage_api_key_input.echoMode() == QLineEdit.EchoMode.Password:



            self.voyage_api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)



            self.voyage_show_key_btn.setText("Hide")



        else:



            self.voyage_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)



            self.voyage_show_key_btn.setText("Show")







    def _toggle_openai_key_visibility(self):



        if self.openai_embedding_api_key_input.echoMode() == QLineEdit.EchoMode.Password:



            self.openai_embedding_api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)



            self.openai_show_key_btn.setText("Hide")



        else:



            self.openai_embedding_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)



            self.openai_show_key_btn.setText("Show")







    def _toggle_cohere_key_visibility(self):



        if self.cohere_api_key_input.echoMode() == QLineEdit.EchoMode.Password:



            self.cohere_api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)



            self.cohere_show_key_btn.setText("Hide")



        else:



            self.cohere_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)



            self.cohere_show_key_btn.setText("Show")







    def _safe_get_ui_value(self, attr_name, default_value):

        """Radically safe UI reader: returns value only if widget is alive and healthy."""

        try:

            try:
                    try:
                        import sip
                    except ImportError:
                        from PyQt6 import sip

            except ImportError:
                    from PyQt6 import sip



            widget = getattr(self, attr_name, None)

            if widget is None or sip.isdeleted(widget):

                return default_value



            from PyQt6.QtWidgets import QCheckBox, QSpinBox, QSlider, QLineEdit, QTextEdit, QComboBox



            if isinstance(widget, QCheckBox):

                return widget.isChecked()

            if isinstance(widget, (QSpinBox, QSlider)):

                return widget.value()

            if isinstance(widget, (QLineEdit, QTextEdit)):

                return widget.text().strip()

            if isinstance(widget, QComboBox):

                data = widget.currentData()

                return data if data is not None else widget.currentText().strip()

            return default_value

        except Exception:

            return default_value



    def save_settings(self):

        """Saves settings with radical safety and consistent notification."""

        from aqt.utils import showInfo



        # Silence signals during save

        widgets = ['search_method_combo', 'answer_provider_combo']

        for name in widgets:

            w = getattr(self, name, None)

            if w: w.blockSignals(True)



        try:

            current_config = load_config()

            current_sc = current_config.get('search_config', {})

            current_style = current_config.get('styling', {})



            # 1. Answer Provider Logic

            answer_with = self._safe_get_ui_value('answer_provider_combo', current_config.get('provider', 'api_key'))



            if answer_with == "local_server":

                provider_type = "local_openai"

                if current_config.get("provider") in ["ollama", "local_openai"]:

                    provider_type = current_config.get("provider")

                api_key = ""

            else:

                api_key = self._safe_get_ui_value('api_key_input', current_config.get('api_key', ''))

                provider_type = self.detect_provider_type(api_key) if api_key else "openai"



            note_type_filter = self._build_ntf_from_ui()



            config = {

                'api_key': api_key,

                'provider': provider_type,

                'styling': {

                    'question_font_size': self._safe_get_ui_value('question_font_spin', current_style.get('question_font_size', 13)),

                    'answer_font_size': self._safe_get_ui_value('answer_font_spin', current_style.get('answer_font_size', 13)),

                    'notes_font_size': self._safe_get_ui_value('notes_font_spin', current_style.get('notes_font_size', 12)),

                    'label_font_size': self._safe_get_ui_value('label_font_spin', current_style.get('label_font_size', 14)),

                    'window_width': self._safe_get_ui_value('width_spin', current_style.get('window_width', 1100)),

                    'window_height': self._safe_get_ui_value('height_spin', current_style.get('window_height', 800)),

                    'section_spacing': self._safe_get_ui_value('section_spacing_spin', current_style.get('section_spacing', 12)),

                    'layout_mode': self._safe_get_ui_value('layout_combo', current_style.get('layout_mode', 'side_by_side')),

                    'answer_spacing': self._safe_get_ui_value('answer_spacing_combo', current_style.get('answer_spacing', 'normal'))

                },

                'note_type_filter': note_type_filter,

                'search_config': {

                    'local_llm_url': self._safe_get_ui_value('local_llm_url', current_sc.get('local_llm_url', 'http://localhost:11434')),

                    'local_llm_model': self._safe_get_ui_value('local_llm_model', current_sc.get('local_llm_model', 'llama3.2')),

                    'search_method': self._safe_get_ui_value('search_method_combo', current_sc.get('search_method', 'hybrid')),

                    'enable_query_expansion': self._safe_get_ui_value('enable_query_expansion_cb', current_sc.get('enable_query_expansion', False)),

                    'use_ai_generic_term_detection': self._safe_get_ui_value('use_ai_generic_term_detection_cb', current_sc.get('use_ai_generic_term_detection', False)),

                    'enable_hyde': self._safe_get_ui_value('enable_hyde_cb', current_sc.get('enable_hyde', False)),

                    'enable_rerank': self._safe_get_ui_value('enable_rerank_cb', current_sc.get('enable_rerank', False)),

                    'use_context_boost': self._safe_get_ui_value('use_context_boost_cb', current_sc.get('use_context_boost', True)),

                    'min_relevance_percent': self._safe_get_ui_value('min_relevance_spin', current_sc.get('min_relevance_percent', 55)),

                    'strict_relevance': self._safe_get_ui_value('strict_relevance_cb', current_sc.get('strict_relevance', True)),

                    'max_results': self._safe_get_ui_value('max_results_spin', current_sc.get('max_results', 50)),

                    'context_chars_per_note': self._safe_get_ui_value('context_chars_per_note_spin', current_sc.get('context_chars_per_note', 0)),

                    'relevance_from_answer': self._safe_get_ui_value('relevance_from_answer_cb', current_sc.get('relevance_from_answer', False)),

                    'hybrid_embedding_weight': self._safe_get_ui_value('hybrid_weight_spin', current_sc.get('hybrid_embedding_weight', 40)),

                    'extra_stop_words': [

                        w.strip().lower()

                        for w in (self._safe_get_ui_value('extra_stop_words_input', "")).split(",")

                        if w.strip()

                    ],

                    'verbose_search_debug': self._safe_get_ui_value('verbose_search_debug_cb', current_sc.get('verbose_search_debug', False)),

                    'embedding_engine': self._safe_get_ui_value('embedding_engine_combo', current_sc.get('embedding_engine', 'voyage')),

                    'voyage_api_key': self._safe_get_ui_value('voyage_api_key_input', current_sc.get('voyage_api_key', '')),

                    'voyage_embedding_model': self._safe_get_ui_value('voyage_embedding_model_combo', current_sc.get('voyage_embedding_model', 'voyage-3.5-lite')),

                    'openai_embedding_api_key': self._safe_get_ui_value('openai_embedding_api_key_input', current_sc.get('openai_embedding_api_key', '')),

                    'openai_embedding_model': self._safe_get_ui_value('openai_embedding_model_input', current_sc.get('openai_embedding_model', 'text-embedding-3-small')),

                    'cohere_api_key': self._safe_get_ui_value('cohere_api_key_input', current_sc.get('cohere_api_key', '')),

                    'cohere_embedding_model': self._safe_get_ui_value('cohere_embedding_model_input', current_sc.get('cohere_embedding_model', 'embed-english-v3.0')),

                    'voyage_batch_size': int(self._safe_get_ui_value('voyage_batch_size_spin', current_sc.get('voyage_batch_size', 64))),

                    'ollama_base_url': self._safe_get_ui_value('ollama_base_url_input', current_sc.get('ollama_base_url', "http://localhost:11434")),

                    'ollama_embed_model': self._safe_get_ui_value('ollama_embed_model_combo', current_sc.get('ollama_embed_model', "nomic-embed-text")),

                    'ollama_batch_size': int(self._safe_get_ui_value('ollama_batch_size_spin', current_sc.get('ollama_batch_size', 64))),

                    'use_dynamic_batch_size': self._safe_get_ui_value('use_dynamic_batch_size_cb', current_sc.get('use_dynamic_batch_size', True)),

                    'ollama_chat_model': self._safe_get_ui_value('ollama_chat_model_combo', current_sc.get('ollama_chat_model', "llama3.2")),

                    'rerank_python_path': self._safe_get_ui_value('rerank_python_path_input', current_sc.get('rerank_python_path', None)),

                }

            }



            # Preserve other config keys

            for k in ['saved_presets']:

                if k in current_config:

                    config[k] = current_config[k]



            if provider_type == "custom":

                config['api_url'] = self._safe_get_ui_value('api_url_input', current_config.get('api_url', ''))



            if save_config(config):

                showInfo("Settings saved successfully!")

                self.accept()

            else:

                showInfo("Error: Could not write config file.")



        except Exception as e:

            log_debug(f"Critical error in save_settings: {e}", is_error=True)

            showInfo(f"Save failed: {e}")

        finally:

            for name in widgets:

                w = getattr(self, name, None)

                if w: w.blockSignals(False)







    def _on_autodetect_python(self):

        """Attempts to find a compatible Python installation on common Windows/Mac paths."""

        from aqt.utils import showInfo, tooltip

        import subprocess

        import os



        # Common Windows paths for Python

        candidates = []

        if os.name == 'nt':

            local_appdata = os.environ.get('LOCALAPPDATA', '')

            base_dir = os.path.join(local_appdata, "Programs", "Python")

            if os.path.exists(base_dir):

                for folder in os.listdir(base_dir):

                    if folder.lower().startswith("python3"):

                        exe = os.path.join(base_dir, folder, "python.exe")

                        if os.path.exists(exe):

                            candidates.append(exe)

            # Check global paths

            for p in [r"C:\Python312\python.exe", r"C:\Python311\python.exe", r"C:\Python310\python.exe"]:

                if os.path.exists(p): candidates.append(p)

        else:

            # Unix/Mac

            for p in ["/usr/bin/python3", "/usr/local/bin/python3", "/opt/homebrew/bin/python3"]:

                if os.path.exists(p): candidates.append(p)



        if not candidates:

            showInfo("No common Python installations found automatically. Please paste the path to your python.exe manually.")

            return



        found_path = None

        for path in candidates:

            try:

                # Check if sentence-transformers is installed in this python

                cmd = [path, "-c", "import sentence_transformers; print('OK')"]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

                if "OK" in result.stdout:

                    found_path = path

                    break

            except Exception:

                continue



        if found_path:

            self.rerank_python_path_input.setText(found_path)

            tooltip(f"Success! Detected 'Rerank-Ready' Python at: {found_path}")

        else:

            # If no ready path, just take the first candidate

            self.rerank_python_path_input.setText(candidates[0])

            showInfo(f"Found Python at {candidates[0]}, but 'sentence-transformers' is not installed there yet. Click 'Install into external Python' to prepare it.")



    def _apply_config_to_ui(self):

        """Populates UI with config values while blocking signals to prevent 'Reversion Disease'."""

        try:
                    try:
                        import sip
                    except ImportError:
                        from PyQt6 import sip

        except ImportError:
                    from PyQt6 import sip



        c = load_config()

        sc = c.get('search_config', {})

        style = c.get('styling', {})



        # Comprehensive list of widgets to silence

        widgets = [

            'search_method_combo', 'max_results_spin', 'hybrid_weight_spin',

            'answer_provider_combo', 'enable_rerank_cb', 'enable_hyde_cb',

            'min_relevance_spin', 'layout_combo', 'answer_spacing_combo',

            'question_font_spin', 'answer_font_spin', 'notes_font_spin',

            'label_font_spin', 'width_spin', 'height_spin', 'section_spacing_spin',

            'use_context_boost_cb', 'strict_relevance_cb', 'relevance_from_answer_cb',

            'embedding_engine_combo', 'ollama_chat_model_combo', 'ollama_embed_model_combo',

            'enable_query_expansion_cb', 'use_ai_generic_term_detection_cb'

        ]



        for name in widgets:

            w = getattr(self, name, None)

            if w and not sip.isdeleted(w):

                w.blockSignals(True)



        try:

            # 1. Search Method & Logic

            method = sc.get('search_method', 'hybrid')

            if hasattr(self, 'search_method_combo') and not sip.isdeleted(self.search_method_combo):

                idx = self.search_method_combo.findData(method)

                if idx >= 0:

                    self.search_method_combo.setCurrentIndex(idx)

                else:

                    # Fallback to text matching

                    for i in range(self.search_method_combo.count()):

                        if method in self.search_method_combo.itemText(i).lower():

                            self.search_method_combo.setCurrentIndex(i)

                            break



            if hasattr(self, 'max_results_spin') and not sip.isdeleted(self.max_results_spin):

                self.max_results_spin.setValue(sc.get('max_results', 50))

            if hasattr(self, 'hybrid_weight_spin') and not sip.isdeleted(self.hybrid_weight_spin):

                self.hybrid_weight_spin.setValue(sc.get('hybrid_embedding_weight', 40))

            if hasattr(self, 'min_relevance_spin') and not sip.isdeleted(self.min_relevance_spin):

                self.min_relevance_spin.setValue(sc.get('min_relevance_percent', 55))



            # 2. Answer Provider & Local Server

            prov = c.get('provider', 'api_key')

            if hasattr(self, 'answer_provider_combo') and not sip.isdeleted(self.answer_provider_combo):

                p_idx = self.answer_provider_combo.findData(prov)

                if p_idx < 0:

                     p_idx = 0 if prov in ['ollama', 'local_openai', 'local_server'] else 1

                self.answer_provider_combo.setCurrentIndex(p_idx)



            if hasattr(self, 'api_key_input') and not sip.isdeleted(self.api_key_input):

                self.api_key_input.setText(c.get('api_key', ''))



            if hasattr(self, 'local_llm_url') and not sip.isdeleted(self.local_llm_url):

                self.local_llm_url.setText(sc.get('local_llm_url', 'http://localhost:11434'))

            if hasattr(self, 'local_llm_model') and not sip.isdeleted(self.local_llm_model):

                self.local_llm_model.setText(sc.get('local_llm_model', 'llama3.2'))



            # 3. Checkboxes (using radical safety)

            self._safe_set_checked(getattr(self, 'enable_query_expansion_cb', None), sc.get('enable_query_expansion', False))

            self._safe_set_checked(getattr(self, 'use_ai_generic_term_detection_cb', None), sc.get('use_ai_generic_term_detection', False))

            self._safe_set_checked(getattr(self, 'enable_hyde_cb', None), sc.get('enable_hyde', False))

            self._safe_set_checked(getattr(self, 'enable_rerank_cb', None), sc.get('enable_rerank', False))

            self._safe_set_checked(getattr(self, 'use_context_boost_cb', None), sc.get('use_context_boost', True))

            self._safe_set_checked(getattr(self, 'strict_relevance_cb', None), sc.get('strict_relevance', True))

            self._safe_set_checked(getattr(self, 'relevance_from_answer_cb', None), sc.get('relevance_from_answer', False))

            self._safe_set_checked(getattr(self, 'verbose_search_debug_cb', None), sc.get('verbose_search_debug', False))



            # --- Persistent Rerank Path Fix ---

            if hasattr(self, "rerank_python_path_input") and not sip.isdeleted(self.rerank_python_path_input):

                path = sc.get('rerank_python_path', "")

                self.rerank_python_path_input.setText(str(path) if path else "")



            # 4. Embedding Engine

            if hasattr(self, 'embedding_engine_combo') and not sip.isdeleted(self.embedding_engine_combo):

                engine = sc.get('embedding_engine', 'voyage')

                e_idx = self.embedding_engine_combo.findData(engine)

                if e_idx >= 0: self.embedding_engine_combo.setCurrentIndex(e_idx)



            if hasattr(self, 'voyage_api_key_input') and not sip.isdeleted(self.voyage_api_key_input):

                self.voyage_api_key_input.setText(sc.get('voyage_api_key', ''))



            if hasattr(self, 'voyage_embedding_model_combo') and not sip.isdeleted(self.voyage_embedding_model_combo):

                v_model = sc.get('voyage_embedding_model', 'voyage-3.5-lite')

                v_idx = self.voyage_embedding_model_combo.findData(v_model)

                if v_idx >= 0: self.voyage_embedding_model_combo.setCurrentIndex(v_idx)



            if hasattr(self, 'ollama_base_url_input') and not sip.isdeleted(self.ollama_base_url_input):

                self.ollama_base_url_input.setText(sc.get('ollama_base_url', "http://localhost:11434"))

            if hasattr(self, 'ollama_embed_model_combo') and not sip.isdeleted(self.ollama_embed_model_combo):

                self.ollama_embed_model_combo.setCurrentText(sc.get('ollama_embed_model', "nomic-embed-text"))

            if hasattr(self, 'ollama_chat_model_combo') and not sip.isdeleted(self.ollama_chat_model_combo):

                self.ollama_chat_model_combo.setCurrentText(sc.get('ollama_chat_model', "llama3.2"))



            # 5. Styling

            styling_widgets = {

                'question_font_spin': 'question_font_size',

                'answer_font_spin': 'answer_font_size',

                'notes_font_spin': 'notes_font_size',

                'label_font_spin': 'label_font_size',

                'width_spin': 'window_width',

                'height_spin': 'window_height',

                'section_spacing_spin': 'section_spacing'

            }

            for widget_name, config_key in styling_widgets.items():

                w = getattr(self, widget_name, None)

                if w and not sip.isdeleted(w):

                    w.setValue(style.get(config_key, 13 if 'font' in config_key else (1100 if 'width' in config_key else 800)))



            if hasattr(self, 'layout_combo') and not sip.isdeleted(self.layout_combo):

                l_idx = self.layout_combo.findData(style.get('layout_mode', 'side_by_side'))

                if l_idx >= 0: self.layout_combo.setCurrentIndex(l_idx)



            if hasattr(self, 'answer_spacing_combo') and not sip.isdeleted(self.answer_spacing_combo):

                s_idx = self.answer_spacing_combo.findData(style.get('answer_spacing', 'normal'))

                if s_idx >= 0: self.answer_spacing_combo.setCurrentIndex(s_idx)



        except Exception as e:

            log_debug(f"Error during unified UI population: {e}", is_error=True)



        # Unsilencing

        for name in widgets:

            w = getattr(self, name, None)

            if w and not sip.isdeleted(w):

                w.blockSignals(False)



        # Final UI Sync

        if hasattr(self, '_on_answer_provider_changed'): self._on_answer_provider_changed()

        if hasattr(self, '_on_embedding_engine_changed'): self._on_embedding_engine_changed()

        if hasattr(self, '_on_search_method_changed'): self._on_search_method_changed()



    def _on_answer_provider_changed(self):

        """Show/hide API key vs Local Server options with radical existence checks."""

        if not hasattr(self, "api_key_section") or not hasattr(self, "local_server_section"):

            return



        provider = self.answer_provider_combo.currentData() or ""



        try:
                    try:
                        import sip
                    except ImportError:
                        from PyQt6 import sip

        except ImportError:
                    from PyQt6 import sip



        if not sip.isdeleted(self.api_key_section):

            self.api_key_section.setVisible(provider == "api_key")

        if not sip.isdeleted(self.local_server_section):

            self.local_server_section.setVisible(provider == "local_server")



    def _refresh_local_models(self):

        """Fetch models from local server (Ollama, LM Studio, etc.)"""

        url = (self.local_llm_url.text() or "").strip()

        if not url:

            showInfo("Please enter a Server URL first.")

            return



        # Auto-correct common mistakes

        if "11434" in url and not url.startswith("http"):

            url = f"http://{url}"



        try:

            # Try Ollama native list first if it looks like Ollama

            if "11434" in url:

                models = get_ollama_models(url)

            else:

                # Try OpenAI compatible /models endpoint

                import requests

                # Handle trailing slashes and /v1

                base = url.rstrip('/')

                models_url = f"{base}/models"

                resp = requests.get(models_url, timeout=5)

                data = resp.json()

                models = [m['id'] for m in data.get('data', [])]



            if models:

                # We don't have a combo anymore, we have a line edit for flexibility,

                # but we'll show a picker or just info.

                from aqt.utils import chooseList

                idx = chooseList("Select a model from your local server:", models)

                if idx >= 0:

                    self.local_llm_model.setText(models[idx])

            else:

                showInfo("Connected but no models found. Make sure a model is loaded in your server.")

        except Exception as e:

            showInfo(f"Could not fetch models: {e}\n\nCheck that your server is running at {url}.")







    def _test_local_server_connection(self):

        """Unified test for local server (Ollama, LM Studio, etc.)"""

        url = (self.local_llm_url.text() or "").strip()

        if not url:

            showInfo("Please enter a Server URL first.")

            return



        import time

        import requests

        start = time.time()

        try:

            # Try a simple GET to the base or models endpoint

            # If it's Ollama, use its tags endpoint; otherwise try /models

            test_url = f"{url.rstrip('/')}/api/tags" if "11434" in url else f"{url.rstrip('/')}/models"

            resp = requests.get(test_url, timeout=5)

            elapsed = (time.time() - start) * 1000



            if resp.status_code == 200:

                showInfo(f"Γ£à Connection successful!\n\nLatency: {elapsed:.0f}ms\nServer: {url}")

            else:

                showInfo(f"ΓÜá∩╕Å Server responded with code {resp.status_code}.\nURL: {test_url}")

        except Exception as e:

            showInfo(f"Γ¥î Connection failed.\n\nError: {e}\n\nMake sure your server is running at {url}")



        base_url = (self.ollama_base_url_input.text() or "http://localhost:11434").strip()



        try:



            t0 = time.perf_counter()



            names = get_ollama_models(base_url)



            elapsed_ms = int((time.perf_counter() - t0) * 1000)



            if names:



                showInfo(f"\u2705 Ollama connection OK\n\nLatency: {elapsed_ms} ms\nModels: {len(names)} available")



            else:



                showInfo("\xe2\u0161\xa0\ufe0f Ollama responded but no models found. Run 'ollama pull <model>' to install.")



        except Exception as e:



            showInfo(f"\xe2\x9d\u0152 Ollama test failed\n\nError: {e}\n\nMake sure Ollama is running (ollama serve) and the URL is correct.")







    def detect_provider_type(self, api_key):



        if api_key.startswith("sk-ant-"):



            return "anthropic"



        elif api_key.startswith("sk-or-"):



            return "openrouter"



        elif api_key.startswith("sk-"):



            return "openai"



        elif api_key.startswith("AI"):



            return "google"



        else:



            return "custom"







    def _check_rerank_available(self, extra_path=None, python_path=None):



        """Check if sentence-transformers CrossEncoder is available.



        If python_path is set (path to python.exe), use that Python for the check.



        Else if extra_path is set, run Anki's Python with that folder on sys.path.



        Else run Anki's Python."""



        try:



            import os



            import subprocess



            import sys



            # Prefer user's Python (e.g. Python 3.11) when set



            if python_path:



                python_path = python_path.strip()



                # Allow folder or executable: if folder, append python.exe on Windows



                if os.path.isdir(python_path):



                    python_exe = os.path.join(python_path, "python.exe")



                    if not os.path.isfile(python_exe):



                        python_exe = os.path.join(python_path, "python")



                    python_path = python_exe if os.path.isfile(python_exe) else python_path



                if not os.path.isfile(python_path):



                    return False



                result = subprocess.run(



                    [python_path, "-c", "from sentence_transformers import CrossEncoder; print('ok')"],



                    capture_output=True, text=True, timeout=30,



                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0



                )



                return result.returncode == 0 and 'ok' in (result.stdout or '')



            env = os.environ.copy()



            if extra_path and os.path.isdir(extra_path):



                check_script = (



                    "import sys, os; "



                    "p = os.environ.get('AI_SEARCH_ST_PATH', ''); "



                    "p and sys.path.insert(0, p); "



                    "from sentence_transformers import CrossEncoder; "



                    "print('ok')"



                )



                env['AI_SEARCH_ST_PATH'] = extra_path



            else:



                check_script = "from sentence_transformers import CrossEncoder; print('ok')"



            result = subprocess.run(



                [sys.executable, "-c", check_script],



                capture_output=True, text=True, timeout=15, env=env,



                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0



            )



            return result.returncode == 0 and 'ok' in (result.stdout or '')



        except Exception:



            return False







    def _update_rerank_tooltip(self):



        """Update Cross-Encoder checkbox tooltip with status and (if unavailable) Python path."""



        import sys



        base = "Re-ranks top 15 results with a cross-encoder for 10-30% better relevance.\n"



        if self._rerank_available:



            self.enable_rerank_cb.setToolTip(base + "Ready to use.")



        else:



            self.enable_rerank_cb.setToolTip(



                base + "Not installed. Set 'Python for Cross-Encoder' to your Python (e.g. Python 3.11) that has "



                "sentence-transformers, or click 'Install Dependencies' to install into Anki's Python:\n" + sys.executable



            )







    def _on_check_rerank_again(self):



        """Re-check sentence-transformers and update Cross-Encoder checkbox state and tooltip."""



        import sys



        import time



        t0 = time.time()



        python_path = (self.rerank_python_path_input.text() or '').strip() or None



        self._rerank_available = self._check_rerank_available(python_path=python_path)



        self.enable_rerank_cb.setEnabled(self._rerank_available)



        self._update_rerank_tooltip()



        if self._rerank_available:



            showInfo("sentence-transformers is available. Cross-Encoder Re-Ranking can be enabled.")



        else:



            msg = (



                "sentence-transformers not found.\n\n"



                "Option A \u2014 Use your own Python (e.g. Python 3.11):\n"



                "1. Set 'Python for Cross-Encoder' above to that python.exe (or its folder).\n"



                "2. Click 'Install into external Python' to install sentence-transformers there.\n"



                "3. Click 'Check again'.\n\n"



                "Option B \u2014 Use Anki's Python:\n"



                "Clear the optional path, click 'Install Dependencies', then 'Check again'.\n\n"



                "Anki's Python: " + sys.executable



            )



            showInfo(msg)







    def _on_install_into_external_python(self):

        """Toggle visibility of external Python path, or install if already visible and path is set."""

        # Toggle visibility

        is_visible = self.python_path_widget.isVisible()



        # If it was hidden, show it and try to help with autodetect

        if not is_visible:

            self.python_path_widget.setVisible(True)

            if not self.rerank_python_path_input.text().strip():

                self._on_autodetect_python()

            return



        # If it is already visible, but the user clicks it again without a path, hide it

        path = (self.rerank_python_path_input.text() or '').strip()

        if is_visible and not path:

            self.python_path_widget.setVisible(False)

            return



        # If it is visible AND there is a path, proceed with installation

        python_exe = _resolve_external_python_exe(path)



        if not path:



            showInfo("Enter a path in 'Python for Cross-Encoder' (python.exe or its folder), then try again.")



            return



        python_exe = _resolve_external_python_exe(path)



        if not python_exe:



            showInfo(f"Path not found or not a valid Python:\n{path}\n\nEnter the path to python.exe or the folder containing it.")



            return



        install_dependencies(python_exe=python_exe)







    def _on_search_method_changed(self):



        """Show/hide Cloud Embeddings and options based on search method."""



        method = self.search_method_combo.currentData() or "hybrid"



        self.embedding_section.setVisible(method in ("embedding", "hybrid"))



        # HyDE only applies to embedding/hybrid



        self.enable_hyde_cb.setVisible(method in ("embedding", "hybrid"))



        # Hybrid weight: used in weighted RRF (\xce\xb1_lexical * 1/(k+r_kw) + \xce\xb1_dense * 1/(k+r_emb))



        self.hybrid_weight_label.setVisible(method == "hybrid")



        self.hybrid_weight_spin.setVisible(method == "hybrid")







    def _refresh_embedding_status(self):

        """Check and display embedding status for the currently selected engine in the UI."""

        try:

            # Get current UI state instead of loading from disk to ensure responsiveness

            engine = self.embedding_engine_combo.currentData() or 'cloud'



            # If in Cloud mode, find which specific provider is selected

            if engine == 'cloud':

                engine = self.cloud_provider_combo.currentData() or 'voyage'



            status_text = ""



            if engine == 'local' or engine == 'ollama':

                base_url = self.ollama_base_url_input.text().strip() or 'http://localhost:11434'

                model = self.ollama_embed_model_combo.currentText().strip() or 'nomic-embed-text'

                status_text = (

                    "CONNECTED: Ollama (Local AI)\n\n"

                    f"URL: {base_url}\n"

                    f"Model: {model}\n\n"

                    "Ensure Ollama is running ('ollama serve') and models are pulled."

                )



            elif engine == "voyage":

                api_key = self.voyage_api_key_input.text().strip()

                if not api_key:

                    status_text = (

                        "DISABLED: Voyage AI\n\n"

                        "Please enter your Voyage API key above to enable high-quality medical search."

                    )

                else:

                    status_text = (

                        "READY: Voyage AI (Cloud)\n\n"

                        "API key detected. Click 'Create/Update' below to index your notes."

                    )



            elif engine == "openai":

                api_key = self.openai_embedding_api_key_input.text().strip()

                model = self.openai_embedding_model_input.text().strip() or "text-embedding-3-small"

                if not api_key:

                    status_text = (

                        "DISABLED: OpenAI\n\n"

                        "Enter your OpenAI API key above to use OpenAI embeddings."

                    )

                else:

                    status_text = (

                        f"READY: OpenAI (Cloud) - Model: {model}\n\n"

                        "Click 'Create/Update' to start embedding."

                    )



            elif engine == "cohere":

                api_key = self.cohere_api_key_input.text().strip()

                if not api_key:

                    status_text = (

                        "DISABLED: Cohere\n\n"

                        "Enter your Cohere API key above to enable embeddings."

                    )

                else:

                    status_text = (

                        "READY: Cohere (Cloud)\n\n"

                        "API key detected. Ready to create embeddings."

                    )



            if hasattr(self, 'embedding_status_label'):

                self.embedding_status_label.setText(status_text)

                if "READY" in status_text or "CONNECTED" in status_text:

                    self.embedding_status_label.setStyleSheet("padding: 10px; border: 1px solid #2ecc71; border-radius: 4px; background: #1a2a1a; color: #2ecc71;")

                else:

                    self.embedding_status_label.setStyleSheet("padding: 10px; border: 1px solid #e74c3c; border-radius: 4px; background: #2a1a1a; color: #e74c3c;")

        except Exception as e:

            if hasattr(self, 'embedding_status_label'):

                self.embedding_status_label.setText(f"Error checking status: {str(e)}")



    def _start_embedding_service(self):



        """Start the embedding service in a separate process"""



        import subprocess



        import sys



        import os



        import urllib.request



        import json



        import time







        # Local embedding service is no longer supported.



        # This method is kept only to avoid breaking older configs.



        showInfo(



            "The local embedding service has been removed.\n\n"



            "This addon now uses only the cloud embeddings API (Voyage) for semantic search.\n"



            "You can generate embeddings via the 'Create/Update Embeddings' button."



        )



        return







        # Check if process is already running



        if self.service_process is not None:



            if sys.platform == 'win32':



                # On Windows, we can't easily check if process is running



                # Just check via HTTP below



                pass



            elif hasattr(self.service_process, 'poll') and self.service_process.poll() is None:



                showInfo("Service is already starting. Please wait...")



                return



            else:



                # Process has ended, reset reference



                self.service_process = None







        # Get addon directory



        addon_dir = os.path.dirname(__file__)







        # Try to start embedding_service.py first (real service)



        service_file = os.path.join(addon_dir, "embedding_service.py")



        fallback_file = os.path.join(addon_dir, "simple_embedding_server.py")







        service_script = None



        service_name = None







        if os.path.exists(service_file):



            service_script = service_file



            service_name = "embedding_service.py (Real Service)"



        elif os.path.exists(fallback_file):



            service_script = fallback_file



            service_name = "simple_embedding_server.py (Test Server)"



        else:



            showInfo(



                f"\xe2\x9d\u0152 Cannot find embedding service files!\n\n"



                f"Expected files:\n"



                f"- {service_file}\n"



                f"- {fallback_file}\n\n"



                f"Please make sure the service files are in the addon directory."



            )



            return







        try:



            # Start the service in a new process



            # On Windows, use cmd.exe start to open a new window



            if sys.platform == 'win32':



                # Create a batch file to run the service (handles paths with spaces better)



                import tempfile



                batch_content = f'''@echo off



title Embedding Service



cd /d "{addon_dir}"



echo Starting embedding service...



echo.



"{sys.executable}" "{service_script}"



if errorlevel 1 (



    echo.



    echo Service exited with an error.



    echo Press any key to close this window...



    pause >nul



)



'''



                # Write batch file



                batch_file = os.path.join(addon_dir, "start_embedding_service.bat")



                try:



                    # Ensure directory exists and file can be written



                    os.makedirs(addon_dir, exist_ok=True)



                    with open(batch_file, 'w', encoding='utf-8', newline='\r\n') as f:  # Windows line endings



                        f.write(batch_content)



                    log_debug(f"Created batch file: {batch_file}")







                    # Try multiple methods to start the service, prioritizing simpler ones



                    service_started = False







                    # Method 1: Use VBScript wrapper (handles paths with spaces perfectly)



                    try:



                        vbs_script = os.path.join(addon_dir, "start_service.vbs")



                        vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")



WshShell.CurrentDirectory = "{addon_dir}"



WshShell.Run "cmd /k ""{batch_file}""", 1, False



Set WshShell = Nothing



'''



                        with open(vbs_script, 'w', encoding='utf-8') as f:



                            f.write(vbs_content)







                        # Execute VBScript - this handles paths with spaces automatically



                        subprocess.Popen(['wscript', vbs_script], shell=False)



                        log_debug(f"Started service via VBScript wrapper: {vbs_script}")



                        service_started = True



                        self.service_process = True



                    except Exception as vbs_err:



                        log_debug(f"Method 1 (VBScript) failed: {vbs_err}")







                    # Method 2: Direct batch file execution using subprocess with shell=True



                    if not service_started:



                        try:



                            # Use shell=True which handles path quoting automatically



                            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):



                                subprocess.Popen(



                                    f'cmd /c start cmd /k "{batch_file}"',



                                    shell=True,



                                    creationflags=subprocess.CREATE_NEW_CONSOLE,



                                    cwd=addon_dir



                                )



                            else:



                                subprocess.Popen(



                                    f'cmd /c start cmd /k "{batch_file}"',



                                    shell=True,



                                    cwd=addon_dir



                                )



                            log_debug(f"Started service via batch file (shell=True): {batch_file}")



                            service_started = True



                            self.service_process = True



                        except Exception as batch_err:



                            log_debug(f"Method 2 (batch file shell) failed: {batch_err}")







                    # Method 3: Direct Python execution with shell=True



                    if not service_started:



                        try:



                            # Use shell=True for automatic path handling



                            cmd_str = f'cmd /c start cmd /k "cd /d "{addon_dir}" && "{sys.executable}" "{service_script}""'



                            if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):



                                subprocess.Popen(



                                    cmd_str,



                                    shell=True,



                                    creationflags=subprocess.CREATE_NEW_CONSOLE



                                )



                            else:



                                subprocess.Popen(cmd_str, shell=True)



                            log_debug(f"Started service via direct Python execution (shell=True)")



                            service_started = True



                            self.service_process = True



                        except Exception as direct_err:



                            log_debug(f"Method 3 (direct Python shell) failed: {direct_err}")







                    # Method 4: os.startfile (Windows only, simplest but no console)



                    if not service_started and sys.platform == 'win32':



                        try:



                            os.startfile(batch_file)



                            log_debug(f"Started service via os.startfile")



                            service_started = True



                            self.service_process = True



                        except Exception as startfile_err:



                            log_debug(f"Method 4 (os.startfile) failed: {startfile_err}")







                    # Method 5: PowerShell as last resort (only if admin needed)



                    if not service_started:



                        try:



                            # Create PowerShell script with proper escaping



                            ps_script = os.path.join(addon_dir, "start_embedding_service_admin.ps1")



                            # Escape backslashes and quotes properly



                            batch_file_escaped = batch_file.replace('\\', '\\\\').replace("'", "''")



                            addon_dir_escaped = addon_dir.replace('\\', '\\\\').replace("'", "''")



                            python_exe_escaped = sys.executable.replace('\\', '\\\\').replace("'", "''")



                            service_script_escaped = service_script.replace('\\', '\\\\').replace("'", "''")







                            # Use $PSScriptRoot for dynamic path (works regardless of folder name)



                            ps_content = f'''# PowerShell script to start embedding service with admin privileges if needed



# This script uses $PSScriptRoot to get the directory where the script is located (works regardless of folder name)



$ErrorActionPreference = "Continue"



$scriptDir = $PSScriptRoot



$batchFile = Join-Path $scriptDir "start_embedding_service.bat"



$pythonExe = '{python_exe_escaped}'



$serviceScript = Join-Path $scriptDir "embedding_service.py"







# Check if running as admin



$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)







if (-not $isAdmin) {{



    Write-Host "Requesting administrator privileges..."



    $cmd = "Set-Location -LiteralPath '$scriptDir'; & '$pythonExe' '$serviceScript'"



    Start-Process powershell -Verb RunAs -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $cmd



}} else {{



    Write-Host "Running with administrator privileges..."



    Set-Location -LiteralPath $scriptDir



    & $pythonExe $serviceScript



}}



'''



                            with open(ps_script, 'w', encoding='utf-8', newline='\r\n') as f:



                                f.write(ps_content)



                            log_debug(f"Created PowerShell script: {ps_script}")







                            # Execute with shell=True for proper path handling



                            subprocess.Popen(



                                f'powershell -ExecutionPolicy Bypass -File "{ps_script}"',



                                shell=True,



                                cwd=addon_dir



                            )



                            log_debug(f"Started service via PowerShell script")



                            service_started = True



                            self.service_process = True



                        except Exception as ps_err:



                            log_debug(f"Method 5 (PowerShell) failed: {ps_err}")







                    if not service_started:



                        raise Exception("All service startup methods failed. Check debug_log.txt for details.")



                except Exception as batch_error:



                    log_debug(f"Failed to create batch file, trying direct method: {batch_error}")



                    # Fallback: try direct method with explicit window



                    try:



                        # Try direct Python execution in new console window



                        # Use full path to Python and service script



                        python_exe = sys.executable



                        service_path = service_script



                        # Escape paths with spaces properly



                        if ' ' in python_exe:



                            python_exe = f'"{python_exe}"'



                        if ' ' in service_path:



                            service_path = f'"{service_path}"'







                        # Create a command that changes directory and runs Python



                        cmd_str = f'cd /d "{addon_dir}" && {python_exe} {service_path}'



                        log_debug(f"Starting service with command: {cmd_str}")







                        # Use CREATE_NEW_CONSOLE flag to ensure new window



                        if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):



                            subprocess.Popen(



                                ['cmd', '/c', 'start', 'cmd', '/k', cmd_str],



                                shell=False,



                                creationflags=subprocess.CREATE_NEW_CONSOLE



                            )



                        else:



                            subprocess.Popen(



                                ['cmd', '/c', 'start', 'cmd', '/k', cmd_str],



                                shell=True



                            )



                        self.service_process = True



                        log_debug("Service started via direct command method")



                    except Exception as direct_error:



                        log_debug(f"Direct method also failed: {direct_error}")



                        # Last resort: try PowerShell



                        try:



                            ps_cmd = f'Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd \'{addon_dir}\'; & \'{sys.executable}\' \'{service_script}\'"'



                            subprocess.Popen(['powershell', '-Command', ps_cmd], shell=False)



                            self.service_process = True



                            log_debug("Service started via PowerShell method")



                        except Exception as ps_error:



                            log_debug(f"PowerShell method also failed: {ps_error}")



                            raise Exception(f"All service startup methods failed. Last error: {ps_error}")



            else:



                self.service_process = subprocess.Popen(



                    [sys.executable, service_script],



                    cwd=addon_dir



                )







            # Wait a moment for the service to start



            time.sleep(3)







            # Check if service is responding (more reliable than checking process)



            # On Windows, we can't easily track the detached process, so we check HTTP



            if sys.platform == 'win32':



                # For Windows, we check via HTTP instead of process polling



                pass  # Will check below



            elif self.service_process.poll() is not None:



                # Process has already terminated (error starting) - only for non-Windows



                showInfo(



                    f"Γ¥î Failed to start service!\n\n"



                    f"Service: {service_name}\n\n"



                    f"The service process exited immediately. Check the console window for error messages.\n\n"



                    f"Common issues:\n"



                    f"- Missing dependencies (pip install flask sentence-transformers)\n"



                    f"- Port 9000 already in use\n"



                    f"- Python path issues"



                )



                self.service_process = None



                return







            # Test if service is responding



            try:



                test_data = json.dumps({"text": "test"}).encode('utf-8')



                test_req = urllib.request.Request(url, test_data, {"Content-Type": "application/json"})



                urllib.request.urlopen(test_req, timeout=3)







                showInfo(



                    f"\u2705 Service started successfully!\n\n"



                    f"Service: {service_name}\n"



                    f"URL: {url}\n\n"



                    f"A console window has been opened showing the service output.\n"



                    f"Keep this window open while using the embedding service."



                )



                # Refresh status



                QTimer.singleShot(500, self._refresh_embedding_status)



            except Exception as e:



                # Service started but not responding yet



                showInfo(



                    f"\xe2\u0161\xa0\ufe0f Service process started but not responding yet.\n\n"



                    f"Service: {service_name}\n"



                    f"URL: {url}\n\n"



                    f"Please wait a few seconds and click '\U0001F50C Test Connection' to verify.\n\n"



                    f"If the service doesn't start, check the console window for errors."



                )



                # Refresh status after a delay



                QTimer.singleShot(3000, self._refresh_embedding_status)







        except Exception as e:



            showInfo(



                f"\xe2\x9d\u0152 Error starting service!\n\n"



                f"Service: {service_name}\n"



                f"Error: {str(e)}\n\n"



                f"Please check:\n"



                f"- Python is installed and in PATH\n"



                f"- Service file exists: {service_script}\n"



                f"- Required dependencies are installed"



            )



            self.service_process = None







    def _test_embedding_connection(self):



        """Test connection to the selected embedding engine (Voyage, OpenAI, Cohere, or Ollama). Shows pass/fail with latency."""



        import time



        test_text = "Test connection"



        engine = self.embedding_engine_combo.currentData() or "voyage"



        sc = {



            "embedding_engine": engine,



            "voyage_api_key": (self.voyage_api_key_input.text() or "").strip(),



            "voyage_embedding_model": (self.voyage_embedding_model_combo.currentData() or "voyage-3.5-lite"),



            "openai_embedding_api_key": (self.openai_embedding_api_key_input.text() or "").strip(),



            "openai_embedding_model": (self.openai_embedding_model_input.text() or "text-embedding-3-small").strip(),



            "cohere_api_key": (self.cohere_api_key_input.text() or "").strip(),



            "cohere_embedding_model": (self.cohere_embedding_model_input.text() or "embed-english-v3.0").strip(),



            "voyage_batch_size": self.voyage_batch_size_spin.value(),



            "ollama_base_url": (self.ollama_base_url_input.text() or "http://localhost:11434").strip(),



            "ollama_embed_model": (self.ollama_embed_model_combo.currentText() or "nomic-embed-text").strip(),



            "ollama_batch_size": self.ollama_batch_size_spin.value(),



        }



        config = {"search_config": sc}



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



                showInfo(



                    f"\u2705 Embedding connection OK \u2014 {engine_name}\n\n"



                    f"Dimension: {dim} | Latency: {elapsed_ms} ms"



                )



                if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                    self.embedding_status_label.setText(f"\u2705 {engine_name} OK ({elapsed_ms} ms)")



            else:



                showInfo(



                    "\xe2\u0161\xa0\ufe0f Connection succeeded but received an empty embedding.\n\n"



                    "Check your engine settings (URL/model or API key) and try again."



                )



                if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                    self.embedding_status_label.setText("\xe2\u0161\xa0\ufe0f Empty embedding")



        except Exception as e:



            if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                self.embedding_status_label.setText("\xe2\x9d\u0152 Test failed")



            if engine == "ollama":



                hint = "Make sure Ollama is running (ollama serve) and the model is pulled (e.g. ollama pull nomic-embed-text)."



            elif engine == "openai":



                hint = "Enter your OpenAI API key above (or set OPENAI_API_KEY) and check internet access."



            elif engine == "cohere":



                hint = "Enter your Cohere API key above (or set COHERE_API_KEY) and check internet access."



            else:



                hint = "Enter your API key above (or set the provider's env var) and check internet access."



            showInfo(



                f"\xe2\x9d\u0152 Embedding test failed!\n\n"



                f"Error: {e}\n\n"



                f"{hint}"



            )







    def _migrate_json_to_db(self):



        """Copy embeddings from legacy JSON file into SQLite DB (no re-embedding)."""



        if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



            self.embedding_status_label.setText("Migrating JSON \u2192 database...")



            QApplication.processEvents()



        try:



            count, err = migrate_embeddings_json_to_db()



            if err:



                showInfo(f"Migration could not complete.\n\n{err}")



                if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                    self.embedding_status_label.setText("")



                return



            showInfo(f"Migrated {count} embeddings from the old JSON file into the database.\n\nYou can keep or delete the old .json file; new data is now in the .db file.")



            if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                self.embedding_status_label.setText(f"Migrated {count} embeddings to database.")



            try:



                QTimer.singleShot(100, self._refresh_embedding_status)



            except Exception:



                pass



        except Exception as e:



            showInfo(f"Migration failed: {e}")



            if hasattr(self, 'embedding_status_label') and self.embedding_status_label:



                self.embedding_status_label.setText("")







    def _create_or_update_embeddings(self):



        """Create or update embeddings for all notes using the selected engine (Voyage, OpenAI, Cohere, or Ollama)."""



        # Persist current UI engine/URL/model so worker uses them (user may have changed without saving dialog)



        config = load_config()



        sc = dict(config.get('search_config') or {})



        sc['embedding_engine'] = self.embedding_engine_combo.currentData() or 'voyage'



        sc['voyage_api_key'] = (self.voyage_api_key_input.text() or '').strip()



        sc['voyage_embedding_model'] = (self.voyage_embedding_model_combo.currentData() or 'voyage-3.5-lite')



        sc['openai_embedding_api_key'] = (self.openai_embedding_api_key_input.text() or '').strip()



        sc['openai_embedding_model'] = (self.openai_embedding_model_input.text() or 'text-embedding-3-small').strip()



        sc['cohere_api_key'] = (self.cohere_api_key_input.text() or '').strip()



        sc['cohere_embedding_model'] = (self.cohere_embedding_model_input.text() or 'embed-english-v3.0').strip()



        sc['voyage_batch_size'] = int(self.voyage_batch_size_spin.value())



        sc['ollama_base_url'] = (self.ollama_base_url_input.text() or "http://localhost:11434").strip()



        sc['ollama_embed_model'] = (self.ollama_embed_model_combo.currentText() or "nomic-embed-text").strip()



        sc['ollama_batch_size'] = int(self.ollama_batch_size_spin.value())



        sc['use_dynamic_batch_size'] = self.use_dynamic_batch_size_cb.isChecked()



        config['search_config'] = sc



        save_config(config)  # persist batch size and embedding settings



        engine = sc.get('embedding_engine') or 'voyage'



        # For Ollama: verify model is available before starting (Ollama loads models on first request)



        if engine == 'ollama':



            base_url = sc.get('ollama_base_url') or 'http://localhost:11434'



            model = sc.get('ollama_embed_model') or 'nomic-embed-text'



            model_base = model.split(':')[0]  # nomic-embed-text:latest -> nomic-embed-text



            try:



                names = get_ollama_models(base_url.strip())



                if not names:



                    showInfo(



                        "\xe2\x9d\u0152 Ollama returned no models.\n\n"



                        "Make sure Ollama is running (ollama serve) and you have pulled at least one model.\n\n"



                        f"For embeddings, run: ollama pull {model_base}"



                    )



                    return



                # Check if our model (or base name) is available



                if not any(model_base in n or n.startswith(model_base) for n in names):



                    showInfo(



                        f"\xe2\x9d\u0152 Ollama embedding model '{model_base}' not found.\n\n"



                        f"Available models: {', '.join(names[:8])}{'...' if len(names) > 8 else ''}\n\n"



                        f"Run: ollama pull {model_base}"



                    )



                    return



            except Exception as e:



                showInfo(



                    f"Γ¥î Cannot reach Ollama at {base_url}\n\n"



                    f"Error: {e}\n\n"



                    "Make sure Ollama is running (ollama serve)."



                )



                return



        # Quick API check first to avoid running a long job with bad config



        try:



            test_embedding = get_embedding_for_query("Test connection")



            if not test_embedding:



                showInfo(



                    "\xe2\x9d\u0152 Embedding engine returned an empty result.\n\n"



                    "Check your engine (URL/model or API key) and try again."



                )



                return



        except Exception as e:



            if engine == 'ollama':



                showInfo(



                    f"\xe2\x9d\u0152 Ollama embedding test failed.\n\n"



                    f"Error: {e}\n\n"



                    "Make sure Ollama is running (ollama serve) and the model is pulled "



                    f"(e.g. ollama pull {model_base})."



                )



            else:



                showInfo(



                    f"Γ¥î Cannot use embeddings API!\n\n"



                    f"Error: {e}\n\n"



                    "Enter your API key for the selected engine above and check internet access."



                )



            return







        # Get note type filter config



        # Always base this on the *current* UI selections so user choices



        # (note types, decks, fields) are remembered between sessions,



        # even if they didn't click the main "Save Settings" button.



        current_ntf = self._build_ntf_from_ui()



        config = load_config()



        config['note_type_filter'] = current_ntf



        # Persist immediately so next Anki restart / addon open uses the same



        # note/deck/field selection.



        save_config(config)



        ntf = current_ntf







        # Count notes that will be processed



        eligibility = analyze_note_eligibility(ntf)

        note_count = eligibility.get('eligible_count', 0)



        if note_count == 0:



            showInfo("No notes found to process. Check your note type and deck filters.")



            return







        # Check for existing checkpoint (only resume if it was for the same embedding engine)



        checkpoint = load_checkpoint()



        resume_available = False



        current_engine_id = get_embedding_engine_id(config)



        if checkpoint and checkpoint.get('engine_id') != current_engine_id:



            checkpoint = None  # different engine: start fresh, don't offer resume



        if checkpoint:



            processed_count = checkpoint.get('processed_count', 0)



            total_notes = checkpoint.get('total_notes', 0)



            if processed_count > 0 and processed_count < total_notes:



                resume_available = True



                reply = QMessageBox.question(



                    self,



                    "Resume Embedding Generation?",



                    f"Found a previous checkpoint:\n\n"



                    f"Processed: {processed_count:,} / {total_notes:,} notes\n"



                    f"Timestamp: {checkpoint.get('timestamp', 'unknown')}\n\n"



                    f"Would you like to resume from where you left off?\n\n"



                    f"(Click 'No' to start over)",



                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,



                    QMessageBox.StandardButton.Yes



                )







                if reply == QMessageBox.StandardButton.Cancel:



                    return



                elif reply == QMessageBox.StandardButton.No:



                    # Clear checkpoint and start fresh



                    clear_checkpoint()



                    checkpoint = None



                    resume_available = False







        if not resume_available:



            reply = QMessageBox.question(



                self,



                "Create/Update Embeddings",



                f"This will generate embeddings for approximately {note_count:,} notes.\n\n"

                f"Currently excluded by filters: {len(eligibility.get('ineligible_notes', [])):,} notes.\n\n"



                f"This may take a while depending on the number of notes.\n\n"



                f"Continue?",



                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,



                QMessageBox.StandardButton.Yes



            )







            if reply != QMessageBox.StandardButton.Yes:



                return







        # Create progress dialog (non-modal so Anki stays responsive)



        progress_dialog = QDialog(self)



        progress_dialog.setWindowTitle("Creating Embeddings")



        progress_dialog.setMinimumWidth(500)



        progress_dialog.setMinimumHeight(350)



        progress_dialog.setModal(False)  # Non-modal so user can continue using Anki



        # Add minimize and maximize buttons



        flags = progress_dialog.windowFlags()



        progress_dialog.setWindowFlags(flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)



        progress_layout = QVBoxLayout(progress_dialog)







        # Track pause state



        progress_dialog._is_paused = False



        progress_dialog._pause_lock = False







        status_label = QLabel("Initializing embedding model...")



        status_label.setWordWrap(True)



        progress_layout.addWidget(status_label)







        progress_bar = QProgressBar()



        progress_bar.setRange(0, note_count)



        progress_bar.setValue(0)



        progress_bar.setTextVisible(True)



        progress_bar.setFormat("%p%")



        progress_layout.addWidget(progress_bar)







        log_text = QTextEdit()



        log_text.setReadOnly(True)



        log_text.setMaximumHeight(200)



        log_text.setFont(QFont("Courier", 9))



        progress_layout.addWidget(log_text)







        # Control buttons



        button_layout = QHBoxLayout()







        pause_button = QPushButton("Pause")



        pause_button.clicked.connect(lambda: self._toggle_pause(progress_dialog, pause_button, log_text))



        button_layout.addWidget(pause_button)







        button_layout.addStretch()







        close_button = QPushButton("Close")



        close_button.setEnabled(False)



        close_button.clicked.connect(progress_dialog.close)



        button_layout.addWidget(close_button)







        progress_layout.addLayout(button_layout)







        # Store references for worker thread



        progress_dialog._status_label = status_label



        progress_dialog._progress_bar = progress_bar



        progress_dialog._log_text = log_text



        progress_dialog._close_button = close_button



        progress_dialog._pause_button = pause_button







        progress_dialog.show()



        QApplication.processEvents()







        # Create and start worker thread for embedding (prevents blocking)



        worker = EmbeddingWorker(



            ntf, note_count, checkpoint, resume_available



        )







        # Connect worker signals to UI updates



        worker.status_update.connect(status_label.setText)



        worker.progress_update.connect(progress_bar.setValue)



        worker.log_message.connect(log_text.append)



        worker.finished_signal.connect(lambda processed, errors, skipped, still_failed: self._on_embedding_finished(



            progress_dialog, processed, errors, skipped, still_failed, note_count



        ))



        worker.error_signal.connect(lambda msg: self._on_embedding_error(progress_dialog, msg))







        # Store worker reference



        progress_dialog._worker = worker







        # Start worker thread



        worker.start()







    def _review_ineligible_notes(self):


        ntf = self._build_ntf_from_ui()


        audit = analyze_note_eligibility(ntf)


        ineligible = audit.get("ineligible_notes", [])


        if not ineligible:


            showInfo(
                "All notes in the current deck/type scope are eligible for embeddings.\n\n"
                f"Eligible notes: {audit.get('eligible_count', 0):,}"
            )


            return


        reason_lines = [
            f"Eligible notes: {audit.get('eligible_count', 0):,}",
            f"Ineligible notes: {len(ineligible):,}",
            f"- Wrong note type: {audit.get('filtered_out_note_type_count', 0):,}",
            f"- No embedding fields selected: {audit.get('no_selected_fields_count', 0):,}",
            f"- Selected fields empty: {audit.get('empty_selected_fields_count', 0):,}",
            "",
            "First ineligible notes:",
        ]


        preview = ineligible[:100]


        for note in preview:


            fields = ", ".join(note.get("field_names") or []) or "(none)"


            model_name = note.get("model_name") or "(not in selected note types)"


            reason_lines.append(
                f"- nid:{note['id']} | {note['reason']} | note type: {model_name} | fields: {fields}"
            )


        if len(ineligible) > len(preview):


            reason_lines.append("")


            reason_lines.append(f"...and {len(ineligible) - len(preview):,} more.")


        note_ids = [str(note["id"]) for note in ineligible]


        browser_note_ids = note_ids[:1000]


        search_query = "nid:" + ",".join(browser_note_ids)


        try:


            browser = dialogs.open("Browser", mw)


            browser.form.searchEdit.lineEdit().setText(search_query)


            browser.onSearchActivated()


            QTimer.singleShot(150, lambda b=browser: self._bring_browser_to_front(b))


            tooltip(f"Opened {len(note_ids)} ineligible notes in browser")


        except Exception as exc:


            log_debug(f"Could not open ineligible notes in browser: {exc}")


        reason_lines.extend([
            "",
            f"Browser opened with first {len(browser_note_ids):,} ineligible notes.",
            "",
            "Browser query:",
            search_query,
        ])


        showText("\n".join(reason_lines), title="Ineligible Notes Audit")


    def _toggle_pause(self, progress_dialog, pause_button, log_text):



        """Toggle pause/resume for embedding process"""



        if progress_dialog._is_paused:



            # Resume



            progress_dialog._is_paused = False



            pause_button.setText("Pause")



            if hasattr(progress_dialog, '_worker') and progress_dialog._worker:



                progress_dialog._worker._is_paused = False



            log_text.append("Resumed processing...")



        else:



            # Pause



            progress_dialog._is_paused = True



            pause_button.setText("\u25b6 Resume")



            if hasattr(progress_dialog, '_worker') and progress_dialog._worker:



                progress_dialog._worker._is_paused = True



            log_text.append("Paused - Click 'Resume' to continue...")







    def _on_embedding_finished(self, progress_dialog, processed, errors, skipped, still_failed_count, note_count):



        """Handle embedding completion"""



        # Invalidate embeddings file cache so next search loads the updated file



        global _embeddings_file_cache, _embeddings_file_cache_path, _embeddings_file_cache_time



        _embeddings_file_cache = None



        _embeddings_file_cache_path = None



        _embeddings_file_cache_time = 0



        status_label = progress_dialog._status_label



        log_text = progress_dialog._log_text



        close_button = progress_dialog._close_button







        status_label.setText(
            f"\u2705 Completed! New embeddings: {processed:,}, already present: {skipped:,} ({errors} errors)"
        )



        log_text.append(f"\n\u2705 Embedding generation complete!")



        log_text.append(f"New embeddings created: {processed:,} notes")



        if skipped > 0:



            log_text.append(f"Skipped (already had embeddings): {skipped:,} notes")



        if errors > 0:



            log_text.append(f"Errors: {errors}")



        if still_failed_count > 0:



            log_text.append(f"Warning: {format_partial_failure_progress(still_failed_count)}")







        # Clear checkpoint only when no notes are still missing (so next run is full; missed ones get retried)



        if still_failed_count == 0:



            clear_checkpoint()







        close_button.setEnabled(True)



        message = (
            "Embedding generation complete!\n"
            f"New embeddings created: {processed:,} notes\n"
            f"Already present: {skipped:,} notes\n"
            f"Errors: {errors}"
        )



        if still_failed_count > 0:



            message += f"\n\nWarning: {format_partial_failure_completion(still_failed_count)}"



        showInfo(message)







    def _on_embedding_error(self, progress_dialog, error_msg):



        """Handle embedding error"""



        status_label = progress_dialog._status_label



        log_text = progress_dialog._log_text



        close_button = progress_dialog._close_button







        status_label.setText(f"\u274c Error: {error_msg}")



        log_text.append(f"\u274c Error: {error_msg}")



        close_button.setEnabled(True)



        showInfo(f"Error during embedding generation: {error_msg}")







# END OF PART 1 - Continue to PART 2

# PART 2 OF 3 - Continue from PART 1



from aqt.qt import *

# Use pyqtSignal directly as it is the most stable across Anki versions

try:

    from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

except ImportError:

    from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot



def show_search_dialog():

    global _ai_search_dialog_instance



    # Re-entrancy guard: Prevent click-bounces or double-triggers

    if hasattr(mw, "_is_spawning_dialog") and mw._is_spawning_dialog:

        return

    mw._is_spawning_dialog = True



    try:

        toggle_sidebar_visibility(False)



        if _ai_search_dialog_instance:

            try:

                if _ai_search_dialog_instance.isVisible():

                    _ai_search_dialog_instance.raise_()

                    _ai_search_dialog_instance.activateWindow()

                    return

                _ai_search_dialog_instance.show()

                return

            except (RuntimeError, AttributeError):

                _ai_search_dialog_instance = None



        _ai_search_dialog_instance = AISearchDialog(mw)

        _ai_search_dialog_instance.finished.connect(_on_search_dialog_closed)

        _ai_search_dialog_instance.show()

    finally:

        # Release the lock after a safe debouncing period

        QTimer.singleShot(400, lambda: setattr(mw, "_is_spawning_dialog", False))



def _on_search_dialog_closed():

    global _ai_search_dialog_instance

    _ai_search_dialog_instance = None

    # Recovery delay: Ensure window handles are released before restoring drawer

    QTimer.singleShot(150, lambda: toggle_sidebar_visibility(True))



def show_settings_dialog(open_to_embeddings=False, auto_start_indexing=False):

    toggle_sidebar_visibility(False)



    dialog = SettingsDialog(mw, open_to_embeddings=open_to_embeddings)

    dialog.setWindowModality(Qt.WindowModality.NonModal)

    # Ensure sidebar returns when settings close

    dialog.finished.connect(lambda _: toggle_sidebar_visibility(True))

    dialog.show()

    dialog.raise_()

    dialog.activateWindow()



    if auto_start_indexing:

        # Trigger the indexing process immediately

        QTimer.singleShot(500, lambda: dialog._create_or_update_embeddings())







def show_debug_log():



    try:



        addon_dir = os.path.dirname(__file__)



        log_file = os.path.join(addon_dir, "debug_log.txt")



        if os.path.exists(log_file):



            if os.name == 'nt':



                os.startfile(log_file)



            else:



                import subprocess



                subprocess.call(['open' if sys.platform == 'darwin' else 'xdg-open', log_file])



        else:



            showInfo("Debug log file not found. Try using the add-on first.")



    except Exception as e:



        showInfo(f"Error opening log file: {e}")







def check_vc_redistributables():



    """Check if Visual C++ Redistributables are installed"""



    import os



    import winreg







    if os.name != 'nt':  # Not Windows



        return True  # Assume OK on non-Windows







    try:



        # Check for Visual C++ 2015-2022 Redistributables (x64)



        # They're registered in the Windows registry



        vc_versions = [



            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"),



            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"),



            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x86"),



            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x86"),



        ]







        # Also check for newer versions (2017-2022)



        for major_version in range(14, 18):  # 14.0 to 17.0



            vc_versions.extend([



                (winreg.HKEY_LOCAL_MACHINE, f"SOFTWARE\\Microsoft\\VisualStudio\\{major_version}.0\\VC\\Runtimes\\x64"),



                (winreg.HKEY_LOCAL_MACHINE, f"SOFTWARE\\WOW6432Node\\Microsoft\\VisualStudio\\{major_version}.0\\VC\\Runtimes\\x64"),



            ])







        # Check if any version is installed - need to actually read a value to verify



        found_any = False



        for hkey, key_path in vc_versions:



            try:



                with winreg.OpenKey(hkey, key_path) as key:



                    # Try to read a value to ensure the key is valid



                    try:



                        version = winreg.QueryValueEx(key, "Version")[0]



                        if version:  # If we got a version, it's installed



                            found_any = True



                            log_debug(f"Found VC++ Redistributables: {key_path}, Version: {version}")



                            break



                    except (FileNotFoundError, OSError):



                        # Key exists but no Version value - still might be installed



                        # Check for other indicators



                        try:



                            # Try to enumerate values (with safety limit to prevent infinite loop)



                            i = 0



                            max_iterations = 1000  # Safety limit



                            while i < max_iterations:



                                try:



                                    name, value, _ = winreg.EnumValue(key, i)



                                    if name and value:



                                        found_any = True



                                        break



                                    i += 1



                                except OSError:



                                    break



                            if found_any:



                                break



                        except:



                            pass



            except (FileNotFoundError, OSError):



                continue







        if found_any:



            return True



        return False  # No VC++ redistributables found



    except Exception as e:



        log_debug(f"Error checking VC++ redistributables: {e}")



        # If we can't check, return None (unknown) instead of assuming True



        return None  # Unknown status







def install_vc_redistributables():



    """Download and install Visual C++ Redistributables"""



    import os



    import sys



    import urllib.request



    import subprocess



    import tempfile







    if os.name != 'nt':  # Not Windows



        showInfo("Visual C++ Redistributables are only needed on Windows.")



        return False







    vc_redist_url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"



    vc_redist_filename = "vc_redist.x64.exe"







    try:



        # Check if already installed - but be more thorough



        vc_status = check_vc_redistributables()



        if vc_status is True:



            # Double-check by trying to actually use a DLL that requires VC++



            # If PyTorch is failing, VC++ might not actually be working



            reply = QMessageBox.question(



                mw,



                "VC++ Redistributables Check",



                "VC++ Redistributables appear to be installed according to the registry.\n\n"



                "However, if you're still experiencing PyTorch DLL errors, they may not be working correctly.\n\n"



                "Options:\n"



                "1. Reinstall VC++ Redistributables anyway (recommended)\n"



                "2. Use keyword-only search (no PyTorch needed)\n"



                "Reinstall VC++ Redistributables?",



                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,



                QMessageBox.StandardButton.Yes



            )



            if reply != QMessageBox.StandardButton.Yes:



                return False



            # Continue with installation anyway



        elif vc_status is None:



            # Unknown status - proceed with installation



            pass



        # If False, continue with installation







        # Create a progress dialog



        progress_dialog = QDialog(mw)



        progress_dialog.setWindowTitle("Installing Visual C++ Redistributables")



        progress_dialog.setMinimumWidth(500)



        progress_dialog.setMinimumHeight(200)



        progress_dialog.setModal(True)



        layout = QVBoxLayout(progress_dialog)







        status_label = QLabel("Downloading Visual C++ Redistributables...")



        status_label.setWordWrap(True)



        layout.addWidget(status_label)







        progress_bar = QProgressBar()



        progress_bar.setRange(0, 0)  # Indeterminate



        layout.addWidget(progress_bar)







        close_btn = QPushButton("Close")



        close_btn.setEnabled(False)



        layout.addWidget(close_btn)







        progress_dialog.show()



        QApplication.processEvents()







        # Download the installer



        temp_dir = tempfile.gettempdir()



        installer_path = os.path.join(temp_dir, vc_redist_filename)







        def download_installer():



            try:



                status_label.setText("Downloading Visual C++ Redistributables installer...\nThis may take a minute.")



                QApplication.processEvents()







                urllib.request.urlretrieve(vc_redist_url, installer_path)







                status_label.setText("Download complete. Launching installer...\n\nYou may need to grant administrator privileges.")



                QApplication.processEvents()







                # Launch the installer



                # /quiet = silent install, /norestart = don't restart



                # /passive = show progress but no user interaction needed



                subprocess.Popen([installer_path, "/passive", "/norestart"], shell=True)







                status_label.setText("\u2705 Installer launched!\n\nPlease follow the installation wizard.\nAfter installation completes, restart Anki.")



                close_btn.setEnabled(True)



                progress_bar.setRange(0, 100)



                progress_bar.setValue(100)







                log_debug("VC++ Redistributables installer launched successfully")



                return True



            except Exception as e:



                status_label.setText(f"\xe2\x9d\u0152 Error: {str(e)}\n\nYou can manually download and install from:\n{vc_redist_url}")



                close_btn.setEnabled(True)



                log_debug(f"Error installing VC++ redistributables: {e}")



                return False







        # Run download in a thread to avoid blocking



        import threading



        thread = threading.Thread(target=download_installer, daemon=True)



        thread.start()







        # Don't wait for thread - let user close dialog when ready



        close_btn.clicked.connect(progress_dialog.close)







        return True



    except Exception as e:



        error_msg = (



            f"Error preparing VC++ Redistributables installation: {str(e)}\n\n"



            f"Please manually download and install from:\n{vc_redist_url}\n\n"



            "After installation, restart Anki."



        )



        showInfo(error_msg)



        log_debug(f"Error in install_vc_redistributables: {e}")



        return False







def get_pytorch_dll_error_guidance():



    """Get guidance message for PyTorch DLL loading errors"""



    import sys



    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"







    vc_status = check_vc_redistributables()



    vc_message = ""



    if vc_status is False:



        vc_message = "\n\xe2\u0161\xa0\ufe0f Visual C++ Redistributables appear to be MISSING!\n   Click 'Install VC++ Redistributables' button below.\n\n"



    elif vc_status is None:



        vc_message = "\n\xe2\u0161\xa0\ufe0f Could not verify Visual C++ Redistributables installation.\n   You may need to install them manually.\n\n"







    guidance = (



        "PyTorch DLL Loading Error Detected\n\n"



        f"Python version: {python_version}\n\n"



        f"{vc_message}"



        "Common causes and solutions:\n\n"



        "1. Missing Visual C++ Redistributables:\n"



        "   - Click 'Install VC++ Redistributables' button below\n"



        "   - Or download manually from:\n"



        "   - https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"



        "2. Python 3.13 compatibility:\n"



        "   - Python 3.13 is very new and PyTorch may not have full support yet\n"



        "   - Try reinstalling PyTorch with CPU-only version:\n"



        "   - Use 'Fix PyTorch DLL Issue' button below\n\n"



        "3. Corrupted installation:\n"



        "   - Try: pip uninstall sentence-transformers torch\n"



        "   - Then reinstall using 'Install Dependencies' in settings\n\n"



        "4. Alternative: Use Anki with Python 3.11 or 3.12 for better compatibility"



    )



    return guidance







def _patch_colorama_for_transformers():



    """Patch colorama ErrorHandler to add flush attribute for transformers compatibility.



    This is a wrapper that ensures the patch is applied (the actual patch runs at module load)."""



    # The actual patching happens at module load time via _patch_colorama_early()



    # This function is kept for backward compatibility and to ensure patch is applied



    # if called before module initialization completes



    try:



        _patch_colorama_early()



    except:



        pass  # Silently fail







def check_dependency_installed(package_name):



    """Check if a Python package is installed"""



    try:



        # Patch colorama before importing sentence_transformers to avoid AttributeError



        if 'sentence_transformers' in package_name or 'transformers' in package_name:



            _patch_colorama_for_transformers()



            _ensure_stderr_patched()



        __import__(package_name.replace('-', '_'))



        return True



    except (ImportError, OSError, ModuleNotFoundError, AttributeError, Exception) as e:



        # OSError can occur when PyTorch DLLs fail to load (e.g., missing Visual C++ Redistributables)



        # ModuleNotFoundError is a subclass of ImportError but we catch it explicitly for clarity



        # AttributeError can occur due to library compatibility issues (e.g., colorama/transformers)



        if isinstance(e, OSError) and 'torch' in str(e).lower():



            log_debug(f"PyTorch DLL error detected: {e}")



        elif isinstance(e, AttributeError):



            log_debug(f"AttributeError during import (likely compatibility issue): {e}")



        return False







def _resolve_external_python_exe(python_path):



    """Resolve 'Python for Cross-Encoder' path to python executable. Returns None if invalid."""



    import os



    path = (python_path or "").strip()



    if not path:



        return None



    if os.path.isfile(path):



        return path



    if os.path.isdir(path):



        exe = os.path.join(path, "python.exe")



        if os.path.isfile(exe):



            return exe



        exe = os.path.join(path, "python")



        if os.path.isfile(exe):



            return exe



    return None











def try_alternative_pytorch_install():



    """Try alternative PyTorch installation methods"""



    import sys



    import subprocess







    methods = [



        {



            "name": "Method 1: PyTorch 2.0.1 (Older, more stable)",



            "command": [sys.executable, "-m", "pip", "install", "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2", "--index-url", "https://download.pytorch.org/whl/cpu"]



        },



        {



            "name": "Method 2: PyTorch 2.1.0 (Mid-version)",



            "command": [sys.executable, "-m", "pip", "install", "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0", "--index-url", "https://download.pytorch.org/whl/cpu"]



        },



        {



            "name": "Method 3: Latest PyTorch (Current default)",



            "command": [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]



        },



        {



            "name": "Method 4: PyTorch without CUDA (pip default)",



            "command": [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]



        }



    ]







    dialog = QDialog(mw)



    dialog.setWindowTitle("Try Alternative PyTorch Installation")



    dialog.setMinimumWidth(500)



    layout = QVBoxLayout(dialog)







    info_label = QLabel(



        "If the standard installation failed, try these alternative methods:\n\n"



        "Select a method to try:"



    )



    info_label.setWordWrap(True)



    layout.addWidget(info_label)







    method_combo = QComboBox()



    for method in methods:



        method_combo.addItem(method["name"])



    layout.addWidget(method_combo)







    button_layout = QHBoxLayout()



    try_btn = QPushButton("Try This Method")



    cancel_btn = QPushButton("Cancel")



    button_layout.addWidget(try_btn)



    button_layout.addWidget(cancel_btn)



    layout.addLayout(button_layout)







    def try_method():



        selected_idx = method_combo.currentIndex()



        method = methods[selected_idx]



        dialog.close()







        # Show progress



        progress = QDialog(mw)



        progress.setWindowTitle("Installing PyTorch")



        progress_layout = QVBoxLayout(progress)



        status = QLabel(f"Trying: {method['name']}\n\nThis may take several minutes...")



        status.setWordWrap(True)



        progress_layout.addWidget(status)



        progress.show()



        QApplication.processEvents()







        try:



            result = subprocess.run(



                method["command"],



                capture_output=True,



                text=True,



                timeout=600



            )







            if result.returncode == 0:



                status.setText("\u2705 Installation successful!\n\nTesting import...")



                QApplication.processEvents()







                # Test import



                try:



                    _patch_colorama_for_transformers()



                    _ensure_stderr_patched()



                    import torch



                    status.setText(f"\u2705 Success! PyTorch {torch.__version__} installed and working.\n\nNow install sentence-transformers.")



                    showInfo(f"PyTorch {torch.__version__} installed successfully!\n\nNow click 'Install Dependencies' to install sentence-transformers.")



                except Exception as e:



                    status.setText(f"\xe2\u0161\xa0\ufe0f Installed but import failed: {e}\n\nTry installing VC++ Redistributables.")



                    showInfo(f"PyTorch installed but import failed: {e}\n\nTry installing VC++ Redistributables first.")



            else:



                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"



                status.setText(f"\xe2\x9d\u0152 Installation failed:\n{error_msg}")



                showInfo(f"Installation failed. Try another method or install VC++ Redistributables.")



        except Exception as e:



            status.setText(f"Γ¥î Error: {e}")



            showInfo(f"Error: {e}")







        close_btn = QPushButton("Close")



        close_btn.clicked.connect(progress.close)



        progress_layout.addWidget(close_btn)







    try_btn.clicked.connect(try_method)



    cancel_btn.clicked.connect(dialog.close)







    dialog.exec()







def fix_pytorch_dll_issue():



    """Fix PyTorch DLL issues by reinstalling with CPU-only version"""



    import sys



    import subprocess







    reply = QMessageBox.question(



        mw,



        "Fix PyTorch DLL Issue",



        "This will reinstall PyTorch with a CPU-only version that's more compatible.\n\n"



        "Steps:\n"



        "1. Uninstall existing PyTorch packages\n"



        "2. Install CPU-only PyTorch from official repository\n"



        "3. Reinstall sentence-transformers\n\n"



        "\xe2\u0161\xa0\ufe0f IMPORTANT: If this fails, you may need to:\n"



        "- Install Visual C++ Redistributables first\n"



        "- Try alternative PyTorch versions\n"



        "- Use keyword-only search (no embeddings needed)\n\n"



        "This may take a few minutes. Continue?",



        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,



        QMessageBox.StandardButton.Yes



    )







    if reply != QMessageBox.StandardButton.Yes:



        return







    # Create progress dialog



    progress_dialog = QDialog(mw)



    progress_dialog.setWindowTitle("Fixing PyTorch DLL Issue")



    progress_dialog.setMinimumWidth(600)



    progress_dialog.setMinimumHeight(500)



    progress_dialog.setModal(False)



    progress_layout = QVBoxLayout(progress_dialog)







    status_label = QLabel("Preparing...")



    status_label.setWordWrap(True)



    progress_layout.addWidget(status_label)







    log_text = QTextEdit()



    log_text.setReadOnly(True)



    log_text.setMaximumHeight(300)



    log_text.setFont(QFont("Courier", 9))



    log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")



    progress_layout.addWidget(log_text)







    close_button = QPushButton("Close")



    close_button.setEnabled(False)



    close_button.clicked.connect(progress_dialog.close)



    progress_layout.addWidget(close_button)







    progress_dialog.show()



    QApplication.processEvents()







    def log(msg):



        log_text.append(msg)



        log_text.verticalScrollBar().setValue(log_text.verticalScrollBar().maximum())



        QApplication.processEvents()



        log_debug(msg)







    try:



        # Step 1: Uninstall PyTorch packages



        status_label.setText("Step 1/3: Uninstalling existing PyTorch packages...")



        log("Uninstalling torch, torchvision, torchaudio...")







        packages_to_uninstall = ['torch', 'torchvision', 'torchaudio', 'sentence-transformers']



        for pkg in packages_to_uninstall:



            try:



                result = subprocess.run(



                    [sys.executable, "-m", "pip", "uninstall", pkg, "-y"],



                    capture_output=True,



                    text=True,



                    timeout=120



                )



                if result.returncode == 0:



                    log(f"\u2705 Uninstalled {pkg}")



                else:



                    log(f"\xe2\u0161\xa0\ufe0f {pkg} may not have been installed")



            except Exception as e:



                log(f"\xe2\u0161\xa0\ufe0f Error uninstalling {pkg}: {e}")







        # Step 2: Install CPU-only PyTorch



        status_label.setText("Step 2/3: Installing CPU-only PyTorch...")



        log("Installing PyTorch CPU-only version from official repository...")



        log("This may take several minutes...")







        result = subprocess.run(



            [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",



             "--index-url", "https://download.pytorch.org/whl/cpu"],



            capture_output=True,



            text=True,



            timeout=600



        )







        if result.returncode == 0:



            log("\u2705 PyTorch CPU-only installed successfully")



        else:



            log(f"Γ¥î Error installing PyTorch:")



            for line in result.stderr.split('\n')[-10:]:



                if line.strip():



                    log(line)



            raise Exception("PyTorch installation failed")







        # Step 3: Reinstall sentence-transformers



        status_label.setText("Step 3/3: Reinstalling sentence-transformers...")



        log("Installing sentence-transformers...")







        result = subprocess.run(



            [sys.executable, "-m", "pip", "install", "sentence-transformers"],



            capture_output=True,



            text=True,



            timeout=300



        )







        if result.returncode == 0:



            log("\u2705 sentence-transformers installed successfully")



        else:



            log(f"Γ¥î Error installing sentence-transformers:")



            for line in result.stderr.split('\n')[-10:]:



                if line.strip():



                    log(line)



            raise Exception("sentence-transformers installation failed")







        # Verify installation



        status_label.setText("Verifying installation...")



        log("Testing import...")







        try:



            _patch_colorama_for_transformers()



            _ensure_stderr_patched()



            from sentence_transformers import SentenceTransformer



            log("\u2705 Import test successful!")



            status_label.setText("\u2705 Fix completed successfully!")



            status_label.setStyleSheet("color: green; font-weight: bold;")



            showInfo("PyTorch DLL issue fixed! You may need to restart Anki for changes to take effect.")



        except Exception as e:



            log(f"\xe2\x9d\u0152 Import test failed: {e}")



            status_label.setText("\xe2\u0161\xa0\ufe0f Installation completed but import test failed")



            status_label.setStyleSheet("color: orange; font-weight: bold;")







            # Add helpful buttons



            button_layout = QHBoxLayout()







            try_alt_btn = QPushButton("Try Alternative PyTorch Version")



            try_alt_btn.clicked.connect(lambda: (progress_dialog.close(), try_alternative_pytorch_install()))



            button_layout.addWidget(try_alt_btn)







            vc_btn = QPushButton("Install VC++ Redistributables")



            vc_btn.clicked.connect(lambda: (progress_dialog.close(), install_vc_redistributables()))



            button_layout.addWidget(vc_btn)







            use_keyword_btn = QPushButton("Use Keyword-Only Search (No PyTorch)")



            use_keyword_btn.clicked.connect(lambda: (



                progress_dialog.close(),



                showInfo("You can use the addon in keyword-only mode!\n\n"



                        "1. Go to Settings\n"



                        "2. Change 'Search Method' to 'Keyword Only'\n"



                        "3. The addon will work without embeddings.")



            ))



            button_layout.addWidget(use_keyword_btn)







            progress_layout.addLayout(button_layout)







            error_msg = (



                f"Installation completed but verification failed: {e}\n\n"



                "Options:\n"



                "1. Try alternative PyTorch version (button above)\n"



                "2. Install VC++ Redistributables (button above)\n"



                "3. Use keyword-only search mode (no embeddings needed)\n"



                "4. Check the log for details"



            )



            showInfo(error_msg)







    except Exception as e:



        log(f"\xe2\x9d\u0152 Error: {e}")



        status_label.setText(f"\xe2\x9d\u0152 Error: {e}")



        status_label.setStyleSheet("color: red; font-weight: bold;")



        showInfo(f"Error fixing PyTorch: {e}")



    finally:



        close_button.setEnabled(True)







def _check_sentence_transformers_installed_subprocess():



    """Check if sentence-transformers is usable in Anki's Python via subprocess (avoids in-process import failures on Python 3.13)."""



    try:



        import subprocess



        import sys



        result = subprocess.run(



            [sys.executable, "-c", "from sentence_transformers import CrossEncoder; print('ok')"],



            capture_output=True, text=True, timeout=15,



            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0



        )



        return result.returncode == 0 and 'ok' in (result.stdout or '')



    except Exception:



        return False











def install_dependencies(python_exe=None):



    """Show manual install instructions for optional dependencies (no auto pip install).



    python_exe: None = Anki's Python; else path to external python.exe for Cross-Encoder."""



    import sys







    if python_exe:



        target_python = python_exe



        target_label = "Python for Cross-Encoder (from Settings)"



    else:



        target_python = sys.executable



        target_label = "Anki's Python"







    # Check if already installed



    try:



        import subprocess



        result = subprocess.run(



            [target_python, "-c", "from sentence_transformers import CrossEncoder; print('ok')"],



            capture_output=True, text=True, timeout=15,



            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0



        )



        if result.returncode == 0 and 'ok' in (result.stdout or ''):



            showInfo("\u2705 sentence-transformers is already available.\n\nClick 'Check again' in Settings to enable Cross-Encoder.")



            return



    except Exception:



        pass







    pip_cmd = f'"{target_python}" -m pip install sentence-transformers'



    msg = (



        "Optional: Cross-Encoder re-ranking (better retrieval quality)\n\n"



        f"Python executable: {target_python}\n\n"



        "Copy this command and run it in a terminal:\n\n"



        f"  {pip_cmd}\n\n"



        f"Where to run: Use the Python above, or the one set as 'Python for Cross-Encoder' in Settings.\n\n"



        "See config.md in the add-on folder for troubleshooting."



    )



    dlg = QMessageBox(mw)



    dlg.setWindowTitle("Manual Install: sentence-transformers")



    dlg.setText(msg)



    dlg.setIcon(QMessageBox.Icon.Information)



    copy_btn = dlg.addButton("Copy command", QMessageBox.ButtonRole.ActionRole)



    dlg.addButton(QMessageBox.StandardButton.Ok)



    dlg.exec()



    if dlg.clickedButton() == copy_btn:



        QApplication.clipboard().setText(pip_cmd)



        tooltip("Command copied to clipboard")







log_debug("=== Anki Semantic Search Add-on Loaded ===")



log_debug(f"Addon directory: {os.path.dirname(__file__)}")



log_debug(f"Addon folder name: {ADDON_NAME}")







# Note: colorama is already patched at module load time via _patch_colorama_early()







# Add menu items



ai_search_menu = QMenu("\U0001F50D Anki Semantic Search", mw)







# search_action = QAction("Search Notes", mw)



# search_action.triggered.connect(show_search_dialog)



# search_action.setToolTip("Open Anki Semantic Search window")



# search_action.setShortcut("Ctrl+Shift+S")



# mw.form.menuTools.addAction(search_action)







mw.addonManager.setConfigAction(ADDON_NAME, show_settings_dialog)







# Background indexer: re-enabled using QueryOp \u2014 collection access only on main thread via QueryOp; worker thread does API + save only (no mw.col).


