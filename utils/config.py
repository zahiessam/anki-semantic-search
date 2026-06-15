"""Config load/save and helpers."""

# ============================================================================
# Imports
# ============================================================================

import os
import json
from .log import log_debug

# ============================================================================
# Default Configuration
# ============================================================================

# Voyage embedding models: voyage-3-lite is faster with fewer dimensions; voyage-3.5-lite is higher quality
VOYAGE_EMBEDDING_MODELS = ["voyage-3-lite", "voyage-3.5-lite"]
DEFAULT_RERANK_MODEL = "NeuML/biomedbert-base-reranker"
RERANK_TOP_K_DEFAULT = 50
RERANK_TOP_K_MIN = 5
RERANK_TOP_K_MAX = 100
RERANK_TIMEOUT_SECONDS_DEFAULT = 90
RERANK_TIMEOUT_SECONDS_MIN = 15
RERANK_TIMEOUT_SECONDS_MAX = 300
RELEVANCE_THRESHOLD_PERCENT_DEFAULT = 65
RELEVANCE_THRESHOLD_PERCENT_MIN = 20
RELEVANCE_THRESHOLD_PERCENT_MAX = 80
RETRIEVAL_RELEVANCE_FLOOR_PERCENT = 20
# Legacy names kept for older imports/tests while the UI moves to relevance_threshold_percent.
MIN_RELEVANCE_PERCENT_DEFAULT = RELEVANCE_THRESHOLD_PERCENT_DEFAULT
MIN_RELEVANCE_PERCENT_MIN = RELEVANCE_THRESHOLD_PERCENT_MIN
MIN_RELEVANCE_PERCENT_MAX = RELEVANCE_THRESHOLD_PERCENT_MAX
ANALYTICS_MODE_DEFAULT = "compact"
ANALYTICS_MODE_CHOICES = ("compact", "full")
MAX_ANALYTICS_FILES_DEFAULT = 50
MAX_ANALYTICS_FILES_MIN = 1
MAX_ANALYTICS_FILES_MAX = 1000

# Safe defaults for shipped config (no secrets, no machine-specific paths)
DEFAULT_CONFIG = {
    "api_key": "",
    "provider": "ollama",
    "styling": {
        "question_font_size": 16,
        "answer_font_size": 16,
        "notes_font_size": 16,
        "label_font_size": 16,
        "window_width": 1100,
        "window_height": 800,
        "section_spacing": 8,
        "layout_mode": "side_by_side",
        "answer_spacing": "normal",
    },
    "note_type_filter": {
        "enabled_note_types": [],
        "search_all_fields": False,
        "note_type_fields": {},
        "use_first_field_fallback": True,
        "enabled_decks": [],
    },
    "review_ask_ai": {
        "context_source": "embedding_fields",
        "note_type_fields": {},
        "search_all_fields": False,
        "use_first_field_fallback": True,
    },
    "search_config": {
        "search_method": "hybrid",
        "enable_query_expansion": False,
        "enable_agentic_rag": False,
        "enable_profile_memory": True,
        "memory_retrieval_mode": "auto_hybrid",
        "memory_retention_days": 30,
        "memory_max_saved_snippets_per_search": 24,
        "memory_max_retrieved_snippets": 5,
        "memory_embedding_enabled": True,
        "agentic_max_retrieval_passes": 3,
        "agentic_max_subqueries": 6,
        "planner_confidence_threshold": 0.6,
        "agentic_planner_mode": "deterministic_v1",
        "agentic_planner_model": "",
        "agentic_planner_timeout_seconds": 25,
        "agentic_planner_max_tokens": 350,
        "enable_hyde": True,
        "enable_rerank": True,
        "use_context_boost": True,
        "relevance_threshold_percent": RELEVANCE_THRESHOLD_PERCENT_DEFAULT,
        "max_results": 50,
        "context_chars_per_note": 0,
        # Assumed local answer model context window. The add-on uses this to
        # dynamically choose light/balanced/deep note context budgets.
        "local_llm_context_tokens": 12288,
        "enable_context_score_cliff": True,
        "context_score_cliff_threshold": 15.0,
        "context_score_cliff_min_notes": 8,
        "enable_context_anchor_rescue": True,
        "context_score_cliff_anchor_rescue_slots": 3,
        "context_score_cliff_anchor_rescue_similarity_floor": None,
        "enable_rescue_specificity_scoring": True,
        "rescue_specificity_threshold": 0.85,
        "rescue_specificity_max_doc_freq": 4,
        "rescue_specificity_max_weight": 0.5,
        # Dedicated local answer model. Kept separate from local embedding
        # model so independent embedding settings cannot overwrite answers.
        "answer_local_model": "",
        "hybrid_embedding_weight": 40,
        "embedding_engine": "ollama",
        "voyage_api_key": "",
        "voyage_embedding_model": "voyage-3.5-lite",
        "openai_embedding_api_key": "",
        "openai_embedding_model": "text-embedding-3-small",
        "cohere_api_key": "",
        "cohere_embedding_model": "embed-english-v3.0",
        "voyage_batch_size": 16,
        "ollama_base_url": "http://localhost:11434",
        "ollama_embed_model": "nomic-embed-text:latest",
        # Optional separate model for query expansion; falls back to ollama_chat_model when empty
        "ollama_query_expansion_model": "",
        "ollama_batch_size": 16,
        "use_dynamic_batch_size": True,
        "ollama_chat_model": "llama3.2:latest",
        "rerank_model": DEFAULT_RERANK_MODEL,
        "rerank_top_k": RERANK_TOP_K_DEFAULT,
        "rerank_timeout_seconds": RERANK_TIMEOUT_SECONDS_DEFAULT,
        "analytics_enabled": False,
        "analytics_mode": ANALYTICS_MODE_DEFAULT,
        "max_analytics_files": MAX_ANALYTICS_FILES_DEFAULT,
        "rerank_python_path": "",
        "sentence_transformers_path": None,
        "show_only_cited": False,
        # Optional extra stop words for domain-specific generic terms in queries
        # e.g. ["pediatrics", "medicine", "clinical"]
        "extra_stop_words": [],
        # Optional verbose debug logging for search internals (query analysis, scores)
        "verbose_search_debug": False,
        # Retrieval V2 is the permanent retrieval engine.
        "retrieval_version": "v2",
        "keyword_scoring_method": "bm25",
        "enable_mmr_diversity": True,
        "mmr_lambda": 0.75,
        "mmr_similarity_method": "token_jaccard",
        # Optional extra synonym groups for query expansion (no UI). Each group is a list of
        # equivalent terms; if any term in the group appears in the query, the rest are appended.
        # Example: [["warfarin", "coumadin"], ["vitamin k", "phytonadione"]]
        "synonym_overrides": [],
        # Hidden/advanced legacy option: one short LLM call per search to detect generic query terms.
        # Deterministic stop-word and intent-aware filtering is always on without this.
        "use_ai_generic_term_detection": False,
        "embedding_same_as_answer": True,
        "embedding_strategy": "cloud",
        "embedding_cloud_provider": "Voyage AI",
        "embedding_cloud_api_key": "",
        "embedding_local_backend": "ollama",
        "embedding_local_url": "",
        "embedding_local_model": "nomic-embed-text",
    },
    "saved_presets": {},
    "current_preset_name": None,
}


# ============================================================================
# Config File Access
# ============================================================================

def _deep_merge(base, override):
    """Merge override into base recursively. Override values take precedence."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def get_addon_name():
    """Get the addon's module name for config storage."""
    addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.basename(addon_dir)


def get_config_file_path():
    """Get path to config file."""
    addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(addon_dir, "config.json")


def save_config(config):
    """Save config to JSON file using atomic write."""
    try:
        config_path = get_config_file_path()
        log_debug(f"Saving config to: {config_path}")
        if isinstance(config, dict):
            search_config = config.get("search_config")
            if isinstance(search_config, dict):
                normalize_search_threshold_config(search_config, log_warnings=False)
        temp_path = config_path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            if os.path.exists(config_path):
                os.replace(temp_path, config_path)
            else:
                os.rename(temp_path, config_path)
        except Exception as write_err:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise write_err
        log_debug("Config file written successfully")
        return True
    except Exception as e:
        log_debug(f"Error saving config file: {e}")
        return False


def load_config():
    """Load config from JSON file, merged with safe defaults. User config persists in addon folder."""
    try:
        config_path = get_config_file_path()
        if not os.path.exists(config_path):
            log_debug(f"Config file does not exist: {config_path}, using defaults")
            merged = _deep_merge(DEFAULT_CONFIG, {})
            search_config = dict(merged.get("search_config") or {})
            normalize_search_threshold_config(search_config, log_warnings=False)
            normalize_analytics_config(search_config, {}, log_warnings=False)
            merged["search_config"] = search_config
            return merged
        with open(config_path, "r", encoding="utf-8-sig") as f:
            file_config = json.load(f)
        for typo in ("retrieval_versoin", "retrival_version"):
            if typo in file_config:
                log_debug(f"Retrieval config warning: ignoring top-level key {typo!r}; put 'retrieval_version' inside search_config")
        merged = _deep_merge(DEFAULT_CONFIG, file_config)
        file_search_config = file_config.get("search_config") or {}
        search_config = dict(merged.get("search_config") or {})
        _protect_answer_model_from_embedding_migration(search_config)
        normalize_search_threshold_config(search_config, log_warnings=True)
        normalize_rerank_config(search_config, log_warnings=True)
        normalize_retrieval_config(search_config, log_warnings=True)
        normalize_analytics_config(search_config, file_search_config, log_warnings=True)
        merged["search_config"] = search_config
        log_debug(f"Config loaded from file (merged with defaults)")
        return merged
    except Exception as e:
        log_debug(f"Error loading config file: {e}")
        merged = _deep_merge(DEFAULT_CONFIG, {})
        search_config = dict(merged.get("search_config") or {})
        normalize_search_threshold_config(search_config, log_warnings=False)
        normalize_analytics_config(search_config, {}, log_warnings=False)
        merged["search_config"] = search_config
        return merged


def _analytics_log_dir_exists():
    try:
        from .paths import get_checkpoint_path

        base = os.path.dirname(get_checkpoint_path())
        return (
            os.path.isdir(os.path.join(base, "search_analytics"))
            or os.path.isdir(os.path.join(base, "search_research"))
        )
    except Exception:
        return False


def _looks_like_embedding_model(model):
    value = (model or "").strip().lower()
    if not value:
        return False
    embedding_markers = (
        "embed",
        "embedding",
        "bge-",
        "e5-",
        "nomic-embed",
        "medembed",
        "minilm",
    )
    return any(marker in value for marker in embedding_markers)


def _protect_answer_model_from_embedding_migration(search_config):
    """Keep legacy local_llm_model from masquerading as the answer model.

    Older code used local_llm_model as the runtime field for local OpenAI-compatible
    embeddings. If independent local embeddings were selected, that could leave the
    embedding model in the answer model field after restart.
    """
    if not isinstance(search_config, dict):
        return
    if search_config.get("answer_local_model"):
        return
    if search_config.get("embedding_same_as_answer", True):
        return
    if (search_config.get("embedding_strategy") or "").strip().lower() != "local":
        return

    answer_model = (search_config.get("local_llm_model") or "").strip()
    embedding_model = (search_config.get("embedding_local_model") or "").strip()
    if answer_model and answer_model == embedding_model and _looks_like_embedding_model(answer_model):
        search_config["local_llm_model"] = ""


def get_config_value(config, key, default):
    """Get config value with default fallback."""
    if not config:
        return default
    return config.get(key, default)


def clamp_relevance_threshold_percent(value, default=RELEVANCE_THRESHOLD_PERCENT_DEFAULT):
    try:
        value = int(value)
    except Exception:
        value = default
    return max(RELEVANCE_THRESHOLD_PERCENT_MIN, min(RELEVANCE_THRESHOLD_PERCENT_MAX, value))


def clamp_min_relevance_percent(value, default=MIN_RELEVANCE_PERCENT_DEFAULT):
    """Backward-compatible alias for the old user-facing setting."""
    return clamp_relevance_threshold_percent(value, default)


def normalize_search_threshold_config(search_config, log_warnings=False):
    """Normalize the single user-facing relevance threshold in-place."""
    sc = search_config if isinstance(search_config, dict) else {}
    source_key = "relevance_threshold_percent"
    if source_key in sc:
        original = sc.get(source_key, RELEVANCE_THRESHOLD_PERCENT_DEFAULT)
    else:
        original = sc.get("min_relevance_percent", RELEVANCE_THRESHOLD_PERCENT_DEFAULT)
    clamped = clamp_relevance_threshold_percent(original)
    sc[source_key] = clamped
    sc.pop("min_relevance_percent", None)
    sc.pop("sensitivity_percent", None)
    sc.pop("relevance_mode", None)
    if log_warnings:
        try:
            original_int = int(original)
        except Exception:
            original_int = RELEVANCE_THRESHOLD_PERCENT_DEFAULT
        if original_int != clamped:
            log_debug(
                "Search threshold config warning: "
                f"{source_key}={original!r} outside "
                f"{RELEVANCE_THRESHOLD_PERCENT_MIN}..{RELEVANCE_THRESHOLD_PERCENT_MAX}; "
                f"clamped to {clamped!r}"
            )
    return sc


# ============================================================================
# Rerank Configuration
# ============================================================================

def normalize_rerank_config(search_config, log_warnings=False):
    """Normalize Cross-Encoder rerank settings in-place and return them."""
    sc = search_config if isinstance(search_config, dict) else {}
    warnings = []

    model = (sc.get("rerank_model") or "").strip()
    if not model:
        model = DEFAULT_RERANK_MODEL
    sc["rerank_model"] = model

    try:
        top_k = int(sc.get("rerank_top_k", RERANK_TOP_K_DEFAULT))
    except Exception:
        warnings.append(f"rerank_top_k must be an integer; using {RERANK_TOP_K_DEFAULT!r}")
        top_k = RERANK_TOP_K_DEFAULT
    clamped = max(RERANK_TOP_K_MIN, min(RERANK_TOP_K_MAX, top_k))
    if clamped != top_k:
        warnings.append(
            f"rerank_top_k={top_k!r} outside {RERANK_TOP_K_MIN}..{RERANK_TOP_K_MAX}; clamped to {clamped!r}"
        )
    sc["rerank_top_k"] = clamped

    try:
        timeout_seconds = int(sc.get("rerank_timeout_seconds", RERANK_TIMEOUT_SECONDS_DEFAULT))
    except Exception:
        warnings.append(
            f"rerank_timeout_seconds must be an integer; using {RERANK_TIMEOUT_SECONDS_DEFAULT!r}"
        )
        timeout_seconds = RERANK_TIMEOUT_SECONDS_DEFAULT
    clamped_timeout = max(
        RERANK_TIMEOUT_SECONDS_MIN,
        min(RERANK_TIMEOUT_SECONDS_MAX, timeout_seconds),
    )
    if clamped_timeout != timeout_seconds:
        warnings.append(
            f"rerank_timeout_seconds={timeout_seconds!r} outside "
            f"{RERANK_TIMEOUT_SECONDS_MIN}..{RERANK_TIMEOUT_SECONDS_MAX}; "
            f"clamped to {clamped_timeout!r}"
        )
    sc["rerank_timeout_seconds"] = clamped_timeout

    if log_warnings:
        for warning in warnings:
            log_debug(f"Rerank config warning: {warning}")
    return sc


def get_rerank_config(config_or_search_config):
    """Return normalized rerank config from full config or search_config."""
    source = config_or_search_config or {}
    if isinstance(source, dict) and "search_config" in source:
        source = source.get("search_config") or {}
    sc = dict(source or {})
    return normalize_rerank_config(sc, log_warnings=False)


# ============================================================================
# Search Analytics Logging Configuration
# ============================================================================

def normalize_analytics_config(search_config, file_search_config=None, log_warnings=False):
    """Normalize local search analytics logging settings in-place and return them."""
    sc = search_config if isinstance(search_config, dict) else {}
    file_sc = file_search_config if isinstance(file_search_config, dict) else sc
    warnings = []

    if "analytics_enabled" not in sc and "research_enabled" in sc:
        sc["analytics_enabled"] = sc.get("research_enabled")
    if "analytics_mode" not in sc and "research_mode" in sc:
        sc["analytics_mode"] = sc.get("research_mode")
    if "max_analytics_files" not in sc and "max_research_files" in sc:
        sc["max_analytics_files"] = sc.get("max_research_files")

    if "analytics_enabled" not in file_sc and "research_enabled" not in file_sc:
        sc["analytics_enabled"] = _analytics_log_dir_exists()
    else:
        sc["analytics_enabled"] = _coerce_bool(
            sc.get("analytics_enabled", False),
            False,
            "analytics_enabled",
            warnings,
        )

    mode = (sc.get("analytics_mode") or ANALYTICS_MODE_DEFAULT).strip().lower()
    if mode not in ANALYTICS_MODE_CHOICES:
        warnings.append(f"analytics_mode={mode!r} is invalid; using {ANALYTICS_MODE_DEFAULT!r}")
        mode = ANALYTICS_MODE_DEFAULT
    sc["analytics_mode"] = mode

    sc["max_analytics_files"] = _coerce_int(
        sc.get("max_analytics_files", MAX_ANALYTICS_FILES_DEFAULT),
        MAX_ANALYTICS_FILES_DEFAULT,
        MAX_ANALYTICS_FILES_MIN,
        MAX_ANALYTICS_FILES_MAX,
        "max_analytics_files",
        warnings,
    )

    if log_warnings:
        for warning in warnings:
            log_debug(f"Analytics config warning: {warning}")
    return sc


def get_analytics_config(config_or_search_config):
    """Return normalized search analytics logging config from full config or search_config."""
    source = config_or_search_config or {}
    if isinstance(source, dict) and "search_config" in source:
        source = source.get("search_config") or {}
    sc = dict(source or {})
    return normalize_analytics_config(sc, sc, log_warnings=False)


def normalize_research_config(search_config, file_search_config=None, log_warnings=False):
    """Backward-compatible alias for older callers."""
    return normalize_analytics_config(search_config, file_search_config, log_warnings)


def get_research_config(config_or_search_config):
    """Backward-compatible alias for older callers."""
    return get_analytics_config(config_or_search_config)


# ============================================================================
# Retrieval V2 Configuration
# ============================================================================

def _coerce_bool(value, default, key, warnings):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    warnings.append(f"{key} must be true/false; using {default!r}")
    return default


def _coerce_float(value, default, min_value, max_value, key, warnings):
    try:
        out = float(value)
    except Exception:
        warnings.append(f"{key} must be a number; using {default!r}")
        return default
    clamped = max(min_value, min(max_value, out))
    if clamped != out:
        warnings.append(f"{key}={out!r} outside {min_value}..{max_value}; clamped to {clamped!r}")
    return clamped


def _coerce_int(value, default, min_value, max_value, key, warnings):
    try:
        out = int(value)
    except Exception:
        warnings.append(f"{key} must be an integer; using {default!r}")
        return default
    clamped = max(min_value, min(max_value, out))
    if clamped != out:
        warnings.append(f"{key}={out!r} outside {min_value}..{max_value}; clamped to {clamped!r}")
    return clamped


def normalize_retrieval_config(search_config, log_warnings=False):
    """Validate retrieval knobs and return a normalized copy."""
    sc = dict(search_config or {})
    warnings = []

    for typo in ("retrieval_versoin", "retrival_version"):
        if typo in sc:
            warnings.append(f"Ignoring unknown retrieval config key {typo!r}; did you mean 'retrieval_version'?")

    # Retrieval V2 is now mandatory.
    sc["retrieval_version"] = "v2"

    scorer = (sc.get("keyword_scoring_method") or "bm25").strip().lower()
    if scorer != "bm25":
        # TF-IDF is removed; force BM25
        scorer = "bm25"
    sc["keyword_scoring_method"] = scorer

    sc["enable_mmr_diversity"] = _coerce_bool(
        sc.get("enable_mmr_diversity", True), True, "enable_mmr_diversity", warnings
    )
    sc["mmr_lambda"] = _coerce_float(sc.get("mmr_lambda", 0.75), 0.75, 0.0, 1.0, "mmr_lambda", warnings)
    sc.pop("mmr_candidate_pool", None)
    method = (sc.get("mmr_similarity_method") or "token_jaccard").strip().lower()
    if method != "token_jaccard":
        warnings.append(f"mmr_similarity_method={method!r} is invalid; using 'token_jaccard'")
        method = "token_jaccard"
    sc["mmr_similarity_method"] = method

    if log_warnings:
        for warning in warnings:
            log_debug(f"Retrieval config warning: {warning}")
    return sc


def get_retrieval_config(config_or_search_config):
    """Return normalized retrieval config from full config or search_config."""
    source = config_or_search_config or {}
    if isinstance(source, dict) and "search_config" in source:
        source = source.get("search_config") or {}
    return normalize_retrieval_config(source, log_warnings=False)


def is_retrieval_v2(config_or_search_config):
    return True


# ============================================================================
# Effective Embedding Configuration
# ============================================================================

# explanation: normalizes display names from the embedding provider dropdown into backend ids.
def _normalize_embedding_cloud_provider(provider):
    value = (provider or "Voyage AI").strip().lower()
    if "openai" in value:
        return "openai"
    if "cohere" in value:
        return "cohere"
    return "voyage"


# explanation: maps the answer provider settings to an embedding backend only when supported.
def _embedding_config_from_answer_provider(config, sc):
    provider = (config.get("provider") or "").strip().lower()
    answer_key = (config.get("api_key") or "").strip()
    effective = dict(sc)

    if provider == "ollama":
        effective["embedding_engine"] = "ollama"
        effective["ollama_base_url"] = (
            sc.get("ollama_base_url")
            or sc.get("local_llm_url")
            or "http://localhost:11434"
        )
        effective["ollama_embed_model"] = (
            sc.get("embedding_local_model")
            or sc.get("ollama_embed_model")
            or sc.get("ollama_chat_model")
            or sc.get("local_llm_model")
            or "nomic-embed-text"
        )
        return effective

    if provider in ("local_openai", "local_server"):
        effective["embedding_engine"] = "local_openai"
        effective["local_llm_url"] = (
            sc.get("embedding_local_url")
            or sc.get("local_llm_url")
            or "http://localhost:1234/v1"
        )
        effective["local_llm_model"] = (
            sc.get("embedding_local_model")
            or sc.get("local_llm_model")
            or sc.get("ollama_embed_model")
            or "text-embedding-3-small"
        )
        return effective

    if provider == "openai":
        effective["embedding_engine"] = "openai"
        effective["openai_embedding_api_key"] = answer_key
        effective["openai_embedding_model"] = (
            sc.get("openai_embedding_model") or "text-embedding-3-small"
        )
        return effective

    effective["embedding_engine"] = "unsupported_same_as_answer"
    return effective


# explanation: returns the correct embedding provider config
# based on whether the user chose "same as answer" or independent.
def get_effective_embedding_config(config: dict) -> dict:
    config = dict(config or {})
    sc = dict(config.get("search_config") or {})

    if sc.get("embedding_same_as_answer", True):
        effective_sc = _embedding_config_from_answer_provider(config, sc)
    else:
        strategy = (sc.get("embedding_strategy") or "cloud").strip().lower()
        effective_sc = dict(sc)

        if strategy == "local":
            local_backend = (sc.get("embedding_local_backend") or "").strip().lower()
            if not local_backend:
                url = (sc.get("embedding_local_url") or sc.get("local_llm_url") or "").lower()
                local_backend = "ollama" if ":11434" in url or (sc.get("embedding_engine") == "ollama") else "custom_openai"

            if local_backend == "ollama":
                effective_sc["embedding_engine"] = "ollama"
                effective_sc["ollama_base_url"] = (
                    sc.get("embedding_local_url")
                    or sc.get("ollama_base_url")
                    or "http://localhost:11434"
                )
                effective_sc["ollama_embed_model"] = (
                    sc.get("embedding_local_model")
                    or sc.get("ollama_embed_model")
                    or "nomic-embed-text"
                )
            else:
                effective_sc["embedding_engine"] = "local_openai"
                effective_sc["local_llm_url"] = (
                    sc.get("embedding_local_url")
                    or sc.get("local_llm_url")
                    or "http://localhost:1234/v1"
                )
                effective_sc["local_llm_model"] = (
                    sc.get("embedding_local_model")
                    or sc.get("local_llm_model")
                    or sc.get("ollama_embed_model")
                    or "text-embedding-3-small"
                )
        else:
            provider_id = _normalize_embedding_cloud_provider(
                sc.get("embedding_cloud_provider")
            )
            api_key = (sc.get("embedding_cloud_api_key") or "").strip()
            effective_sc["embedding_engine"] = provider_id
            if provider_id == "openai":
                effective_sc["openai_embedding_api_key"] = api_key
                effective_sc["openai_embedding_model"] = (
                    sc.get("openai_embedding_model") or "text-embedding-3-small"
                )
            elif provider_id == "cohere":
                effective_sc["cohere_api_key"] = api_key
                effective_sc["cohere_embedding_model"] = (
                    sc.get("cohere_embedding_model") or "embed-english-v3.0"
                )
            else:
                effective_sc["voyage_api_key"] = api_key
                effective_sc["voyage_embedding_model"] = (
                    sc.get("voyage_embedding_model") or "voyage-3.5-lite"
                )

    effective = dict(config)
    effective["search_config"] = effective_sc
    return effective


# explanation: validates that the embedding provider fields
# are sufficiently filled before allowing embedding creation
def validate_embedding_config(config: dict) -> tuple[bool, str]:
    effective = get_effective_embedding_config(config)
    sc = effective.get("search_config") or {}
    engine = (sc.get("embedding_engine") or "").strip().lower()

    if engine == "unsupported_same_as_answer":
        provider = (effective.get("provider") or "selected answer provider").strip()
        return (
            False,
            f"{provider} cannot create embeddings here. Turn off 'Use same provider as answering' and choose Voyage AI, OpenAI, Cohere, or a local /embeddings server.",
        )

    if engine in ("local_openai", "ollama"):
        url_key = "ollama_base_url" if engine == "ollama" else "local_llm_url"
        if not (sc.get(url_key) or "").strip():
            return False, "Enter a local embedding server URL before creating embeddings."
        return True, ""

    key_by_engine = {
        "voyage": "voyage_api_key",
        "openai": "openai_embedding_api_key",
        "cohere": "cohere_api_key",
    }
    key_name = key_by_engine.get(engine, "voyage_api_key")
    if not (sc.get(key_name) or "").strip():
        provider_names = {
            "voyage": "Voyage AI",
            "openai": "OpenAI",
            "cohere": "Cohere",
        }
        return False, f"Enter a {provider_names.get(engine, 'cloud')} API key before creating embeddings."

    return True, ""
