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
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_TOP_K_DEFAULT = 25
RERANK_TOP_K_MIN = 5
RERANK_TOP_K_MAX = 100
RERANK_TIMEOUT_SECONDS_DEFAULT = 90
RERANK_TIMEOUT_SECONDS_MIN = 15
RERANK_TIMEOUT_SECONDS_MAX = 300

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
    "search_config": {
        "search_method": "hybrid",
        # Overall relevance mode for search results:
        # - "focused": very on-topic, fewer notes (strict)
        # - "balanced": default mix of precision and coverage
        # - "broad": wider net, more tolerant of weaker matches
        "relevance_mode": "balanced",
        "enable_query_expansion": False,
        "enable_hyde": True,
        "enable_rerank": True,
        "use_context_boost": True,
        "min_relevance_percent": 55,
        "max_results": 50,
        "context_chars_per_note": 0,
        # Assumed local answer model context window. The add-on uses this to
        # dynamically choose light/balanced/deep note context budgets.
        "local_llm_context_tokens": 12288,
        # Dedicated local answer model. Kept separate from local embedding
        # model so independent embedding settings cannot overwrite answers.
        "answer_local_model": "",
        "relevance_from_answer": True,
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
        "rerank_python_path": "",
        "sentence_transformers_path": None,
        "sensitivity_percent": 87,
        "show_only_cited": False,
        # Optional extra stop words for domain-specific generic terms in queries
        # e.g. ["pediatrics", "medicine", "clinical"]
        "extra_stop_words": [],
        # Optional verbose debug logging for search internals (query analysis, scores)
        "verbose_search_debug": False,
        # Retrieval V2 is opt-in. "legacy" preserves the historical ranking path.
        "retrieval_version": "legacy",
        "keyword_scoring_method": "bm25",
        "enable_mmr_diversity": True,
        "mmr_lambda": 0.75,
        "mmr_candidate_pool": 50,
        "mmr_similarity_method": "token_jaccard",
        # Optional extra synonym groups for query expansion (no UI). Each group is a list of
        # equivalent terms; if any term in the group appears in the query, the rest are appended.
        # Example: [["warfarin", "coumadin"], ["vitamin k", "phytonadione"]]
        "synonym_overrides": [],
        # Optional: one short LLM call per search to detect generic query terms to exclude
        "use_ai_generic_term_detection": False,
        "embedding_same_as_answer": True,
        "embedding_strategy": "cloud",
        "embedding_cloud_provider": "Voyage AI",
        "embedding_cloud_api_key": "",
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
            return _deep_merge(DEFAULT_CONFIG, {})
        with open(config_path, "r", encoding="utf-8-sig") as f:
            file_config = json.load(f)
        for typo in ("retrieval_versoin", "retrival_version"):
            if typo in file_config:
                log_debug(f"Retrieval config warning: ignoring top-level key {typo!r}; put 'retrieval_version' inside search_config")
        merged = _deep_merge(DEFAULT_CONFIG, file_config)
        file_search_config = file_config.get("search_config") or {}
        search_config = dict(merged.get("search_config") or {})
        mode = (search_config.get("relevance_mode") or "").strip().lower()
        if "relevance_mode" not in file_search_config:
            search_config["relevance_mode"] = "balanced"
        _protect_answer_model_from_embedding_migration(search_config)
        normalize_rerank_config(search_config, log_warnings=True)
        normalize_retrieval_config(search_config, log_warnings=True)
        merged["search_config"] = search_config
        log_debug(f"Config loaded from file (merged with defaults)")
        return merged
    except Exception as e:
        log_debug(f"Error loading config file: {e}")
        return _deep_merge(DEFAULT_CONFIG, {})


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
    """Validate retrieval knobs and return a normalized copy.

    Invalid values intentionally fall back to legacy-safe behavior so a typo in
    config.json cannot break searching during study.
    """
    sc = dict(search_config or {})
    warnings = []

    for typo in ("retrieval_versoin", "retrival_version"):
        if typo in sc:
            warnings.append(f"Ignoring unknown retrieval config key {typo!r}; did you mean 'retrieval_version'?")

    version = (sc.get("retrieval_version") or "legacy").strip().lower()
    if version not in ("legacy", "v2"):
        warnings.append(f"retrieval_version={version!r} is invalid; using 'legacy'")
        version = "legacy"
    sc["retrieval_version"] = version

    scorer = (sc.get("keyword_scoring_method") or ("bm25" if version == "v2" else "tfidf")).strip().lower()
    if scorer not in ("bm25", "tfidf"):
        warnings.append(f"keyword_scoring_method={scorer!r} is invalid; using {'bm25' if version == 'v2' else 'tfidf'!r}")
        scorer = "bm25" if version == "v2" else "tfidf"
    if version != "v2":
        scorer = "tfidf"
    sc["keyword_scoring_method"] = scorer

    sc["enable_mmr_diversity"] = _coerce_bool(
        sc.get("enable_mmr_diversity", True), True, "enable_mmr_diversity", warnings
    )
    sc["mmr_lambda"] = _coerce_float(sc.get("mmr_lambda", 0.75), 0.75, 0.0, 1.0, "mmr_lambda", warnings)
    sc["mmr_candidate_pool"] = _coerce_int(
        sc.get("mmr_candidate_pool", 50), 50, 5, 200, "mmr_candidate_pool", warnings
    )
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
    return get_retrieval_config(config_or_search_config).get("retrieval_version") == "v2"


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
            effective_sc["embedding_engine"] = "local_openai"
            effective_sc["local_llm_url"] = (
                sc.get("embedding_local_url")
                or sc.get("local_llm_url")
                or "http://localhost:11434/v1"
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
