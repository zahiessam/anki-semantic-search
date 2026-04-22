"""Config load/save and helpers."""
import os
import json
from .log import log_debug

# Voyage embedding models: voyage-3-lite is faster with fewer dimensions; voyage-3.5-lite is higher quality
VOYAGE_EMBEDDING_MODELS = ["voyage-3-lite", "voyage-3.5-lite"]

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
        "strict_relevance": True,
        "max_results": 50,
        "context_chars_per_note": 0,
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
        "rerank_python_path": "",
        "sentence_transformers_path": None,
        "sensitivity_percent": 87,
        "show_only_cited": False,
        # Optional extra stop words for domain-specific generic terms in queries
        # e.g. ["pediatrics", "medicine", "clinical"]
        "extra_stop_words": [],
        # Optional verbose debug logging for search internals (query analysis, scores)
        "verbose_search_debug": False,
        # Optional extra synonym groups for query expansion (no UI). Each group is a list of
        # equivalent terms; if any term in the group appears in the query, the rest are appended.
        # Example: [["warfarin", "coumadin"], ["vitamin k", "phytonadione"]]
        "synonym_overrides": [],
        # Optional: one short LLM call per search to detect generic query terms to exclude
        "use_ai_generic_term_detection": False,
    },
    "saved_presets": {},
}


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
            return dict(DEFAULT_CONFIG)
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = json.load(f)
        merged = _deep_merge(DEFAULT_CONFIG, file_config)
        log_debug(f"Config loaded from file (merged with defaults)")
        return merged
    except Exception as e:
        log_debug(f"Error loading config file: {e}")
        return dict(DEFAULT_CONFIG)


def get_config_value(config, key, default):
    """Get config value with default fallback."""
    if not config:
        return default
    return config.get(key, default)
