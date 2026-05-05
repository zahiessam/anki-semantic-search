"""Source/model label helpers for search results and research reports."""

from ..utils import get_effective_embedding_config


class SearchSourceLabelsMixin:
    """Owns compact source labels for answer, embedding, reranker, and result status."""

    def _get_answer_source_text(self, config):
        """Return a short hint: where the answer came from (online API name or local model)."""
        if not config:
            return ""

        provider = config.get("provider", "openai")

        if provider == "ollama":
            sc = config.get("search_config") or {}
            model = (sc.get("ollama_chat_model") or "llama3.2").strip()
            return f"Ollama (local) \u2014 {model}"

        if provider in ("local_openai", "local_server"):
            sc = config.get("search_config") or {}
            model = (
                sc.get("answer_local_model")
                or sc.get("local_llm_model")
                or config.get("local_llm_model")
                or "local-model"
            ).strip()
            base_url = (
                sc.get("local_llm_url")
                or config.get("local_llm_url")
                or config.get("api_url")
                or ""
            ).strip()
            label = "Local Server (Ollama, LM Studio, Jan)"
            if "11434" in base_url:
                label = "Ollama (local)"
            elif "1234" in base_url:
                label = "LM Studio/Jan (local)"
            return f"{label} \u2014 {model}"

        names = {
            "anthropic": "Anthropic (Claude)",
            "openai": "OpenAI (GPT)",
            "google": "Google (Gemini)",
            "openrouter": "OpenRouter",
            "custom": "Custom / OpenAI-compatible",
        }
        name = names.get(provider, "API")
        model = self.get_best_model(provider)
        return f"{name} \u2014 {model}"

    def _get_embedding_source_text(self, config):
        """Return a compact label for the embedding provider/model actually configured."""
        if not config:
            return "unknown"

        effective = get_effective_embedding_config(config)
        sc = effective.get("search_config") or {}
        engine = (sc.get("embedding_engine") or "unknown").strip().lower()

        if engine == "local_openai":
            model = (
                sc.get("local_llm_model")
                or sc.get("embedding_local_model")
                or "local-embedding-model"
            )
            return f"local_openai:{model}"

        if engine == "ollama":
            model = sc.get("ollama_embed_model") or sc.get("embedding_local_model") or "nomic-embed-text"
            return f"ollama:{model}"

        if engine == "voyage":
            return f"voyage:{sc.get('voyage_embedding_model') or 'voyage-3.5-lite'}"

        if engine == "openai":
            return f"openai:{sc.get('openai_embedding_model') or 'text-embedding-3-small'}"

        if engine == "cohere":
            return f"cohere:{sc.get('cohere_embedding_model') or 'embed-english-v3.0'}"

        return engine or "unknown"

    def _get_rerank_source_text(self, search_config, effective_method):
        """Return reranker model and whether it actually ran."""
        if "Re-ranked" not in (effective_method or ""):
            return "off"

        model = (search_config.get("rerank_model") or "cross-encoder/ms-marco-MiniLM-L6-v2").strip()
        status = "ok" if getattr(self, "_last_rerank_success", False) else "skipped"
        return f"{model} ({status})"

    def _build_result_source_text(self, config, effective_method):
        search_config = config.get("search_config") or {}
        mode = (search_config.get("relevance_mode") or "balanced").strip().lower()
        mode_display = {"focused": "Focused", "balanced": "Balanced", "broad": "Broad"}.get(
            mode,
            mode.capitalize() if mode else "Balanced",
        )

        answer = self._get_answer_source_text(config) or "none"
        embeddings = self._get_embedding_source_text(config)
        reranker = self._get_rerank_source_text(search_config, effective_method)
        return (
            f"Results from: {effective_method} \u00b7 {mode_display} "
            f"\u00b7 Answer: {answer} \u00b7 Embeddings: {embeddings} \u00b7 Reranker: {reranker}"
        )
