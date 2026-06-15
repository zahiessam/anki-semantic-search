"""Source/model label helpers for search results and analytics reports."""

from ..utils import clamp_relevance_threshold_percent, get_effective_embedding_config


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
            label = "Custom local server"
            if "11434" in base_url:
                label = "Ollama (local)"
            elif "1234" in base_url:
                label = "LM Studio (local)"
            elif "1337" in base_url:
                label = "Jan (local)"
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
        if not bool(search_config.get("enable_rerank", False)):
            return "off"

        model = (search_config.get("rerank_model") or "NeuML/biomedbert-base-reranker").strip()
        if getattr(self, "_last_rerank_success", False):
            status = "ok"
        elif getattr(self, "_pending_rerank", False):
            status = "running"
        else:
            status = "skipped"
        return f"{model} ({status})"

    def _build_result_source_text(self, config, effective_method):
        search_config = config.get("search_config") or {}
        def short_answer_label(value):
            text = value or "none"
            provider = text.split("\u2014", 1)[0].strip()
            model = text.split("\u2014", 1)[1].strip() if "\u2014" in text else ""
            if model:
                if len(model) > 28:
                    model = model[:25].rstrip() + "..."
                return f"{provider}: {model}"
            return provider

        def short_embedding_label(value):
            text = value or "unknown"
            status = ""
            if "(" in text and text.endswith(")"):
                text, status = text.rsplit("(", 1)
                status = status.rstrip(")")
            engine = text.split(":", 1)[0].strip()
            if status:
                status = (
                    status.replace("Embedding ", "")
                    .replace("active-engine ", "")
                    .replace("rows found", "rows")
                )
                return f"{engine} ({status})"
            return engine

        def short_reranker_label(value):
            text = value or "off"
            if "(" in text and text.endswith(")"):
                return text.rsplit("(", 1)[1].rstrip(")")
            return text

        try:
            slider_threshold = None
            slider = getattr(self, "sensitivity_slider", None)
            if slider is not None:
                try:
                    slider_threshold = slider.value()
                except Exception:
                    slider_threshold = None
            live_threshold = getattr(self, "_effective_relevance_threshold_percent", None)
            if slider_threshold is not None:
                raw_threshold = slider_threshold
            elif live_threshold is not None:
                raw_threshold = live_threshold
            else:
                raw_threshold = search_config.get("relevance_threshold_percent", 65)
            threshold = clamp_relevance_threshold_percent(raw_threshold)
        except Exception:
            threshold = 65

        answer = self._get_answer_source_text(config) or "none"
        embeddings = self._get_embedding_source_text(config)
        fallback_matches = int(getattr(self, "_last_embedding_note_id_fallback_matches", 0) or 0)
        exact_matches = int(getattr(self, "_last_embedding_exact_matches", 0) or 0)
        rows_checked = int(getattr(self, "_last_embedding_rows_checked", 0) or 0)
        if fallback_matches and not exact_matches:
            embeddings = f"{embeddings} (Embedding fallback by note-id; update embeddings)"
        elif fallback_matches:
            noun = "chunk" if fallback_matches == 1 else "chunks"
            embeddings = f"{embeddings} (Embedding partial: {fallback_matches} stale/missing {noun}; update embeddings)"
        elif exact_matches:
            embeddings = f"{embeddings} (Embedding current)"
        elif (
            rows_checked == 0
            and (effective_method or "").lower().startswith("keyword")
            and getattr(self, "_last_requested_search_method", None) in ("embedding", "hybrid")
        ):
            embeddings = f"{embeddings} (Embedding unavailable: no active-engine rows found)"
        reranker = self._get_rerank_source_text(search_config, effective_method)
        answer_short = short_answer_label(answer)
        embeddings_short = short_embedding_label(embeddings)
        reranker_short = short_reranker_label(reranker)
        if bool(getattr(self, "_sources_rank_mode", False)):
            return (
                f"Related notes: {effective_method} \u00b7 Threshold {threshold}% "
                f"\u00b7 Embeddings {embeddings_short} \u00b7 Reranker {reranker_short}"
            )
        return (
            f"Ask Notes: {effective_method} \u00b7 Threshold {threshold}% "
            f"\u00b7 Answer {answer_short} \u00b7 Embeddings {embeddings_short} \u00b7 Reranker {reranker_short}"
        )
