"""Analytics logging helpers for the AI search dialog."""

import datetime

from ..utils import get_analytics_config, log_debug
from ..utils.search_analytics_log import (
    compact_results,
    new_search_run_id,
    write_search_analytics_report,
)


class SearchAnalyticsMixin:
    """Owns per-search analytics report state and persistence."""

    def _init_search_analytics_report(self, query, config):
        sc = (config or {}).get("search_config") or {}
        analytics = get_analytics_config(sc)
        if not analytics.get("analytics_enabled", False):
            self._search_analytics_report = None
            return

        started = datetime.datetime.now()
        self._search_analytics_started_at = started

        self._search_analytics_report = {
            "run_id": new_search_run_id(),
            "started_at": started.isoformat(),
            "query": query,
            "config": config,
            "analytics": {
                "mode": analytics.get("analytics_mode", "compact"),
                "max_analytics_files": analytics.get("max_analytics_files", 50),
            },
            "models": {
                "answer": self._get_answer_source_text(config),
                "embeddings": self._get_embedding_source_text(config),
                "reranker": sc.get("rerank_model") or "off",
            },
            "settings": {
                "search_method": sc.get("search_method"),
                "relevance_threshold_percent": sc.get("relevance_threshold_percent"),
                "enable_query_expansion": sc.get("enable_query_expansion"),
                "enable_agentic_rag": sc.get("enable_agentic_rag"),
                "enable_profile_memory": sc.get("enable_profile_memory"),
                "agentic_planner_mode": sc.get("agentic_planner_mode"),
                "agentic_planner_model": sc.get("agentic_planner_model"),
                "agentic_planner_timeout_seconds": sc.get("agentic_planner_timeout_seconds"),
                "agentic_planner_max_tokens": sc.get("agentic_planner_max_tokens"),
                "planner_confidence_threshold": sc.get("planner_confidence_threshold"),
                "enable_hyde": sc.get("enable_hyde"),
                "enable_rerank": sc.get("enable_rerank"),
                "rerank_top_k": sc.get("rerank_top_k"),
                "rerank_timeout_seconds": sc.get("rerank_timeout_seconds"),
                "hybrid_embedding_weight": sc.get("hybrid_embedding_weight"),
                "max_results": sc.get("max_results"),
                "retrieval_version": sc.get("retrieval_version"),
                "keyword_scoring_method": sc.get("keyword_scoring_method"),
                "enable_mmr_diversity": sc.get("enable_mmr_diversity"),
                "mmr_lambda": sc.get("mmr_lambda"),
            },
            "memory": {
                "enabled": bool(sc.get("enable_profile_memory", True)),
                "snapshot_size": 0,
                "profile_keys_used": [],
                "session_cache_item_count": 0,
            },
            "stages": [],
        }

    def _analytics_mode(self):
        report = getattr(self, "_search_analytics_report", None)
        if not isinstance(report, dict):
            return "compact"

        analytics = report.get("analytics") or {}
        return "full" if analytics.get("mode") == "full" else "compact"

    def _analytics_results(self, scored_notes, limit=50):
        mode = self._analytics_mode()
        if mode == "full" and scored_notes is not None:
            try:
                limit = len(scored_notes)
            except Exception:
                pass
        return compact_results(scored_notes, limit=limit, mode=mode)

    def _analytics_stage(self, name, **data):
        report = getattr(self, "_search_analytics_report", None)
        if not isinstance(report, dict):
            return

        elapsed_ms = None
        started = getattr(self, "_search_analytics_started_at", None)
        if started is not None:
            try:
                elapsed_ms = int((datetime.datetime.now() - started).total_seconds() * 1000)
            except Exception:
                elapsed_ms = None

        item = {
            "name": name,
            "timestamp": datetime.datetime.now().isoformat(),
            **data,
        }
        if elapsed_ms is not None:
            item["elapsed_ms"] = elapsed_ms

        report.setdefault("stages", []).append(item)

    def _write_search_analytics_report(self, **updates):
        report = getattr(self, "_search_analytics_report", None)
        if not isinstance(report, dict):
            return None

        report.update(updates)

        path = write_search_analytics_report(report)
        if path:
            log_debug(f"Search analytics report written: {path}")

        return path
