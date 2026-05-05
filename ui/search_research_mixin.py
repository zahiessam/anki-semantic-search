"""Research logging helpers for the AI search dialog."""

import datetime

from ..utils import get_research_config, log_debug
from ..utils.search_research_log import (
    compact_results,
    new_search_run_id,
    write_search_research_report,
)


class SearchResearchMixin:
    """Owns per-search research report state and persistence."""

    def _init_search_research_report(self, query, config):
        sc = (config or {}).get("search_config") or {}
        research = get_research_config(sc)
        if not research.get("research_enabled", False):
            self._search_research_report = None
            return

        started = datetime.datetime.now()
        self._search_research_started_at = started

        self._search_research_report = {
            "run_id": new_search_run_id(),
            "started_at": started.isoformat(),
            "query": query,
            "config": config,
            "research": {
                "mode": research.get("research_mode", "compact"),
                "max_research_files": research.get("max_research_files", 50),
            },
            "models": {
                "answer": self._get_answer_source_text(config),
                "embeddings": self._get_embedding_source_text(config),
                "reranker": sc.get("rerank_model") or "off",
            },
            "settings": {
                "search_method": sc.get("search_method"),
                "relevance_mode": sc.get("relevance_mode"),
                "enable_query_expansion": sc.get("enable_query_expansion"),
                "enable_hyde": sc.get("enable_hyde"),
                "enable_rerank": sc.get("enable_rerank"),
                "rerank_top_k": sc.get("rerank_top_k"),
                "rerank_timeout_seconds": sc.get("rerank_timeout_seconds"),
                "hybrid_embedding_weight": sc.get("hybrid_embedding_weight"),
                "min_relevance_percent": sc.get("min_relevance_percent"),
                "max_results": sc.get("max_results"),
                "retrieval_version": sc.get("retrieval_version"),
                "keyword_scoring_method": sc.get("keyword_scoring_method"),
                "enable_mmr_diversity": sc.get("enable_mmr_diversity"),
                "mmr_lambda": sc.get("mmr_lambda"),
                "mmr_candidate_pool": sc.get("mmr_candidate_pool"),
            },
            "stages": [],
        }

    def _research_mode(self):
        report = getattr(self, "_search_research_report", None)
        if not isinstance(report, dict):
            return "compact"

        research = report.get("research") or {}
        return "full" if research.get("mode") == "full" else "compact"

    def _research_results(self, scored_notes, limit=50):
        mode = self._research_mode()
        if mode == "full" and scored_notes is not None:
            try:
                limit = len(scored_notes)
            except Exception:
                pass
        return compact_results(scored_notes, limit=limit, mode=mode)

    def _research_stage(self, name, **data):
        report = getattr(self, "_search_research_report", None)
        if not isinstance(report, dict):
            return

        elapsed_ms = None
        started = getattr(self, "_search_research_started_at", None)
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

    def _write_search_research_report(self, **updates):
        report = getattr(self, "_search_research_report", None)
        if not isinstance(report, dict):
            return None

        report.update(updates)

        path = write_search_research_report(report)
        if path:
            log_debug(f"Search research report written: {path}")

        return path
