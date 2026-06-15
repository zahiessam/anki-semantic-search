"""Search execution entrypoint and background worker handoff callbacks."""

import json
import urllib.request

from aqt import mw
from aqt.qt import QApplication
from aqt.utils import tooltip

from .dependency_install import is_numpy_available, show_numpy_install_guidance
from .image_attachments import IMAGE_SUPPORT_ERROR, merge_visual_findings_query, snapshot_image_payloads
from .search_workers import (
    AgenticPlanWorker,
    MAX_RERANK_COUNT,
    EmbeddingSearchWorker,
    KeywordFilterWorker,
    RerankWorker,
    _do_rerank,
)
from .search_dialog_shared import get_notes_content_with_col, get_safe_config
from ..core.agentic_planner import (
    build_agentic_plan,
    evaluate_agentic_evidence,
    has_entity_signal,
    possible_media_in_note,
)
from ..core.agentic_runtime import apply_runtime_budgets, validate_required_local_tools
from ..core.adaptive_agentic_planner import (
    AGENTIC_PLANNER_TIMEOUT_SECONDS_DEFAULT,
    apply_adaptive_defaults,
    build_model_planner_with_retry,
    embedding_health_snapshot,
    is_adaptive_v2_enabled,
    planner_timeout_seconds,
    should_call_model_planner,
)
from ..core.memory import (
    SessionMemory,
    build_memory_snapshot,
    load_memory_profile,
    memory_profile_summary,
    retrieve_memory_snippets,
)
from ..core.engine import get_embedding_engine_id, get_ollama_model_capabilities, load_embedding_engine_counts
from ..core.keyword_scoring import compute_bm25_scores, extract_keywords_improved
from ..utils import (
    get_embeddings_db_path,
    get_retrieval_config,
    load_config,
    log_debug,
)
from ..utils.anthropic_response import extract_anthropic_text


class SearchExecutionMixin:
    def _planner_model_name(self, config):
        config = config or {}
        sc = config.get("search_config") or {}
        explicit = (sc.get("agentic_planner_model") or "").strip()
        if explicit:
            return explicit
        provider = (config.get("provider") or "ollama").strip().lower()
        if provider == "ollama":
            return (sc.get("ollama_chat_model") or "llama3.2").strip()
        if provider in ("local_openai", "local_server"):
            return (sc.get("answer_local_model") or sc.get("local_llm_model") or "model-identifier").strip()
        return self.get_best_model(provider) if hasattr(self, "get_best_model") else "gpt-4o-mini"

    def _call_agentic_model_planner(self, config, system_prompt, user_prompt, max_tokens_override=None, timeout_override=None):
        config = config or {}
        sc = config.get("search_config") or {}
        provider = (config.get("provider") or "ollama").strip().lower()
        model = self._planner_model_name(config)
        if timeout_override is None:
            timeout = planner_timeout_seconds(sc)
        else:
            timeout = max(1, min(60, int(timeout_override or AGENTIC_PLANNER_TIMEOUT_SECONDS_DEFAULT)))
        token_source = max_tokens_override if max_tokens_override is not None else sc.get("agentic_planner_max_tokens", 350)
        max_tokens = max(100, min(1000, int(token_source or 350)))

        if provider == "ollama":
            base_url = (sc.get("ollama_base_url") or "http://localhost:11434").strip().rstrip("/")
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
            }
            req = urllib.request.Request(
                base_url + "/api/chat",
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return ((result.get("message") or {}).get("content") or "").strip(), model

        if provider in ("local_openai", "local_server"):
            base_url = (sc.get("local_llm_url") or "http://localhost:1234/v1").strip().rstrip("/")
            url = base_url + ("" if base_url.endswith("/chat/completions") else "/chat/completions")
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return (result["choices"][0]["message"]["content"] or "").strip(), model

        api_key = config.get("api_key", "")
        if not api_key:
            raise RuntimeError("planner_api_key_missing")

        if provider == "anthropic":
            data = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(data).encode("utf-8"),
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return extract_anthropic_text(result, source="Agentic planner").strip(), model

        if provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
        elif provider == "google":
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
            data = {
                "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0},
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            parts = (((result.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
            return "".join(part.get("text", "") for part in parts).strip(), model
        else:
            url = config.get("api_url") or "https://api.openai.com/v1/chat/completions"

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return (result["choices"][0]["message"]["content"] or "").strip(), model

    def _build_agentic_search_plan(self, query, history, config):
        search_config = dict((config.get("search_config") or {}))
        prior_queries = [
            (item.get("content") or "").strip()
            for item in reversed(history or [])
            if item.get("role") == "user" and item.get("mode") == "notes" and (item.get("content") or "").strip()
        ]
        session_memory = getattr(self, "_agentic_session_memory", None)
        if session_memory is None:
            session_memory = SessionMemory()
            self._agentic_session_memory = session_memory
        selected_text = getattr(self, "_pending_selected_answer_context", "") or ""
        review_context = getattr(self, "_review_context_text", "") or ""
        if query and has_entity_signal(query, search_config):
            session_memory.maybe_clear_followup_for_query(query, search_config)
        memory_enabled = bool(search_config.get("enable_profile_memory", True))
        memory_profile = load_memory_profile() if memory_enabled else None
        memory_query = " ".join(
            part for part in (
                query or "",
                getattr(session_memory, "resolved_followup_target", "") or "",
            )
            if part
        )
        memory_diagnostics = {}
        retrieval_mode = search_config.get("memory_retrieval_mode", "text")
        if not search_config.get("memory_embedding_enabled", True):
            retrieval_mode = "text"
        fact_snippets = retrieve_memory_snippets(
            memory_query,
            limit=int(search_config.get("memory_max_retrieved_snippets", 5) or 5),
            mode=retrieval_mode,
            config=config,
            diagnostics=memory_diagnostics,
        ) if memory_enabled else []
        memory_snapshot = build_memory_snapshot(
            config,
            history,
            review_context=review_context,
            selected_text=selected_text,
            session_memory=session_memory,
            profile=memory_profile,
            fact_snippets=fact_snippets,
        )
        session_context = {
            "prior_query": prior_queries[0] if prior_queries else "",
            "memory_snapshot": memory_snapshot,
            "memory_diagnostics": memory_diagnostics,
        }
        health = embedding_health_snapshot(config)
        if health.get("status") == "healthy":
            try:
                engine_id = get_embedding_engine_id(config)
                counts = load_embedding_engine_counts()
                if int(counts.get(engine_id, 0) or 0) <= 0:
                    health = {"status": "missing_index", "reason": "no_embeddings_for_current_engine"}
            except Exception as exc:
                log_debug(f"Agentic embedding health DB check skipped: {exc}")
        deterministic = build_agentic_plan(query, session_context, search_config)
        deterministic["memory_snapshot"] = memory_snapshot
        deterministic["memory_used"] = bool(memory_enabled and memory_snapshot)
        deterministic["embedding_health"] = health.get("status")
        deterministic["planner_model"] = ""
        if not is_adaptive_v2_enabled(search_config):
            plan = dict(deterministic)
            plan.update({
                "planner_version": "deterministic_v1",
                "planner_source": "deterministic",
                "retrieval_mode": "hybrid",
                "retrieval_mode_reason": "deterministic_v1_default",
                "use_rerank": bool(search_config.get("enable_rerank", True)),
                "pass2_policy": "if_coverage_gap",
                "memory_used": bool(memory_enabled and memory_snapshot),
                "required_local_tools": ["hybrid_embedding_search", "evidence_evaluation"],
                "budget_reason": "deterministic_v1_local_bounded",
                "coverage_targets": [],
                "plan_validation_errors": [],
                "memory_diagnostics": memory_diagnostics,
            })
            return apply_runtime_budgets(validate_required_local_tools(plan), search_config=search_config)

        plan = apply_adaptive_defaults(deterministic, query, health, search_config)
        plan["memory_snapshot"] = memory_snapshot
        plan["memory_used"] = bool(memory_enabled and memory_snapshot)
        plan["embedding_health"] = health.get("status")
        if should_call_model_planner(deterministic, query, session_context, search_config):
            result = build_model_planner_with_retry(
                query,
                deterministic,
                plan,
                session_context,
                health,
                memory_snapshot,
                search_config,
                lambda system_prompt, user_prompt, max_tokens_override=None, timeout_override=None:
                    self._call_agentic_model_planner(
                        config,
                        system_prompt,
                        user_prompt,
                        max_tokens_override=max_tokens_override,
                        timeout_override=timeout_override,
                    ),
            )
            model_plan = result.get("plan")
            if model_plan is None:
                plan["planner_source"] = "fallback"
                plan["fallback_reason"] = deterministic.get("fallback_reason") or "model_plan_invalid"
                plan["llm_planner_error"] = result.get("llm_planner_error") or "model_plan_invalid"
                plan["plan_validation_errors"] = result.get("errors") or []
                plan["planner_model"] = result.get("model_name") or self._planner_model_name(config)
                log_debug(f"Agentic adaptive planner invalid after retry: {plan['plan_validation_errors']}")
            else:
                plan = model_plan
                plan["memory_snapshot"] = memory_snapshot
                plan["memory_used"] = bool(plan.get("memory_used") or (memory_enabled and memory_snapshot))
                plan["planner_model"] = result.get("model_name") or self._planner_model_name(config)
                if result.get("errors"):
                    plan["plan_validation_errors"] = list(dict.fromkeys(
                        list(plan.get("plan_validation_errors") or []) + list(result.get("errors") or [])
                    ))
        else:
            plan["planner_model"] = ""
        plan["memory_diagnostics"] = memory_diagnostics
        return apply_runtime_budgets(validate_required_local_tools(plan), search_config=search_config)

    def _apply_agentic_retrieval_to_config(self, config, plan):
        config = dict(config or {})
        search_config = dict(config.get("search_config") or {})
        mode = (plan or {}).get("retrieval_mode") or "hybrid"
        search_config["search_method"] = "keyword" if mode == "bm25_only" else "hybrid"
        if plan and "use_rerank" in plan:
            search_config["enable_rerank"] = bool(plan.get("use_rerank"))
        config["search_config"] = search_config
        return config

    def _fallback_agentic_plan(self, query, search_config_for_plan, reason="planner_error"):
        return {
            "agentic_enabled": True,
            "intent": "specific_fact",
            "confidence": 0.0,
            "threshold": float(search_config_for_plan.get("planner_confidence_threshold", 0.6) or 0.6),
            "subqueries": [query],
            "follow_up": False,
            "image_intent": False,
            "retrieval_plan": "fallback_current_hybrid",
            "fallback_reason": reason,
            "planner_version": "deterministic_v1",
            "planner_source": "fallback",
            "retrieval_mode": "hybrid",
            "retrieval_mode_reason": reason,
            "pass2_policy": "if_coverage_gap",
            "coverage_targets": [],
            "plan_validation_errors": [],
            "selected_model_image_support": self._selected_model_image_support(getattr(self, "_search_pending_config", None) or {}),
        }

    def _record_agentic_plan_analytics(self, search_config_for_plan):
        if self._agentic_plan:
            self._analytics_stage("agentic_plan_built", **self._agentic_plan)
            try:
                report = getattr(self, "_search_analytics_report", None)
                if isinstance(report, dict):
                    snapshot = self._agentic_plan.get("memory_snapshot") or {}
                    profile_keys = sorted((snapshot.get("active_scope") or {}).keys()) if isinstance(snapshot, dict) else []
                    report["memory"] = {
                        "enabled": bool(search_config_for_plan.get("enable_profile_memory", True)),
                        "snapshot_size": len(json.dumps(snapshot, ensure_ascii=False)) if snapshot else 0,
                        "profile_keys_used": profile_keys,
                        "session_cache_item_count": len(getattr(getattr(self, "_agentic_session_memory", None), "fact_cache", []) or []),
                        "durable_snippets_retrieved": len((snapshot or {}).get("fact_snippets") or []),
                        "fact_snippets_in_snapshot": len((snapshot or {}).get("fact_snippets") or []),
                        "fact_snippets_retrieved": len((snapshot or {}).get("fact_snippets") or []),
                        "profile_memory_used": bool(search_config_for_plan.get("enable_profile_memory", True) and snapshot),
                    }
                    memory_diagnostics = self._agentic_plan.get("memory_diagnostics") or {}
                    if isinstance(memory_diagnostics, dict):
                        report["memory"].update(memory_diagnostics)
                    try:
                        summary = memory_profile_summary()
                        report["memory"]["durable_snippet_count"] = summary.get("snippet_count", 0)
                    except Exception:
                        pass
            except Exception as exc:
                log_debug(f"Memory analytics update skipped: {exc}")

    def _apply_agentic_plan_and_stage(self, plan, config):
        self._agentic_plan = dict(plan or {})
        self._agentic_plan["selected_model_image_support"] = self._selected_model_image_support(config)
        retrieval_mode = self._agentic_plan.get("retrieval_mode")
        if retrieval_mode == "bm25_only":
            stage = "Searching notes with BM25..."
        elif retrieval_mode == "hybrid":
            stage = "Searching notes with Hybrid retrieval..."
        else:
            stage = "Searching notes..."
        if hasattr(self, "_set_search_stage"):
            self._set_search_stage(stage, chat_message=True)
        if hasattr(self, "_append_system_chat_message"):
            if self._agentic_plan.get("embedding_health") == "partial":
                self._append_system_chat_message(
                    "Embeddings are incomplete. Search still worked, but some notes may be missing. Run Create/Update Embeddings for full coverage.",
                    kind="warning",
                )
        log_debug(f"Agentic RAG plan: {self._agentic_plan}")

    def _on_agentic_plan_ready(self, plan):
        config = getattr(self, "_search_pending_config", None) or load_config()
        search_config_for_plan = dict((config.get("search_config") or {}))
        self._agentic_plan_worker = None
        self._apply_agentic_plan_and_stage(plan, config)
        self._record_agentic_plan_analytics(search_config_for_plan)
        config = self._apply_agentic_retrieval_to_config(config, self._agentic_plan)
        self._search_pending_config = config
        self._start_note_load_for_search(self._search_pending_query, config)

    def _on_agentic_plan_error(self, error):
        config = getattr(self, "_search_pending_config", None) or load_config()
        search_config_for_plan = dict((config.get("search_config") or {}))
        self._agentic_plan_worker = None
        log_debug(f"Agentic RAG planner failed, using normal hybrid flow: {error}")
        self._agentic_plan = self._fallback_agentic_plan(
            getattr(self, "_search_pending_query", "") or "",
            search_config_for_plan,
            "planner_error",
        )
        self._agentic_planner_runtime_warning = "Adaptive planner failed before retrieval. Used deterministic retrieval instead."
        self._record_agentic_plan_analytics(search_config_for_plan)
        config = self._apply_agentic_retrieval_to_config(config, self._agentic_plan)
        self._search_pending_config = config
        self._start_note_load_for_search(self._search_pending_query, config)

    def _start_note_load_for_search(self, query, config):
        import time

        try:
            if self._agentic_plan:
                config = self._apply_agentic_retrieval_to_config(config, self._agentic_plan)
            search_config = dict(config.get('search_config', {}) or {})
            search_config['search_method'] = search_config.get('search_method') or 'hybrid'
            config['search_config'] = search_config
            retrieval = get_retrieval_config(config)
            if search_config.get("verbose_search_debug"):
                log_debug(
                    "Retrieval diagnostics: "
                    f"version={retrieval.get('retrieval_version')}, "
                    f"scorer={retrieval.get('keyword_scoring_method')}, "
                    f"mmr_enabled={retrieval.get('enable_mmr_diversity')}"
                )

            self._last_requested_search_method = search_config.get('search_method') or 'hybrid'
            self._search_pending_query = query
            self._original_search_query = query
            self._search_pending_config = config
            self._search_pending_async = True

            if hasattr(self, "_set_search_stage"):
                self._set_search_stage("Loading notes...", progress=0)
            else:
                self.status_label.setText("Loading notes...")

            if not hasattr(self, "_set_search_stage"):
                self._show_centile_progress("Loading notes...", 0)

            self._start_estimated_progress_timer(30, 5, 95)

            log_debug("Starting to load notes in background...")
            self._note_load_started = time.time()

            from aqt.operations import QueryOp

            op = QueryOp(
                parent=mw,
                op=lambda col: get_notes_content_with_col(col, config),
                success=self._on_get_notes_done,
            )
            op.run_in_background()
            return
        except Exception as e:
            log_debug(f"Unexpected error starting note load: {type(e).__name__}: {str(e)}")
            import traceback
            log_debug(f"Traceback: {traceback.format_exc()}")
            self.answer_box.setText(
                f"Unexpected Error:\n{str(e)}\n\n"
                "Please check the debug log for details."
            )
            self.status_label.setText("Error occurred")
            self._search_pending_async = False
            self.search_btn.setEnabled(True)
            if hasattr(self, "ask_ai_btn"):
                self.ask_ai_btn.setEnabled(True)
            if hasattr(self, "clear_chat_btn"):
                self.clear_chat_btn.setEnabled(True)
            if hasattr(self, "_set_search_chat_busy"):
                self._set_search_chat_busy(False)

    def _selected_model_image_support(self, config):
        config = config or {}
        sc = config.get("search_config") or {}
        provider = (config.get("provider") or "").lower()
        model = ""
        base_url = ""
        if provider == "ollama":
            model = (sc.get("ollama_chat_model") or "llama3.2").strip()
            base_url = (sc.get("ollama_base_url") or "http://localhost:11434").strip()
        elif provider in ("local_openai", "local_server") and "11434" in (sc.get("local_llm_url") or ""):
            model = (sc.get("answer_local_model") or sc.get("local_llm_model") or "").strip()
            base_url = (sc.get("local_llm_url") or "http://localhost:11434").strip()
        if not model:
            return {"model": model, "image_support": "unknown"}
        try:
            caps = get_ollama_model_capabilities(base_url, [model], timeout=2).get(model)
            if caps is not None:
                return {"model": model, "image_support": bool("vision" in caps)}
        except Exception as exc:
            log_debug(f"Agentic RAG image capability check skipped: {exc}")
        return {"model": model, "image_support": "unknown"}

    def _maybe_apply_agentic_pass2(self, scored_notes, notes, search_config, effective_method):
        plan = getattr(self, "_agentic_plan", None)
        if not plan or plan.get("fallback_reason"):
            return scored_notes, effective_method
        if hasattr(self, "_set_search_stage"):
            self._set_search_stage("Checking evidence coverage...", chat_message=True)
        try:
            decision = evaluate_agentic_evidence(plan, scored_notes, search_config)
        except Exception as exc:
            log_debug(f"Agentic RAG evidence gate failed: {exc}")
            decision = {
                "should_run_pass2": False,
                "evidence_status": "gate_error",
                "coverage_gaps": [],
                "error": str(exc),
            }
        self._agentic_evidence_decision = decision
        self._analytics_stage("agentic_evidence_gate", **decision)
        if not decision.get("should_run_pass2"):
            self._mark_agentic_result_metadata(scored_notes, pass_number=1, query_label="main")
            return scored_notes, effective_method

        used_queries = {str(getattr(self, "_search_pending_query", "") or "").strip().lower()}
        subqueries = self._agentic_subqueries_for_gaps(plan, decision, used_queries, fallback_to_plan=True)[:3]
        if not subqueries:
            return scored_notes, effective_method

        if hasattr(self, "_set_search_stage"):
            self._set_search_stage("Running second pass for missing coverage...", chat_message=True, progress=0)
        merged = self._merge_agentic_subquery_results(scored_notes, notes, subqueries, search_config, pass_number=2)
        self._agentic_passes_run = 2
        used_queries.update(str(q or "").strip().lower() for q in subqueries if str(q or "").strip())
        self._analytics_stage(
            "agentic_pass2_merged",
            subqueries=subqueries,
            merge_strategy="dedupe_note_id_keep_highest_score",
            total_candidates=len(merged),
            top_results=self._analytics_results(merged, limit=50),
        )
        final_method = f"{effective_method} + Agentic RAG"

        try:
            max_passes = int((plan.get("runtime_budgets") or {}).get("max_retrieval_passes", 2) or 2)
        except Exception:
            max_passes = 2
        if max_passes >= 3:
            pass3_decision = evaluate_agentic_evidence(plan, merged, search_config)
            self._analytics_stage("agentic_pass3_evidence_gate", **pass3_decision)
            pass3_subqueries = self._agentic_pass3_subqueries(plan, pass3_decision, used_queries)[:2]
            if pass3_subqueries:
                if hasattr(self, "_set_search_stage"):
                    self._set_search_stage("Running final coverage pass...", chat_message=True, progress=0)
                merged = self._merge_agentic_subquery_results(merged, notes, pass3_subqueries, search_config, pass_number=3)
                self._agentic_passes_run = 3
                self._analytics_stage(
                    "agentic_pass3_merged",
                    subqueries=pass3_subqueries,
                    pass3_reason="unmet_coverage_target",
                    total_candidates=len(merged),
                    top_results=self._analytics_results(merged, limit=50),
                )
                final_method = f"{final_method} + pass 3"
        return merged, final_method

    def _agentic_subqueries_for_gaps(self, plan, decision, used_queries=None, fallback_to_plan=False):
        used_queries = set(used_queries or set())
        gaps = decision.get("coverage_gaps") or []
        subquery_plans = plan.get("subquery_plans") or []
        gap_keys = {str(gap).strip().lower() for gap in gaps if str(gap).strip()}
        subqueries = []
        for item in subquery_plans:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip().lower()
            query_text = str(item.get("query") or "").strip()
            key = query_text.lower()
            if query_text and key not in used_queries and (key in gap_keys or label in gap_keys):
                subqueries.append(query_text)
        if fallback_to_plan and not subqueries:
            subqueries = [
                q for q in gaps
                if q and str(q).strip().lower() not in used_queries
            ]
        if fallback_to_plan and not subqueries:
            subqueries = [
                q for q in (plan.get("subqueries") or [])[1:]
                if q and str(q).strip().lower() not in used_queries
            ]
        return list(dict.fromkeys(subqueries))

    def _agentic_pass3_subqueries(self, plan, decision, used_queries=None):
        if plan.get("fallback_reason") or int(getattr(self, "_agentic_passes_run", 1) or 1) < 2:
            return []
        used_queries = set(used_queries or set())
        targets = decision.get("unmet_coverage_targets") or []
        if not targets:
            return []
        target_keys = {
            str(target.get("label") or "").strip().lower()
            for target in targets if isinstance(target, dict)
        }
        subqueries = []
        for item in plan.get("subquery_plans") or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip().lower()
            query_text = str(item.get("query") or "").strip()
            if query_text and query_text.lower() not in used_queries and label in target_keys:
                subqueries.append(query_text)
        for target in targets:
            if not isinstance(target, dict):
                continue
            terms = [str(term).strip() for term in (target.get("required_terms") or []) if str(term).strip()]
            query_text = " ".join(terms[:6]).strip() or str(target.get("label") or "").strip()
            if query_text and query_text.lower() not in used_queries:
                subqueries.append(query_text)
        return list(dict.fromkeys(subqueries))

    def _merge_agentic_subquery_results(self, scored_notes, notes, subqueries, search_config, pass_number=2):
        by_id = {}
        for score, note in scored_notes or []:
            note = dict(note)
            note["_agentic_pass"] = 1
            note["_agentic_queries"] = list(dict.fromkeys((note.get("_agentic_queries") or []) + ["main"]))
            note["_possible_media"] = possible_media_in_note(note)
            by_id[note.get("id")] = (score, note)

        for subquery in subqueries:
            keywords, _stems, phrases = extract_keywords_improved(subquery, search_config)
            query_terms = list(dict.fromkeys(keywords + phrases))
            bm25_scores, _high_freq = compute_bm25_scores(notes, keywords)
            scored = []
            for note in notes or []:
                text = (note.get("content") or "").lower()
                phrase_bonus = sum(10 for phrase in phrases if phrase and phrase.lower() in text)
                score = float(bm25_scores.get(note.get("id"), 0) or 0) * 20.0 + phrase_bonus
                if score <= 0 and any(term.lower() in text for term in query_terms):
                    score = 5.0
                if score > 0:
                    scored.append((score, note))
            max_score = max((score for score, _note in scored), default=0.0)
            if max_score > 0:
                scored = [(score / max_score * 100.0, note) for score, note in scored]
            for score, note in sorted(scored, key=lambda item: item[0], reverse=True)[:40]:
                note = dict(note)
                note["_agentic_pass"] = pass_number
                note["_agentic_queries"] = list(dict.fromkeys((note.get("_agentic_queries") or []) + [subquery]))
                note["_possible_media"] = possible_media_in_note(note)
                note_id = note.get("id")
                prev = by_id.get(note_id)
                if prev is None or score > prev[0]:
                    if prev is not None:
                        note["_agentic_queries"] = list(dict.fromkeys((prev[1].get("_agentic_queries") or []) + note["_agentic_queries"]))
                    by_id[note_id] = (score, note)
                else:
                    prev_note = prev[1]
                    prev_note["_agentic_queries"] = list(dict.fromkeys((prev_note.get("_agentic_queries") or []) + [subquery]))

        merged = list(by_id.values())
        merged.sort(key=lambda item: item[0], reverse=True)
        return merged

    def _mark_agentic_result_metadata(self, scored_notes, pass_number=1, query_label="main"):
        for _score, note in scored_notes or []:
            note["_agentic_pass"] = pass_number
            note["_agentic_queries"] = list(dict.fromkeys((note.get("_agentic_queries") or []) + [query_label]))
            note["_possible_media"] = possible_media_in_note(note)

    def perform_search(self):



        """Perform search with proper error handling and UI updates"""



        log_debug("=== Perform Search Called ===")
        if hasattr(self, "_expand_sources_panel"):
            self._expand_sources_panel(manual=False)
        import time
        self._retrieval_search_started = time.time()
        self._show_all_dynamic_results = False
        self._last_dynamic_note_budget = None
        if hasattr(self, "show_all_dynamic_results_btn"):
            self.show_all_dynamic_results_btn.setVisible(False)







        # Check for config



        config = self.get_config()



        log_debug(f"Retrieved config for search: {get_safe_config(config)}")







        if not config:



            log_debug("ERROR: No config found")



            if hasattr(self, "_append_system_chat_message"):
                self._append_system_chat_message("Please configure your API key first. Click the Settings button.", kind="error")
            else:
                self.answer_box.setText("Please configure your API key first. Click the \u2699 button.")



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            tooltip("API not configured")



            return







        typed_query = self.search_input.toPlainText().strip()
        if hasattr(self, "_strip_selected_answer_snippet_from_query"):
            typed_query = self._strip_selected_answer_snippet_from_query(typed_query)
        if hasattr(self, "_snapshot_composer_image_payloads"):
            composer_image_payloads = self._snapshot_composer_image_payloads()
        else:
            composer_image_payloads = snapshot_image_payloads(getattr(self, "_composer_image_payloads", []) or [])
        selected_context = self._composer_selected_answer_context() if hasattr(self, "_composer_selected_answer_context") else ""
        if not typed_query and selected_context:
            typed_query = "Explain this selected text."
        self._pending_selected_answer_context = selected_context
        self._composer_selected_answer_context_text = ""
        if hasattr(self, "_update_selected_answer_context_chip"):
            self._update_selected_answer_context_chip()
        history = list(getattr(self, "_chat_history", []) or [])
        related_only = getattr(self, "_search_mode", "ask_notes") == "related_notes"
        related_card_query = str(getattr(self, "_pending_related_notes_query", "") or "").strip()
        query = (
            related_card_query
            if related_only else
            typed_query
        )
        if not query and hasattr(self, "_latest_user_question"):
            query = self._latest_user_question()
        if not query and composer_image_payloads:
            query = "Find notes related to this image."



        if not query:



            tooltip("Please enter a search query")



            return

        if getattr(self, "_direct_ai_worker", None) is not None or getattr(self, "_ask_ai_worker", None) is not None:
            tooltip("Please wait for the current answer to finish")
            return

        self.search_btn.setEnabled(False)
        if hasattr(self, "ask_ai_btn"):
            self.ask_ai_btn.setEnabled(False)
        if hasattr(self, "clear_chat_btn"):
            self.clear_chat_btn.setEnabled(False)
        if hasattr(self, "attach_image_btn"):
            self.attach_image_btn.setEnabled(False)

        self._notes_answer_query = query
        self._related_notes_only = related_only
        self._sources_rank_mode = bool(related_only)
        retrieval_query = query
        if hasattr(self, "_append_user_chat_message") and not related_only:
            self._append_user_chat_message(query, mode="notes", image_payloads=composer_image_payloads)
            self.search_input.clear()
            self._set_chat_transient("Preparing note search...", mode="info")
        if history and not related_only and hasattr(self, "rewrite_notes_search_query"):
            try:
                retrieval_query = self.rewrite_notes_search_query(query, history, config) or query
                if retrieval_query != query:
                    log_debug(f"Smart Ask Notes rewritten query: display={query!r}, retrieval={retrieval_query!r}")
                    self._set_chat_transient("Searching notes with conversation context...", mode="info")
                else:
                    self._set_chat_transient("Searching notes...", mode="info")
            except Exception as exc:
                log_debug(f"Smart Ask Notes rewrite failed, using visible query: {exc}")
                retrieval_query = query
                self._set_chat_transient("Searching notes...", mode="info")
        else:
            self._set_chat_transient("Searching notes...", mode="info")

        if composer_image_payloads and hasattr(self, "generate_image_search_terms"):
            self._set_chat_transient("Reading image for note search...", mode="info")
            if hasattr(self, "_set_search_stage"):
                self._set_search_stage("Reading image for note search...", chat_message=True, progress=0)
            try:
                image_terms = self.generate_image_search_terms(
                    retrieval_query if not related_only else merge_visual_findings_query(related_card_query, typed_query),
                    config,
                    composer_image_payloads,
                )
            except Exception as exc:
                log_debug(f"Image search term generation failed: {exc}")
                self._set_chat_transient(None)
                if hasattr(self, "_append_system_chat_message"):
                    message = str(exc) if str(exc) == IMAGE_SUPPORT_ERROR else f"Could not read the image for note search:\n{exc}"
                    self._append_system_chat_message(message, kind="error")
                if hasattr(self, "_set_search_chat_busy"):
                    self._set_search_chat_busy(False)
                else:
                    self.search_btn.setEnabled(True)
                    if hasattr(self, "ask_ai_btn"):
                        self.ask_ai_btn.setEnabled(True)
                    if hasattr(self, "clear_chat_btn"):
                        self.clear_chat_btn.setEnabled(True)
                    if hasattr(self, "attach_image_btn"):
                        self.attach_image_btn.setEnabled(True)
                if hasattr(self, "status_label"):
                    self.status_label.setText("Ready")
                return
            if related_only:
                retrieval_query = merge_visual_findings_query(related_card_query, typed_query, image_terms=image_terms)
            else:
                retrieval_query = merge_visual_findings_query(retrieval_query, image_terms=image_terms)
            if image_terms:
                answer_segments = [query]
                answer_segments.append(f"Visual findings from attached image: {image_terms}")
                self._notes_answer_query = merge_visual_findings_query(*answer_segments)
                self._notes_image_terms = image_terms
            self._set_chat_transient("Searching notes...", mode="info")

        self._notes_display_query = query
        self._notes_retrieval_query = retrieval_query
        query = retrieval_query

        self._agentic_plan = None
        self._agentic_evidence_decision = None
        self._agentic_passes_run = 1
        search_config_for_plan = dict((config.get("search_config") or {}))

        self._init_search_analytics_report(query, config)

        self._analytics_stage("search_started")

        # Disable search button to prevent multiple clicks



        self.search_btn.setEnabled(False)
        if hasattr(self, "ask_ai_btn"):
            self.ask_ai_btn.setEnabled(False)
        if hasattr(self, "clear_chat_btn"):
            self.clear_chat_btn.setEnabled(False)
        if hasattr(self, "attach_image_btn"):
            self.attach_image_btn.setEnabled(False)



        if not self._agentic_plan:
            if hasattr(self, "_set_search_stage"):
                self._set_search_stage("Searching notes...", chat_message=True)
            else:
                self.status_label.setText("Searching notes...")



        if not hasattr(self, "_render_chat_transcript"):
            self.answer_box.clear()



        self.results_list.setRowCount(0)  # Clear table



        self._update_view_all_button_state()



        if hasattr(self, 'search_method_result_label'):



            self.search_method_result_label.setText("")



        self._last_rerank_success = False

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

        if hasattr(self, 'view_btn'):



            self.view_btn.setEnabled(False)



        if hasattr(self, 'selected_count_label'):



            self.selected_count_label.setText("(0 selected)")



        self._last_formatted_answer = None



        if hasattr(self, 'search_progress_bar') and self.search_progress_bar:



            self.search_progress_bar.setVisible(False)



        if hasattr(self, 'search_progress_label') and self.search_progress_label:



            self.search_progress_label.setVisible(False)



        try:



            self._search_pending_query = query
            self._original_search_query = query



            self._search_pending_config = config
            if composer_image_payloads and hasattr(self, "_clear_composer_image_attachment"):
                self._clear_composer_image_attachment()



            if search_config_for_plan.get("enable_agentic_rag", False):
                if hasattr(self, "_set_search_stage"):
                    self._set_search_stage("Planning retrieval...", chat_message=True, progress=0)
                self._search_pending_async = True
                self._agentic_plan_worker = AgenticPlanWorker(
                    self._build_agentic_search_plan,
                    query,
                    history,
                    config,
                )
                self._agentic_plan_worker.finished_signal.connect(self._on_agentic_plan_ready)
                self._agentic_plan_worker.error_signal.connect(self._on_agentic_plan_error)
                self._agentic_plan_worker.start()
                return

            if hasattr(self, "_set_search_stage"):
                self._set_search_stage("Searching notes...", chat_message=True)
            else:
                self.status_label.setText("Searching notes...")

            self._start_note_load_for_search(query, config)



            return



        except Exception as e:



            log_debug(f"Unexpected error in perform_search: {type(e).__name__}: {str(e)}")



            import traceback



            log_debug(f"Traceback: {traceback.format_exc()}")



            self.answer_box.setText(



                f"Unexpected Error:\n{str(e)}\n\n"



                "Please check the debug log for details."



            )



            self.status_label.setText("Error occurred")



            self._search_pending_async = False



        finally:



            if not getattr(self, '_search_pending_async', False) and not getattr(self, '_pending_rerank', False):



                self.search_btn.setEnabled(True)
                if hasattr(self, "_set_search_chat_busy"):
                    self._set_search_chat_busy(False)



            self._hide_busy_progress()



    def _on_get_notes_done(self, payload):



        """Called when background get_notes_content_with_col finishes. Starts keyword_filter in worker."""



        import time



        self._search_pending_async = False



        self._stop_estimated_progress_timer()



        self._hide_busy_progress()



        if payload is None or not isinstance(payload, (list, tuple)) or len(payload) != 3:



            self.search_btn.setEnabled(True)
            if hasattr(self, "_set_search_chat_busy"):
                self._set_search_chat_busy(False)



            self.status_label.setText("Ready")



            return



        notes, fields_description, cache_key = payload

        config = getattr(self, '_search_pending_config', None) or load_config()
        if (config.get("search_config") or {}).get("verbose_search_debug"):
            started = getattr(self, '_note_load_started', None)
            elapsed = (time.time() - started) if started else 0
            log_debug(f"Retrieval diagnostics: note_load_seconds={elapsed:.3f}, loaded_rows={len(notes)}")



        self.fields_description = fields_description



        self._cached_notes = notes



        self._cached_notes_key = cache_key



        unique_note_count = len(set(n['id'] for n in notes)) if notes else 0



        if not notes:



            message = f"No notes with {fields_description} content found in your collection."
            if hasattr(self, "_append_system_chat_message"):
                self._set_chat_transient(None)
                self._append_system_chat_message(message, kind="warning")
            else:
                self.answer_box.setText(message)



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.status_label.setText("Ready")



            self.search_btn.setEnabled(True)
            if hasattr(self, "_set_search_chat_busy"):
                self._set_search_chat_busy(False)



            return



        self.total_notes_searched = unique_note_count



        self._search_pending_notes = notes
        self._original_search_query = self._search_pending_query
        self._last_embedding_query = None



        if hasattr(self, "_set_search_stage"):
            self._set_search_stage(f"Searching {unique_note_count} notes...")
        else:
            self.status_label.setText(f"Searching {unique_note_count} notes...")



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
            if hasattr(self, "_set_search_chat_busy"):
                self._set_search_chat_busy(False)



            return



        if isinstance(result, tuple) and result[0] == "PENDING_EMBEDDING":



            # Embedding search will run in background worker; show progress and return



            _, embedding_query, notes_for_embedding, state = result



            self._search_pending_state = state



            self._search_pending_notes = notes
            self._last_embedding_query = embedding_query



            setattr(self, "_last_embedding_error", None)
            self._last_embedding_rows_checked = 0
            self._last_embedding_exact_matches = 0
            self._last_embedding_note_id_fallback_matches = 0



            self.current_query = state["query"]



            config = state["config"]



            if hasattr(self, 'search_progress_bar') and self.search_progress_bar:



                self.search_progress_bar.setVisible(True)



                self.search_progress_bar.setRange(0, 100)



                self.search_progress_bar.setValue(0)



            if hasattr(self, 'search_progress_label') and self.search_progress_label:



                self.search_progress_label.setVisible(True)



                self.search_progress_label.setText("Starting embedding search...")



            if hasattr(self, "_set_search_stage"):
                self._set_search_stage("Searching notes with Hybrid retrieval...", chat_message=True, progress=0)
            else:
                self.status_label.setText("Searching notes with Hybrid retrieval...")

            if len(notes_for_embedding or []) > 2000 and not getattr(self, "_shown_numpy_performance_tip", False):
                if not is_numpy_available():
                    self._shown_numpy_performance_tip = True
                    show_numpy_install_guidance(
                        self,
                        reason=(
                            "NumPy is missing from Anki's Python; semantic search will use a slower fallback "
                            "for this large collection."
                        ),
                    )



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
            scored_notes, effective_method = self._maybe_apply_agentic_pass2(
                scored_notes,
                notes,
                config.get("search_config") or {},
                effective_method,
            )



            self.current_query = query



            self._pending_rerank = True



            search_config = config.get('search_config') or {}
            pre_budget = self._dynamic_note_budget(
                query,
                scored_notes,
                search_config,
                config.get("provider", "openai"),
                phase="rerank",
                pinned_count=len((getattr(self, "selected_note_ids", set()) or set()) | (getattr(self, "_pinned_note_ids", set()) or set())),
                rerank_used=False,
            )
            rerank_limit = pre_budget.get("rerank_candidate_limit", MAX_RERANK_COUNT)
            self._last_dynamic_note_budget = pre_budget



            self._rerank_continue = (notes, effective_method, search_config)
            self._analytics_pre_rerank_results = self._analytics_results(scored_notes, limit=100)

            self._analytics_stage(
                "pre_cross_encoder_candidates",
                effective_method=effective_method,
                total_candidates=len(scored_notes or []),
                top_results=self._analytics_pre_rerank_results[:50],
            )



            if hasattr(self, 'status_label') and self.status_label:



                self.status_label.setText("Reranking strongest matches...")



            if hasattr(self, "_set_chat_transient"):
                self._set_chat_transient("Reranking strongest matches...", mode="info")

            self._show_centile_progress("Reranking strongest matches...", 0)



            self._start_estimated_progress_timer(25, 5, 95)



            try:



                from aqt.operations import QueryOp



                op = QueryOp(



                    parent=mw,



                    op=lambda col: _do_rerank(query, scored_notes, rerank_limit, search_config),



                    success=lambda pair: self._on_rerank_done(pair[0], pair[1]),



                )



                op.run_in_background()



            except Exception:



                self._rerank_worker = RerankWorker(query, scored_notes, rerank_limit, search_config)



                self._rerank_worker.finished_signal.connect(self._on_rerank_done)



                self._rerank_worker.start()



            return



        scored_notes, effective_method, total_above_threshold = result
        scored_notes, effective_method = self._maybe_apply_agentic_pass2(
            scored_notes,
            notes,
            config.get("search_config") or {},
            effective_method,
        )



        self.current_query = query



        self._perform_search_continue(notes, scored_notes, effective_method, total_above_threshold)



        self.search_btn.setEnabled(True)
        if hasattr(self, "_set_search_chat_busy"):
            self._set_search_chat_busy(False)


