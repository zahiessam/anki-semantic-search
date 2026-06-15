"""Search continuation flow after retrieval has produced scored notes."""

import datetime

from aqt.qt import QApplication

from .search_workers import AskAIWorker
from ..core.engine import (
    estimate_tokens,
    get_cached_query_embedding,
    get_embedding_engine_id,
    load_embedding_engine_counts,
)
from ..utils import (
    format_dimension_mismatch_hint,
    get_embeddings_db_path,
    get_retrieval_config,
    load_config,
    load_search_history,
    log_debug,
)
from ..core.agentic_runtime import (
    bounded_context_limit,
    extract_session_fact_cache,
)
from ..core.memory import (
    memory_profile_summary,
    save_fact_snippets_from_context,
    schedule_memory_embeddings_for_ids,
    update_profile_from_search,
)
from ..utils.text import merge_overlapping_note_chunks


class SearchContinuationMixin:
    def _agentic_result_method_label(self, effective_method):
        plan = getattr(self, "_agentic_plan", None) or {}
        if plan.get("planner_version") != "adaptive_v2":
            return ""
        mode = (plan.get("retrieval_mode") or "").strip()
        if mode == "bm25_only":
            method = "BM25-only"
        elif mode == "hybrid":
            method = "Hybrid"
        else:
            method = effective_method or "retrieval"
        passes = int(getattr(self, "_agentic_passes_run", 1) or 1)
        if passes >= 2 and "pass 2" not in method.lower():
            method = f"{method} + pass 2"
        prefix = "Agentic V2 fallback" if plan.get("planner_source") == "fallback" else "Agentic V2"
        label = f"{prefix} \u00b7 {method}"
        if plan.get("planner_source") == "fallback" and plan.get("llm_planner_error"):
            label = f"{label} (planner fallback)"
        return label

    def _agentic_conflict_signal(self, scored_notes):
        top = list(scored_notes or [])[:2]
        if len(top) < 2:
            return {"conflict": False}
        (score_a, note_a), (score_b, note_b) = top
        close_rank = abs(float(score_a or 0) - float(score_b or 0)) <= 8.0
        different_notes = note_a.get("id") != note_b.get("id")
        bm25_a = float(note_a.get("_keyword_score") or note_a.get("_retrieval_score") or score_a or 0)
        bm25_b = float(note_b.get("_keyword_score") or note_b.get("_retrieval_score") or score_b or 0)
        divergent_keyword = abs(bm25_a - bm25_b) >= 15.0
        return {
            "conflict": bool(different_notes and close_rank and divergent_keyword),
            "top_note_ids": [note_a.get("id"), note_b.get("id")],
            "score_delta": round(abs(float(score_a or 0) - float(score_b or 0)), 3),
            "keyword_delta": round(abs(bm25_a - bm25_b), 3),
        }

    def _perform_search_continue(self, notes, scored_notes, effective_method, total_above_threshold):



        """Continue search after keyword_filter (or after embedding worker): display results, call AI, etc."""



        config = load_config()



        query = getattr(self, 'current_query', '')
        answer_query = getattr(self, '_notes_answer_query', None) or query



        log_debug(f"Filtered to {len(scored_notes)} potentially relevant notes (method: {effective_method}, total above threshold: {total_above_threshold})")

        self._analytics_stage(
            "retrieval_finalized",
            effective_method=effective_method,
            total_candidates=len(scored_notes or []),
            total_above_threshold=total_above_threshold,
            top_results=self._analytics_results(scored_notes, limit=50),
        )
        if getattr(self, "_agentic_plan", None):
            self._analytics_stage(
                "agentic_retrieval_finalized",
                passes_run=getattr(self, "_agentic_passes_run", 1),
                plan=getattr(self, "_agentic_plan", None),
                evidence=getattr(self, "_agentic_evidence_decision", None),
                conflict_signal=self._agentic_conflict_signal(scored_notes),
            )
        try:
            if (config.get("search_config") or {}).get("enable_profile_memory", True):
                update_profile_from_search(
                    config,
                    getattr(self, "_agentic_plan", None),
                    {
                        "result_count": len(scored_notes or []),
                        "passes_run": getattr(self, "_agentic_passes_run", 1),
                    },
                )
        except Exception as exc:
            log_debug(f"Profile memory update skipped: {exc}")

        retrieval = get_retrieval_config(config)
        agentic_plan = getattr(self, "_agentic_plan", None) or {}
        if agentic_plan.get("intent") == "broad_review":
            retrieval = dict(retrieval)
            retrieval["mmr_lambda"] = float(agentic_plan.get("broad_review_mmr_lambda") or 0.6)
        for score, note in scored_notes:
            note['_retrieval_score'] = score
        scored_notes = self._apply_mmr_diversity(scored_notes, retrieval, config)
        if scored_notes and getattr(self, '_mmr_applied', False):

            self._analytics_stage(
                "mmr_diversity_applied",
                candidate_pool_size=getattr(self, "_mmr_candidate_pool_size", None),
                candidates_after_mmr=len(scored_notes),
                top_results=self._analytics_results(scored_notes, limit=50),
            )

            self._scored_notes_for_context = None
        if (config.get("search_config") or {}).get("verbose_search_debug"):
            log_debug(
                "Retrieval V2 active: "
                f"scorer={retrieval.get('keyword_scoring_method')}, "
                f"candidates={len(scored_notes)}"
            )


        search_config = config.get("search_config") or {}
        final_budget = self._dynamic_note_budget(
            query,
            scored_notes,
            search_config,
            config.get("provider", "openai"),
            phase="final",
            pinned_count=len((getattr(self, "selected_note_ids", set()) or set()) | (getattr(self, "_pinned_note_ids", set()) or set())),
            rerank_used=bool(getattr(self, "_last_rerank_success", False)),
        )
        self._last_dynamic_note_budget = final_budget

        self.all_scored_notes = scored_notes



        self._total_above_threshold = total_above_threshold



        self._last_search_method = effective_method



        if hasattr(self, 'search_method_result_label'):



            agentic_label = self._agentic_result_method_label(effective_method)
            label_text = agentic_label or self._build_result_source_text(config, effective_method)



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



                    hint = "Embedding unavailable: no active-engine rows found"
                    try:
                        active_engine = get_embedding_engine_id(config)
                        engine_counts = load_embedding_engine_counts()
                        other_counts = {
                            engine: count
                            for engine, count in engine_counts.items()
                            if engine != active_engine and int(count or 0) > 0
                        }
                        if not int(engine_counts.get(active_engine, 0) or 0) and other_counts:
                            other_engine, count = max(other_counts.items(), key=lambda item: int(item[1] or 0))
                            hint = (
                                f"Wrong engine: {int(count):,} embeddings exist for {other_engine}, "
                                f"but active engine is {active_engine}. Switch back or rebuild."
                            )
                    except Exception:
                        pass



                base_label = agentic_label or f"Results from: {effective_method}"
                self.search_method_result_label.setText(f"{base_label} ({hint})")



        if getattr(self, "_related_notes_only", False):
            self._related_notes_only = False
            self.set_search_mode("ask_notes") if hasattr(self, "set_search_mode") else None
            self._pending_related_notes_query = ""
            self._sources_rank_mode = True
            self._cited_note_ids = set()
            self._cited_refs = set()
            self._display_scored_notes = None
            self._context_note_ids = []
            self._context_note_id_and_chunk = None
            self._context_note_identity_keys = None
            self._set_chat_transient(None) if hasattr(self, "_set_chat_transient") else None
            if hasattr(self, "filter_and_display_notes"):
                self.filter_and_display_notes()
            if hasattr(self, "_expand_sources_panel"):
                self._expand_sources_panel(manual=True)
            self.status_label.setText("Related notes ready" if scored_notes else "No related notes found")
            self._set_search_chat_busy(False) if hasattr(self, "_set_search_chat_busy") else None
            if hasattr(self, "clear_chat_btn"):
                self.clear_chat_btn.setEnabled(True)
            if hasattr(self, "_notify_long_job_done"):
                count = len(scored_notes or [])
                self._notify_long_job_done(
                    "Related notes complete",
                    f"{count} related notes ready" if count else "No related notes found",
                )
            return



        if not scored_notes:



            n_searched = getattr(self, 'total_notes_searched', None) or len(set(n['id'] for n in notes))



            message = f"No notes found matching keywords from your query. Searched {n_searched} notes ({getattr(self, 'fields_description', 'Text & Extra')})."
            if hasattr(self, "_append_system_chat_message"):
                self._set_chat_transient(None)
                self._append_system_chat_message(message, kind="warning")
                self._set_search_chat_busy(False)
            else:
                self.answer_box.setText(message)



            if hasattr(self, 'answer_source_label'):



                self.answer_source_label.setText("")



            self.status_label.setText("No matches found")

            self._analytics_stage(
                "no_matches",
                searched_notes=n_searched,
                fields=getattr(self, 'fields_description', 'Text & Extra'),
            )

            self._write_search_analytics_report(
                completed_at=datetime.datetime.now().isoformat(),
                final_status="no_matches",
                searched_notes=n_searched,
                final_results=[],
            )
            if hasattr(self, "_notify_long_job_done"):
                self._notify_long_job_done("Ask Notes complete", "No matching notes found")



            return



        # Use un-aggregated list for AI context when we have chunks so the AI can cite specific sections



        raw_for_context = getattr(self, '_scored_notes_for_context', None)



        if raw_for_context:



            relevant_notes = [note for _, note in raw_for_context]



        else:



            relevant_notes = [note for _, note in scored_notes]



        review_context_active = bool((getattr(self, "_review_context_text", "") or "").strip())
        history_result = None if review_context_active else load_search_history(query)
        if review_context_active:
            log_debug("Skipping cached search result because current review context is active")
        if history_result and not history_result.get("context_note_identity_keys"):
            seen_note_ids = set()
            has_duplicate_note_ids = False
            for _score, note in scored_notes or []:
                note_id = note.get("id")
                if note_id in seen_note_ids:
                    has_duplicate_note_ids = True
                    break
                seen_note_ids.add(note_id)
            if has_duplicate_note_ids:
                log_debug(
                    "Ignoring legacy cached search result because it lacks chunk identity "
                    "and current results contain duplicate note IDs"
                )
                history_result = None



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



            answer = history_result.get('answer', '')



            relevant_indices = []



            used_history = True



            self._context_note_ids = history_result.get('context_note_ids') or []



            raw_context_pairs = history_result.get('context_note_id_and_chunk') or []
            self._context_note_id_and_chunk = [
                (item[0], item[1] if len(item) > 1 else None)
                for item in raw_context_pairs
                if isinstance(item, (list, tuple)) and item
            ] or None
            raw_identity_keys = history_result.get('context_note_identity_keys') or []
            self._context_note_identity_keys = [
                (item[0], item[1] if len(item) > 1 else None, item[2] if len(item) > 2 else None)
                for item in raw_identity_keys
                if isinstance(item, (list, tuple)) and item
            ] or None



            self._display_scored_notes = None  # History uses aggregated list



            if 'scored_notes' in history_result:



                history_scored = []



                note_key_map = {
                    (note.get('id'), note.get('chunk_index'), note.get('content_hash')): note
                    for _, note in scored_notes
                }
                note_id_map = {note['id']: note for _, note in scored_notes}



                for score, hist_note in history_result['scored_notes']:



                    note_id = hist_note.get('id')



                    hist_key = (
                        note_id,
                        hist_note.get('chunk_index'),
                        hist_note.get('content_hash'),
                    )

                    if hist_key in note_key_map:



                        history_scored.append((score, note_key_map[hist_key]))



                    elif note_id in note_id_map:



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



            relevant_note_ids = set(history_result.get('relevant_note_ids', []))



            self._cited_note_ids = relevant_note_ids



            self._cited_refs = {
                idx + 1
                for idx, note_id in enumerate(self._context_note_ids or [])
                if note_id in relevant_note_ids
            }



            if answer and str(answer).strip():



                log_debug(
                    "Displaying cached answer "
                    f"(answer length: {len(answer)}, cited notes: {len(relevant_note_ids)})"
                )



                self._display_answer_and_notes_after_rerank(answer, config, used_history, notes)



                return



            log_debug("Cached search result had an empty answer; showing matching notes only", is_error=True)



            message = "Cached search result had no saved answer. Run a fresh search after clearing history."
            if hasattr(self, "_append_system_chat_message"):
                self._set_chat_transient(None)
                self._append_system_chat_message(message, kind="warning")
            else:
                self.answer_box.setPlainText(message)



            self.filter_and_display_notes()



            return



        else:



            # Cap notes/chunks sent to the AI (avoids rate limits and token overflow; chunked results can be huge)



            search_config = config.get('search_config') or {}



            provider_for_context = config.get('provider', 'openai')



            local_context_plan = self._local_context_usage_plan(
                query, len(relevant_notes), provider_for_context, search_config
            )


            selected_ids = set(getattr(self, 'selected_note_ids', set()) or [])



            pinned_ids = set(getattr(self, '_pinned_note_ids', set()) or [])



            priority_ids = selected_ids | pinned_ids



            context_budget = self._dynamic_note_budget(
                query,
                scored_notes,
                search_config,
                provider_for_context,
                local_context_plan=local_context_plan,
                phase="final",
                pinned_count=len(priority_ids),
                rerank_used=bool(getattr(self, "_last_rerank_success", False)),
            )
            self._last_dynamic_note_budget = context_budget



            if local_context_plan:



                max_context = max(3, int(context_budget.get("context_limit") or local_context_plan['max_notes']))



            else:



                max_context = max(5, int(context_budget.get("context_limit") or 12))
            max_context = bounded_context_limit(max_context)

            agentic_plan = getattr(self, "_agentic_plan", None) or {}
            if agentic_plan.get("intent") == "broad_review":
                broad_cap = max(3, int(agentic_plan.get("broad_review_max_context_notes") or 12))
                max_context = min(max_context, broad_cap)

            original_context_limit = max_context
            cliff_plan = self._context_score_cliff_plan(
                scored_notes,
                search_config,
                rerank_used=bool(getattr(self, "_last_rerank_success", False)),
            )
            cliff_cutoff = cliff_plan.get("cutoff_rank")
            if cliff_cutoff:
                max_context = min(max_context, int(cliff_cutoff))
            cliff_plan["original_context_limit"] = original_context_limit
            cliff_plan["final_context_limit"] = max_context

            context_filter = self._filter_context_notes_by_display_relevance(
                relevant_notes,
                priority_ids=priority_ids,
            )
            context_candidate_notes = context_filter["notes"]
            zero_dr_excluded_note_ids = []
            priority_ids_for_zero_dr = set(priority_ids or [])
            for note in relevant_notes or []:
                if not isinstance(note, dict):
                    continue
                note_id = note.get("id")
                if note_id in priority_ids_for_zero_dr:
                    continue
                display_relevance = note.get("_display_relevance")
                if isinstance(display_relevance, (int, float)) and display_relevance <= 0:
                    zero_dr_excluded_note_ids.append(note_id)


            if priority_ids:



                prioritized = [n for n in context_candidate_notes if n['id'] in priority_ids]



                remaining = [n for n in context_candidate_notes if n['id'] not in priority_ids]



                remaining_limit = max(0, max_context - len(prioritized))



                context_notes = prioritized + remaining[:remaining_limit]



            else:



                context_notes = list(context_candidate_notes)[:max_context]

            rescue_enabled = bool(search_config.get("enable_context_anchor_rescue", True))
            rescue_method = "skipped"
            rescue_skipped_reason = None
            rescued_notes = []
            rescued_note_scores = []
            rescue_similarity_floor = None
            query_cache_key = None
            query_cache_hit = None
            rescue_original_query = getattr(self, "_original_search_query", None) or query
            rescue_meta = {}

            if rescue_enabled:
                selected_ids = {n.get("id") for n in context_notes if n.get("id") is not None}
                rescue_candidates = [n for n in context_candidate_notes if n.get("id") not in selected_ids]
                cached_query = get_cached_query_embedding(rescue_original_query, config)
                query_cache_key = cached_query.get("cache_key")
                query_cache_hit = bool(cached_query.get("cache_hit"))
                log_debug(
                    "Context rescue query embedding cache: "
                    f"original_query={rescue_original_query!r}, "
                    f"key={query_cache_key!r}, hit={query_cache_hit}"
                )
                query_embedding = cached_query.get("embedding")
                if query_embedding:
                    all_note_ids = [
                        n.get("id")
                        for n in list(context_notes) + list(rescue_candidates)
                        if n.get("id") is not None
                    ]
                    note_embeddings = self._get_note_embeddings_for_rescue(
                        all_note_ids,
                        db_path=get_embeddings_db_path(),
                        engine_id=get_embedding_engine_id(config),
                    )
                    if note_embeddings:
                        rescued_notes, rescue_meta = self._score_rescue_candidates(
                            query_embedding,
                            rescue_candidates,
                            context_notes,
                            note_embeddings,
                            search_config,
                            all_candidate_notes=context_candidate_notes,
                            query=query,
                        )
                        rescue_similarity_floor = rescue_meta.get("rescue_similarity_floor")
                        rescued_note_scores = rescue_meta.get("candidate_scores") or []
                        if rescue_similarity_floor is None:
                            rescue_skipped_reason = "selected_note_embeddings_unavailable"
                        else:
                            rescue_method = "embedding_similarity"
                    else:
                        rescue_skipped_reason = "note_embeddings_unavailable"
                else:
                    rescue_skipped_reason = "embedding_unavailable_original_query_cache_miss"

                if rescue_method == "skipped":
                    fallback_notes, fallback_meta = self._fallback_anchor_rescue_candidates(
                        query,
                        rescue_candidates,
                        context_notes,
                        search_config,
                        all_candidate_notes=context_candidate_notes,
                    )
                    if fallback_notes:
                        rescued_notes = fallback_notes
                        rescue_method = "specificity_only_fallback"
                        rescue_skipped_reason = None
                    else:
                        rescue_skipped_reason = rescue_skipped_reason or "no_specificity_anchors"
                    cliff_plan["anchor_terms_detected"] = fallback_meta.get("anchor_terms_detected") or []
                    cliff_plan["rescue_scoring_method"] = fallback_meta.get("rescue_scoring_method")
                    cliff_plan["specificity_weight_used"] = fallback_meta.get("specificity_weight_used")
                    cliff_plan["rescued_note_specificity_scores"] = fallback_meta.get("rescued_note_specificity_scores") or []

                if rescued_notes:
                    for rescued in rescued_notes:
                        if rescue_method == "embedding_similarity":
                            rescued["_rescued_by_anchor_similarity"] = True
                        elif rescue_method == "specificity_only_fallback":
                            rescued["_rescued_by_anchor_fallback"] = True
                    context_notes = list(context_notes) + list(rescued_notes)

            cliff_plan.update({
                "anchor_rescue_enabled": rescue_enabled,
                "rescue_method": rescue_method,
                "rescue_original_query": rescue_original_query,
                "rescue_query_embedding_cache_key": repr(query_cache_key) if query_cache_key is not None else None,
                "rescue_query_embedding_cache_hit": query_cache_hit,
                "rescue_similarity_floor": rescue_similarity_floor,
                "rescued_note_ids": [n.get("id") for n in rescued_notes],
                "rescued_note_count": len(rescued_notes),
                "rescued_note_scores": rescued_note_scores,
                "rescue_skipped_reason": rescue_skipped_reason,
                "anchor_terms_detected": cliff_plan.get("anchor_terms_detected", rescue_meta.get("anchor_terms_detected") or []),
                "rescue_scoring_method": cliff_plan.get("rescue_scoring_method", rescue_meta.get("rescue_scoring_method")),
                "specificity_weight_used": cliff_plan.get("specificity_weight_used", rescue_meta.get("specificity_weight_used") or 0.0),
                "rescued_note_specificity_scores": cliff_plan.get("rescued_note_specificity_scores", rescue_meta.get("rescued_note_specificity_scores") or []),
                "filtered_by_specificity": rescue_meta.get("filtered_by_specificity", 0),
                "zero_dr_excluded_count": len(zero_dr_excluded_note_ids),
                "zero_dr_excluded_note_ids": zero_dr_excluded_note_ids[:25],
                "context_display_relevance_zero_excluded_count": context_filter["excluded_count"],
                "context_display_relevance_zero_excluded_note_ids": context_filter["excluded_note_ids"][:25],
            })
            self._last_context_score_cliff = cliff_plan
            self._analytics_stage("context_score_cliff_evaluated", **cliff_plan)
            log_debug(
                "Context score cliff: "
                f"enabled={cliff_plan.get('enabled')}, "
                f"threshold={cliff_plan.get('threshold')}, "
                f"score_source={cliff_plan.get('score_source')}, "
                f"display_relevance={cliff_plan.get('display_relevance_count')}/"
                f"{cliff_plan.get('candidate_count')}, "
                f"largest_gap={cliff_plan.get('largest_gap'):.3f}, "
                f"cutoff_rank={cliff_plan.get('cutoff_rank')}, "
                f"context_limit={original_context_limit}->{max_context}, "
                f"rescue_method={rescue_method}, "
                f"rescued_note_ids={cliff_plan.get('rescued_note_ids')}, "
                f"rescued_note_scores={rescued_note_scores}"
            )

            context_notes = merge_overlapping_note_chunks(context_notes)
            try:
                session_memory = getattr(self, "_agentic_session_memory", None)
                if session_memory is not None:
                    label = "main"
                    plan = getattr(self, "_agentic_plan", None) or {}
                    subqueries = plan.get("subqueries") or []
                    if subqueries:
                        label = str(subqueries[0] or "main")
                    session_memory.fact_cache = extract_session_fact_cache(context_notes, label)
                    report = getattr(self, "_search_analytics_report", None)
                    if isinstance(report, dict):
                        memory_report = dict(report.get("memory") or {})
                        memory_report["session_cache_item_count"] = len(session_memory.fact_cache)
                        report["memory"] = memory_report
                    self._analytics_stage(
                        "session_fact_cache_extracted",
                        session_cache_item_count=len(session_memory.fact_cache),
                    )
            except Exception as exc:
                log_debug(f"Session fact cache extraction skipped: {exc}")
            review_note = self._review_context_note_for_answer() if hasattr(self, "_review_context_note_for_answer") else None
            if review_note:
                review_note_id = review_note.get("id")
                context_notes = [n for n in context_notes if n.get("id") != review_note_id]
                context_notes.insert(0, review_note)
                self._sources_rank_mode = False
            max_context = len(context_notes)


            # Alignment Fix: Ensure all notes sent to the AI are visible in the table.
            # This prevents gaps in Ref numbering and ensure citations like [20] match visible rows.
            for n in context_notes:
                n['_passes_broad'] = True

            # Alignment Fix: If we prioritized (pinned) notes, they must move to the top
            # of the table as well so that they are labeled Ref 1, Ref 2, etc.
            if priority_ids:
                # Find all context notes in scored_notes
                ctx_ids = {n['id'] for n in context_notes}
                remaining_scored = [pair for pair in scored_notes if pair[1]['id'] not in ctx_ids]

                # Rebuild scored_notes with context_notes (in AI order) at the top
                # note: scores for context_notes are preserved inside display_pairs logic below
                new_scored = []
                # map ids to scores for context notes
                id_to_orig_score = {n['id']: pair[0] for pair in scored_notes if n['id'] == pair[1]['id']}
                for cn in context_notes:
                    new_scored.append((id_to_orig_score.get(cn['id'], 100 if cn.get("_review_context_note") else 0), cn))
                new_scored.extend(remaining_scored)
                scored_notes = new_scored
                self.all_scored_notes = scored_notes

            context_note_ids = [n['id'] for n in context_notes]



            # Store (note_id, chunk_index) in context order so Ref column and citation [N] match



            self._context_note_id_and_chunk = [(n['id'], n.get('chunk_index')) for n in context_notes]
            self._context_note_identity_keys = [
                (n['id'], n.get('chunk_index'), n.get('content_hash'))
                for n in context_notes
            ]
            log_debug(
                "Ref gap debug: context_built "
                f"context_notes={len(context_notes)}, "
                f"context_identity_keys={len(self._context_note_identity_keys)}, "
                f"first_refs={self._context_note_identity_keys[:12]}"
            )



            # Store (score, note) for each context item so display can show all refs the AI can cite



            if raw_for_context:



                note_to_score = {}
                note_id_to_best_score = {}



                for s, n in raw_for_context:



                    key = (n['id'], n.get('chunk_index'), n.get('content_hash'))



                    note_to_score[key] = s
                    nid = n.get('id')
                    if nid is not None and (nid not in note_id_to_best_score or s > note_id_to_best_score[nid]):
                        note_id_to_best_score[nid] = s



                display_pairs = []



                for n in context_notes:



                    k = (n['id'], n.get('chunk_index'), n.get('content_hash'))



                    display_pairs.append((note_to_score.get(k, note_id_to_best_score.get(n.get('id'), 0)), n))



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



                merged_indexes = n.get('_merged_chunk_indexes') or []

                if len(merged_indexes) > 1:

                    sections = ", ".join(str(int(idx) + 1) for idx in merged_indexes)

                    return f"Note {i+1} (sections {sections} of note ID {n['id']}): {text}"



                if chunk_idx is not None:



                    return f"Note {i+1} (section {chunk_idx + 1} of note ID {n['id']}): {text}"



                return f"Note {i+1}: {text}"



            context_lines = [_context_line(i, n) for i, n in enumerate(context_notes)]



            if local_context_plan:



                context_lines = self._fit_context_lines_to_token_budget(
                    context_lines, local_context_plan['context_token_budget']
                )



                if len(context_lines) < len(context_notes):



                    context_notes = context_notes[:len(context_lines)]



                    context_note_ids = context_note_ids[:len(context_lines)]



                    self._context_note_id_and_chunk = self._context_note_id_and_chunk[:len(context_lines)]
                    self._context_note_identity_keys = self._context_note_identity_keys[:len(context_lines)]



                    if self._display_scored_notes:



                        self._display_scored_notes = self._display_scored_notes[:len(context_lines)]

                    log_debug(
                        "Ref gap debug: context_trimmed_for_token_budget "
                        f"context_lines={len(context_lines)}, context_notes={len(context_notes)}, "
                        f"context_identity_keys={len(self._context_note_identity_keys)}, "
                        f"display_scored_notes={len(self._display_scored_notes or [])}"
                    )



                log_debug(
                    "Local context plan: "
                    f"{local_context_plan['mode']}, notes={len(context_notes)}, "
                    f"context_tokens~{estimate_tokens(chr(10).join(context_lines))}, "
                    f"answer_tokens={local_context_plan['max_output_tokens']}"
                )



            try:
                if (config.get("search_config") or {}).get("enable_profile_memory", True):
                    plan = getattr(self, "_agentic_plan", None) or {}
                    subqueries = plan.get("subqueries") or []
                    label = str(subqueries[0] or "main") if subqueries else "main"
                    save_result = save_fact_snippets_from_context(
                        context_notes,
                        source_query=answer_query,
                        subquery_label=label,
                        max_snippets=int((config.get("search_config") or {}).get("memory_max_saved_snippets_per_search", 24) or 24),
                        retention_days=int((config.get("search_config") or {}).get("memory_retention_days", 30) or 30),
                        config=config,
                    )
                    schedule_result = schedule_memory_embeddings_for_ids(
                        save_result.get("saved_ids") or [],
                        config=config,
                    )
                    report = getattr(self, "_search_analytics_report", None)
                    if isinstance(report, dict):
                        memory_report = dict(report.get("memory") or {})
                        memory_report["durable_snippets_saved"] = save_result.get("saved", 0)
                        memory_report["durable_snippets_pruned"] = save_result.get("pruned", 0)
                        memory_report["embedding_background_scheduled"] = bool(
                            memory_report.get("embedding_background_scheduled")
                            or schedule_result.get("scheduled")
                        )
                        memory_report["embedding_background_scheduled_count"] = int(
                            memory_report.get("embedding_background_scheduled_count", 0) or 0
                        ) + int(schedule_result.get("count") or 0)
                        try:
                            summary = memory_profile_summary()
                            memory_report["durable_snippet_count"] = summary.get("snippet_count", 0)
                        except Exception:
                            pass
                        report["memory"] = memory_report
                    self._analytics_stage(
                        "durable_memory_snippets_saved",
                        durable_snippets_saved=save_result.get("saved", 0),
                        durable_snippets_pruned=save_result.get("pruned", 0),
                    )
            except Exception as exc:
                log_debug(f"Durable memory snippet save skipped: {exc}")

            context = "\n\n".join(context_lines)



            n_notes = len(context_notes)



            if hasattr(self, "_set_search_stage"):
                self._set_search_stage("Preparing answer from selected notes...", chat_message=True)
            else:
                self.status_label.setText("Preparing answer from selected notes...")



            self._show_centile_progress("Answering from your notes...", 0)



            self._start_estimated_progress_timer(30, 5, 95)



            if hasattr(self, "_set_chat_transient"):
                self._set_chat_transient("Answering from your notes...", mode="info")
            else:
                self.answer_box.setPlainText("Thinking...")



            provider = config.get('provider', 'openai')



            if provider == "anthropic":



                try:



                    self._start_anthropic_stream(answer_query, context_notes, context, config, relevant_notes, scored_notes, context_note_ids)



                    return



                except Exception as e:



                    log_debug(f"Anthropic streaming not available, falling back to non-streaming: {e}")



            # Run ask_ai in background so UI stays responsive (avoids "Not Responding" during 10\xe2\u20ac\u201c300 s request)



            self._ask_ai_relevant_notes = context_notes



            self._ask_ai_scored_notes = scored_notes



            self._ask_ai_context_note_ids = context_note_ids



            self._ask_ai_used_history = used_history



            self._ask_ai_notes = notes



            self._ask_ai_config = config
            self._ask_ai_context = context
            if hasattr(self, "_set_search_stage"):
                self._set_search_stage("Answering from your notes...", chat_message=True)
            log_debug(
                "AI call context diagnostics: "
                f"context_notes={len(context_notes)}, "
                f"context_note_ids={list(context_note_ids or [])}, "
                f"prompt_context_non_empty={bool((context or '').strip())}"
            )



            self._ask_ai_worker = AskAIWorker(self, answer_query, context_notes, context, config)



            self._streamed_answer = ""



            self._ask_ai_worker.chunk_signal.connect(self._append_stream_chunk)



            self._ask_ai_worker.success_signal.connect(self._on_ask_ai_success)



            self._ask_ai_worker.error_signal.connect(self._on_ask_ai_error)



            self._ask_ai_worker.finished.connect(self._on_ask_ai_worker_finished)



            self._ask_ai_worker.start()



            return


