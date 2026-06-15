import re

from ..core.engine import estimate_tokens


GENERAL_COMPLEX_MARKERS = (
    'compare', 'differentiate', 'difference', 'mechanism', 'pathway',
    'steps', 'sequence', 'algorithm', 'diagnosis', 'management',
    'treatment', 'why', 'explain', 'list', 'table', 'vs', 'versus',
    'all', 'causes', 'risk factors', 'complications'
)

CLINICAL_COMPLEX_MARKERS = (
    'presentation', 'distinguish', 'mimics', 'differential',
    'first-line', 'side-effect', 'side-effects', 'adverse effect',
    'contraindication', 'contraindications', 'innervation', 'blood supply',
    'gold-standard', 'triad', 'criteria', 'diagnostic criteria',
)


class SearchContextPlanningMixin:
    def _filter_context_notes_by_display_relevance(self, notes, priority_ids=None):
        """Exclude ordinary notes below the user-facing relevance threshold from AI context."""
        priority_ids = set(priority_ids or [])
        try:
            from ..utils import clamp_relevance_threshold_percent, load_config

            live_value = getattr(self, "_effective_relevance_threshold_percent", None)
            search_config = (load_config() or {}).get("search_config") or {}
            min_relevance = clamp_relevance_threshold_percent(
                live_value if live_value is not None else search_config.get("relevance_threshold_percent", 65)
            )
        except Exception:
            min_relevance = 65
        kept = []
        excluded_ids = []

        for note in notes or []:
            if not isinstance(note, dict):
                kept.append(note)
                continue

            note_id = note.get("id")
            if note_id in priority_ids or "_display_relevance" not in note:
                kept.append(note)
                continue

            display_relevance = note.get("_display_relevance")
            if not isinstance(display_relevance, (int, float)):
                kept.append(note)
                continue

            if display_relevance < min_relevance:
                excluded_ids.append(note_id)
                continue

            kept.append(note)

        return {
            "notes": kept,
            "excluded_count": len(excluded_ids),
            "excluded_note_ids": excluded_ids,
        }

    def _query_complexity_score(self, query, available_notes=0, search_config=None):
        """Shared query complexity signal for context and dynamic retrieval budgets."""
        search_config = search_config or {}
        query_text = query or ""
        query_tokens = estimate_tokens(query_text)
        lower_query = query_text.lower()

        complexity = 0
        if query_tokens > 18:
            complexity += 1
        if query_tokens > 35:
            complexity += 1
        if any(marker in lower_query for marker in GENERAL_COMPLEX_MARKERS):
            complexity += 1
        if any(marker in lower_query for marker in CLINICAL_COMPLEX_MARKERS):
            complexity += 1
        if available_notes > 12:
            complexity += 1
        if search_config.get('enable_hyde') or search_config.get('enable_query_expansion'):
            complexity += 1
        return complexity

    def _local_context_usage_plan(self, query, available_notes, provider, search_config):
        """Choose a local-model context budget based on question complexity."""
        if provider not in ("local_openai", "local_server", "ollama"):
            return None

        try:
            n_ctx = int(search_config.get('local_llm_context_tokens') or 12288)
        except Exception:
            n_ctx = 12288
        n_ctx = max(4096, min(n_ctx, 32768))

        complexity = self._query_complexity_score(query, available_notes, search_config)

        if complexity <= 1:
            mode = "light"
            max_output_tokens = min(1024, max(512, n_ctx // 6))
            desired_context_tokens = min(2400, max(1200, n_ctx - max_output_tokens - 900))
            max_notes = 8
        elif complexity <= 3:
            mode = "balanced"
            max_output_tokens = min(2048, max(768, n_ctx // 5))
            desired_context_tokens = min(5600, max(2200, n_ctx - max_output_tokens - 900))
            max_notes = 16
        else:
            mode = "deep"
            max_output_tokens = min(4096, max(1024, n_ctx // 4))
            desired_context_tokens = min(10000, max(3500, n_ctx - max_output_tokens - 900))
            max_notes = 32

        return {
            "mode": mode,
            "model_context_tokens": n_ctx,
            "context_token_budget": desired_context_tokens,
            "max_output_tokens": max_output_tokens,
            "max_notes": max_notes,
        }

    def _dynamic_note_budget(
        self,
        query,
        scored_notes,
        search_config,
        provider,
        local_context_plan=None,
        phase="final",
        pinned_count=0,
        rerank_used=False,
    ):
        """Choose display, context, and rerank budgets from scores and query complexity."""
        search_config = search_config or {}
        scored_notes = list(scored_notes or [])
        available = len(scored_notes)
        mode = "threshold"
        mode_cfg = {"min": 10, "base_display": 50, "max_display": 100, "base_context": 12, "max_context": 24, "elbow": 0.15, "rerank": 50}

        complexity = self._query_complexity_score(query, available, search_config)
        expansion = min(3, max(0, complexity - 1))
        user_max = int(search_config.get("max_results") or 50)
        user_max = max(5, min(100, user_max))

        display_limit = mode_cfg["base_display"] + expansion * 5
        context_limit = mode_cfg["base_context"] + expansion * 3
        rerank_candidate_limit = mode_cfg["rerank"] + expansion * 4

        if phase == "rerank":
            rerank_candidate_limit = min(60, max(mode_cfg["min"], rerank_candidate_limit))
        else:
            rerank_candidate_limit = min(60, max(mode_cfg["min"], rerank_candidate_limit))

        max_display_limit = min(mode_cfg["max_display"], user_max, available)
        display_limit = min(mode_cfg["max_display"], user_max, max(mode_cfg["min"], display_limit))
        context_limit = min(mode_cfg["max_context"], max(mode_cfg["min"], context_limit))
        if local_context_plan:
            context_limit = min(context_limit, int(local_context_plan.get("max_notes") or context_limit))

        cutoff_index = min(available, display_limit)
        reason = "threshold"
        min_floor = min(mode_cfg["min"], available)
        elbow_threshold = mode_cfg["elbow"]
        for idx in range(max(0, min_floor - 1), max(0, min(available - 1, display_limit - 1))):
            current = float(scored_notes[idx][0] or 0)
            nxt = float(scored_notes[idx + 1][0] or 0)
            rel_drop = (current - nxt) / max(abs(current), 1.0)
            if rel_drop > elbow_threshold:
                cutoff_index = idx + 1
                reason = f"threshold_elbow_{int(round(rel_drop * 100))}pct"
                break

        cutoff_index = min(available, max(min_floor, cutoff_index))
        hidden_count = max(0, min(available, display_limit) - cutoff_index)
        if getattr(self, "_show_all_dynamic_results", False):
            cutoff_index = min(available, display_limit)
            hidden_count = 0
            reason = "threshold_show_all"

        return {
            "display_limit": cutoff_index,
            "max_display_limit": max_display_limit,
            "context_limit": max(0, int(context_limit)),
            "rerank_candidate_limit": int(rerank_candidate_limit),
            "cutoff_index": cutoff_index,
            "hidden_count": hidden_count,
            "reason": reason,
            "can_show_all": hidden_count > 0,
            "mode": mode,
            "complexity": complexity,
            "pinned_count": int(pinned_count or 0),
            "rerank_used": bool(rerank_used),
        }

    def _cliff_scored_notes(self, scored_notes):
        """Return scores for cliff detection on one consistent 0-100 scale."""
        scored_notes = list(scored_notes or [])
        candidate_count = len(scored_notes)
        display_relevance_count = sum(
            1 for _score, note in scored_notes if note and note.get("_display_relevance") is not None
        )

        if candidate_count and display_relevance_count == candidate_count:
            return {
                "scored_notes": [
                    (max(0.0, min(100.0, float(note.get("_display_relevance") or 0))), note)
                    for _score, note in scored_notes
                ],
                "score_source": "display_relevance",
                "display_relevance_count": display_relevance_count,
                "candidate_count": candidate_count,
            }

        score_source = "normalized_tuple_score"
        if display_relevance_count:
            score_source = "normalized_tuple_score_partial_display_relevance"

        try:
            top_score = float(scored_notes[0][0] or 0) if scored_notes else 0.0
        except Exception:
            top_score = 0.0
        scale = 100.0 / top_score if top_score > 0 else 0.0

        normalized = []
        for score, note in scored_notes:
            try:
                normalized_score = float(score or 0) * scale if scale else 0.0
            except Exception:
                normalized_score = 0.0
            normalized.append((max(0.0, min(100.0, normalized_score)), note))

        return {
            "scored_notes": normalized,
            "score_source": score_source,
            "display_relevance_count": display_relevance_count,
            "candidate_count": candidate_count,
        }

    def _context_score_cliff_plan(self, scored_notes, search_config=None, rerank_used=False):
        """Find a large post-rerank score drop that should cap AI context."""
        search_config = search_config or {}
        enabled = bool(search_config.get("enable_context_score_cliff", True)) and bool(rerank_used)
        try:
            threshold = float(search_config.get("context_score_cliff_threshold", 15.0))
        except Exception:
            threshold = 15.0
        threshold = max(0.0, threshold)
        try:
            min_notes = int(search_config.get("context_score_cliff_min_notes", 8))
        except Exception:
            min_notes = 8
        min_notes = max(1, min_notes)
        tail_artifact_score_floor = 10.0

        cliff_scores = self._cliff_scored_notes(scored_notes)
        scored_notes = cliff_scores["scored_notes"]
        plan = {
            "enabled": enabled,
            "threshold": threshold,
            "min_notes": min_notes,
            "tail_artifact_score_floor": tail_artifact_score_floor,
            "tail_artifact_definition": "next_score_at_or_below_floor",
            "largest_gap": 0.0,
            "first_qualifying_gap": None,
            "cutoff_rank": None,
            "candidate_gaps": [],
            "selected_gap_reason": "none",
            "ignored_tail_gap_count": 0,
            "score_source": cliff_scores["score_source"],
            "display_relevance_count": cliff_scores["display_relevance_count"],
            "candidate_count": cliff_scores["candidate_count"],
        }
        if not enabled or len(scored_notes) <= min_notes:
            return plan

        first_meaningful_gap = None
        first_tail_gap = None
        for idx in range(len(scored_notes) - 1):
            current = float(scored_notes[idx][0] or 0)
            nxt = float(scored_notes[idx + 1][0] or 0)
            gap = max(0.0, current - nxt)
            if gap > plan["largest_gap"]:
                plan["largest_gap"] = gap

            retained_count = idx + 1
            is_tail_artifact = nxt <= tail_artifact_score_floor
            if retained_count >= min_notes and gap > threshold:
                gap_info = {
                    "rank": retained_count,
                    "from_score": current,
                    "to_score": nxt,
                    "gap": gap,
                    "is_tail_artifact": is_tail_artifact,
                }
                plan["candidate_gaps"].append(gap_info)
                if is_tail_artifact:
                    plan["ignored_tail_gap_count"] += 1
                    if first_tail_gap is None:
                        first_tail_gap = gap_info
                elif first_meaningful_gap is None:
                    first_meaningful_gap = gap_info

        selected_gap = None
        if first_meaningful_gap is not None:
            selected_gap = first_meaningful_gap
            plan["selected_gap_reason"] = "first_meaningful_gap"
        elif first_tail_gap is not None:
            selected_gap = first_tail_gap
            plan["selected_gap_reason"] = "tail_fallback"

        if selected_gap is not None:
            plan["first_qualifying_gap"] = selected_gap
            plan["cutoff_rank"] = selected_gap["rank"]

        return plan

    def _get_note_embeddings_for_rescue(self, note_ids, db_path=None, engine_id=None):
        """Return stored embeddings for note ids using the same SQLite bulk loader."""
        wanted = {int(note_id) for note_id in (note_ids or []) if note_id is not None}
        if not wanted:
            return {}

        try:
            from ..core.engine import _blob_to_embedding, load_embeddings_bulk
        except Exception:
            return {}

        out = {}
        try:
            for row in load_embeddings_bulk(db_path=db_path, engine_id=engine_id):
                note_id = row.get("note_id")
                try:
                    note_id = int(note_id)
                except Exception:
                    continue
                if note_id not in wanted or note_id in out:
                    continue
                blob = row.get("embedding_blob")
                if blob:
                    out[note_id] = _blob_to_embedding(blob)
                if len(out) >= len(wanted):
                    break
        except Exception:
            return {}
        return out

    def _cosine_similarity(self, query_embedding, note_embedding):
        if query_embedding is None or note_embedding is None:
            return None
        try:
            if len(query_embedding) != len(note_embedding):
                return None
        except Exception:
            return None

        try:
            import numpy as np

            q = np.asarray(query_embedding, dtype=float)
            n = np.asarray(note_embedding, dtype=float)
            denom = float(np.linalg.norm(q) * np.linalg.norm(n))
            if denom <= 0:
                return None
            return float(np.dot(q, n) / denom)
        except Exception:
            dot = 0.0
            q_norm = 0.0
            n_norm = 0.0
            try:
                for qv, nv in zip(query_embedding, note_embedding):
                    qf = float(qv)
                    nf = float(nv)
                    dot += qf * nf
                    q_norm += qf * qf
                    n_norm += nf * nf
            except Exception:
                return None
            denom = (q_norm ** 0.5) * (n_norm ** 0.5)
            if denom <= 0:
                return None
            return dot / denom

    def _normalize_specificity_text(self, text):
        text = (text or "").lower()
        text = re.sub(r"\b([a-z])\.\s*([a-z][a-z0-9]+)\b", r"\1 \2", text)
        text = re.sub(r"[-_/]+", " ", text)
        text = re.sub(r"(?<=\b[a-z])\.(?=\s*[a-z])", " ", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _specificity_terms_from_query(self, query):
        normalized = self._normalize_specificity_text(query)
        tokens = [token for token in normalized.split() if len(token) >= 3 or token in {"m", "h", "e", "s", "n"}]
        terms = []
        for size in (1, 2, 3):
            for idx in range(0, max(0, len(tokens) - size + 1)):
                term_tokens = tokens[idx:idx + size]
                term = " ".join(term_tokens).strip()
                if len(term.replace(" ", "")) >= 3:
                    terms.append(term)
        return list(dict.fromkeys(terms))

    def _note_specificity_text(self, note):
        if not isinstance(note, dict):
            return ""
        return self._normalize_specificity_text(
            " ".join(
                str(note.get(key) or "")
                for key in ("display_content", "content_preview", "content")
            )
        )

    def _analyze_query_specificity(self, query, all_candidate_notes, search_config=None):
        search_config = search_config or {}
        if not search_config.get("enable_rescue_specificity_scoring", True):
            return {
                "anchor_terms": [],
                "all_terms": [],
                "total_candidates": len(all_candidate_notes or []),
            }

        try:
            threshold = float(search_config.get("rescue_specificity_threshold", 0.85))
        except Exception:
            threshold = 0.85
        threshold = max(0.0, min(1.0, threshold))
        try:
            max_doc_freq = int(search_config.get("rescue_specificity_max_doc_freq", 4))
        except Exception:
            max_doc_freq = 4
        max_doc_freq = max(0, max_doc_freq)

        terms = self._specificity_terms_from_query(query)
        notes = list(all_candidate_notes or [])
        total = len(notes)
        if not terms or total <= 0:
            return {"anchor_terms": [], "all_terms": [], "total_candidates": total}

        note_texts = [self._note_specificity_text(note) for note in notes]
        all_terms = []
        anchor_terms = []
        for term in terms:
            doc_freq = sum(1 for text in note_texts if term in text)
            specificity = 1.0 - (float(doc_freq) / float(total))
            item = {
                "term": term,
                "specificity_score": specificity,
                "doc_freq": doc_freq,
            }
            all_terms.append(item)
            if doc_freq > 0 and (specificity >= threshold or doc_freq <= max_doc_freq):
                anchor_terms.append(item)

        anchor_terms.sort(key=lambda item: (-item["specificity_score"], item["doc_freq"], item["term"]))
        return {
            "anchor_terms": anchor_terms,
            "all_terms": all_terms,
            "total_candidates": total,
        }

    def _note_specificity_score(self, note, anchor_terms):
        anchor_terms = list(anchor_terms or [])
        if not note or not anchor_terms:
            return 0.0
        text = self._note_specificity_text(note)
        total_weight = sum(float(item.get("specificity_score") or 0.0) for item in anchor_terms)
        if total_weight <= 0:
            return 0.0
        matched = 0.0
        for item in anchor_terms:
            term = item.get("term")
            if term and term in text:
                matched += float(item.get("specificity_score") or 0.0)
        return max(0.0, min(1.0, matched / total_weight))

    def _score_rescue_candidates(
        self,
        query_embedding,
        candidate_notes,
        cliff_selected_notes,
        note_embeddings,
        search_config=None,
        all_candidate_notes=None,
        query=None,
    ):
        search_config = search_config or {}
        candidate_notes = list(candidate_notes or [])
        cliff_selected_notes = list(cliff_selected_notes or [])
        note_embeddings = note_embeddings or {}

        try:
            slots = int(search_config.get("context_score_cliff_anchor_rescue_slots", 3))
        except Exception:
            slots = 3
        slots = max(0, slots)
        if slots <= 0:
            return [], {"rescue_similarity_floor": None, "candidate_scores": []}

        specificity = self._analyze_query_specificity(
            query or "",
            all_candidate_notes if all_candidate_notes is not None else list(cliff_selected_notes) + list(candidate_notes),
            search_config,
        )
        anchor_terms = specificity.get("anchor_terms") or []
        try:
            max_specificity_weight = float(search_config.get("rescue_specificity_max_weight", 0.5))
        except Exception:
            max_specificity_weight = 0.5
        max_specificity_weight = max(0.0, min(1.0, max_specificity_weight))
        if anchor_terms and search_config.get("enable_rescue_specificity_scoring", True):
            specificity_weight = min(
                max_specificity_weight,
                max(float(item.get("specificity_score") or 0.0) for item in anchor_terms),
            )
        else:
            specificity_weight = 0.0
        scoring_method = "adaptive_hybrid" if specificity_weight > 0 else "cosine_only"

        def _hybrid_score(note, cosine):
            specificity_score = self._note_specificity_score(note, anchor_terms)
            hybrid = (float(cosine) * (1.0 - specificity_weight)) + (specificity_score * specificity_weight)
            return hybrid, specificity_score

        selected_scores = []
        for note in cliff_selected_notes:
            note_id = note.get("id") if isinstance(note, dict) else None
            try:
                emb = note_embeddings.get(int(note_id)) if note_id is not None else None
            except Exception:
                emb = None
            sim = self._cosine_similarity(query_embedding, emb)
            if sim is not None:
                hybrid, specificity_score = _hybrid_score(note, sim)
                selected_scores.append(hybrid)

        if not selected_scores:
            return [], {
                "rescue_similarity_floor": None,
                "candidate_scores": [],
                "anchor_terms_detected": anchor_terms,
                "rescue_scoring_method": scoring_method,
                "specificity_weight_used": specificity_weight,
            }

        dynamic_floor = min(selected_scores)
        absolute_floor = search_config.get("context_score_cliff_anchor_rescue_similarity_floor")
        if absolute_floor is not None:
            try:
                rescue_floor = float(absolute_floor)
            except Exception:
                rescue_floor = dynamic_floor
            rescue_floor = max(0.0, min(1.0, rescue_floor))
        else:
            rescue_floor = dynamic_floor

        try:
            from ..utils import clamp_relevance_threshold_percent

            live_value = getattr(self, "_effective_relevance_threshold_percent", None)
            min_relevance = float(
                clamp_relevance_threshold_percent(
                    live_value if live_value is not None else search_config.get("relevance_threshold_percent", 65)
                )
            )
        except Exception:
            min_relevance = 65.0

        selected_ids = {
            note.get("id")
            for note in cliff_selected_notes
            if isinstance(note, dict) and note.get("id") is not None
        }
        scored = []
        filtered_by_specificity = 0
        for rank, note in enumerate(candidate_notes):
            if not isinstance(note, dict):
                continue
            note_id = note.get("id")
            if note_id is None or note_id in selected_ids:
                continue
            display_relevance = note.get("_display_relevance")
            if display_relevance is not None:
                try:
                    if float(display_relevance) < min_relevance:
                        continue
                except Exception:
                    pass
            try:
                emb = note_embeddings.get(int(note_id))
            except Exception:
                emb = None
            sim = self._cosine_similarity(query_embedding, emb)
            if sim is None:
                continue
            hybrid, specificity_score = _hybrid_score(note, sim)
            if hybrid > rescue_floor:
                scored.append((hybrid, rank, note, sim, specificity_score))
            elif specificity_weight > 0 and specificity_score <= 0:
                filtered_by_specificity += 1

        scored.sort(key=lambda item: (-item[0], item[1]))
        rescued = []
        candidate_scores = []
        for hybrid, _rank, note, sim, specificity_score in scored[:slots]:
            rescued.append(note)
            candidate_scores.append({
                "note_id": note.get("id"),
                "similarity": sim,
                "specificity_score": specificity_score,
                "hybrid_score": hybrid,
            })

        return rescued, {
            "rescue_similarity_floor": rescue_floor,
            "candidate_scores": candidate_scores,
            "anchor_terms_detected": anchor_terms,
            "rescue_scoring_method": scoring_method,
            "specificity_weight_used": specificity_weight,
            "rescued_note_specificity_scores": candidate_scores,
            "filtered_by_specificity": filtered_by_specificity,
        }

    def _fallback_anchor_rescue_candidates(self, query, candidate_notes, selected_notes, search_config=None, all_candidate_notes=None):
        search_config = search_config or {}
        specificity = self._analyze_query_specificity(
            query,
            all_candidate_notes if all_candidate_notes is not None else list(selected_notes or []) + list(candidate_notes or []),
            search_config,
        )
        anchors = specificity.get("anchor_terms") or []
        if not anchors:
            return [], {
                "anchor_terms_detected": [],
                "rescue_scoring_method": "specificity_only",
                "specificity_weight_used": 0.0,
            }
        try:
            slots = int(search_config.get("context_score_cliff_anchor_rescue_slots", 3))
        except Exception:
            slots = 3
        try:
            from ..utils import clamp_relevance_threshold_percent

            live_value = getattr(self, "_effective_relevance_threshold_percent", None)
            min_relevance = float(
                clamp_relevance_threshold_percent(
                    live_value if live_value is not None else search_config.get("relevance_threshold_percent", 65)
                )
            )
        except Exception:
            min_relevance = 65.0
        selected_ids = {note.get("id") for note in selected_notes or [] if note.get("id") is not None}
        scored = []
        for rank, note in enumerate(candidate_notes or []):
            note_id = note.get("id")
            if note_id is None or note_id in selected_ids:
                continue
            display_relevance = note.get("_display_relevance")
            if display_relevance is not None:
                try:
                    if float(display_relevance) < min_relevance:
                        continue
                except Exception:
                    pass
            score = self._note_specificity_score(note, anchors)
            if score > 0:
                scored.append((score, rank, note))
        scored.sort(key=lambda item: (-item[0], item[1]))
        rescued = [note for _score, _rank, note in scored[:max(0, slots)]]
        return rescued, {
            "anchor_terms_detected": anchors,
            "rescue_scoring_method": "specificity_only",
            "specificity_weight_used": 1.0 if anchors else 0.0,
            "rescued_note_specificity_scores": [
                {"note_id": note.get("id"), "specificity_score": score}
                for score, _rank, note in scored[:max(0, slots)]
            ],
        }

    def _fit_context_lines_to_token_budget(self, context_lines, token_budget):
        """Keep note blocks within a token budget while preserving note order."""
        fitted = []
        used = 0
        token_budget = max(600, int(token_budget or 0))

        for line in context_lines:
            line_tokens = estimate_tokens(line)
            remaining = token_budget - used
            if remaining <= 120:
                break
            if line_tokens <= remaining:
                fitted.append(line)
                used += line_tokens
                continue

            keep_chars = max(240, remaining * 4)
            fitted.append(line[:keep_chars].rstrip() + " ...")
            break

        return fitted or context_lines[:1]
