from aqt.qt import QColor, QTableWidgetItem, Qt

from ..utils import clamp_relevance_threshold_percent, load_config, log_debug


class SearchResultDisplayMixin:
    def _result_identity_key(self, note):
        return (note.get('id'), note.get('chunk_index'), note.get('content_hash'))

    def _source_table_rank_mode(self):
        return bool(getattr(self, "_sources_rank_mode", False))

    def _current_source_table_sort(self):
        table = getattr(self, "results_list", None)
        if table is None:
            return None
        try:
            header = table.horizontalHeader()
            section = int(header.sortIndicatorSection())
            if section < 0 or section >= table.columnCount():
                return None
            return section, header.sortIndicatorOrder()
        except Exception:
            return None

    def _default_source_table_sort(self, citation_order_display, has_context_refs):
        if citation_order_display or has_context_refs:
            return 1, Qt.SortOrder.AscendingOrder
        return 4, Qt.SortOrder.DescendingOrder

    def _apply_source_table_mode_labels(self):
        table = getattr(self, "results_list", None)
        if table is None:
            return
        try:
            if self._source_table_rank_mode():
                table.setHorizontalHeaderLabels(["\u2713", "Rank", "Content", "Note ID", "Relevance"])
                label = getattr(self, "show_only_cited_cb", None)
                if label is not None:
                    label.setEnabled(False)
                    label.setToolTip("Only applies to Ask AI answers with citations.")
            else:
                table.setHorizontalHeaderLabels(["\u2713", "Ref", "Content", "Note ID", "Relevance"])
                label = getattr(self, "show_only_cited_cb", None)
                if label is not None:
                    label.setEnabled(True)
                    label.setToolTip("Show only notes that the AI explicitly cited in its answer.")
        except Exception:
            pass

    def _ensure_review_context_source_visible(self, scored_notes):
        scored_notes = list(scored_notes or [])
        if not getattr(self, "_context_note_identity_keys", None):
            return scored_notes
        review_note = self._review_context_note_for_answer() if hasattr(self, "_review_context_note_for_answer") else None
        if not review_note:
            return scored_notes
        review_key = self._result_identity_key(review_note)
        has_review = any(self._result_identity_key(note) == review_key for _score, note in scored_notes)
        if has_review:
            return scored_notes
        return [(100, review_note)] + scored_notes

    def _debug_ref_gap(self, stage, scored_notes):
        try:
            context_keys = list(getattr(self, '_context_note_identity_keys', None) or [])
            if not context_keys:
                context_pairs = list(getattr(self, '_context_note_id_and_chunk', None) or [])
                context_keys = [(nid, cidx, None) for nid, cidx in context_pairs]
            if not context_keys:
                return
            displayed_keys = {
                self._result_identity_key(note)
                for _score, note in (scored_notes or [])
            }
            displayed_pairs = {
                (note.get('id'), note.get('chunk_index'), None)
                for _score, note in (scored_notes or [])
            }
            missing = []
            for idx, key in enumerate(context_keys, start=1):
                if key not in displayed_keys and key not in displayed_pairs:
                    missing.append((idx, key))
            if missing:
                sample = missing[:12]
                log_debug(
                    "Ref gap debug: "
                    f"stage={stage}, missing_refs={sample}, missing_count={len(missing)}, "
                    f"context_count={len(context_keys)}, displayed_count={len(scored_notes or [])}, "
                    f"mode={getattr(self, 'relevance_mode', None)}, "
                    f"effective_mode={getattr(self, '_effective_relevance_mode', None)}, "
                    f"show_only_cited={bool(getattr(getattr(self, 'show_only_cited_cb', None), 'isChecked', lambda: False)())}"
                )
            else:
                log_debug(
                    "Ref gap debug: "
                    f"stage={stage}, missing_count=0, context_count={len(context_keys)}, "
                    f"displayed_count={len(scored_notes or [])}"
                )
        except Exception as exc:
            log_debug(f"Ref gap debug failed at {stage}: {exc}")

    def _on_show_all_dynamic_results(self):
        self._show_all_dynamic_results = True
        if hasattr(self, "show_all_dynamic_results_btn"):
            self.show_all_dynamic_results_btn.setVisible(False)
        self.filter_and_display_notes()

    def _dynamic_budget_message(self):
        budget = getattr(self, "_last_dynamic_note_budget", None) or {}
        hidden = int(budget.get("hidden_count") or 0)
        if hidden > 0:
            threshold = self._global_relevance_threshold_percent()
            return f"{hidden} results hidden by the display limit or Threshold: {threshold}%."
        return ""

    def _note_pair_key(self, note):
        return (note.get('id'), note.get('chunk_index'))

    def _pinned_result_sets(self):
        identity_keys = set()
        pair_keys = set()
        note_ids = set()

        for key in list(getattr(self, '_context_note_identity_keys', None) or []):
            try:
                if len(key) >= 3:
                    identity_keys.add(tuple(key[:3]))
                    note_ids.add(key[0])
                elif len(key) >= 2:
                    pair_keys.add(tuple(key[:2]))
                    note_ids.add(key[0])
            except Exception:
                pass

        for key in list(getattr(self, '_context_note_id_and_chunk', None) or []):
            try:
                if len(key) >= 2:
                    pair_keys.add(tuple(key[:2]))
                    note_ids.add(key[0])
            except Exception:
                pass

        for note_id in list(getattr(self, '_context_note_ids', None) or []):
            note_ids.add(note_id)

        for note_id in list(getattr(self, '_cited_note_ids', None) or []):
            note_ids.add(note_id)

        return identity_keys, pair_keys, note_ids

    def _is_pinned_result_note(self, note, pinned_sets=None):
        identity_keys, pair_keys, note_ids = pinned_sets or self._pinned_result_sets()
        return (
            self._result_identity_key(note) in identity_keys
            or self._note_pair_key(note) in pair_keys
            or note.get('id') in note_ids
        )

    def _passes_relevance_mode(self, note, mode):
        return True

    def _global_relevance_threshold_percent(self):
        slider = getattr(self, "sensitivity_slider", None)
        if slider is not None:
            try:
                return clamp_relevance_threshold_percent(slider.value())
            except Exception:
                pass

        live_value = getattr(self, "_effective_relevance_threshold_percent", None)
        if live_value is not None:
            return clamp_relevance_threshold_percent(live_value)
        try:
            config = load_config()
            search_config = (config or {}).get("search_config") or {}
            value = search_config.get("relevance_threshold_percent", 65)
        except Exception:
            value = 65
        try:
            value = int(value)
        except Exception:
            value = 65
        return clamp_relevance_threshold_percent(value)

    def _global_min_relevance_percent(self):
        return self._global_relevance_threshold_percent()

    def _display_relevance_for_note(self, score, note, max_score):
        percentage = note.get("_display_relevance")
        if percentage is None:
            try:
                percentage = int((float(score) / float(max_score)) * 100) if max_score > 0 else 0
            except Exception:
                percentage = 0
        try:
            return max(0, min(100, int(percentage)))
        except Exception:
            return 0

    def _mode_display_threshold(self, mode, min_relevance):
        return self._global_relevance_threshold_percent()

    def _user_max_display_results(self):
        try:
            config = load_config()
            search_config = (config or {}).get("search_config") or {}
            value = int(search_config.get("max_results") or 50)
        except Exception:
            value = 50
        return max(5, min(100, value))

    def _display_limit_for_mode(self, budget, mode, pinned_count, fill_count, filtered_count, show_all=False):
        if show_all:
            return max(max(0, filtered_count), pinned_count)
        user_max = self._user_max_display_results()
        base_limit = min(max(0, filtered_count), user_max)
        return max(base_limit, pinned_count)

    def _build_relevance_display_plan(self, scored_notes, mode=None, budget=None, show_all=False):
        scored_notes = list(scored_notes or [])
        mode = "threshold"

        pinned_sets = self._pinned_result_sets()
        max_score = max((score for score, _note in scored_notes), default=1)
        min_relevance = self._global_relevance_threshold_percent()
        mode_threshold = min_relevance
        pinned_keys = set()
        mode_keys = set()
        mode_filtered_notes = []
        floor_keys = set()

        for score, note in scored_notes:
            key = self._result_identity_key(note)
            is_pinned = self._is_pinned_result_note(note, pinned_sets)
            display_relevance = self._display_relevance_for_note(score, note, max_score)
            passes_threshold = display_relevance >= min_relevance

            if is_pinned and passes_threshold:
                pinned_keys.add(key)

            if not is_pinned and passes_threshold:
                floor_keys.add(key)

            if passes_threshold and display_relevance >= mode_threshold:
                mode_keys.add(key)
                mode_filtered_notes.append((score, note))

        if show_all:
            fill_keys = {
                self._result_identity_key(note)
                for _score, note in scored_notes
                if self._result_identity_key(note) not in pinned_keys
            }
        else:
            fill_keys = {key for key in mode_keys if key not in pinned_keys}
        display_limit = self._display_limit_for_mode(
            budget,
            mode,
            len(pinned_keys),
            len(fill_keys),
            len(pinned_keys | fill_keys) if show_all else len(pinned_keys | mode_keys),
            show_all=show_all,
        )
        fill_slots = max(0, display_limit - len(pinned_keys))

        selected_fill_keys = set()
        if fill_slots > 0:
            for score, note in scored_notes:
                key = self._result_identity_key(note)
                if key in fill_keys:
                    selected_fill_keys.add(key)
                    if len(selected_fill_keys) >= fill_slots:
                        break

        selected_keys = pinned_keys | selected_fill_keys
        final_notes = [
            (score, note)
            for score, note in scored_notes
            if self._result_identity_key(note) in selected_keys
        ]

        all_candidate_keys = {
            self._result_identity_key(note)
            for _score, note in scored_notes
        }
        selectable_count = len(pinned_keys | fill_keys) if show_all else len(pinned_keys | mode_keys)
        hidden_count = 0 if show_all else max(0, len(all_candidate_keys - selected_keys))
        if budget is not None:
            budget["hidden_count"] = hidden_count

        try:
            log_debug(
                "Relevance display plan: "
                f"threshold={min_relevance}, threshold_filtered_count={len(mode_filtered_notes)}, "
                f"pinned_count={len(pinned_keys)}, final_visible_count={len(final_notes)}, "
                f"hidden_count={hidden_count}"
            )
        except Exception:
            pass

        return {
            "mode": mode,
            "min_relevance": min_relevance,
            "mode_threshold": mode_threshold,
            "mode_filtered_notes": mode_filtered_notes,
            "focused_backfill_count": 0,
            "pinned_count": len(pinned_keys),
            "final_notes": final_notes,
            "hidden_count": hidden_count,
            "display_limit": display_limit,
            "selectable_count": selectable_count,
        }

    def _analytics_relevance_threshold_data(self, scored_notes=None):
        """Summarize what the active relevance threshold shows for this result set."""
        scored_notes = list(scored_notes if scored_notes is not None else getattr(self, "all_scored_notes", []) or [])
        if not scored_notes:
            return {}

        try:
            config = load_config()
        except Exception:
            config = {}
        search_config = dict((config or {}).get("search_config") or {})
        provider = (config or {}).get("provider", "openai")
        original_show_all = getattr(self, "_show_all_dynamic_results", False)

        out = {}
        try:
            self._show_all_dynamic_results = False
            budget = {}
            if hasattr(self, "_dynamic_note_budget"):
                try:
                    budget = self._dynamic_note_budget(
                        getattr(self, "current_query", "") or "",
                        scored_notes,
                        dict(search_config),
                        provider,
                        phase="final",
                        pinned_count=len(
                            (getattr(self, "selected_note_ids", set()) or set())
                            | (getattr(self, "_pinned_note_ids", set()) or set())
                        ),
                        rerank_used=bool(getattr(self, "_last_rerank_success", False)),
                    )
                except Exception as exc:
                    log_debug(f"Could not compute analytics relevance threshold budget: {exc}")
                    budget = {}

            plan = self._build_relevance_display_plan(
                scored_notes,
                mode="threshold",
                budget=dict(budget or {}),
                show_all=False,
            )
            final_notes = plan.get("final_notes") or []
            mode_filtered_notes = plan.get("mode_filtered_notes") or []
            out = {
                "threshold_percent": int(plan.get("min_relevance") or 0),
                "threshold_pct": int(plan.get("min_relevance") or 0),
                "threshold_source": getattr(self, "_relevance_threshold_source", "session_default"),
                "show_all_override": bool(original_show_all),
                "threshold_filtered_count": len(mode_filtered_notes),
                "pinned_count": int(plan.get("pinned_count") or 0),
                "final_visible_count": len(final_notes),
                "visible": len(final_notes),
                "hidden_count": int(plan.get("hidden_count") or 0),
                "hidden": int(plan.get("hidden_count") or 0),
                "display_limit": int(plan.get("display_limit") or 0),
                "selectable_count": int(plan.get("selectable_count") or 0),
                "final_note_ids": [note.get("id") for _score, note in final_notes],
                "threshold_filtered_note_ids": [note.get("id") for _score, note in mode_filtered_notes[:100]],
                "top_results": self._analytics_results(final_notes, limit=100) if hasattr(self, "_analytics_results") else [],
            }
        finally:
            self._show_all_dynamic_results = original_show_all

        return out

    def _analytics_relevance_mode_data(self, scored_notes=None):
        return self._analytics_relevance_threshold_data(scored_notes)

    def filter_and_display_notes(self):
        self._apply_source_table_mode_labels()



        checked_note_ids = set(getattr(self, 'selected_note_ids', set()) or [])



        # Use chunk-level display list when AI received more items than aggregated display (fixes Ref 35 vs 32)



        display_source = getattr(self, '_display_scored_notes', None)



        ctx = getattr(self, '_context_note_ids', None) or []
        has_context_refs = bool(
            getattr(self, '_context_note_identity_keys', None)
            or getattr(self, '_context_note_id_and_chunk', None)
            or ctx
        )



        all_scored = list(getattr(self, 'all_scored_notes', None) or [])



        citation_order_display = False



        if display_source and ctx:



            notes_to_display = list(display_source)
            notes_to_display = self._ensure_review_context_source_visible(notes_to_display)



            seen_context_keys = {
                self._result_identity_key(note)
                for _score, note in notes_to_display
            }



            notes_to_display.extend(
                (score, note)
                for score, note in all_scored
                if self._result_identity_key(note) not in seen_context_keys
            )



            citation_order_display = True



        elif all_scored:



            notes_to_display = all_scored
            notes_to_display = self._ensure_review_context_source_visible(notes_to_display)



        else:



            return



        if not notes_to_display:



            return

        self._debug_ref_gap("notes_to_display_initial", notes_to_display)







        # Store current row count and sort before clearing



        old_row_count = self.results_list.rowCount()
        saved_sort = self._current_source_table_sort() if old_row_count > 0 else None







        # Clear and repopulate the table without letting transient unchecked items erase
        # the user's checked-note selections.



        if hasattr(self, "_reset_note_preview_popup"):
            self._reset_note_preview_popup()

        self.results_list.blockSignals(True)



        self.results_list.setRowCount(0)







        max_score = max((score for score, _note in notes_to_display), default=1)
        sensitivity_threshold = self._global_relevance_threshold_percent()
        filtered_notes = list(notes_to_display)

        self._debug_ref_gap("after_score_or_context_filter", filtered_notes)







        # Filter by the single relevance threshold using one shared selection
        # path for pinned context refs and dynamic display limits.
        pre_mode_filtered_notes = list(filtered_notes)
        mode = "threshold"



        # Optionally restrict to the exact refs cited in the AI answer ([1], [2], ...).



        if (
            not self._source_table_rank_mode()
            and getattr(self, 'show_only_cited_cb', None)
            and self.show_only_cited_cb.isChecked()
        ):



            cited = getattr(self, '_cited_note_ids', None) or set()



            cited_refs = getattr(self, '_cited_refs', None) or set()



            context_id_and_chunk = getattr(self, '_context_note_id_and_chunk', None) or []
            context_identity_keys = getattr(self, '_context_note_identity_keys', None) or []



            cited_keys = {
                context_identity_keys[ref - 1]
                for ref in cited_refs
                if isinstance(ref, int) and 1 <= ref <= len(context_identity_keys)
            }
            cited_pairs = {
                context_id_and_chunk[ref - 1]
                for ref in cited_refs
                if isinstance(ref, int) and 1 <= ref <= len(context_id_and_chunk)
            }



            if cited_keys:



                filtered_notes = [
                    (score, note)
                    for score, note in pre_mode_filtered_notes
                    if self._result_identity_key(note) in cited_keys
                ]



            elif cited_pairs:



                filtered_notes = [
                    (score, note)
                    for score, note in pre_mode_filtered_notes
                    if (note['id'], note.get('chunk_index')) in cited_pairs
                ]



            elif cited:



                filtered_notes = [(score, note) for score, note in pre_mode_filtered_notes if note['id'] in cited]

            self._debug_ref_gap("after_show_only_cited_filter", filtered_notes)

        hidden_message = ""
        budget = getattr(self, "_last_dynamic_note_budget", None) or {}
        show_all = bool(getattr(self, "_show_all_dynamic_results", False))
        show_only_cited = bool(
            not self._source_table_rank_mode()
            and getattr(self, 'show_only_cited_cb', None)
            and self.show_only_cited_cb.isChecked()
        )
        display_plan = None
        display_plan = self._build_relevance_display_plan(
            filtered_notes,
            mode=mode,
            budget=budget,
            show_all=show_all,
        )
        self._last_relevance_display_plan = display_plan
        self._debug_ref_gap("after_threshold_filter", display_plan["mode_filtered_notes"])
        filtered_notes = display_plan["final_notes"]
        hidden_message = self._dynamic_budget_message()

        self._debug_ref_gap("after_dynamic_display_limit", filtered_notes)
        if hasattr(self, "show_all_dynamic_results_btn"):
            self.show_all_dynamic_results_btn.setVisible(bool(hidden_message))
            if hidden_message:
                self.show_all_dynamic_results_btn.setToolTip(
                    hidden_message + " Show lower-relevance source notes; AI answer context stays unchanged."
                )







        # Set row count



        self.results_list.setRowCount(len(filtered_notes))



        # Disable sorting while populating so rows stay 0..N and every row gets the correct content (fixes empty rows when toggling "Show only cited notes")



        self.results_list.setSortingEnabled(False)







        # 1-based position in context (order sent to AI) so [1], [2], [19] in answer match this #



        cited_ids = set() if self._source_table_rank_mode() else (getattr(self, '_cited_note_ids', set()) or set())
        cited_refs = set() if self._source_table_rank_mode() else (getattr(self, '_cited_refs', set()) or set())



        # Build Ref = context position so citation [N] matches row labeled N (works after re-rank and with chunks)



        context_id_and_chunk = getattr(self, '_context_note_id_and_chunk', None)



        if context_id_and_chunk:



            context_identity_keys = getattr(self, '_context_note_identity_keys', None) or []
            ref_from_context = {
                key: i + 1
                for i, key in enumerate(context_identity_keys)
            } if context_identity_keys else {
                (nid, cidx): i + 1
                for i, (nid, cidx) in enumerate(context_id_and_chunk)
            }



            non_context_refs = {}



            next_ref = len(context_id_and_chunk) + 1



            for _score, note in filtered_notes:



                key = self._result_identity_key(note) if context_identity_keys else (note['id'], note.get('chunk_index'))



                if key not in ref_from_context and key not in non_context_refs:



                    non_context_refs[key] = next_ref



                    next_ref += 1



            def get_ref(note, row):



                key = self._result_identity_key(note) if context_identity_keys else (note['id'], note.get('chunk_index'))



                return ref_from_context.get(key, non_context_refs.get(key, row + 1))



        else:



            order_for_note = {nid: i + 1 for i, nid in enumerate(ctx)} if ctx else {}



            def get_ref(note, row):



                return order_for_note.get(note['id'], row + 1)







        for row, (score, note) in enumerate(filtered_notes):



            # Content-based relevance when set (HyDE/query similarity); else rank-based



            percentage = note.get('_display_relevance')



            if percentage is None:



                percentage = int((score / max_score) * 100) if max_score > 0 else 0



            else:



                percentage = max(0, min(100, int(percentage)))



            # Show the same relevance value used by the threshold filter so the
            # slider and the table stay in one consistent scale.
            display_pct = percentage







            # Column 1: citation Ref for Ask AI answers, or rank for related-note retrieval.



            order_num = get_ref(note, row)



            order_item = QTableWidgetItem()



            order_item.setData(Qt.ItemDataRole.DisplayRole, order_num)



            order_item.setData(Qt.ItemDataRole.UserRole, note['id'])



            order_item.setFlags(order_item.flags() & ~Qt.ItemFlag.ItemIsEditable)



            order_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)



            is_cited_ref = order_num in cited_refs or note['id'] in cited_ids



            if is_cited_ref:



                order_item.setForeground(QColor('#3498db'))



                font = order_item.font()



                font.setBold(True)



                order_item.setFont(font)



            if self._source_table_rank_mode():
                why_ref = "Related-note rank"
                order_item.setToolTip(f"Why this result?\nRanked related note. Note ID: {note['id']}")
            else:
                why_ref = "Cited in answer: Yes" if is_cited_ref else "Cited in answer: No"
                order_item.setToolTip(f"Why this result?\n{why_ref}\nCitation [N] in answer matches Ref. Note ID: {note['id']}")



            select_item = QTableWidgetItem()
            select_item.setCheckState(
                Qt.CheckState.Checked if note['id'] in checked_note_ids else Qt.CheckState.Unchecked
            )
            select_item.setData(Qt.ItemDataRole.UserRole, note['id'])
            select_item.setFlags(
                (select_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                & ~Qt.ItemFlag.ItemIsEditable
            )
            select_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            select_item.setToolTip("Select this note for View Selected")
            self.results_list.setItem(row, 0, select_item)

            self.results_list.setItem(row, 1, order_item)







            # Column 2: Content






            # Use full note content when available (chunk-level display); so preview shows start of full note, not chunk



            raw_for_display = (
                note.get('_full_display_content')
                or note.get('display_content')
                or note.get('_full_content')
                or note['content']
            )



            display_content = self.reveal_cloze_for_display(raw_for_display)



            content_item = QTableWidgetItem(display_content)



            content_item.setData(Qt.ItemDataRole.UserRole, note['id'])



            content_item.setData(Qt.ItemDataRole.UserRole + 1, display_pct)



            content_item.setData(Qt.ItemDataRole.UserRole + 2, display_content)



            matching_terms = self._get_matching_terms_for_note(note, getattr(self, 'current_query', ''))



            content_item.setToolTip("")
            content_item.setData(Qt.ItemDataRole.UserRole + 3, {
                "id": note['id'],
                "display_content": display_content,
                "relevance": display_pct,
                "why_ref": why_ref,
                "matching_terms": matching_terms,
            })



            content_item.setFlags(content_item.flags() & ~Qt.ItemFlag.ItemIsEditable)



            self.results_list.setItem(row, 2, content_item)







            # Column 3: Note ID



            note_id_item = QTableWidgetItem(str(note['id']))



            note_id_item.setData(Qt.ItemDataRole.UserRole, note['id'])



            note_id_item.setFlags(note_id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)



            note_id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)



            note_id_item.setToolTip(f"Note ID: {note['id']}\nDouble-click to open in browser")



            self.results_list.setItem(row, 3, note_id_item)







            # Column 4: Relevance (same percent used by the threshold filter)



            percentage_item = QTableWidgetItem()



            percentage_item.setData(Qt.ItemDataRole.DisplayRole, display_pct)



            percentage_item.setData(Qt.ItemDataRole.UserRole, display_pct)



            percentage_item.setFlags(percentage_item.flags() & ~Qt.ItemFlag.ItemIsEditable)



            percentage_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)



            if display_pct >= 80:



                percentage_item.setForeground(QColor("#27ae60"))



                relevance_desc = "High relevance"



            elif display_pct >= 50:



                percentage_item.setForeground(QColor("#f39c12"))



                relevance_desc = "Medium relevance"



            else:



                percentage_item.setForeground(QColor("#e74c3c"))



                relevance_desc = "Low relevance"



            why_pct = f"Why this result?\nRelevance: {display_pct}% ({relevance_desc})\n{why_ref}"
            match_status = (note.get("_embedding_match_status") or "").strip().lower()
            if match_status == "exact":
                why_pct += "\nEmbedding match: Exact current chunk."
            elif match_status == "fallback":
                why_pct += "\nEmbedding match: Fallback by note ID. Rebuild embeddings for exact chunk ranking."
            elif match_status == "partial":
                why_pct += "\nEmbedding match: Partial chunk coverage."
            elif match_status == "missing":
                why_pct += "\nEmbedding match: Missing for this chunk."



            if matching_terms:



                why_pct += f"\nMatching terms: {', '.join(matching_terms[:6])}{'...' if len(matching_terms) > 6 else ''}"



            # region agent log



            try:



                cq = getattr(self, "current_query", "") or ""



                if "trisom" in cq.lower():



                    from .search_dialog_shared import _agent_debug_log



                    _agent_debug_log(



                        run_id="pre-fix",



                        hypothesis_id="H3",



                        location="__init__.filter_and_display_notes",



                        message="note_row_display",



                        data={



                            "note_id": note.get("id"),



                            "row": row,



                            "raw_score": score,



                            "display_relevance": display_pct,



                            "matching_terms": matching_terms[:6] if matching_terms else [],



                        },



                    )



            except Exception:



                pass



            # endregion



            percentage_item.setToolTip(why_pct)



            self.results_list.setItem(row, 4, percentage_item)







        # Update status to match table: "Showing X of Y" so it always matches Matching notes row count



        filtered_count = len(filtered_notes)
        self._last_visible_result_count = filtered_count



        total_in_result = len(notes_to_display)
        self._last_total_result_count = total_in_result
        min_score = (sensitivity_threshold / 100.0) * max_score if max_score > 0 else 0



        # No slider: omit score % from status; with slider would show " (score \xe2\u2030\xa5 X%)"



        sensitivity_text = ""



        if self.sensitivity_slider is not None:



            effective_pct = round(100 * min_score / max_score) if (max_score > 0 and sensitivity_threshold > 0) else None



            sensitivity_text = ""



            if self.sensitivity_value_label is not None:



                if sensitivity_threshold == 0:



                    self.sensitivity_value_label.setText("0%")



                elif sensitivity_threshold > 0:



                    self.sensitivity_value_label.setText(f"{sensitivity_threshold}%")



        searched_suffix = ""



        if hasattr(self, 'total_notes_searched') and self.total_notes_searched is not None:



            searched_suffix = f" (searched {self.total_notes_searched} in {getattr(self, 'fields_description', 'Text & Extra')})"



        threshold_label = f"Threshold: {self._global_relevance_threshold_percent()}%"



        # #region agent log



        try:



            from .search_dialog_shared import _session_debug_log



            _session_debug_log(



                "H1",



                "filter_and_display_notes.status_threshold",



                "status bar threshold",



                data={"relevance_threshold_percent": self._global_relevance_threshold_percent()},



            )






        except Exception:



            pass



        # #endregion



        visible_status = f"Showing {filtered_count} of {total_in_result} | {threshold_label}"
        detail_parts = [visible_status]
        if hidden_message:
            detail_parts.append(hidden_message)
        if searched_suffix:
            detail_parts.append(searched_suffix.strip(" ()"))

        self.status_label.setText(visible_status)
        self.status_label.setToolTip("\n".join(detail_parts))







        # Enable/disable toggle button based on list content



        has_items = self.results_list.rowCount() > 0



        if hasattr(self, 'toggle_select_btn'):



            self.toggle_select_btn.setEnabled(has_items)







        self.results_list.blockSignals(False)



        # Restore selections from persistence



        if hasattr(self, 'selected_note_ids') and self.selected_note_ids:



            self.restore_selections()







        # Update selection count and button text



        self.update_selection_count()







        # Update View All button state and tooltip (enabled when list has rows)



        self._update_view_all_button_state()







        # Re-enable sorting and keep the user's current sort choice when the
        # threshold slider rebuilds the table.



        self.results_list.setSortingEnabled(True)

        sort_section, sort_order = saved_sort or self._default_source_table_sort(
            citation_order_display,
            has_context_refs,
        )
        self.results_list.sortItems(sort_section, sort_order)
