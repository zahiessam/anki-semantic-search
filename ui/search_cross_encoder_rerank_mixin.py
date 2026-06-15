from aqt.utils import showInfo

from .search_workers import MAX_RERANK_COUNT, _do_rerank
from ..utils import (
    RETRIEVAL_RELEVANCE_FLOOR_PERCENT,
    clamp_relevance_threshold_percent,
    load_config,
    log_debug,
)
from ..utils.search_analytics_log import rank_deltas


class SearchCrossEncoderRerankMixin:
    def _on_rerank_done(self, scored_notes, success):



        """Called when RerankWorker finishes; apply min_relevance/max_results and continue search."""



        self._pending_rerank = False



        self._stop_estimated_progress_timer()



        self._hide_busy_progress()



        self.search_btn.setEnabled(True)
        if hasattr(self, "_set_search_chat_busy"):
            self._set_search_chat_busy(False)

        self._last_rerank_success = bool(success is True)







        if success == "LIBRARY_LOAD_FAILED":






            showInfo(



                "Reranking was skipped: sentence-transformers/torch could not be loaded (e.g. DLL error on Windows).\n\n"



                "This usually means the Visual C++ Redistributable is missing or corrupted.\n\n"



                "Search results are still shown using the initial ranking. To fix reranking, try:\n"



                "1. Tools \u2192 Anki Semantic Search \u2192 Install extra model (reinstalls dependencies)\n"



                "2. Check 'Python for Cross-Encoder' in Settings if using an external Python."



            )



            # Continue with original notes as success=False but success=="LIBRARY_LOAD_FAILED"



            # we already have scored_notes as the original ones.



            success = False







        self._analytics_stage(
            "cross_encoder_rerank",
            success=bool(success is True),
            rerank_status="ok" if success is True else "skipped_or_failed",
            top_results=self._analytics_results(scored_notes, limit=50),
            rank_deltas=rank_deltas(
                getattr(self, "_analytics_pre_rerank_results", []),
                self._analytics_results(scored_notes, limit=100),
            ),
        )

        # Ensure the final re-ranked order is used for numbering and AI context.
        # Clearing this forces _perform_search_continue to use the new scored_notes.
        self._scored_notes_for_context = None

        notes, effective_method, search_config = getattr(self, '_rerank_continue', (None, '', {}))



        if notes is None:



            return



        MAX_STORED_FOR_MODES = 100



        min_relevance = clamp_relevance_threshold_percent(
            getattr(self, "_effective_relevance_threshold_percent", 65)
        )



        min_relevance_stored = RETRIEVAL_RELEVANCE_FLOOR_PERCENT



        all_above_stored = [x for x in scored_notes if x[0] >= min_relevance_stored]



        scored_notes = all_above_stored[:MAX_STORED_FOR_MODES] if all_above_stored else scored_notes[:5]



        total_above_threshold = sum(1 for x in scored_notes if x[0] >= min_relevance)



        self._perform_search_continue(notes, scored_notes, effective_method, total_above_threshold)







    def _rerank_with_cross_encoder(self, query, scored_notes, top_k=15):



        """



        Re-rank top results using a cross-encoder (delegates to _do_rerank).



        Limited to top 15 to avoid CPU bottleneck. Use RerankWorker for non-blocking UI.



        """



        config = load_config()



        sc = config.get('search_config') or {}



        top_k = min(top_k, MAX_RERANK_COUNT)



        return _do_rerank(query, scored_notes, top_k, sc)







    def _passes_focused_balanced_broad(



        self,



        matched_keywords,



        final_score,



        emb_score,



        max_emb_score,



        keywords,



        search_method,



        embeddings_available,



        min_emb_frac=0.25,



        very_high_emb_frac=0.9,



    ):



        """Compute whether a note would pass Focused, Balanced, or Broad inclusion. Returns (passes_focused, passes_balanced, passes_broad)."""



        n_kw = len(keywords) if keywords else 0



        # Focused



        min_kw_focused = max(2, int(n_kw * 0.4)) if n_kw else 1



        if n_kw <= 2:



            min_kw_focused = 1



        min_score_focused = 18



        # Balanced



        min_kw_balanced = max(1, int(n_kw * 0.25)) if n_kw else 1



        min_score_balanced = 10



        # Broad



        min_kw_broad = max(1, int(n_kw * 0.2)) if n_kw else 1



        min_score_broad = 8







        if search_method == "embedding" and embeddings_available:



            if emb_score > 0:



                return (True, True, True)



            return (False, False, False)







        if search_method == "hybrid" and embeddings_available and max_emb_score > 0:



            very_high = emb_score >= very_high_emb_frac * max_emb_score



            decent = emb_score >= min_emb_frac * max_emb_score



            # Focused: (decent + (kw or score)) or very_high; and (not strict or matched_kw>0 or very_high)



            pf = (



                (decent and (matched_keywords >= min_kw_focused or final_score > min_score_focused)) or very_high



            ) and (matched_keywords > 0 or very_high)



            # Balanced



            pb_al = (decent and (matched_keywords >= min_kw_balanced or final_score > min_score_balanced)) or very_high



            # Broad



            pb_br = (decent and (matched_keywords >= min_kw_broad or final_score > min_score_broad)) or very_high



            return (pf, pb_al, pb_br)







        # Keyword-only or fallback



        pf = matched_keywords >= min_kw_focused



        if n_kw <= 2:



            pb_al = matched_keywords >= 1



            pb_br = matched_keywords >= 1



        else:



            pb_al = matched_keywords >= min_kw_balanced or final_score > min_score_balanced



            pb_br = matched_keywords >= min_kw_broad or final_score > min_score_broad



        return (pf, pb_al, pb_br)
