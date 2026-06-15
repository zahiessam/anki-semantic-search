"""Continuation after embedding retrieval returns results."""

import re

from aqt.utils import tooltip

from . import note_content
from .search_workers import RRF_K
from ..utils import (
    RETRIEVAL_RELEVANCE_FLOOR_PERCENT,
    clamp_relevance_threshold_percent,
    log_debug,
)



class SearchKeywordContinueMixin:
    def keyword_filter_continue(self, state, embedding_results, progress_callback=None):



        """Continue keyword_filter after embedding search worker finishes. Uses state from keyword_filter.



        progress_callback(idx, total) is called every 500 notes when provided (e.g. from a worker thread).



        When notes count is large and we have embedding results, only score a subset (top by TF-IDF + embedding note ids) to avoid long freezes."""



        import re



        notes = state["notes"]



        query = state["query"]



        keywords = state["keywords"]



        stems = state["stems"]



        phrases = state["phrases"]

        query_intent = state.get("query_intent")

        try:

            self._query_intent = query_intent

        except Exception:

            pass



        bm25_scores = state["bm25_scores"]



        search_method = state["search_method"]



        original_search_method = state["original_search_method"]



        search_config = state["search_config"]



        use_context_boost = state["use_context_boost"]



        config = state["config"]



        embedding_scores = None



        embeddings_available = False



        if embedding_results:



            embedding_scores = {note['id']: score for score, note in embedding_results}



            embeddings_available = True



            # Limit to subset when very large so "Combining results" does not take minutes



            COMBINE_MAX_NOTES = 6000



            if len(notes) > COMBINE_MAX_NOTES:



                emb_ids = {note['id'] for _, note in embedding_results}



                top_by_bm25 = sorted(notes, key=lambda n: bm25_scores.get(n['id'], 0), reverse=True)[:COMBINE_MAX_NOTES]



                top_ids = {n['id'] for n in top_by_bm25}



                subset_ids = top_ids | emb_ids



                notes = [n for n in notes if n['id'] in subset_ids]



                log_debug(f"keyword_filter_continue: limited to {len(notes)} notes (top {COMBINE_MAX_NOTES} by BM25 + embedding results)")



        else:



            if search_method == 'embedding' and not hasattr(self, '_embedding_warning_shown'):



                tooltip(



                    "No embeddings found for this search. Using keyword search.\n\n"



                    "If you already ran Create/Update Embeddings: the selected decks/note types may not match.",



                    period=5000,



                )



                self._embedding_warning_shown = True



            elif search_method == 'hybrid' and not hasattr(self, '_hybrid_warning_shown'):



                tooltip(



                    "No embeddings for these notes. Using keyword-only search.\n\n"



                    "Run Create/Update Embeddings (Settings \u2192 Search & Embeddings) for the selected decks/note types.",



                    period=4000,



                )



                self._hybrid_warning_shown = True



        scored_notes = []



        keyword_scored_list = []



        max_score = 0



        max_emb_score = max(embedding_scores.values()) if embedding_scores else 0.0



        min_emb_frac = 0.25



        very_high_emb_frac = 0.9



        use_rrf = (search_method == 'hybrid' and embedding_scores and embeddings_available)



        total_notes = len(notes)



        high_freq_keywords = getattr(self, "_query_high_freq_keywords", set()) or set()



        if not isinstance(high_freq_keywords, set):



            try:



                high_freq_keywords = set(high_freq_keywords)



            except Exception:



                high_freq_keywords = set()



        for idx, note in enumerate(notes):



            if progress_callback and idx > 0 and idx % 500 == 0:



                try:



                    progress_callback(idx + 1, total_notes)



                except Exception:



                    pass



            content_lower = note['content'].lower()



            keyword_score = 0



            matched_keywords = 0



            for keyword in keywords:



                if keyword in high_freq_keywords:



                    continue



                whole_word = bool(re.search(r'\b' + re.escape(keyword) + r'\b', content_lower))



                if whole_word:



                    matched_keywords += 1



                    count = content_lower.count(keyword)



                    keyword_score += min(count * 12, 45) + 10



                elif keyword in content_lower:



                    matched_keywords += 1



                    keyword_score += 4



                stem = self._simple_stem(keyword)



                if stem != keyword and stem in content_lower:



                    keyword_score += 3



            for phrase in phrases:



                tokens = phrase.split()



                if tokens and all(t in high_freq_keywords for t in tokens):



                    continue



                if phrase in content_lower:



                    keyword_score += 18



            keyword_score += bm25_scores.get(note['id'], 0) * 2

            intent_boost = note_content.apply_query_intent_boost_for_dialog(self, content_lower, keywords)
            if intent_boost:
                keyword_score += intent_boost
                note['_intent_boost'] = intent_boost
            else:
                note.pop('_intent_boost', None)



            if search_method == 'embedding':



                final_score = embedding_scores.get(note['id'], 0) if (embedding_scores and embeddings_available) else keyword_score



            else:



                final_score = keyword_score



            if use_context_boost:



                final_score = self._context_aware_boost(note, final_score)



            if use_rrf:



                emb_score = embedding_scores.get(note['id'], 0) if embedding_scores else 0



                passes_focused, passes_balanced, passes_broad = self._passes_focused_balanced_broad(



                    matched_keywords, final_score, emb_score, max_emb_score, keywords,



                    search_method, embeddings_available, min_emb_frac, very_high_emb_frac,



                )



                note['_passes_focused'] = passes_focused



                note['_passes_balanced'] = passes_balanced



                note['_passes_broad'] = passes_broad



                keyword_scored_list.append((keyword_score, final_score, note))



                continue



            # Inclusion: include if passes Broad; attach flags for UI mode filtering.



            emb_score = embedding_scores.get(note['id'], 0) if embedding_scores else 0



            passes_focused, passes_balanced, passes_broad = self._passes_focused_balanced_broad(



                matched_keywords, final_score, emb_score, max_emb_score, keywords,



                search_method, embeddings_available, min_emb_frac, very_high_emb_frac,



            )



            note['_passes_focused'] = passes_focused



            note['_passes_balanced'] = passes_balanced



            note['_passes_broad'] = passes_broad



            if passes_broad:



                scored_notes.append((final_score, note))



                max_score = max(max_score, final_score)



        if progress_callback and total_notes > 0:



            try:



                progress_callback(total_notes, total_notes)



            except Exception:



                pass



        # Optional verbose debug logging for search behavior tuning



        if search_config.get("verbose_search_debug", False):



            try:



                high_freq = getattr(self, "_query_high_freq_keywords", set()) or set()



                if not isinstance(high_freq, set):



                    high_freq = set(high_freq)



            except Exception:



                high_freq = set()



            top_notes = scored_notes[:5]



            debug_rows = []



            for score, note in top_notes:



                nid = note.get("id")



                emb_score = embedding_scores.get(nid, 0) if embedding_scores else 0



                bm25 = bm25_scores.get(nid, 0)



                snippet = (note.get("content", "") or "")[:160].replace("\n", " ")



                debug_rows.append(



                    {



                        "note_id": nid,



                        "score": round(score, 2),



                        "embedding_score": round(emb_score, 4) if isinstance(emb_score, (int, float)) else emb_score,



                        "bm25": round(bm25, 4) if isinstance(bm25, (int, float)) else bm25,



                        "snippet": snippet,



                    }



                )



            log_debug(



                f"verbose_search_debug: query={query!r}, keywords={keywords}, phrases={phrases[:6]}, "



                f"high_freq_keywords={sorted(list(high_freq))[:12]}, top_notes={debug_rows}"



            )



        if use_rrf and embedding_results:



            k = RRF_K



            emb_weight = max(0, min(1, (search_config.get('hybrid_embedding_weight', 50) or 50) / 100.0))



            kw_weight = 1.0 - emb_weight



            keyword_ranked = sorted(keyword_scored_list, key=lambda x: (x[0], x[1]), reverse=True)



            kw_rank = {}
            kw_note_by_id = {}



            for rank, (_, _, note) in enumerate(keyword_ranked, start=1):



                nid = note['id']



                kw_rank[nid] = min(kw_rank.get(nid, rank), rank)
                kw_note_by_id.setdefault(nid, note)



            emb_rank = {}
            emb_note_by_id = {}



            for rank, (_, note) in enumerate(embedding_results, start=1):



                nid = note['id']



                emb_rank[nid] = min(emb_rank.get(nid, rank), rank)
                emb_note_by_id.setdefault(nid, note)



            all_ids = set(kw_rank) | set(emb_rank)



            rrf_scores = []



            for nid in all_ids:



                rrf = 0



                if nid in kw_rank:



                    rrf += kw_weight * (1.0 / (k + kw_rank[nid]))



                if nid in emb_rank:



                    rrf += emb_weight * (1.0 / (k + emb_rank[nid]))



                if rrf > 0:



                    note = kw_note_by_id.get(nid) or emb_note_by_id.get(nid)



                    if note:



                        rrf_scores.append((rrf, note))



            scored_notes = sorted(rrf_scores, reverse=True, key=lambda x: x[0])



            for _s, note in scored_notes:



                if '_passes_broad' not in note:



                    note['_passes_broad'] = True



                    note['_passes_balanced'] = True



                    note['_passes_focused'] = False



            max_score = scored_notes[0][0] if scored_notes else 1



        if max_score > 0 and max_score != 100:



            scored_notes = [(score / max_score * 100, note) for score, note in scored_notes]



        scored_notes.sort(reverse=True, key=lambda x: x[0])



        has_chunks_cont = any(n.get('chunk_index') is not None for _, n in scored_notes)



        if has_chunks_cont:



            self._scored_notes_for_context = list(scored_notes)



        else:



            self._scored_notes_for_context = None



        scored_notes = self._aggregate_scored_notes_by_note_id(scored_notes)



        if original_search_method == "keyword_rerank":



            effective_method = "Keyword + Re-rank"



        elif search_method == "embedding" and embeddings_available:



            effective_method = "Embedding only"



        elif search_method == "hybrid" and embeddings_available:



            effective_method = "Hybrid"



        else:



            effective_method = "Keyword only"



        if scored_notes and (search_config.get('enable_rerank', False) or original_search_method == 'keyword_rerank'):



            return ("PENDING_RERANK", notes, scored_notes, effective_method, 0)



        MAX_STORED_FOR_MODES = 100



        min_relevance = clamp_relevance_threshold_percent(
            getattr(self, "_effective_relevance_threshold_percent", 65)
        )



        min_relevance_stored = RETRIEVAL_RELEVANCE_FLOOR_PERCENT



        all_above_stored = [x for x in scored_notes if x[0] >= min_relevance_stored]



        scored_notes = all_above_stored[:MAX_STORED_FOR_MODES] if all_above_stored else scored_notes[:5]



        total_above_threshold = sum(1 for x in scored_notes if x[0] >= min_relevance)



        return scored_notes, effective_method, total_above_threshold


