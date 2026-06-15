"""Initial keyword, HyDE, embedding-prep, and retrieval scoring setup."""

import re
import time

from aqt.qt import QApplication

from . import note_content
from .search_workers import RRF_K
from ..utils import (
    RETRIEVAL_RELEVANCE_FLOOR_PERCENT,
    clamp_relevance_threshold_percent,
    load_config,
    log_debug,
)



class SearchKeywordFilterMixin:
    def keyword_filter(self, query, notes):



        """



        Enhanced semantic search with multiple methods:



        - Improved keyword extraction (stemming, n-grams, TF-IDF)



        - Optional embedding-based search using cloud embeddings (Voyage)



        - Hybrid approach combining both methods



        - Context-aware ranking



        """



        import time
        import re
        combine_started = time.time()
        keyword_started = time.time()







        # Get search configuration



        config = dict(getattr(self, "_search_pending_config", None) or load_config())



        search_config = dict(config.get('search_config', {}) or {})
        search_config['search_method'] = search_config.get('search_method') or 'hybrid'
        config['search_config'] = search_config



        if getattr(self, "_effective_relevance_threshold_percent", None) is None:
            self._effective_relevance_threshold_percent = 65
        if not getattr(self, "_relevance_threshold_source", None):
            self._relevance_threshold_source = "session_default"



        original_search_method = search_config.get('search_method') or 'hybrid'



        search_method = original_search_method  # 'keyword', 'keyword_rerank', 'embedding', 'hybrid'



        use_context_boost = search_config.get('use_context_boost', True)



        # keyword_rerank = keyword scoring then cross-encoder rerank (no embeddings)



        if search_method == 'keyword_rerank':



            search_method = 'keyword'  # use keyword path; effective_method will show "Keyword + Re-rank"







        # Always run synonym expansion (built-in medical aliases + config synonym_overrides).



        # Optional AI-based expansion runs inside _expand_query when enable_query_expansion is on.



        original_query = query



        query = self._expand_query(query, config)



        self._search_expanded_query = query



        if query != original_query:



            self._analytics_stage(
                "query_expanded",
                original_query=original_query,
                expanded_query=query,
            )







        # Optional: AI-mediated generic term exclusion (one short LLM call per search)



        if search_config.get('use_ai_generic_term_detection', False):



            try:



                self._query_ai_excluded_terms = self._get_ai_excluded_terms(query, config)



            except Exception:



                self._query_ai_excluded_terms = set()



        else:



            self._query_ai_excluded_terms = set()







        # Improved keyword extraction



        keywords, stems, phrases = self._extract_keywords_improved(query)
        query_intent = getattr(self, "_query_intent", None)







        if not keywords and not phrases:



            return ([(1, note) for note in notes[:50]], "Keyword only", min(50, len(notes)))







        # Compute TF-IDF scores



        bm25_scores = self._compute_bm25_scores(notes, keywords)
        if search_config.get("verbose_search_debug", False):
            log_debug(
                "Retrieval diagnostics: "
                f"keyword_scoring_seconds={time.time() - keyword_started:.3f}, "
                f"keyword_scorer={getattr(self, '_query_keyword_scoring_method', 'tfidf')}, "
                f"notes={len(notes)}, keywords={len(keywords)}, phrases={len(phrases)}"
            )







        # Get embedding scores if available and method requires it



        embedding_scores = None



        embeddings_available = False



        # HyDE: optional hypothetical document for better semantic retrieval



        embedding_query = query



        if search_method in ('embedding', 'hybrid') and search_config.get('enable_hyde', False):



            hyde_doc = self._generate_hyde_document(query, config)
            try:
                self._analytics_stage(
                    "hyde_generated",
                    hyde_prompt_type=getattr(self, "_last_hyde_prompt_type", "standard"),
                    hyde_length=int(getattr(self, "_last_hyde_length", len(hyde_doc or "")) or 0),
                    hyde_has_clinical_anchor=bool(getattr(self, "_last_hyde_has_clinical_anchor", False)),
                )
            except Exception:
                pass



            if hyde_doc:



                embedding_query = hyde_doc



                log_debug("Using HyDE hypothetical document for embedding search")







        if search_method in ('embedding', 'hybrid'):



            # For speed and better relevance, only run the slower



            # embedding-based search on the top N TF-IDF candidates.



            max_notes_for_embedding = 2000



            notes_sorted = None



            if len(notes) > max_notes_for_embedding:



                notes_sorted = sorted(



                    notes,



                    key=lambda n: bm25_scores.get(n['id'], 0),



                    reverse=True,



                )



                notes_for_embedding = notes_sorted[:max_notes_for_embedding]



            else:



                notes_for_embedding = notes



            # Run embedding search in background worker so UI stays responsive



            state = dict(



                notes=notes, query=query, keywords=keywords, stems=stems, phrases=phrases,

                query_intent=query_intent,



                bm25_scores=bm25_scores, search_method=search_method,



                original_search_method=original_search_method, search_config=search_config,



                use_context_boost=use_context_boost, config=config,



                notes_sorted=notes_sorted, max_notes_for_embedding=max_notes_for_embedding,



            )



            return ("PENDING_EMBEDDING", embedding_query, notes_for_embedding, state)







        # Score notes using selected method (keyword-only path)



        scored_notes = []



        keyword_scored_list = []  # For RRF: (keyword_score, note) for hybrid



        max_score = 0



        max_emb_score = max(embedding_scores.values()) if embedding_scores else 0.0



        min_emb_frac = 0.25  # hybrid weighted fallback



        very_high_emb_frac = 0.9  # near-best semantic match gets through even w/ weak keywords



        use_rrf = (search_method == 'hybrid' and embedding_scores and embeddings_available)



        high_freq_keywords = getattr(self, "_query_high_freq_keywords", set()) or set()



        if not isinstance(high_freq_keywords, set):



            try:



                high_freq_keywords = set(high_freq_keywords)



            except Exception:



                high_freq_keywords = set()



        # Search cascade: when all query keywords are high-freq (generic), down-weight keyword-only score



        query_all_high_freq = bool(keywords and all(k in high_freq_keywords for k in keywords))







        for idx, note in enumerate(notes):

            content_lower = note['content'].lower()



            keyword_score = 0



            matched_keywords = 0







            # Improved keyword matching with stemming



            for keyword in keywords:



                # Skip auto-detected high-frequency filler terms for this query



                if keyword in high_freq_keywords:



                    continue



                # Exact whole word match

                whole_word = bool(re.search(r'\b' + re.escape(keyword) + r'\b', content_lower))



                if whole_word:



                    matched_keywords += 1



                    count = content_lower.count(keyword)



                    keyword_score += min(count * 12, 45) + 10



                elif keyword in content_lower:



                    matched_keywords += 1



                    keyword_score += 4







                # Check stemmed versions



                stem = self._simple_stem(keyword)



                if stem != keyword and stem in content_lower:



                    keyword_score += 3







            # Phrase matching (bigrams and trigrams)



            for phrase in phrases:



                # Don't give phrase bonus if all its tokens are high-frequency fillers



                tokens = phrase.split()



                if tokens and all(t in high_freq_keywords for t in tokens):



                    continue



                if phrase in content_lower:



                    keyword_score += 18







            # Add TF-IDF component



            tfidf_score = bm25_scores.get(note['id'], 0) * 2  # Weight BM25



            keyword_score += tfidf_score

            intent_boost = note_content.apply_query_intent_boost_for_dialog(self, content_lower, keywords)
            if intent_boost:
                keyword_score += intent_boost
                note['_intent_boost'] = intent_boost
            else:
                note.pop('_intent_boost', None)







            # Combine with embedding score if available



            if search_method == 'embedding':



                if embedding_scores and embeddings_available:



                    # Use only embedding score



                    final_score = embedding_scores.get(note['id'], 0)



                else:



                    # Fallback to keyword if embeddings not available



                    # This ensures the search still works even without embeddings



                    final_score = keyword_score



            elif search_method == 'hybrid':



                final_score = keyword_score



            else:



                final_score = keyword_score







            # Cascade: don't treat generic-keyword-only matches as highly relevant



            if query_all_high_freq and final_score == keyword_score:



                final_score = final_score * 0.3







            # Apply context-aware boost



            if use_context_boost:



                final_score = self._context_aware_boost(note, final_score)







            # For RRF hybrid: collect keyword scores; attach Focused/Balanced/Broad flags for UI filtering.



            if use_rrf:



                emb_score = embedding_scores.get(note['id'], 0) if embedding_scores else 0



                passes_focused, passes_balanced, passes_broad = self._passes_focused_balanced_broad(



                    matched_keywords, final_score, emb_score, max_emb_score, keywords,



                    search_method, embeddings_available, min_emb_frac, very_high_emb_frac,



                )



                note['_passes_focused'] = passes_focused



                note['_passes_balanced'] = passes_balanced



                note['_passes_broad'] = passes_broad



                keyword_scored_list.append((keyword_score, final_score, note))  # final_score has context boost



                continue







            # Inclusion criteria for non-RRF: include if passes Broad (superset); attach all three flags for UI filtering.



            emb_score = embedding_scores.get(note['id'], 0) if embedding_scores else 0



            passes_focused, passes_balanced, passes_broad = self._passes_focused_balanced_broad(



                matched_keywords, final_score, emb_score, max_emb_score, keywords,



                search_method, embeddings_available, min_emb_frac, very_high_emb_frac,



            )



            note['_passes_focused'] = passes_focused



            note['_passes_balanced'] = passes_balanced



            note['_passes_broad'] = passes_broad







            # region agent log



            try:



                q_lower = (getattr(self, "_search_pending_query", None) or query or "").lower()



                if "trisom" in q_lower:



                    from .search_dialog_shared import _agent_debug_log



                    _agent_debug_log(



                        run_id="post-fix_candidate",



                        hypothesis_id="H4",



                        location="__init__.keyword_filter",



                        message="candidate_inclusion_decision",



                        data={



                            "note_id": note.get("id"),



                            "matched_keywords": matched_keywords,



                            "final_score": final_score,



                            "emb_score": emb_score,



                            "search_method": search_method,



                            "len_keywords": len(keywords),



                            "passes_focused": passes_focused,



                            "passes_balanced": passes_balanced,



                            "passes_broad": passes_broad,



                        },



                    )



            except Exception:



                pass



            # endregion







            if passes_broad:



                scored_notes.append((final_score, note))



                max_score = max(max_score, final_score)







        # RRF (Reciprocal Rank Fusion): combine keyword and vector rankings. Standard formula 1/(k+rank); more effective than weighted averaging.



        # With chunks, use best rank per note id (min rank across chunks).



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



            # Notes that came only from embedding_results may lack _passes_*; show them in Broad/Balanced.



            for _s, note in scored_notes:



                if '_passes_broad' not in note:



                    note['_passes_broad'] = True



                    note['_passes_balanced'] = True



                    note['_passes_focused'] = False



            max_score = scored_notes[0][0] if scored_notes else 1







        # Normalize scores to 0-100 range if needed



        if max_score > 0 and max_score != 100:



            scored_notes = [(score / max_score * 100, note) for score, note in scored_notes]







        scored_notes.sort(reverse=True, key=lambda x: x[0])



        # Keep un-aggregated list for AI context so it can cite specific sections (chunks)



        has_chunks = any(n.get('chunk_index') is not None for _, n in scored_notes)



        if has_chunks:



            self._scored_notes_for_context = list(scored_notes)



        else:



            self._scored_notes_for_context = None



        # Aggregate chunks by note id: one entry per note (best score), display full content



        scored_notes = self._aggregate_scored_notes_by_note_id(scored_notes)







        # Effective method shown to user (may differ from config if embeddings unavailable)



        if original_search_method == "keyword_rerank":



            effective_method = "Keyword + Re-rank"



        elif search_method == "embedding" and embeddings_available:



            effective_method = "Embedding only"



        elif search_method == "hybrid" and embeddings_available:



            effective_method = "Hybrid"



        else:



            effective_method = "Keyword only"







        # Optional: Cross-encoder re-ranking in background (avoids UI freeze)



        if scored_notes and (search_config.get('enable_rerank', False) or original_search_method == 'keyword_rerank'):



            return ("PENDING_RERANK", scored_notes, effective_method, 0, notes)







        # Stored superset for threshold changes: keep retrieval broad so rerank/display can refilter locally.



        MAX_STORED_FOR_MODES = 100



        min_relevance = clamp_relevance_threshold_percent(
            getattr(self, "_effective_relevance_threshold_percent", 65)
        )



        min_relevance_stored = RETRIEVAL_RELEVANCE_FLOOR_PERCENT



        all_above_stored = [x for x in scored_notes if x[0] >= min_relevance_stored]



        scored_notes = all_above_stored[:MAX_STORED_FOR_MODES] if all_above_stored else scored_notes[:5]



        total_above_threshold = sum(1 for x in scored_notes if x[0] >= min_relevance)
        if search_config.get("verbose_search_debug", False):
            log_debug(
                "Retrieval diagnostics: "
                f"combine_seconds={time.time() - combine_started:.3f}, "
                f"embedding_candidates={len(embedding_results or [])}, "
                f"final_candidates={len(scored_notes)}, method={effective_method}"
            )



        return scored_notes, effective_method, total_above_threshold


