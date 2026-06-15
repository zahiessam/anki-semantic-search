class SearchResultExplainabilityMixin:
    def _get_matching_terms_for_note(self, note, query):
        """Return list of query terms that appear in the note (for 'Why this result?' explainability)."""
        if not query or not hasattr(self, '_extract_keywords_improved'):
            return []

        try:
            keywords, stems, phrases = self._extract_keywords_improved(query)
            content_lower = (note.get('content') or note.get('display_content') or '').lower()
            if not content_lower:
                return []

            matched = set()

            # Phrase matches first; these are usually the most informative.
            for p in phrases:
                if p and p.lower() in content_lower:
                    matched.add(p)

            # Exact keyword matches
            for w in keywords:
                wl = (w or '').lower()
                if wl and wl in content_lower:
                    matched.add(w)

            # Stem-based matches: map stem -> representative original query word
            stem_to_display = dict(stems or {})
            for w in keywords:
                stem = self._simple_stem(w)
                if stem:
                    stem_to_display.setdefault(stem, w)

            for stem, display in stem_to_display.items():
                # Skip extremely short stems to avoid over-matching
                if not stem or len(stem) <= 3:
                    continue
                if stem in content_lower:
                    matched.add(display)

            if not matched:
                return []

            # Filter out generic/low-information terms and very short tokens.
            # Also exclude per-query high-frequency terms (set during search) and AI-detected
            # generic terms so we don't show uninformative words in "Why this result?".
            stop_words = self._get_extended_stop_words()

            high_freq = getattr(self, '_query_high_freq_keywords', None) or set()
            if not isinstance(high_freq, set):
                try:
                    high_freq = set(high_freq)
                except Exception:
                    high_freq = set()

            ai_excluded = getattr(self, '_query_ai_excluded_terms', None) or set()
            if not isinstance(ai_excluded, set):
                try:
                    ai_excluded = set(ai_excluded)
                except Exception:
                    ai_excluded = set()

            def _is_meaningful(term: str) -> bool:
                t = (term or '').strip().lower()
                if not t:
                    return False
                if t in stop_words:
                    return False
                if t in high_freq:
                    return False
                if t in ai_excluded:
                    return False
                # Drop very short tokens by default (can whitelist later if needed)
                if len(t) <= 3:
                    return False
                return True

            filtered = [t for t in matched if _is_meaningful(t)]
            if not filtered:
                return []

            # Prefer more specific/phrase-like terms: phrases first, then by length
            filtered.sort(key=lambda t: (0 if ' ' in t else 1, -len(t), t.lower()))

            # region agent log
            try:
                if "trisom" in (query or "").lower():
                    from .search_dialog_shared import _agent_debug_log

                    _agent_debug_log(
                        run_id="pre-fix",
                        hypothesis_id="H2",
                        location="__init__._get_matching_terms_for_note",
                        message="matching_terms_for_note",
                        data={
                            "note_id": note.get("id"),
                            "query": query,
                            "matched_all": sorted(matched),
                            "filtered": filtered[:12],
                        },
                    )
            except Exception:
                pass
            # endregion

            return filtered[:12]

        except Exception:
            return []
