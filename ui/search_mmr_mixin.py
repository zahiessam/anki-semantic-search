import re

from ..core.keyword_scoring import get_extended_stop_words
from ..utils import log_debug


class SearchMMRMixin:
    def _mmr_token_set(self, note, stop_words, common_tokens):
        text = (note.get('display_content') or note.get('content') or '').lower()
        tokens = re.findall(r"\b\w+\b", text)
        meaningful = []
        for token in tokens:
            if len(token) <= 2 or token in stop_words or token in common_tokens:
                continue
            meaningful.append(token)
            if len(meaningful) >= 200:
                break
        return set(meaningful)

    def _automatic_mmr_candidate_pool(self, scored_count, retrieval):
        """Choose a hidden MMR pool that scales with the requested result count."""
        if scored_count <= 0:
            return 0

        try:
            requested_results = int(retrieval.get("max_results", 50) or 50)
        except Exception:
            requested_results = 50

        target_pool = max(50, min(100, requested_results * 2))
        return min(scored_count, target_pool)

    def _apply_mmr_diversity(self, scored_notes, retrieval, config):
        """Apply dependency-free MMR in Retrieval V2.

        The candidate pool is intentionally hidden from the UI. It adapts to
        the requested result count, stays within 50-100 notes when possible,
        and shrinks when fewer candidate notes are available.
        """
        self._mmr_applied = False
        self._mmr_candidate_pool_size = 0
        if not scored_notes or not retrieval.get("enable_mmr_diversity", True):
            return scored_notes

        pool_size = self._automatic_mmr_candidate_pool(len(scored_notes), retrieval)
        self._mmr_candidate_pool_size = pool_size
        if pool_size <= 2:
            return scored_notes

        search_config = (config or {}).get("search_config") or {}
        verbose = bool(search_config.get("verbose_search_debug"))
        if verbose:
            log_debug(
                f"MMR diversity: pool_size=auto({pool_size}), candidates={len(scored_notes)}, "
                f"max_results={retrieval.get('max_results')}"
            )

        pool = list(scored_notes[:pool_size])
        rest = list(scored_notes[pool_size:])
        max_score = max((score for score, _ in pool), default=1.0) or 1.0
        lambda_value = float(retrieval.get("mmr_lambda", 0.75))

        stop_words = get_extended_stop_words(search_config)
        token_docs = []
        token_df = {}
        for _, note in pool:
            text = (note.get('display_content') or note.get('content') or '').lower()
            token_set = {
                token
                for token in re.findall(r"\b\w+\b", text)
                if len(token) > 2 and token not in stop_words
            }
            token_docs.append(token_set)
            for token in token_set:
                token_df[token] = token_df.get(token, 0) + 1

        common_cutoff = max(1, int(0.6 * len(pool)))
        common_tokens = {token for token, count in token_df.items() if count > common_cutoff}
        token_sets = [self._mmr_token_set(note, stop_words, common_tokens) for _, note in pool]

        selected_indices = [0]
        remaining = set(range(1, len(pool)))

        def jaccard(a, b):
            if not a or not b:
                return 0.0
            return len(a & b) / max(1, len(a | b))

        while remaining:
            best_idx = None
            best_score = None
            for idx in remaining:
                relevance = max(0.0, pool[idx][0]) / max_score
                redundancy = max(jaccard(token_sets[idx], token_sets[sel]) for sel in selected_indices)
                mmr_score = lambda_value * relevance - (1.0 - lambda_value) * redundancy
                if best_score is None or mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        self._mmr_applied = True
        return [pool[idx] for idx in selected_indices] + rest
