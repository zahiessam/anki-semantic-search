# Stage 2 Refactor Recommendations

This report lists organization and de-duplication candidates only. No file moves or logic merges were made in Stage 1.

## Completed In Stage 2

- Extracted keyword scoring helpers from `core/engine.py` into `core/keyword_scoring.py`, while preserving the existing `core.engine` import surface.
- Extracted embedding dimension mismatch detection into `core/errors.py` and reused it from `core/engine.py`, `core/workers.py`, and `ui/dialogs.py`.
- Replaced the duplicate `_semantic_chunk_text` implementation in `ui/dialogs.py` with the shared `utils.text.semantic_chunk_text` helper.
- Extracted dependency repair/install helpers into `ui/dependency_install.py`.
- Extracted public dialog launch coordination into `ui/dialog_entrypoints.py`, with compatibility wrappers still exposed from `ui/dialogs.py`.
- Extracted `SettingsDialog` into `ui/settings_dialog.py`, with a compatibility import still exposed from `ui/dialogs.py`.
- Extracted local search worker threads and standalone rerank helpers into `ui/search_workers.py`, with compatibility imports still exposed from `ui/dialogs.py`.
- Extracted answer prompt construction helpers into `ui/answer_prompts.py`.
- Extracted answer HTML formatting and citation link generation into `ui/answer_formatting.py`, while keeping `AISearchDialog.format_answer()` as a wrapper.
- Extracted citation click handling, browser opening, and answer clipboard helpers into `ui/answer_navigation.py`, while keeping `AISearchDialog` wrappers.
- Extracted the dynamic search workflow container and method installer into `ui/search_workflow.py`, while keeping the same runtime method-copy behavior.
- Extracted note content loading, text helpers, and dialog-facing keyword/TF-IDF wrappers into `ui/note_content.py`.
- Extracted query expansion, AI generic-term exclusion, HyDE generation, and relevance-mode helpers into `ui/query_enhancement.py`.
- Extracted dialog-facing embedding lookup/search, note metadata, and context boost helpers into `ui/embedding_helpers.py`.
- Extracted search dialog state, history, scope banner, settings launcher, and option handlers into `ui/search_dialog_state.py`.

## Safe File-Splitting Candidates

### `ui/dialogs.py`

- Split search UI into `ui/search_dialog_main.py` for `AISearchDialog` construction, search controls, selection controls, history, and scope banner behavior.
- Split answer rendering into `ui/answer_formatting.py` for answer HTML formatting, citation parsing, clipboard output, and browser navigation helpers.
- Split search workflow callbacks into `ui/search_workflow.py` for embedding progress, keyword filtering, rerank callbacks, result display, and copied method compatibility.
- Settings UI has been split into `ui/settings_dialog.py`; next cleanup is updating any direct imports once compatibility wrappers are no longer needed.
- Dependency repair helpers have been split into `ui/dependency_install.py`; next cleanup is smoke-testing install/repair buttons in Anki.
- Search worker threads have been split into `ui/search_workers.py`; next cleanup is comparing them with `core/workers.py` before removing any remaining duplicate concepts.

### `core/engine.py`

- Split embedding providers into `core/embedding_providers.py` for Voyage, OpenAI, Cohere, Ollama, and local OpenAI-compatible request functions.
- Split Anki collection queries into `core/collection_queries.py` for note type counts, deck names/counts, deck query building, and eligibility analysis.
- Split embedding storage into `core/embedding_store.py` for SQLite table creation, blob conversion, save/load, JSON migration, and batch flushing.
- Split keyword scoring into `core/keyword_scoring.py` for stemming, stop words, keyword extraction, TF-IDF scoring, and score aggregation.

### `core/workers.py`

- Split embedding workers into `core/embedding_workers.py` for embedding search and embedding index creation.
- Split rerank workers into `core/rerank_workers.py` for rerank availability checks, keyword filtering, cross-encoder rerank, and answer relevance rerank.
- Split answer workers into `core/answer_workers.py` for non-streaming provider calls and Anthropic streaming.

## Common-Code Merge Candidates

- Replace `AISearchDialog._simple_stem`, `_get_extended_stop_words`, `_extract_keywords_improved`, `_compute_tfidf_scores`, and `_aggregate_scored_notes_by_note_id` with imports from the keyword scoring helpers already exposed through `core.engine`.
- Replace local chunking helpers in `ui/dialogs.py` with `utils.text.semantic_chunk_text` once the call signatures and chunk-size semantics are confirmed equivalent.
- Remove duplicate worker class definitions from `ui/dialogs.py` after confirming all UI signal expectations are covered by `core.workers`.
- Use `core.engine._is_embedding_dimension_mismatch` everywhere and delete the duplicate helper in `ui/dialogs.py`.
- Keep the dynamic method-copy compatibility block until the copied methods are moved into a real mixin or controller class and `AISearchDialog` owns those methods directly.

## Suggested Refactor Order

1. Smoke-test the completed `dependency_install`, `dialog_entrypoints`, `settings_dialog`, `search_workers`, and `search_workflow` splits inside Anki.
2. Extract answer formatting only after saving representative answer/citation output for comparison.
3. Replace dynamic method copying with explicit mixins only after search workflow smoke tests pass.
4. Remove compatibility shims only after Anki smoke tests pass with the new module boundaries.

## Required Smoke Tests After Each Split

- Add-on imports without errors in Anki.
- Search dialog opens and can start a search.
- Settings dialog opens, loads current config, and saves without changing unrelated config keys.
- Embedding/indexing flow starts and reports progress.
- Optional dependency checks still show the expected guidance dialogs.
