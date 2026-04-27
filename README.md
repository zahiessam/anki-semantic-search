# Anki Semantic Search

Semantic search and AI answers for your Anki collection. The add-on helps you find relevant notes by meaning, then can generate an answer grounded in the matching cards.

AnkiWeb code: `751846187`

## What It Does

- Searches your selected Anki notes by keyword, embeddings, or hybrid retrieval.
- Generates AI answers from the matching notes.
- Shows cited notes and lets you open selected results in the Anki Browser.
- Supports local-first workflows with Ollama, LM Studio, Jan, or other OpenAI-compatible local servers.
- Supports cloud providers for higher-quality answers or embeddings.
- Lets you choose exactly which decks, note types, and fields are searchable.
- Builds and refreshes an embedding index for semantic search.
- Includes optional medical/clinical retrieval tuning, re-ranking, query expansion, HyDE, and context-aware ranking.
- Keeps search history and lets you reload previous searches.

## Requirements

- Anki 2.1.66 or newer.
- Restart Anki after installing or updating the add-on.
- One of the following:
  - Local server for private/local answers or embeddings.
  - Cloud API key for cloud answers or embeddings.

Optional features may require Python packages listed in `requirements-optional.txt`.

## Quick Start

1. Install the add-on from AnkiWeb with code `751846187`, or place this folder in `Anki2/addons21/`.
2. Restart Anki.
3. Open the add-on from the side drawer or from Anki's add-ons/config area.
4. Open Settings.
5. Configure an answer provider in the API Settings tab.
6. Configure an embedding provider in the API Settings tab or keep it set to use the same provider when supported.
7. Choose decks, note types, and fields in Note Types & Fields.
8. Go to Search & Embeddings and click Create/Update Embeddings.
9. Open the search dialog and ask a question.

## Access Points

- Side drawer button in the Anki main window.
- Add-ons manager config entry.
- Settings button inside the search dialog.

## Main Search Workflow

1. Type a question in the search box.
2. Press Search or use `Ctrl+Enter`.
3. Review Matching Notes.
4. Select notes if you want to inspect or open them.
5. Read the AI answer and use citation links to jump to supporting notes.
6. Use Show only cited notes when you want to narrow the result table to notes used in the answer.

Search UI features:

- Search history dropdown.
- Clear history button.
- Matching notes table with relevance bars.
- Preview length slider for result snippets.
- Select all and deselect all controls.
- View Selected and View All buttons for opening notes in Anki Browser.
- Copy button for the AI answer.
- Source/status labels showing answer/search state.

## Settings Tabs

Settings use a shared dark theme across tabs, with consistent rows, controls, tables, section headers, and scrollbars. The API and Styling tabs are width-constrained for comfortable reading on wide monitors, while Search & Embeddings and Note Types & Fields remain full-width work surfaces.

### API Settings

Configure answer and embedding providers.

Answer providers:

- Local Server: Ollama, LM Studio, Jan, LocalAI, vLLM, or other OpenAI-compatible servers.
- Cloud API: OpenAI-compatible APIs, Anthropic Claude, Google Gemini, OpenRouter, and custom endpoints supported by the add-on configuration.

Embedding providers:

- Use same provider as answering when supported.
- Local server embedding endpoint.
- Voyage AI.
- OpenAI embeddings.
- Cohere embeddings.

Useful controls:

- API key input with Show/Hide.
- Automatic provider detection.
- Local server URL and model name.
- Local server autodetect for running Ollama, LM Studio, or Jan instances and loaded models.
- Local connection test.
- Cloud embedding provider selection.
- Embedding API key input.

### Search & Embeddings

Controls retrieval behavior and embedding indexing.

Search strategies:

- Keyword Only.
- Keyword + Re-rank.
- Hybrid (RRF).
- Embedding Only.

AI-assisted retrieval:

- Query Expansion: adds related medical/semantic terms.
- Filter Filler Words: uses AI to reduce generic query terms.
- HyDE: generates a hypothetical answer/document first, then searches with it.

Clinical accuracy tuning:

- Minimum relevance threshold.
- Max results pool.
- Hybrid embedding weight.
- Relevance from answer.
- Relevance mode: Focused, Balanced, or Broad.

Re-ranking:

- Cross-Encoder re-ranking through `sentence-transformers`.
- External Python path support, recommended on Windows to avoid Anki torch/DLL issues.
- Autodetect, browse, install/show command, and check-again controls.
- Context-aware ranking.

Technical diagnostics:

- Verbose search debug.
- Extra stop words.
- Max characters per note.
- Advanced Anki Python dependency install option.

Embedding actions:

- Create/Update Embeddings.
- Review Ineligible Notes.
- Test Connection.
- Reset to Clinical Defaults.

### Note Types & Fields

Choose what can be searched and embedded.

- Include all note types or choose specific note types.
- Include all decks or choose specific decks.
- Choose fields per note type.
- Search all fields option.
- Use first field fallback when a note type has no selected fields.
- Count notes with current settings.
- Save, load, and delete presets.
- Refresh note type, field, and deck lists.

This tab is the main place to prevent irrelevant cards from entering the search index.

### Styling

Controls visual and layout preferences.

- Question input font size.
- AI answer font size.
- Matching notes font size.
- Label font size.
- Default window width and height.
- Answer/notes layout mode.
- Section spacing.
- Answer line spacing.
- Centered, compact settings layout for easier use on wide displays.

## Embeddings And Indexing

Semantic search needs embeddings. Click Create/Update Embeddings after:

- First install.
- Changing embedding provider/model.
- Changing searchable decks, note types, or fields.
- Adding many new notes.
- Editing note content you want reflected in semantic search.

The add-on stores embedding data under `user_files/` and uses status checks to detect stale or incomplete indexing.

If a note is excluded from indexing, use Review Ineligible Notes to inspect why.

## Local AI Setup

### Ollama

1. Install Ollama.
2. Pull a chat model, for example:

```powershell
ollama pull llama3.2
```

3. Pull an embedding model, for example:

```powershell
ollama pull nomic-embed-text
```

4. In Settings, use:

```text
http://localhost:11434
```

### LM Studio, Jan, LocalAI, vLLM, or OpenAI-Compatible Servers

Use the server's OpenAI-compatible base URL, commonly:

```text
http://localhost:1234/v1
```

Make sure the server exposes the endpoints needed for your selected role:

- Chat/completions for answers.
- Embeddings endpoint for semantic search.

## Cloud Provider Setup

Enter API keys in Settings. Typical choices:

- Anthropic Claude for answers.
- OpenAI-compatible APIs for answers.
- Voyage AI for embeddings.
- OpenAI for embeddings.
- Cohere for embeddings.

Provider detection helps identify some keys automatically. Custom endpoints can be set through the API URL field where supported.

## Optional Dependencies

Optional packages are not required for basic search, but enable extra features.

Install into the correct Python environment:

- Anki Python for in-process UI features.
- External Python for Cross-Encoder re-ranking when configured.

Packages:

```powershell
pip install sentence-transformers anthropic pyspellchecker
```

Feature mapping:

- `sentence-transformers`: Cross-Encoder re-ranking.
- `anthropic`: Anthropic streaming support.
- `pyspellchecker`: spell check and right-click suggestions in the search box.

On Windows, using an external Python for `sentence-transformers` is recommended because torch can fail to load inside Anki's Python.

## Privacy

What may be sent externally depends on your provider choices.

- Local-only setup: if both answers and embeddings use a local server, note content stays on your machine.
- Cloud embeddings: selected searchable note fields are sent to the embedding provider when building/searching embeddings.
- Cloud answers: your query and selected matching note context are sent to the answer provider.

API keys are stored in `config.json` in the add-on folder. Do not share this file.

Logs and user data are stored under `user_files/` where possible. Logs redact sensitive values.

## Troubleshooting

### Search finds nothing

- Check Note Types & Fields.
- Confirm the deck is included.
- Rebuild embeddings after changing search scope.
- Lower the minimum relevance threshold.
- Try Hybrid (RRF).

### Results look irrelevant

- Use Note Types & Fields to limit searchable fields.
- Enable Query Expansion for medical synonyms.
- Enable Cross-Encoder re-ranking if installed.
- Try Focused relevance mode.
- Review extra stop words.

### Embeddings fail

- Use Test Connection in Search & Embeddings.
- Check API key or local server URL.
- Confirm the embedding model exists.
- For Ollama, pull the embedding model first.
- If provider/model changed, rebuild embeddings.

### Cross-Encoder is not ready

- Set an external Python path.
- Click Install / show command for external Python.
- Run the shown command if needed.
- Click Check again.

### Text in Matching Notes runs together

The add-on formats note display text separately from search text so HTML blocks and selected fields remain readable in previews and tooltips. Reopen the search dialog or rerun the search after updating.

## Developer Map

High-level modules:

- `__init__.py`: add-on entry point and menu/bootstrap integration.
- `llm.py`: answer provider calls.
- `rerank_helper.py`: helper entry point for external re-ranking.

Core:

- `core/engine.py`: embedding providers, storage, note/deck queries, checkpoint migration, and embedding helper APIs.
- `core/workers.py`: background workers for embedding search, indexing, keyword filtering, re-ranking, answer generation, and Anthropic streaming.
- `core/keyword_scoring.py`: keyword extraction, stemming, stop words, TF-IDF scoring, and score aggregation.
- `core/errors.py`: shared error classification helpers.
- `core/compat.py`: runtime compatibility helpers.

UI:

- `ui/dialogs.py`: compatibility layer and `AISearchDialog` wrapper methods.
- `ui/search_dialog_lifecycle.py`: search dialog construction/lifecycle.
- `ui/search_dialog_ui.py`: search dialog layout and controls.
- `ui/search_dialog_state.py`: history, scope banner, state, and settings launch helpers.
- `ui/search_workflow.py`: search workflow, result display, filtering, selection, and callback logic.
- `ui/search_workers.py`: UI-facing search worker helpers.
- `ui/search_dialog.py`: custom table delegates for content and relevance bars.
- `ui/settings_dialog.py`: Settings UI and persistence.
- `ui/theme.py`: shared Anki-aware theme tokens and Settings stylesheet helpers.
- `ui/widgets.py`: shared UI widgets, collapsible sections, sidebar, and field-row helpers.
- `ui/note_content.py`: loading selected note fields and display/search text preparation.
- `ui/query_enhancement.py`: query expansion, AI generic-term detection, HyDE, and relevance-mode helpers.
- `ui/embedding_helpers.py`: dialog-facing embedding lookup and metadata helpers.
- `ui/answer_prompts.py`: answer prompt construction.
- `ui/answer_formatting.py`: answer HTML formatting and citation rendering.
- `ui/answer_navigation.py`: citation navigation, browser opening, and clipboard helpers.
- `ui/dependency_install.py`: dependency checks, install guidance, Visual C++ checks, and PyTorch repair guidance.
- `ui/dialog_entrypoints.py`: public dialog open/close helpers.
- `ui/sidebar_bootstrap.py`: side drawer bootstrap.

Utils:

- `utils/config.py`: default config, config load/save, embedding config normalization, validation.
- `utils/paths.py`: profile-aware paths for embeddings and checkpoints.
- `utils/text.py`: HTML cleanup, display text cleanup, cloze reveal, chunking, and regex helpers.
- `utils/history.py`: search history persistence.
- `utils/log.py`: redacted debug logging.
- `utils/embeddings_status.py`: embedding status messages, stale index hints, and user-facing copy.

## Data Files

- `config.json`: user settings and API keys.
- `user_files/`: embeddings database/cache, checkpoints, logs, and runtime user data.

Do not publish `config.json` or `user_files/` with real personal data or API keys.

## License

MIT

## Credits

Author: Zahi-Essam

Built for medical and general Anki search workflows with semantic retrieval, local AI support, and cloud provider support.
