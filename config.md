# Anki Semantic Search – Configuration & Privacy

## Overview

Anki Semantic Search lets you search your Anki notes semantically and get AI-generated answers. This document describes what data is sent to providers, how to configure API keys, and how to install optional dependencies.

---

## Privacy & Data Sent to Providers

**AI answers may send selected note content to the chosen provider.** When you run a search:

- **Embedding providers** (Voyage, OpenAI, Cohere, Ollama): Receive note text from the fields you selected in Settings → Note Type Filter. This is used to compute semantic similarity.
- **LLM providers** (OpenAI, Anthropic, Ollama, etc.): Receive your query plus the text of notes that match your search. This context is sent so the AI can answer based on your notes.

**Local options:** If you use **Ollama** for both embeddings and answers, no data leaves your machine.

**Local Memory:** The add-on can store compact snippets from final answer context in `user_files/agent_memory.db`. These snippets are used only for future retrieval planning and are never treated as citable answer notes. If hybrid memory retrieval is enabled with a cloud embedding provider, memory snippets may be sent to that provider for embeddings.

**Preview payload:** In the Settings dialog, use the "Preview payload" or similar option to see exactly what would be sent before running a search.

---

## API Keys

### Where to set them

1. Open **Tools → Anki Semantic Search → Settings** (or the add-on menu).
2. In the **Provider** section, choose your provider and enter the API key.
3. In the **Search / Embedding** section, set keys for embedding providers (Voyage, OpenAI, Cohere) if you use them.

### Environment variables (optional)

- `VOYAGE_API_KEY` – used for Voyage embeddings if not set in Settings.
- Other providers use the keys from the Settings UI only.

### Security

- API keys are stored in `config.json` in the add-on folder. Do not share this file.
- Keys are redacted in debug logs.
- **Rotate/revoke** any keys that may have been exposed (e.g. in older versions or shared configs).

---

## Optional Dependencies (Manual Install)

The add-on works without optional packages, but some features require them. Install manually using Anki’s Python or the Python you use for the Cross-Encoder rerank.

### 1. sentence-transformers (Cross-Encoder reranking)

- **Purpose:** Better retrieval quality via re-ranking.
- **Install:**  
  `pip install sentence-transformers`  
  Or, if using a separate Python for rerank:  
  `"path/to/that/python" -m pip install sentence-transformers`
- **Where to run:** In the same Python environment that you set as "Python for Cross-Encoder" in Settings, or in Anki’s Python if you use in-process rerank.

### 2. anthropic (Anthropic streaming)

- **Purpose:** Real-time streaming of AI answers when using Anthropic.
- **Install:**  
  `pip install anthropic`
- **Where to run:** In Anki’s Python environment. On Windows, you can find it via:  
  `python -c "import sys; print(sys.executable)"`  
  while Anki is running, or use the path shown in the add-on’s dependency dialog.

### 3. pyspellchecker (Spell check in search box)

- **Purpose:** Underline misspellings and right-click suggestions in the search input.
- **Install:**  
  `pip install pyspellchecker`
- **Where to run:** In Anki’s Python environment.

### Troubleshooting

- **"Module not found"** – Ensure you install into the correct Python (Anki’s or the Cross-Encoder Python).
- **Windows:** The add-on can show the exact `pip` command and Python path in a popup when a dependency is missing.
- See the add-on’s **Tools → Anki Semantic Search → Install Optional Dependencies** for guided instructions.

---

## Config File Location

User config is stored in:

- **File:** `config.json` in the add-on folder (e.g. `Anki2/addons21/AI_search/config.json`).
- Config is merged with safe defaults on load; your changes persist in this file.

### Local LLM context budgeting

For local OpenAI-compatible servers such as LM Studio, set:

```json
"local_llm_context_tokens": 12288
```

inside `search_config` to match the model's loaded context length. The add-on uses this as an upper budget and automatically sends less context for simple questions and more context for complex, multi-note questions.

### Relevance threshold

The user-facing result cutoff is:

```json
"relevance_threshold_percent": 65
```

This value is clamped to `0..80`. It controls both visible search rows and which ordinary notes are eligible for AI answer context. Retrieval keeps a hidden low internal floor so reranking still receives enough candidates.

### Local memory settings

Relevant `search_config` keys:

```json
"enable_profile_memory": true,
"memory_retrieval_mode": "auto_hybrid",
"memory_retention_days": 30,
"memory_max_saved_snippets_per_search": 24,
"memory_max_retrieved_snippets": 5,
"memory_embedding_enabled": true,
"agentic_max_retrieval_passes": 3,
"agentic_max_subqueries": 6
```

`memory_retrieval_mode` can be `text` or `auto_hybrid`. New installs default to `auto_hybrid`; text mode remains available for local SQLite scoring only. Auto hybrid uses memory embeddings when available and falls back to text mode if embedding fails, times out, or is stale.
