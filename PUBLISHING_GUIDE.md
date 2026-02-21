# Publishing Guide — GitHub & AnkiWeb

## AnkiWeb Description (copy-paste)

Use this when creating/editing your add-on on [AnkiWeb](https://ankiweb.net):

```markdown
**Search your Anki notes by meaning, not just keywords.** Find relevant cards with AI-powered semantic search and get answers generated from your own notes.

## Features

- **Semantic search** — Finds notes by conceptual similarity (e.g. "causes of heart failure" matches related content even without exact words)
- **AI answers** — Get summaries and answers grounded in your selected notes
- **Multiple providers** — Use Ollama (100% local, no API keys), or Voyage/OpenAI/Cohere/Anthropic with API keys
- **Flexible configuration** — Choose note types, fields, decks, relevance mode, and optional re-ranking
- **Privacy-friendly** — Use Ollama for embeddings and answers to keep all data on your machine

## Quick Start (Ollama — free, local)

1. Install [Ollama](https://ollama.com) and run: `ollama pull llama3.2` and `ollama pull nomic-embed-text`
2. In Anki: **Tools → Anki Semantic Search → Settings** → set embeddings and answers to **Ollama (local)**
3. **Tools → Anki Semantic Search → Create/Update Embeddings** (run once or after adding many notes)
4. **Tools → Anki Semantic Search → Search Notes** → enter a query and search

## Requirements

- Anki 2.1.66+ (tested on 25.09.2)
- Embedding engine: Ollama (local) or Voyage/OpenAI/Cohere (API keys)
- AI provider: Ollama (local) or OpenAI/Anthropic (API keys)

## Optional Enhancements

| Feature | Install |
|---------|---------|
| Cross-encoder re-ranking (better results) | `pip install sentence-transformers` |
| Anthropic streaming | `pip install anthropic` |
| Spell check in search box | `pip install pyspellchecker` |

## Credits

Developed by **Zahi-Essam**. Built with Cursor AI and API costs (~$50) to develop this add-on.
```

---

## 1. Naming

| Item | Recommendation |
|------|----------------|
| **Add-on name** | Anki Semantic Search |
| **Package ID** | AI_search (already in manifest.json) |
| **GitHub repo** | `anki-ai-semantic-search` or `AI_Semantic_Search` |

Suggested GitHub repo name: **`anki-ai-semantic-search`** — lowercase, hyphenated, easy to find.

---

## 2. Publish to GitHub

### Step 1: Create the repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `anki-ai-semantic-search`
3. Description: `Semantic search over Anki notes with AI-generated answers. Find cards by meaning, not keywords.`
4. Choose **Public**, add a **MIT** license and **README** if you want (you can overwrite with this project’s files)
5. Create repository (do **not** initialize with .gitignore if you already have one)

### Step 2: Initialize git and push

From the add-on folder:

```powershell
cd "e:\AnkiData\Anki2\addons21\AI_search"

# Initialize git (if not already)
git init

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/anki-ai-semantic-search.git

# Add files (respects .gitignore)
git add .

# Commit
git commit -m "v1.0.0 - Initial release"

# Push
git branch -M main
git push -u origin main
```

### Step 3: Create a release

1. On GitHub: **Releases** → **Create a new release**
2. Tag: `v1.0.0`
3. Title: `v1.0.0 — Initial release`
4. Description: `1.0.0 — Initial release. Semantic search, AI answers, optional re-ranking.`
5. Upload the `.ankiaddon` file: run `python build_release.py` and attach the generated zip

---

## 3. Publish to AnkiWeb

### Step 1: Create/add your add-on

1. Go to [ankiweb.net](https://ankiweb.net) and log in
2. **Add-ons** → **Shared** → **Upload an add-on**
3. Upload the `.ankiaddon` file from `Anki2/addons21/` (or run `python build_release.py` first)

### Step 2: Fill in the add-on page

- **Name:** Anki Semantic Search
- **Description:** Copy the AnkiWeb description from above
- **Source code:** `https://github.com/YOUR_USERNAME/anki-ai-semantic-search`
- **License:** MIT
- **Screenshots:** Add 1–2 if you have them

### Step 3: After approval

AnkiWeb may review the add-on. Once approved, users can install it from Anki via **Tools → Add-ons → Get Add-ons** and your code ID.

---

## 4. Files to update before publishing

- [ ] `manifest.json` — Verify `version`, `author` (e.g. `Zahi-Essam`)
- [ ] `LICENSE` — Add your name/year if desired
- [ ] `README.md` — Ensure credits and links are correct
- [ ] Run `python build_release.py` to create the release zip
