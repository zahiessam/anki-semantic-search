# Anki Semantic Search

Semantic search over your Anki notes with AI-generated answers. Find relevant cards by meaning, not just keywords.

## Quick Start

1. **Install** the add-on via AnkiWeb or by placing the add-on folder in your Anki addons directory.
2. **Open** Tools → Anki Semantic Search (or use the menu entry).
3. **Configure** Settings to choose:
   - Note types and fields to search
   - Embedding engine (Ollama, Voyage, OpenAI, Cohere)
   - AI provider for answers (Ollama, OpenAI, Anthropic, etc.)
4. **Enter** your query and run a search.

### Quick start (Ollama)

Minimal path to run everything locally with no API keys:

1. Install and run [Ollama](https://ollama.com); pull a model (e.g. `ollama pull llama3.2` and `ollama pull nomic-embed-text`).
2. In the add-on: **Tools → Anki Semantic Search → Settings**. In **API Settings**, set **Answer with** to **Ollama (local)**. In **Search Settings**, set the embedding engine to **Ollama (local)** and ensure the Ollama URL is correct (default `http://localhost:11434`). Save.
3. **Tools → Anki Semantic Search → Create/Update Embeddings** to build embeddings for your note types/decks (run once, or after adding many notes).
4. **Tools → Anki Semantic Search → Search Notes**, enter a query, and click Search.

## Requirements

- **Anki 2.1.66+** (tested on 25.09.2)
- **Embedding engine:** Ollama (local, no key) or Voyage/OpenAI/Cohere (API key required)
- **AI provider:** Ollama (local) or OpenAI/Anthropic (API key required)

## Optional Features

| Feature | Package | Install |
|--------|---------|---------|
| Cross-encoder reranking | sentence-transformers | `pip install sentence-transformers` |
| Anthropic streaming | anthropic | `pip install anthropic` |
| Spell check in search | pyspellchecker | `pip install pyspellchecker` |

See [config.md](config.md) for full details and troubleshooting.

## Privacy

AI answers may send selected note content to the chosen provider. Use **Ollama** for both embeddings and answers to keep everything local. See [config.md](config.md) for details.

## Changelog

- **1.0.0** – Initial release.

## Credits

- **Author:** Zahi-Essam  
- **License:** MIT  
- Built with Cursor AI and API costs (~$50) to develop this add-on.

## License

See [LICENSE](LICENSE).
