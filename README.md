# Anki Semantic Search

Semantic search over your Anki notes with AI-generated answers. Find relevant cards by meaning, not just keywords.

## Quick Start (New in v1.1.0)

If you are updating from an older version, the add-on will automatically migrate your embedding databases and logs into the new `user_files/` directory to keep your root directory clean.

1. **Open** Tools → Anki Semantic Search → Settings.
2. **Standardized Setup**: Use the new unified Cloud AI section to configure Voyage, OpenAI, or Cohere.
3. **Local AI**: If using Ollama, ensure it is running in the background. The UI now hides local-only scanning widgets when Cloud AI is active.
4. **Re-index**: If you notice missing results, go to **Create/Update Embeddings**. The new version uses a profile-specific database naming convention for better multi-profile support.
5. **Search**: Enter your query in the Search dialog. The status banner at the top will now give you real-time feedback on your API/Local connectivity.

### Quick start (Ollama)

Minimal path to run everything locally with no API keys:

1. Install and run [Ollama](https://ollama.com); pull a model (e.g. `ollama pull llama3.2` and `ollama pull nomic-embed-text`).
2. In the add-on: **Tools → Anki Semantic Search → Settings**. In **API Settings**, set **Answer with** to **Ollama (local)**. In **Search Settings**, set the embedding engine to **Ollama (local)**. Save.
3. **Tools → Anki Semantic Search → Create/Update Embeddings** to build embeddings.
4. **Tools → Anki Semantic Search → Search Notes**, enter a query, and click Search.

## Requirements

- **Anki 2.1.66+**
- **Embedding engine:** Ollama (local) or Voyage/OpenAI/Cohere (Cloud)
- **AI provider:** Ollama (local) or OpenAI/Anthropic/Gemini (Cloud)

## Privacy & Data Isolation

- **Isolation Protocol:** All user-specific files (SQLite databases, search history, debug logs) are redirected to the `user_files/` subdirectory.
- **Security:** `config.json` and `user_files/` are ignored by Git to prevent accidental leaking of API keys or personal note data.
- **Local-First:** Use **Ollama** for both embeddings and answers to keep all data on your machine.

## Changelog

### v1.1.0 (The "Refinement" Update)
- **UI Standardization**: Unified Cloud AI provider panels using a helper-method architecture, ensuring consistent behavior and easier maintenance.
- **Data Isolation Protocol**: Moved all `.db`, `.log`, and `.json` user data into a dedicated `user_files/` directory.
- **Emoji Sanitization**: Replaced Unicode emojis with ASCII/Plain text symbols to fix "weird character" rendering errors on Windows/Qt environments.
- **Zombie Widget Containment**: Grouped local AI scanning labels and buttons into visibility groups; they now hide completely when a Cloud AI provider is selected.
- **Real-time Status Banner**: Rewrote the status engine to pull live data from the UI state rather than stale disk configs.
- **Multi-Profile Support**: Improved profile-specific hash detection for embedding databases.

### v1.0.0
- Initial release with support for Ollama, OpenAI, Voyage, and Cohere.

## Credits

- **Author:** Zahi-Essam  
- **License:** MIT  
- **Development Tools**: Built with **Cursor AI**, **Android Studio with Gemini**, and roughly $50 in API testing costs to ensure high-quality medical search performance.

## License

See [LICENSE](LICENSE).
