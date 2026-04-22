# Anki Semantic Search 🧠🔍

Semantic search over your Anki notes with AI-generated answers. Find relevant cards by **meaning**, not just keywords.

---

## 🚀 Quick Start

### 1. Installation
- Install via AnkiWeb [Code: 751846187] or drop the folder into your addons directory.
- Restart Anki.

### 2. Configure Your Engines (Tools → Anki Semantic Search → Settings)

You need to set up two things (they can be different!):

| **The Librarian (Search/Embeddings)** | **The Doctor (Answering Queries)** |
| :--- | :--- |
| *How the AI finds your cards.* | *How the AI explains the results.* |
| **Options:** Ollama (Free), Voyage, OpenAI, Cohere. | **Options:** Ollama, OpenAI, Anthropic, Gemini. |
| **Setup:** Choose one in "Search Settings". | **Setup:** Choose one in "API Settings". |

> **💡 Pro Tip:** You can search locally with **Ollama** (Private/Free) but use **OpenAI** to generate the final answer for the best of both worlds.

### 3. Select Note Types & Build Index
- In Settings, go to **Search Settings** and **select which Note Types** you want to search (e.g., "Medical", "Cloze").
- Go to **Tools → Anki Semantic Search → Create/Update Embeddings**. 
- Click **Start**. This "teaches" the AI about your cards.

### 4. Start Searching
Use the **Search Notes** menu or shortcut. Ask a question: *"Mechanism of action for ACE inhibitors?"* and the AI will find the cards and summarize an answer.

---

## 🛠 Requirements
- **Anki 2.1.66+** (Qt6).
- **Local AI:** [Ollama](https://ollama.com) must be installed and running.
- **Cloud AI:** An API key for your chosen provider (OpenAI, Voyage, etc.).

## 🔒 Privacy & Data Management
- **Isolation Protocol:** Your search data and logs are stored in `user_files/` to keep your Anki profile folder tidy.
- **Cloud Privacy:** If using Cloud providers, only the cards relevant to your search are sent to the AI for answering. Use Ollama for 100% local privacy.

## 📜 Changelog (v1.1.0)
- **Standardized UI:** Easier setup for Cloud providers.
- **Emoji Fix:** Removed "weird letters" from Windows menus.
- **Clean Storage:** User data moved to `user_files/` for better organization.
- **Real-time Status:** A new status bar in settings tells you exactly if your API key or Ollama is working.

## 🎓 Credits
- **Author:** Zahi-Essam  
- **Tools:** Built with Cursor AI & Android Studio (Gemini).  
- **License:** MIT
