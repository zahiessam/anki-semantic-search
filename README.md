# Anki Semantic Search 🧠🔍

Semantic search over your Anki notes with AI-generated answers. Find relevant cards by **meaning**, not just keywords.

---

## 🚀 How to Access

This add-on is integrated into your Anki workflow via two main routes:

1.  **The Side Drawer:** Look for the **Semantic Search icon** in the left panel/drawer of your Anki main window.
2.  **Add-ons Manager:** Go to **Tools → Add-ons**, select **Anki Semantic Search**, and click **Config** (or double-click the name) to access settings.

---

## 🚀 Quick Start

### 1. Installation
- Install via AnkiWeb [Code: 751846187] or drop the folder into your addons directory.
- Restart Anki.

### 2. Configure Your Engines
Open the **Settings** (via the side drawer icon or Add-ons Config):

| **The Librarian (Search/Embeddings)** | **The Doctor (Answering Queries)** |
| :--- | :--- |
| *How the AI finds your cards.* | *How the AI explains the results.* |
| **Options:** Ollama (Free), Voyage, OpenAI, Cohere. | **Options:** Local Server, OpenAI, Anthropic, Gemini. |
| **Setup:** Choose one in "Search Settings". | **Setup:** Choose one in "API Settings". |

> **💡 Pro Tip:** You can search locally with **Ollama** (Private/Free) but use a high-powered **Cloud LLM** to generate the final answer for the best of both worlds.

### 3. Select Note Types & Build Index
- In Settings, go to **Search Settings** and **select which Note Types** you want to search (e.g., "Medical", "Cloze").
- Go to **Create/Update Embeddings** (found in the main Search dialog or Side Drawer).
- Click **Start**. This "teaches" the AI about your cards.

### 4. Start Searching
Open the **Search Dialog** from the side drawer. Ask a question: *"Mechanism of action for ACE inhibitors?"* and the AI will find the cards and summarize an answer.

---

## 🛠 Local AI Support (Privacy First)
Keep your search 100% private and free by using local AI servers:
- **Ollama (Recommended):** Native support for high-speed local embeddings and chat.
- **OpenAI-Compatible Servers:** Connect to **LM Studio**, **Jan**, **LocalAI**, or **vLLM** by entering your local URL (e.g., `http://localhost:1234/v1`) in the "Local OpenAI" settings.

## 🛠 Requirements
- **Anki 2.1.66+** (Qt6).
- **Local AI:** [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai) running locally.
- **Cloud AI:** An API key for your chosen provider (OpenAI, Voyage, etc.).

## 🔒 Privacy & Data Management
- **Isolation Protocol:** Your search data and logs are stored in `user_files/` to keep your Anki profile folder tidy.
- **Cloud Privacy:** If using Cloud providers, only the cards relevant to your search are sent to the AI for answering. Use a local server for 100% privacy.

## 📜 Changelog (v1.1.0)
- **Standardized UI:** Easier setup for Cloud providers.
- **Local Server Expansion:** Improved compatibility with LM Studio and other OpenAI-style local servers.
- **Emoji Fix:** Removed "weird letters" from Windows menus.
- **Clean Storage:** User data moved to `user_files/` for better organization.
- **Real-time Status:** A new status bar in settings tells you exactly if your API key or Local Server is working.

## 🎓 Credits
- **Author:** Zahi-Essam  
- **Tools:** Built with Cursor AI & Android Studio (Gemini).  
- **License:** MIT
