"""Constants used by the Settings dialog."""

ANSWER_CLOUD_PROVIDERS = [
    ("Anthropic (Claude)", "anthropic", "sk-ant-..."),
    ("OpenAI (GPT)", "openai", "sk-..."),
    ("Google (Gemini)", "google", "AI..."),
    ("OpenRouter", "openrouter", "sk-or-..."),
    ("Custom / OpenAI-compatible", "custom", "custom key"),
]

EMBEDDING_CLOUD_PROVIDERS = [
    ("Voyage AI (Recommended)", "Voyage AI", "pa-..."),
    ("OpenAI", "OpenAI", "sk-..."),
    ("Cohere", "Cohere", "co-..."),
]

RERANK_MODEL_PRESETS = [
    ("Clinical Accuracy Boost - BiomedBERT", "NeuML/biomedbert-base-reranker"),
    ("Fast general - MiniLM", "cross-encoder/ms-marco-MiniLM-L6-v2"),
    ("Stronger general - BGE v2 M3", "BAAI/bge-reranker-v2-m3"),
]

ANSWER_KEY_PROVIDER_PREFIXES = (
    ("sk-ant-", "anthropic"),
    ("sk-or-", "openrouter"),
    ("AI", "google"),
    ("sk-", "openai"),
)

EMBEDDING_KEY_PROVIDER_PREFIXES = (
    ("sk-ant-", None),
    ("sk-or-", None),
    ("pa-", "Voyage AI"),
    ("co-", "Cohere"),
    ("sk-", "OpenAI"),
)
