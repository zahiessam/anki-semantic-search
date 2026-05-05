"""Model selection defaults for search answer providers."""


class SearchModelSelectionMixin:
    """Owns default model names used when provider-specific config is absent."""

    def get_best_model(self, provider):
        models = {
            'anthropic': 'claude-sonnet-4-20250514',
            'openai': 'gpt-4o-mini',
            'google': 'gemini-1.5-flash',
            'openrouter': 'google/gemini-flash-1.5',
            'ollama': 'llama3.2'
        }

        return models.get(provider, 'gpt-4o-mini')
