import json
import urllib.request
import urllib.error
from .utils.log import log_debug

class LLMProvider:
    def generate(self, payload):
        raise NotImplementedError()

class OpenAICompatibleProvider(LLMProvider):
    """Handles OpenAI, LM Studio, Jan, and any OpenAI-compatible local server."""
    def __init__(self, api_key, base_url, model=None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, payload):
        url = f"{self.base_url}/chat/completions"
        # Ensure model is set in payload if provided in init
        if self.model and "model" not in payload:
            payload["model"] = self.model

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res["choices"][0]["message"]["content"]
        except Exception as e:
            log_debug(f"LLM Request failed at {url}: {str(e)}")
            raise

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key, model=None):
        self.api_key = api_key
        self.model = model or "claude-3-5-sonnet-20241022"

    def generate(self, payload):
        url = "https://api.anthropic.com/v1/messages"
        # Anthropic uses a different payload structure than OpenAI
        system = ""
        user_msg = ""
        for msg in payload.get("messages", []):
            if msg["role"] == "system":
                system = msg["content"]
            if msg["role"] == "user":
                user_msg = msg["content"]

        anthropic_payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": user_msg}]
        }

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = json.dumps(anthropic_payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["content"][0]["text"]

class OllamaProvider(LLMProvider):
    """Direct Ollama API (non-OpenAI path)."""
    def __init__(self, base_url, model=None):
        self.base_url = base_url.rstrip("/")
        self.model = model or "llama3.2"

    def generate(self, payload):
        url = f"{self.base_url}/api/chat"
        # Convert OpenAI style to Ollama style if needed
        ollama_payload = {
            "model": self.model,
            "messages": payload.get("messages", []),
            "stream": False,
            "options": {"temperature": payload.get("temperature", 0.7)}
        }

        data = json.dumps(ollama_payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["message"]["content"]

def get_provider(name, api_key, config=None):
    """Factory to get the right provider based on settings."""
    if config is None:
        from aqt import mw
        config = mw.addonManager.getConfig(__name__) or {}

    if name == "Anthropic":
        return AnthropicProvider(api_key, model=config.get("anthropic_model"))

    if name == "Ollama" or name == "ollama":
        base_url = config.get("ollama_base_url", "http://localhost:11434")
        model = config.get("ollama_chat_model", "llama3.2")
        return OllamaProvider(base_url, model=model)

    if name == "LM Studio" or name == "Local OpenAI" or name == "local_openai":
        # Check both top-level and search_config for these keys
        sc = config.get("search_config", {})
        base_url = config.get("local_llm_url") or sc.get("local_llm_url") or "http://localhost:1234/v1"
        model = config.get("local_llm_model") or sc.get("local_llm_model") or "model-identifier"
        return OpenAICompatibleProvider(api_key="", base_url=base_url, model=model)

    # Default to OpenAI
    return OpenAICompatibleProvider(api_key, "https://api.openai.com/v1", model=config.get("openai_chat_model"))

def create_payload(provider_name, system_prompt, user_prompt, temperature=0.7, model=None):
    """Standardized payload creator."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature
    }
