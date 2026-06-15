"""Low-cost diagnostics for cloud answer API providers."""

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

from ..utils.log import log_debug


DEFAULT_ANSWER_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
    "openrouter": "google/gemini-flash-1.5",
    "custom": "gpt-4o-mini",
}


@dataclass
class CloudDiagnosticResult:
    ok: bool
    status: str
    provider: str
    check_type: str
    endpoint: str
    latency_ms: int = 0
    detail: str = ""
    http_status: Optional[int] = None


def classify_provider_error(status_code=None, body="", reason=""):
    """Normalize provider/network errors into stable UI categories."""
    text = f"{body or ''} {reason or ''}".lower()

    if status_code in (401, 403) or any(
        marker in text
        for marker in (
            "invalid api key",
            "incorrect api key",
            "unauthorized",
            "forbidden",
            "authentication",
            "permission denied",
            "api key not valid",
            "invalid x-api-key",
        )
    ):
        return "auth_failed"

    if any(
        marker in text
        for marker in (
            "insufficient_quota",
            "quota exceeded",
            "quota_exceeded",
            "billing",
            "payment required",
            "credit balance",
            "out of credits",
            "no credits",
            "usage limit",
        )
    ):
        return "quota_or_billing"

    if status_code == 429 or "rate limit" in text or "too many requests" in text:
        return "rate_limited"

    if status_code == 404 or any(
        marker in text
        for marker in (
            "model_not_found",
            "model not found",
            "model unavailable",
            "does not exist",
            "not found for api version",
        )
    ):
        return "model_unavailable"

    if status_code and status_code >= 500:
        return "server_error"

    if any(
        marker in text
        for marker in (
            "timed out",
            "timeout",
            "getaddrinfo failed",
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname provided",
            "winerror 11001",
            "winerror 10051",
            "winerror 10060",
            "winerror 10065",
            "connection refused",
            "connection reset",
            "network is unreachable",
        )
    ):
        return "network_error"

    return "server_error"


def provider_status_message(status):
    return {
        "ok": "Connection OK",
        "auth_failed": "Authentication failed. Check that the API key is valid and not revoked.",
        "quota_or_billing": "Quota or billing problem. The key was accepted, but the account cannot complete requests.",
        "rate_limited": "Rate limited. The provider is reachable, but requests are temporarily blocked.",
        "model_unavailable": "Selected model is unavailable. Check the provider/model setting.",
        "network_error": "Network error. Check internet access, DNS, proxy/VPN, or the custom endpoint.",
        "server_error": "Provider server error. Try again later or check the endpoint.",
    }.get(status, "Provider check failed.")


def _provider_label(provider):
    return {
        "anthropic": "Anthropic",
        "openai": "OpenAI",
        "google": "Google Gemini",
        "openrouter": "OpenRouter",
        "custom": "Custom OpenAI-compatible",
    }.get((provider or "").lower(), provider or "Cloud API")


def _request_json(method, url, payload=None, headers=None, timeout=8):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return resp.status, json.loads(raw) if raw else {}


def _openai_compatible_models_url(api_url):
    value = (api_url or "").strip()
    if not value:
        value = "https://api.openai.com/v1"
    if "://" not in value:
        value = "https://" + value
    value = value.rstrip("/")
    lower = value.lower()
    for suffix in ("/chat/completions", "/completions", "/embeddings"):
        if lower.endswith(suffix):
            value = value[: -len(suffix)]
            lower = value.lower()
            break
    if lower.endswith("/models"):
        return value
    if not lower.endswith("/v1"):
        value = value + "/v1"
    return value + "/models"


def _redact_url_secret(url):
    try:
        parsed = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        redacted_pairs = [
            (key, "***" if key.lower() in ("key", "api_key", "apikey", "token") else value)
            for key, value in query
        ]
        redacted_query = urllib.parse.urlencode(redacted_pairs, safe="*")
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, redacted_query, parsed.fragment))
    except Exception:
        return url or ""


def test_cloud_answer_connection(provider, api_key, api_url="", model=None, timeout=8):
    """Run the cheapest available cloud answer-provider check."""
    provider = (provider or "openai").strip().lower()
    api_key = (api_key or "").strip()
    model = (model or DEFAULT_ANSWER_MODELS.get(provider) or "gpt-4o-mini").strip()
    label = _provider_label(provider)

    if not api_key and provider != "custom":
        return CloudDiagnosticResult(
            ok=False,
            status="auth_failed",
            provider=label,
            check_type="API key validation",
            endpoint="",
            detail="Enter an API key first.",
        )

    if provider == "openai":
        method = "GET"
        endpoint = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = None
        check_type = "models endpoint"
    elif provider == "openrouter":
        method = "GET"
        endpoint = "https://openrouter.ai/api/v1/auth/key"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = None
        check_type = "key status endpoint"
    elif provider == "google":
        method = "GET"
        endpoint = "https://generativelanguage.googleapis.com/v1/models?" + urllib.parse.urlencode({"key": api_key})
        headers = {}
        payload = None
        check_type = "models endpoint"
    elif provider == "anthropic":
        method = "POST"
        endpoint = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
        check_type = "minimal message request"
    else:
        if not (api_url or "").strip():
            return CloudDiagnosticResult(
                ok=False,
                status="network_error",
                provider=label,
                check_type="models endpoint",
                endpoint="",
                detail="Enter a custom API URL first.",
            )
        method = "GET"
        endpoint = _openai_compatible_models_url(api_url)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        payload = None
        check_type = "models endpoint"

    start = time.perf_counter()
    safe_endpoint = _redact_url_secret(endpoint)
    try:
        log_debug(f"Testing cloud API connection: provider={provider}, check={check_type}, endpoint={safe_endpoint}")
        status_code, _data = _request_json(method, endpoint, payload=payload, headers=headers, timeout=timeout)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return CloudDiagnosticResult(
            ok=True,
            status="ok",
            provider=label,
            check_type=check_type,
            endpoint=safe_endpoint,
            latency_ms=elapsed_ms,
            detail=provider_status_message("ok"),
            http_status=status_code,
        )
    except urllib.error.HTTPError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = str(exc)
        status = classify_provider_error(exc.code, body, str(exc))
        log_debug(f"Cloud API test failed: provider={provider}, status={status}, http={exc.code}, body={body[:500]}")
        return CloudDiagnosticResult(
            ok=False,
            status=status,
            provider=label,
            check_type=check_type,
            endpoint=safe_endpoint,
            latency_ms=elapsed_ms,
            detail=provider_status_message(status),
            http_status=exc.code,
        )
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        reason = getattr(exc, "reason", exc)
        status = classify_provider_error(None, "", str(reason))
        log_debug(f"Cloud API network test failed: provider={provider}, status={status}, error={reason}")
        return CloudDiagnosticResult(
            ok=False,
            status=status,
            provider=label,
            check_type=check_type,
            endpoint=safe_endpoint,
            latency_ms=elapsed_ms,
            detail=provider_status_message(status),
        )
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        status = classify_provider_error(None, "", str(exc))
        log_debug(f"Cloud API test failed: provider={provider}, status={status}, error={exc}")
        return CloudDiagnosticResult(
            ok=False,
            status=status,
            provider=label,
            check_type=check_type,
            endpoint=safe_endpoint,
            latency_ms=elapsed_ms,
            detail=provider_status_message(status),
        )
