"""Explicit remote inference exception; does not relax the local network policy."""
from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx

from app.llm.vllm_provider import VLLMProvider

if TYPE_CHECKING:
    from app.core.config import Settings


def validate_remote_endpoint(endpoint: str) -> None:
    message = "Remote inference requires a public HTTPS base URL without credentials, query or fragment"
    try:
        parsed = urlsplit(endpoint)
        parsed.port
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or any(char.isspace() for char in endpoint) or "\\" in endpoint
                or "%" in parsed.netloc):
            raise ValueError(message)
        host = parsed.hostname.lower().rstrip(".")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            # Reject internal names and legacy numeric IP spellings such as
            # 127.1, which some resolvers interpret as loopback addresses.
            if ("." not in host or host[-1].isdigit()
                    or host.endswith((".localhost", ".local", ".internal", ".test", ".invalid"))):
                raise ValueError(message)
        else:
            if not address.is_global:
                raise ValueError(message)
    except ValueError:
        raise ValueError(message) from None


def validate_remote_settings(settings: Settings) -> None:
    if not settings.remote_inference_enabled:
        raise ValueError("Set SOVEREIGN_REMOTE_INFERENCE_ENABLED=true to explicitly allow remote inference")
    validate_remote_endpoint(settings.remote_vllm_url)
    if not settings.remote_vllm_model.strip():
        raise ValueError("SOVEREIGN_REMOTE_VLLM_MODEL must name the model served by the GPU host")
    _validate_key(settings.remote_vllm_api_key.get_secret_value())


def _validate_key(api_key: str) -> None:
    if (not api_key.strip() or api_key.strip().lower() in {"local", "changeme", "replace-me"}
            or any(char.isspace() for char in api_key)):
        raise ValueError("Configure a non-placeholder SOVEREIGN_REMOTE_VLLM_API_KEY without whitespace")


class RemoteVLLMProvider(VLLMProvider):
    """Authenticated HTTPS vLLM with streaming, cancellation and fail-closed errors."""

    provider_name = "remote_vllm"
    inference_label = "Remote"
    _validate_endpoint = staticmethod(validate_remote_endpoint)

    def __init__(self, endpoint: str, *, enabled: bool = False, api_key: str, **kwargs) -> None:
        if not enabled:
            raise ValueError("Remote inference is disabled")
        _validate_key(api_key)
        kwargs["allow_fallback"] = False
        super().__init__(endpoint, api_key=api_key, **kwargs)

    @staticmethod
    def _error_detail(error: Exception) -> str:
        # Do not reflect server messages, request URLs or credentials into audit/UI.
        if isinstance(error, httpx.HTTPStatusError):
            return f"GPU endpoint returned HTTP {error.response.status_code}"
        return type(error).__name__

    async def _is_model_available(self, client: httpx.AsyncClient, model: str) -> None:
        # Avoid a second internet round trip before every generation. /models is
        # checked explicitly by health/readiness; listing isn't proof of warmth.
        return None
