"""Vertex AI model wrappers with kagent transport configuration."""

from __future__ import annotations

import os
from functools import cached_property
from typing import Optional

import httpx
from anthropic import AsyncAnthropicVertex
from google.adk.models.anthropic_llm import Claude as ClaudeLLM
from google.adk.models.google_llm import Gemini as GeminiLLM
from google.adk.utils._google_client_headers import get_tracking_headers
from google.genai import Client, types

from ._ssl import KAgentTLSMixin


def _google_cloud_location() -> str | None:
    return os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GOOGLE_CLOUD_REGION")


def _merge_headers(extra_headers: Optional[dict[str, str]]) -> dict[str, str]:
    headers = get_tracking_headers()
    if extra_headers:
        headers.update(extra_headers)
    return headers


class _KAgentVertexTransportMixin(KAgentTLSMixin):
    """Shared TLS/header fields for Vertex-backed Python SDK clients."""

    extra_headers: Optional[dict[str, str]] = None
    api_key_passthrough: Optional[bool] = None


class KAgentGeminiVertexLlm(_KAgentVertexTransportMixin, GeminiLLM):
    """Gemini-on-Vertex model that preserves ADC while applying kagent TLS settings."""

    model_config = {"arbitrary_types_allowed": True}

    def _http_options(self, *, api_version: str | None = None) -> types.HttpOptions:
        verify = self._tls_verify()
        kwargs = {}
        if verify is not None:
            kwargs = {
                "client_args": {"verify": verify},
                "async_client_args": {"verify": verify},
            }
        return types.HttpOptions(
            headers=_merge_headers(self.extra_headers),
            retry_options=self.retry_options,
            base_url=self.base_url,
            api_version=api_version,
            **kwargs,
        )

    @cached_property
    def api_client(self) -> Client:
        return Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=_google_cloud_location(),
            http_options=self._http_options(),
        )

    @cached_property
    def _live_api_client(self) -> Client:
        return Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=_google_cloud_location(),
            http_options=self._http_options(api_version=self._live_api_version),
        )


class KAgentClaudeVertexLlm(_KAgentVertexTransportMixin, ClaudeLLM):
    """Claude-on-Vertex model that preserves Vertex auth while applying kagent TLS settings."""

    model_config = {"arbitrary_types_allowed": True}

    @cached_property
    def _anthropic_client(self) -> AsyncAnthropicVertex:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        region = _google_cloud_location()
        if not project_id or not region:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION (or GOOGLE_CLOUD_REGION) "
                "must be set for using Anthropic on Vertex."
            )

        kwargs = {
            "project_id": project_id,
            "region": region,
            "default_headers": _merge_headers(self.extra_headers),
        }
        verify = self._tls_verify()
        if verify is not None:
            kwargs["http_client"] = httpx.AsyncClient(verify=verify)
        return AsyncAnthropicVertex(**kwargs)
