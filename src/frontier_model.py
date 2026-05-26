"""Frontier model wrapper via Anthropic Claude API."""

from __future__ import annotations

import os
from typing import Iterator, Optional

import anthropic


SUPPORTED_MODELS: dict[str, str] = {
    "Claude Sonnet 4.6 (recommended)": "claude-sonnet-4-6",
    "Claude Haiku 4.5": "claude-haiku-4-5-20251001",
    "Claude Opus 4.7": "claude-opus-4-7",
}

DEFAULT_MODEL = "claude-sonnet-4-6"


class FrontierModel:
    """Thin wrapper around the Anthropic Messages API for chat completion.

    Separates the ``system`` role from the conversation history automatically
    (Anthropic's API treats system as a top-level parameter, not a message).
    Supports streaming and non-streaming modes.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=resolved_key)

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
    ) -> str | Iterator[str]:
        """Send a chat request and return full text or a streaming iterator."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        system, conv = self._split_system(messages)

        kwargs: dict = {
            "model": self.model_id,
            "max_tokens": max_tok,
            "temperature": temp,
            "messages": conv,
        }
        if system:
            kwargs["system"] = system

        if stream:
            return self._stream(**kwargs)
        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def _stream(self, **kwargs) -> Iterator[str]:
        with self.client.messages.stream(**kwargs) as stream:
            yield from stream.text_stream

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
        """Pull out any system message; return (system_text, remaining_messages)."""
        system_parts: list[str] = []
        conv: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conv.append(msg)
        return "\n\n".join(system_parts), conv

    # ── Metadata ───────────────────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        return self.model_id

    @staticmethod
    def list_models() -> dict[str, str]:
        return SUPPORTED_MODELS
