"""Frontier model wrapper — Google Gemini via google-genai SDK."""

from __future__ import annotations

import os
from typing import Iterator, Optional

from google import genai
from google.genai import types


SUPPORTED_MODELS: dict[str, str] = {
    "Gemini 2.0 Flash (recommended)": "gemini-2.0-flash",
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 1.5 Pro": "gemini-1.5-pro",
}

DEFAULT_MODEL = "gemini-2.0-flash"


class FrontierModel:
    """Thin wrapper around the Google Gemini API for chat completion.

    Converts OpenAI-style {role, content} message lists to Gemini's
    Contents format automatically.  System messages are passed as
    system_instruction.  Supports streaming and non-streaming modes.
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
        resolved_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=resolved_key)

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
    ) -> str | Iterator[str]:
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        system_text, contents = self._convert_messages(messages)

        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_tok,
            system_instruction=system_text if system_text else None,
        )

        if stream:
            return self._stream(contents, config)

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config,
        )
        return response.text or ""

    def _stream(self, contents, config) -> Iterator[str]:
        for chunk in self.client.models.generate_content_stream(
            model=self.model_id,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[str, list]:
        """Split system message out; convert rest to Gemini Contents."""
        system_parts: list[str] = []
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                # Gemini uses "user" and "model" (not "assistant")
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                ))
        return "\n\n".join(system_parts), contents

    # ── Metadata ───────────────────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        return self.model_id

    @staticmethod
    def list_models() -> dict[str, str]:
        return SUPPORTED_MODELS
