"""Open-source model wrapper via HuggingFace Inference API."""

from __future__ import annotations

import os
from typing import Iterator, Optional

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError


SUPPORTED_MODELS: dict[str, str] = {
    "Qwen2.5-0.5B-Instruct (recommended)": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Phi-3-mini-4k-Instruct": "microsoft/Phi-3-mini-4k-instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
}

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class OSSModel:
    """Thin wrapper around HuggingFace Serverless Inference for chat completion.

    Supports streaming and non-streaming modes.  Falls back gracefully when the
    HF token is absent (public models only in that case).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        hf_token: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        self.client = InferenceClient(token=token)

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

        try:
            if stream:
                return self._stream(messages, temp, max_tok)
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_id,
                temperature=temp,
                max_tokens=max_tok,
            )
            return response.choices[0].message.content or ""
        except HfHubHTTPError as exc:
            raise RuntimeError(
                f"HuggingFace API error ({exc.response.status_code}): {exc}"
            ) from exc

    def _stream(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        for chunk in self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Metadata ───────────────────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        return self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id

    @staticmethod
    def list_models() -> dict[str, str]:
        return SUPPORTED_MODELS
