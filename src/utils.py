"""Shared utility helpers."""

from __future__ import annotations

import os
from typing import Optional


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI personal assistant. "
    "Provide accurate, thoughtful responses. Acknowledge uncertainty when unsure. "
    "Decline politely to assist with harmful, illegal, or unethical requests. "
    "Treat every person with equal respect regardless of background."
)


def gradio_history_to_messages(
    history: list[list[Optional[str]]],
    system_prompt: str = "",
) -> list[dict]:
    """Convert Gradio ChatInterface history to OpenAI-style message list.

    Args:
        history: List of [user_text, assistant_text] pairs from Gradio.
        system_prompt: Optional system message to prepend.
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for pair in history:
        user_msg = pair[0] if pair[0] is not None else ""
        asst_msg = pair[1] if pair[1] is not None else ""
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if asst_msg:
            messages.append({"role": "assistant", "content": asst_msg})
    return messages


def format_error(exc: Exception, context: str = "") -> str:
    """Return a user-friendly error string from an exception."""
    prefix = f"[{context}] " if context else ""
    return f"⚠️ {prefix}Error: {type(exc).__name__}: {exc}"


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)
