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
    history,
    system_prompt: str = "",
) -> list[dict]:
    """Convert Gradio history to OpenAI-style message list.

    Handles both Gradio 6 format (list of {role, content} dicts)
    and old Gradio format (list of [user, assistant] pairs).
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for item in history:
        if isinstance(item, dict):
            # Gradio 6 new format: {"role": "user"|"assistant", "content": "..."}
            role = item.get("role", "")
            content = item.get("content") or ""
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        else:
            # Legacy format: [user_msg, assistant_msg]
            user_msg = item[0] if item[0] is not None else ""
            asst_msg = item[1] if item[1] is not None else ""
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
