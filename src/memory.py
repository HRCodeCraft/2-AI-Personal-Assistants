"""Conversation memory with sliding-window context management."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """Thread-safe conversation buffer with configurable sliding window.

    Keeps the last `max_turns` user/assistant exchanges.  A system message is
    stored separately and always prepended when exporting messages so it never
    counts against the turn budget.
    """

    def __init__(self, max_turns: int = 20, system_message: str = "") -> None:
        self._messages: list[Message] = []
        self.max_turns = max_turns
        self.system_message = system_message

    # ── Mutation ───────────────────────────────────────────────────────────────

    def add_user_message(self, content: str) -> None:
        self._messages.append(Message(role="user", content=content))
        self._evict()

    def add_assistant_message(self, content: str) -> None:
        self._messages.append(Message(role="assistant", content=content))
        self._evict()

    def set_system_message(self, message: str) -> None:
        self.system_message = message

    def clear(self) -> None:
        self._messages.clear()

    def _evict(self) -> None:
        """Drop oldest messages once we exceed the turn budget."""
        limit = self.max_turns * 2  # each turn = 1 user + 1 assistant
        if len(self._messages) > limit:
            self._messages = self._messages[-limit:]

    # ── Export ─────────────────────────────────────────────────────────────────

    def get_messages(self, include_system: bool = True) -> list[dict]:
        """Return the full message list in OpenAI / Anthropic chat format."""
        out: list[dict] = []
        if include_system and self.system_message:
            out.append({"role": "system", "content": self.system_message})
        out.extend(m.to_dict() for m in self._messages)
        return out

    def get_gradio_history(self) -> list[list[Optional[str]]]:
        """Return conversation as Gradio ChatInterface-compatible history.

        Each element is [user_text, assistant_text].  If the last message is
        from the user (no reply yet), the assistant slot is ``None``.
        """
        history: list[list[Optional[str]]] = []
        msgs = self._messages
        i = 0
        while i < len(msgs):
            if msgs[i].role == "user":
                user_text = msgs[i].content
                assistant_text: Optional[str] = None
                if i + 1 < len(msgs) and msgs[i + 1].role == "assistant":
                    assistant_text = msgs[i + 1].content
                    i += 2
                else:
                    i += 1
                history.append([user_text, assistant_text])
            else:
                i += 1
        return history

    def to_json(self) -> str:
        return json.dumps(
            {
                "system_message": self.system_message,
                "messages": [m.to_dict() for m in self._messages],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, data: str) -> "ConversationMemory":
        obj = json.loads(data)
        mem = cls(system_message=obj.get("system_message", ""))
        for m in obj.get("messages", []):
            mem._messages.append(Message(role=m["role"], content=m["content"]))
        return mem

    def __len__(self) -> int:
        return len(self._messages)
