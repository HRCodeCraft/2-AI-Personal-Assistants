"""Standalone Gradio app — Frontier Assistant (Anthropic Claude)."""

from __future__ import annotations

import os
import sys
from typing import Iterator

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.frontier_model import FrontierModel, SUPPORTED_MODELS, DEFAULT_MODEL
from src.utils import gradio_history_to_messages, format_error, DEFAULT_SYSTEM_PROMPT

# ── Shared model instance ─────────────────────────────────────────────────────
_model: FrontierModel | None = None


def _get_model(model_id: str) -> FrontierModel:
    global _model
    if _model is None or _model.model_id != model_id:
        _model = FrontierModel(model_id=model_id)
    return _model


# ── Core chat function ────────────────────────────────────────────────────────

def respond(
    message: str,
    history: list[list[str | None]],
    system_prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Iterator[str]:
    """Stream a reply; yield the accumulated string on each chunk."""
    model_id = SUPPORTED_MODELS.get(model_name, DEFAULT_MODEL)
    model = _get_model(model_id)

    messages = gradio_history_to_messages(history, system_prompt)
    messages.append({"role": "user", "content": message})

    try:
        accumulated = ""
        for chunk in model.chat(messages, temperature=temperature, max_tokens=max_tokens, stream=True):
            accumulated += chunk
            yield accumulated
    except Exception as exc:
        yield format_error(exc, "Frontier Model")


# ── Gradio layout ─────────────────────────────────────────────────────────────

_THEME = gr.themes.Soft(primary_hue="orange", secondary_hue="amber", neutral_hue="slate")


def create_frontier_app() -> gr.Blocks:
    with gr.Blocks(title="Frontier Assistant — Claude") as demo:
        gr.Markdown(
            """
# 🧠 Frontier Assistant
**Powered by Gemini (Google) — State-of-the-Art Language Model**

Multi-turn conversation · Streaming · Context-aware
"""
        )

        with gr.Accordion("⚙️ Settings", open=False):
            model_dropdown = gr.Dropdown(
                choices=list(SUPPORTED_MODELS.keys()),
                value=list(SUPPORTED_MODELS.keys())[0],
                label="Model",
                info="Select Gemini model variant",
            )
            system_prompt = gr.Textbox(
                value=DEFAULT_SYSTEM_PROMPT,
                label="System Prompt",
                lines=3,
                placeholder="Instructions that shape the assistant's personality…",
            )
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.7, step=0.05,
                    label="Temperature", info="Higher → more creative"
                )
                max_tokens = gr.Slider(
                    minimum=64, maximum=4096, value=1024, step=64,
                    label="Max Tokens"
                )

        chatbot = gr.Chatbot(
            label="Conversation",
            height=520,
            layout="bubble",
            buttons=["copy"],
        )

        gr.ChatInterface(
            fn=respond,
            chatbot=chatbot,
            additional_inputs=[system_prompt, model_dropdown, temperature, max_tokens],
            additional_inputs_accordion=gr.Accordion(visible=False),
            submit_btn="Send ↵",
            examples=[
                ["What is the capital of Australia?"],
                ["Explain quantum entanglement in simple terms."],
                ["Write a short poem about the ocean."],
                ["What are the pros and cons of renewable energy?"],
            ],
        )

        gr.Markdown(
            "_Powered by [Anthropic Claude](https://anthropic.com). "
            "Set `GOOGLE_API_KEY` in `.env` to use this assistant. Free key at [aistudio.google.com](https://aistudio.google.com/apikey)._"
        )

    return demo


if __name__ == "__main__":
    app = create_frontier_app()
    app.queue()
    app.launch(
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7861")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        theme=_THEME,
    )
