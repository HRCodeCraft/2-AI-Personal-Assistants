"""Standalone Gradio app — Open-Source Assistant (HuggingFace / Qwen)."""

from __future__ import annotations

import os
import sys
from typing import Iterator

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.oss_model import OSSModel, SUPPORTED_MODELS, DEFAULT_MODEL
from src.utils import gradio_history_to_messages, format_error, DEFAULT_SYSTEM_PROMPT

# ── Shared model instance (reused across Gradio sessions) ─────────────────────
_model: OSSModel | None = None


def _get_model(model_id: str) -> OSSModel:
    global _model
    if _model is None or _model.model_id != model_id:
        _model = OSSModel(model_id=model_id)
    return _model


# ── Core chat function ─────────────────────────────────────────────────────────

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
        yield format_error(exc, "OSS Model")


# ── Gradio layout ──────────────────────────────────────────────────────────────

def create_oss_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="purple",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, title="OSS Assistant — Qwen / HuggingFace") as demo:
        gr.Markdown(
            """
# 🤖 Open-Source Assistant
**Powered by Qwen 2.5 via HuggingFace Serverless Inference**

Multi-turn conversation · Streaming · Context-aware
"""
        )

        with gr.Accordion("⚙️ Settings", open=False):
            model_dropdown = gr.Dropdown(
                choices=list(SUPPORTED_MODELS.keys()),
                value=list(SUPPORTED_MODELS.keys())[0],
                label="Model",
                info="Select open-source model (all served via HuggingFace Inference API)",
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
                    minimum=64, maximum=2048, value=1024, step=64,
                    label="Max Tokens"
                )

        chatbot = gr.Chatbot(
            label="Conversation",
            height=520,
            show_copy_button=True,
            avatar_images=(None, "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"),
            bubble_full_width=False,
        )

        gr.ChatInterface(
            fn=respond,
            chatbot=chatbot,
            additional_inputs=[system_prompt, model_dropdown, temperature, max_tokens],
            additional_inputs_accordion=gr.Accordion(visible=False),
            submit_btn="Send ↵",
            retry_btn="↺ Retry",
            undo_btn="↩ Undo",
            clear_btn="🗑 Clear",
            examples=[
                "What is the capital of Australia?",
                "Explain quantum entanglement in simple terms.",
                "Write a short poem about the ocean.",
                "What are the pros and cons of renewable energy?",
            ],
        )

        gr.Markdown(
            "_Model served via [HuggingFace Serverless Inference](https://huggingface.co/docs/api-inference). "
            "Set `HF_TOKEN` in `.env` for higher rate limits._"
        )

    return demo


if __name__ == "__main__":
    app = create_oss_app()
    app.queue()
    app.launch(
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
    )
