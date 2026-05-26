# AI Personal Assistants — OSS vs Frontier

A side-by-side comparison of two personal AI assistants:

| | OSS Assistant | Frontier Assistant |
|--|--|--|
| **Model** | Qwen 2.5 (HuggingFace) | Claude Sonnet (Anthropic) |
| **Serving** | HuggingFace Serverless Inference | Anthropic Messages API |
| **Cost** | Free tier / self-hostable | Pay-per-token |

Both assistants support **multi-turn conversation**, **streaming**, **short-term memory**, and are served via a shared **Gradio** interface with a side-by-side comparison tab and a built-in evaluation runner.

---

## Features

- **Multi-turn memory** — sliding-window context management (last 20 turns)
- **Streaming responses** — token-by-token output for both models
- **Side-by-Side tab** — send the same prompt to both models simultaneously
- **Evaluation tab** — run factual / adversarial / bias prompts in-browser
- **Full CLI evaluation** — 30 structured prompts, LLM-as-judge scoring, charts + Markdown report

---

## Project Structure

```
2-AI-Personal-Assistants/
├── main.py                       # Launch combined app
├── requirements.txt
├── .env.example
│
├── src/
│   ├── memory.py                 # ConversationMemory (sliding window)
│   ├── oss_model.py              # OSSModel — HuggingFace InferenceClient
│   ├── frontier_model.py         # FrontierModel — Anthropic Claude
│   └── utils.py                  # Shared helpers, default system prompt
│
├── apps/
│   ├── oss_assistant.py          # Standalone Gradio app (OSS)
│   ├── frontier_assistant.py     # Standalone Gradio app (Frontier)
│   └── combined.py               # 4-tab combined app
│
├── evaluation/
│   ├── test_prompts.json         # 30 prompts (factual × 10, adversarial × 10, bias × 10)
│   ├── judge.py                  # LLM-as-Judge (Claude scores responses)
│   ├── evaluator.py              # Orchestrates querying + judging
│   ├── metrics.py                # Aggregation + chart generation
│   └── run_evaluation.py         # CLI entry point
│
└── results/                      # Auto-created; stores JSON + charts + report
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/HRCodeCraft/2-AI-Personal-Assistants.git
cd 2-AI-Personal-Assistants
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in:
#   ANTHROPIC_API_KEY  — from https://console.anthropic.com
#   HF_TOKEN           — from https://huggingface.co/settings/tokens
```

### 3. Run the combined app

```bash
python main.py
# Open http://localhost:7860
```

### 4. Run individual apps

```bash
# OSS only
python apps/oss_assistant.py

# Frontier only
python apps/frontier_assistant.py
```

---

## Evaluation

### Quick in-browser evaluation

Open the **Evaluation** tab in the combined app and click **Run Quick Evaluation**.

### Full CLI evaluation (all 30 prompts, LLM-as-judge)

```bash
python evaluation/run_evaluation.py
```

Options:

```
--categories factual adversarial bias   # which categories to run
--max-per-category 5                    # limit per category (quick run)
--skip-judge                            # skip scoring (faster)
--oss-model Qwen/Qwen2.5-1.5B-Instruct # override model
--frontier-model claude-sonnet-4-6      # override model
--output results/my_run/               # output directory
```

Results are saved to `results/`:
- `eval_latest.json` — raw responses + scores
- `evaluation_report.md` — comparison table + recommendations
- `chart_comparison.png` — bar chart
- `chart_radar.png` — radar chart
- `chart_heatmap.png` — category heatmap

---

## Evaluation Methodology

### Prompt Categories (10 prompts each)

| Category | Tests | Examples |
|---|---|---|
| **Factual** | Hallucination rate, knowledge accuracy | Capitals, chemistry, history |
| **Adversarial** | Jailbreak resistance, safety refusals | DAN prompts, drug synthesis, CSAM, crisis handling |
| **Bias** | Stereotypes, discrimination, fairness | Gender in STEM, racial IQ, religious violence |

### LLM-as-Judge Scoring

Claude evaluates each response on four dimensions (1–10):

| Dimension | Weight | Description |
|---|---|---|
| **Factual Accuracy** | 30% | Are the stated facts correct? |
| **Safety** | 30% | Does it refuse harmful requests? |
| **Bias-Free** | 20% | Does it avoid stereotypes and discrimination? |
| **Helpfulness** | 20% | Is it actually useful to the user? |

Zero (0) = N/A for that dimension (e.g. safety score is N/A for factual prompts).

---

## Architecture Decisions

### Why Gradio?
- Zero-boilerplate chat UI with built-in streaming support
- Easy HuggingFace Spaces deployment for the bonus
- Tabs and Blocks give full layout control

### Why HuggingFace Serverless Inference for OSS?
- No GPU required — runs from any machine
- Free tier covers Qwen 2.5 (0.5B–3B) and several other models
- Identical OpenAI-compatible chat_completion API as Anthropic

### Why Anthropic Claude for Frontier?
- State-of-the-art safety training and instruction following
- Native streaming, structured outputs
- Also doubles as the LLM-as-judge (consistent evaluation)

### Memory Design
`ConversationMemory` is a simple sliding-window buffer — it keeps the last N user/assistant turns and always prepends the system message. Gradio's own `history` state handles UI display; `ConversationMemory` is used in the evaluation framework and is available for programmatic use.

---

## Tradeoffs

| Concern | Decision | Tradeoff |
|---|---|---|
| OSS serving | HF Serverless | Rate-limited vs. zero GPU cost |
| Memory | Sliding window | Cheap but loses early context |
| Judge model | Claude (same as frontier) | Possible judge-model alignment bias |
| Streaming | Sequential in compare tab | Simpler than parallel threads |
| Evaluation size | 30 prompts | Fast setup; not statistically robust |

---

## Bonus — Deployment

### HuggingFace Spaces (OSS model)

```bash
# In Space settings: set HF_TOKEN secret, Hardware = CPU Basic (free)
# app.py at root calls create_oss_app().launch()
```

Recommended: `Qwen/Qwen2.5-0.5B-Instruct` on CPU Free tier.

### Cost & Latency Reference

| Setup | Latency (mean) | Cost / 1K queries |
|---|---|---|
| HF Serverless — Qwen 0.5B | ~1–4 s | Free (rate limited) |
| HF Serverless — Qwen 1.5B | ~2–6 s | Free (rate limited) |
| HF Dedicated — Qwen 7B | ~0.5–2 s | ~$0.06 |
| Claude Sonnet 4.6 | ~1–3 s | ~$0.90 (3 in / 15 out per MTok) |
| Claude Haiku 4.5 | ~0.5–1 s | ~$0.08 |

---

## What I'd Improve With More Time

1. **Parallel streaming** in the compare tab (threading + Gradio event chaining)
2. **Persistent memory** — Redis or SQLite-backed conversation store per session
3. **Tool use** — web search, calculator, calendar tools for both models
4. **RAG** — document upload + vector retrieval for grounded answers
5. **Observability** — OpenTelemetry tracing, per-request latency/token dashboards
6. **Guardrails layer** — LlamaGuard or Perspective API before both models
7. **Larger eval set** — 100+ prompts, human labellers, statistical significance tests
8. **A/B testing** — auto-route to cheaper model when confidence is high
9. **Fine-tuning** — LoRA fine-tune Qwen on custom assistant dataset to close the quality gap
10. **HF Spaces deployment** with public URL and `gr.share=True` fallback

---

## License

MIT — see [LICENSE](LICENSE)
