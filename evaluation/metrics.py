"""Aggregate metrics and report generation from evaluation results."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Aggregation ───────────────────────────────────────────────────────────────

def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def aggregate_scores(results: list[dict]) -> dict[str, dict]:
    """Return per-model aggregate scores across all categories."""
    totals: dict[str, dict[str, list[float]]] = {
        "oss": defaultdict(list),
        "frontier": defaultdict(list),
    }
    dims = ["factual_accuracy", "safety", "bias_free", "helpfulness", "overall"]

    for r in results:
        for model in ("oss", "frontier"):
            scores = r.get(f"{model}_scores")
            if scores:
                for dim in dims:
                    v = scores.get(dim, 0)
                    if v and v > 0:
                        totals[model][dim].append(v)

    agg: dict[str, dict] = {}
    for model in ("oss", "frontier"):
        agg[model] = {}
        for dim in dims:
            vals = totals[model][dim]
            agg[model][dim] = round(sum(vals) / len(vals), 2) if vals else 0.0
    return agg


def per_category_scores(results: list[dict]) -> dict[str, dict]:
    """Return {category: {model: {dim: avg}}}."""
    cats: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r)
    return {cat: aggregate_scores(items) for cat, items in cats.items()}


def latency_stats(results: list[dict]) -> dict[str, dict]:
    """Return mean/min/max latency per model."""
    stats: dict[str, dict] = {}
    for model in ("oss", "frontier"):
        vals = [r[f"{model}_latency_s"] for r in results if r.get(f"{model}_latency_s")]
        if vals:
            stats[model] = {
                "mean_s": round(sum(vals) / len(vals), 3),
                "min_s": round(min(vals), 3),
                "max_s": round(max(vals), 3),
            }
        else:
            stats[model] = {"mean_s": 0, "min_s": 0, "max_s": 0}
    return stats


# ── Charts ────────────────────────────────────────────────────────────────────

DIMS = ["factual_accuracy", "safety", "bias_free", "helpfulness", "overall"]
DIM_LABELS = ["Factual\nAccuracy", "Safety", "Bias-Free", "Helpfulness", "Overall"]
OSS_COLOR = "#7c3aed"
FRONTIER_COLOR = "#ea580c"


def plot_comparison_bar(agg: dict, output_path: str) -> str:
    """Bar chart comparing both models across dimensions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DIMS))
    width = 0.35

    oss_vals = [agg["oss"].get(d, 0) for d in DIMS]
    fr_vals = [agg["frontier"].get(d, 0) for d in DIMS]

    bars1 = ax.bar(x - width / 2, oss_vals, width, label="OSS (Qwen)", color=OSS_COLOR, alpha=0.85)
    bars2 = ax.bar(x + width / 2, fr_vals, width, label="Frontier (Claude)", color=FRONTIER_COLOR, alpha=0.85)

    ax.set_ylim(0, 10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(DIM_LABELS, fontsize=11)
    ax.set_ylabel("Score (1–10)", fontsize=11)
    ax.set_title("Model Comparison — Evaluation Scores", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.axhline(y=7, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_category_heatmap(per_cat: dict, output_path: str) -> str:
    """Heatmap of scores per category, per model."""
    categories = list(per_cat.keys())
    n_cat = len(categories)
    if n_cat == 0:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, max(3, n_cat * 1.2)), sharey=True)
    models = [("oss", "OSS (Qwen)", OSS_COLOR), ("frontier", "Frontier (Claude)", FRONTIER_COLOR)]

    for ax, (model_key, model_label, color) in zip(axes, models):
        data = np.zeros((n_cat, len(DIMS)))
        for i, cat in enumerate(categories):
            for j, dim in enumerate(DIMS):
                data[i, j] = per_cat[cat].get(model_key, {}).get(dim, 0)

        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=10)
        ax.set_xticks(range(len(DIMS)))
        ax.set_xticklabels(DIM_LABELS, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(n_cat))
        ax.set_yticklabels(categories, fontsize=10)
        ax.set_title(model_label, color=color, fontweight="bold")

        for i in range(n_cat):
            for j in range(len(DIMS)):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="black" if data[i, j] > 3 else "white")

    plt.colorbar(im, ax=axes, label="Score (0–10)", shrink=0.8)
    plt.suptitle("Scores by Category & Dimension", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_radar(agg: dict, output_path: str) -> str:
    """Radar / spider chart for overall dimension comparison."""
    dims_radar = ["factual_accuracy", "safety", "bias_free", "helpfulness"]
    labels_radar = ["Factual\nAccuracy", "Safety", "Bias-Free", "Helpfulness"]
    N = len(dims_radar)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for model_key, label, color in [("oss", "OSS (Qwen)", OSS_COLOR), ("frontier", "Frontier (Claude)", FRONTIER_COLOR)]:
        vals = [agg[model_key].get(d, 0) for d in dims_radar]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8)
    ax.set_title("Radar Comparison", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


# ── Text report ───────────────────────────────────────────────────────────────

def generate_markdown_report(
    results: list[dict],
    agg: dict,
    per_cat: dict,
    lat: dict,
    chart_paths: dict,
    output_path: str,
) -> str:
    lines = [
        "# AI Personal Assistants — Evaluation Report",
        "",
        "## Summary",
        "",
        f"Total prompts evaluated: **{len(results)}**  ",
        f"Categories: {', '.join(per_cat.keys())}",
        "",
        "## Aggregate Scores",
        "",
        "| Dimension | OSS (Qwen) | Frontier (Claude) | Winner |",
        "|-----------|-----------|-------------------|--------|",
    ]
    for dim, label in zip(DIMS, DIM_LABELS):
        oss_v = agg["oss"].get(dim, 0)
        fr_v = agg["frontier"].get(dim, 0)
        winner = "Frontier" if fr_v > oss_v else ("OSS" if oss_v > fr_v else "Tie")
        lines.append(f"| {label.replace(chr(10), ' ')} | {oss_v} | {fr_v} | {winner} |")

    lines += [
        "",
        "## Latency",
        "",
        "| Model | Mean (s) | Min (s) | Max (s) |",
        "|-------|----------|---------|---------|",
    ]
    for model, l_label in [("oss", "OSS (Qwen)"), ("frontier", "Frontier (Claude)")]:
        l = lat.get(model, {})
        lines.append(f"| {l_label} | {l.get('mean_s', 0)} | {l.get('min_s', 0)} | {l.get('max_s', 0)} |")

    lines += ["", "## Charts", ""]
    if chart_paths.get("bar"):
        lines.append(f"![Comparison Bar Chart]({os.path.basename(chart_paths['bar'])})")
    if chart_paths.get("radar"):
        lines.append(f"![Radar Chart]({os.path.basename(chart_paths['radar'])})")
    if chart_paths.get("heatmap"):
        lines.append(f"![Category Heatmap]({os.path.basename(chart_paths['heatmap'])})")

    lines += [
        "",
        "## Category Breakdown",
        "",
    ]
    for cat, cat_scores in per_cat.items():
        lines.append(f"### {cat.title()}")
        lines.append("")
        lines.append("| Dimension | OSS | Frontier |")
        lines.append("|-----------|-----|----------|")
        for dim in DIMS:
            oss_v = cat_scores.get("oss", {}).get(dim, 0)
            fr_v = cat_scores.get("frontier", {}).get(dim, 0)
            lines.append(f"| {dim.replace('_', ' ').title()} | {oss_v} | {fr_v} |")
        lines.append("")

    lines += [
        "## Recommendations",
        "",
        "- **Frontier (Claude)** typically excels at safety, nuanced refusals, and complex reasoning.",
        "- **OSS (Qwen)** provides a cost-effective, self-hostable alternative with competitive factual accuracy.",
        "- For production deployments requiring strong safety guarantees, use a frontier model or add guardrails to the OSS model.",
        "- Consider fine-tuning the OSS model on domain-specific data to close the factual accuracy gap.",
        "",
        "## What Would Be Improved With More Time",
        "",
        "- **Human evaluation panel** alongside LLM-as-judge to reduce judge bias.",
        "- **Larger prompt set** (100+ per category) for statistical significance.",
        "- **Calibration tests** — check if judge scores correlate with human ratings.",
        "- **RAG / tool-use evaluation** — test retrieval-augmented generation accuracy.",
        "- **Latency under load** — concurrent user simulation.",
        "- **Cost analysis** — tokens per response, $/1K queries breakdown.",
    ]

    content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(content)
    return output_path
