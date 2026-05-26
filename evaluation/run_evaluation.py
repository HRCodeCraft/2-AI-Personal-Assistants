"""CLI entry point for running the full evaluation pipeline.

Usage:
  python evaluation/run_evaluation.py
  python evaluation/run_evaluation.py --categories factual adversarial --max-per-category 5
  python evaluation/run_evaluation.py --skip-judge --output results/dry_run/
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from src.oss_model import OSSModel
from src.frontier_model import FrontierModel
from evaluation.judge import LLMJudge
from evaluation.evaluator import Evaluator
from evaluation.metrics import (
    aggregate_scores,
    per_category_scores,
    latency_stats,
    plot_comparison_bar,
    plot_category_heatmap,
    plot_radar,
    generate_markdown_report,
    load_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI assistant evaluation")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["factual", "adversarial", "bias"],
        choices=["factual", "adversarial", "bias"],
        help="Which prompt categories to include",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Max prompts per category (None = all)",
    )
    parser.add_argument(
        "--oss-model",
        default=os.getenv("OSS_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
        help="HuggingFace model ID for OSS assistant",
    )
    parser.add_argument(
        "--frontier-model",
        default=os.getenv("FRONTIER_MODEL_ID", "claude-sonnet-4-6"),
        help="Anthropic model ID for frontier assistant",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for both models",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per response",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring (faster, no scores)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Directory to save results and charts",
    )
    parser.add_argument(
        "--from-file",
        default=None,
        help="Skip inference, generate report from existing results JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 65)
    print("  AI PERSONAL ASSISTANTS — EVALUATION PIPELINE")
    print("=" * 65)
    print(f"  OSS model      : {args.oss_model}")
    print(f"  Frontier model : {args.frontier_model}")
    print(f"  Categories     : {args.categories}")
    print(f"  Skip judge     : {args.skip_judge}")
    print(f"  Output dir     : {args.output}")
    print("=" * 65)

    if args.from_file:
        print(f"\nLoading results from {args.from_file}…")
        results = load_results(args.from_file)
    else:
        oss = OSSModel(model_id=args.oss_model, temperature=args.temperature, max_tokens=args.max_tokens)
        frontier = FrontierModel(model_id=args.frontier_model, temperature=args.temperature, max_tokens=args.max_tokens)
        judge = LLMJudge() if not args.skip_judge else None

        evaluator = Evaluator(
            oss_model=oss,
            frontier_model=frontier,
            judge=judge,
            output_dir=args.output,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        result_objs = evaluator.run(
            categories=args.categories,
            max_per_category=args.max_per_category,
            skip_judge=args.skip_judge,
        )
        results = [r.to_dict() for r in result_objs]

    # ── Generate report ────────────────────────────────────────────────────────
    if not args.skip_judge and any(r.get("oss_scores") for r in results):
        print("\nGenerating report and charts…")
        agg = aggregate_scores(results)
        per_cat = per_category_scores(results)
        lat = latency_stats(results)

        charts: dict = {}
        charts["bar"] = plot_comparison_bar(agg, os.path.join(args.output, "chart_comparison.png"))
        charts["radar"] = plot_radar(agg, os.path.join(args.output, "chart_radar.png"))
        if len(per_cat) > 1:
            charts["heatmap"] = plot_category_heatmap(per_cat, os.path.join(args.output, "chart_heatmap.png"))

        report_path = generate_markdown_report(
            results=results,
            agg=agg,
            per_cat=per_cat,
            lat=lat,
            chart_paths=charts,
            output_path=os.path.join(args.output, "evaluation_report.md"),
        )
        print(f"📊 Report saved → {report_path}")
        print(f"📈 Charts saved → {args.output}/")

        # Print summary table
        print("\n── SUMMARY ──────────────────────────────────────────────")
        print(f"{'Dimension':<22} {'OSS':>8} {'Frontier':>10}")
        print("─" * 44)
        for dim in ["factual_accuracy", "safety", "bias_free", "helpfulness", "overall"]:
            oss_v = agg["oss"].get(dim, 0)
            fr_v = agg["frontier"].get(dim, 0)
            winner = " ←" if fr_v > oss_v else (" ←" if oss_v > fr_v else "")
            fr_str = f"{fr_v:.2f}{winner if fr_v >= oss_v else ''}"
            oss_str = f"{oss_v:.2f}{winner if oss_v > fr_v else ''}"
            print(f"{dim.replace('_', ' ').title():<22} {oss_str:>8} {fr_str:>10}")
        print("─" * 44)
        lat_oss = lat.get("oss", {}).get("mean_s", 0)
        lat_fr = lat.get("frontier", {}).get("mean_s", 0)
        print(f"{'Mean Latency (s)':<22} {lat_oss:>8.3f} {lat_fr:>10.3f}")
    else:
        print("\n(Skipped judge scoring — no report generated)")

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
