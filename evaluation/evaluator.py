"""Core evaluation runner: queries both models, scores with judge, saves results."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.oss_model import OSSModel
from src.frontier_model import FrontierModel
from evaluation.judge import LLMJudge, JudgeScores


@dataclass
class EvaluationResult:
    prompt_id: str
    category: str
    prompt: str
    oss_response: str
    frontier_response: str
    oss_scores: Optional[JudgeScores] = None
    frontier_scores: Optional[JudgeScores] = None
    oss_latency_s: float = 0.0
    frontier_latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "category": self.category,
            "prompt": self.prompt,
            "oss_response": self.oss_response,
            "frontier_response": self.frontier_response,
            "oss_scores": self.oss_scores.to_dict() if self.oss_scores else None,
            "frontier_scores": self.frontier_scores.to_dict() if self.frontier_scores else None,
            "oss_latency_s": round(self.oss_latency_s, 3),
            "frontier_latency_s": round(self.frontier_latency_s, 3),
        }


class Evaluator:
    """Orchestrates model querying, judging, and result persistence."""

    def __init__(
        self,
        oss_model: OSSModel,
        frontier_model: FrontierModel,
        judge: LLMJudge,
        prompts_path: str | None = None,
        output_dir: str = "results",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> None:
        self.oss = oss_model
        self.frontier = frontier_model
        self.judge = judge
        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        os.makedirs(output_dir, exist_ok=True)

        if prompts_path is None:
            prompts_path = os.path.join(os.path.dirname(__file__), "test_prompts.json")
        with open(prompts_path) as f:
            self.prompts_db: dict = json.load(f)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        categories: list[str] | None = None,
        max_per_category: int | None = None,
        skip_judge: bool = False,
    ) -> list[EvaluationResult]:
        """Run evaluation over the specified categories.

        Args:
            categories: Which categories to include (default: all).
            max_per_category: Limit prompts per category (useful for quick runs).
            skip_judge: If True, skip LLM judging (faster, no scores).
        """
        cats = categories or list(self.prompts_db.keys())
        results: list[EvaluationResult] = []

        for cat in cats:
            prompts = self.prompts_db.get(cat, [])
            if max_per_category:
                prompts = prompts[:max_per_category]

            print(f"\n{'═' * 60}")
            print(f"  Category: {cat.upper()}  ({len(prompts)} prompts)")
            print(f"{'═' * 60}")

            for item in tqdm(prompts, desc=cat):
                result = self._evaluate_single(item, cat, skip_judge)
                results.append(result)

        self._save_results(results)
        return results

    # ── Private ───────────────────────────────────────────────────────────────

    def _query_model(self, model, messages: list[dict]) -> tuple[str, float]:
        """Return (response_text, latency_seconds)."""
        t0 = time.perf_counter()
        try:
            text = ""
            for chunk in model.chat(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            ):
                text += chunk
            return text, time.perf_counter() - t0
        except Exception as exc:
            return f"[ERROR] {type(exc).__name__}: {exc}", time.perf_counter() - t0

    def _evaluate_single(self, item: dict, category: str, skip_judge: bool) -> EvaluationResult:
        prompt_id = item.get("id", "unknown")
        prompt = item["prompt"]
        messages = [{"role": "user", "content": prompt}]

        oss_resp, oss_lat = self._query_model(self.oss, messages)
        fr_resp, fr_lat = self._query_model(self.frontier, messages)

        result = EvaluationResult(
            prompt_id=prompt_id,
            category=category,
            prompt=prompt,
            oss_response=oss_resp,
            frontier_response=fr_resp,
            oss_latency_s=oss_lat,
            frontier_latency_s=fr_lat,
        )

        if not skip_judge:
            try:
                result.oss_scores = self.judge.evaluate(
                    prompt, oss_resp, category, prompt_id, "oss"
                )
            except Exception as exc:
                print(f"  ⚠ Judge failed for OSS ({prompt_id}): {exc}")

            try:
                result.frontier_scores = self.judge.evaluate(
                    prompt, fr_resp, category, prompt_id, "frontier"
                )
            except Exception as exc:
                print(f"  ⚠ Judge failed for Frontier ({prompt_id}): {exc}")

        return result

    def _save_results(self, results: list[EvaluationResult]) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"eval_{timestamp}.json")
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\n✅ Results saved → {path}")
        # Also write a stable 'latest' alias
        latest = os.path.join(self.output_dir, "eval_latest.json")
        with open(latest, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
