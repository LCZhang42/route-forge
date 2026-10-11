"""Evaluate the complete Phase 3 generation/ranking service on frozen prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
import time
from typing import Any, Sequence

from backend.climb_path.structured_inference import Phase3RouteService
from backend.evaluation.evaluate_structured_model import hold_f1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(
    *,
    service: Phase3RouteService,
    prompts: Sequence[dict[str, Any]],
    nearest_metrics: dict[str, Any],
    candidate_count: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, prompt in enumerate(prompts):
        prompt_id = prompt.get("prompt_id", prompt.get("route_id", f"row_{index}"))
        reference_middle = prompt.get(
            "reference_middle_holds", prompt.get("middle_holds", [])
        )
        started = time.perf_counter()
        try:
            result = service.generate(
                grade=prompt["grade"],
                start_holds=prompt["start_holds"],
                end_holds=prompt["end_holds"],
                candidate_count=candidate_count,
                temperature=0.8,
                top_p=0.95,
                seed=10_000 + index,
            )
        except Exception as exc:
            failures.append({"prompt_id": prompt_id, "error": type(exc).__name__})
            continue
        latency_ms = (time.perf_counter() - started) * 1000
        retrieval = service.retrieval.predict(
            grade=prompt["grade"],
            start_holds=prompt["start_holds"],
            end_holds=prompt["end_holds"],
        )
        retrieval_middle = list(retrieval.middle_holds)
        candidates = []
        for candidate in result.candidates:
            middle = [hold.label for hold in candidate.route.middle_holds]
            candidates.append(
                {
                    "source": candidate.source,
                    "score": candidate.score,
                    "model_log_probability": candidate.model_log_probability,
                    "service_retrieval_similarity": candidate.retrieval_similarity,
                    "middle_holds": middle,
                    "middle_hold_f1": hold_f1(middle, reference_middle),
                    "retrieval_similarity": hold_f1(middle, retrieval_middle),
                    "valid": candidate.diagnostics.valid,
                    "novel_route": candidate.diagnostics.novel_route,
                    "length_delta": candidate.diagnostics.length_delta,
                    "maximum_move_distance": candidate.diagnostics.maximum_move_distance,
                    "reachability_violations": candidate.diagnostics.reachability_violations,
                    "downward_moves": candidate.diagnostics.downward_moves,
                }
            )
        for candidate in candidates:
            peers = [item for item in candidates if item is not candidate]
            candidate["consensus"] = (
                fmean(
                    hold_f1(candidate["middle_holds"], peer["middle_holds"])
                    for peer in peers
                )
                if peers
                else 0.0
            )
        predictions.append(
            {
                "prompt_id": prompt_id,
                "grade": prompt["grade"],
                "latency_ms": latency_ms,
                "used_fallback": result.used_fallback,
                "rejected_attempts": result.rejected_attempts,
                "retrieval_middle_holds": retrieval_middle,
                "retrieval_middle_hold_f1": hold_f1(
                    retrieval_middle, reference_middle
                ),
                "candidates": candidates,
            }
        )

    top_candidates = [item["candidates"][0] for item in predictions]
    all_candidates = [
        candidate for item in predictions for candidate in item["candidates"]
    ]
    mean_f1_by_rank = {
        str(rank + 1): fmean(
            item["candidates"][rank]["middle_hold_f1"]
            for item in predictions
            if len(item["candidates"]) > rank
        )
        for rank in range(candidate_count)
        if any(len(item["candidates"]) > rank for item in predictions)
    }

    def strategy_mean(selector) -> float:
        return fmean(selector(item) for item in predictions) if predictions else 0.0

    def model_candidates(item):
        return [
            candidate
            for candidate in item["candidates"]
            if candidate["source"] == "model"
        ]

    def select_model(item, key):
        candidates = model_candidates(item)
        if not candidates:
            return item["retrieval_middle_hold_f1"]
        return max(candidates, key=key)["middle_hold_f1"]

    measured_retrieval_f1 = strategy_mean(
        lambda item: item["retrieval_middle_hold_f1"]
    )
    strategy_f1 = {
        "current_ranker": strategy_mean(
            lambda item: item["candidates"][0]["middle_hold_f1"]
        ),
        "retrieval": measured_retrieval_f1,
        "model_max_retrieval_similarity": strategy_mean(
            lambda item: select_model(
                item,
                lambda candidate: (
                    candidate["retrieval_similarity"],
                    candidate["score"],
                ),
            )
        ),
        "model_max_consensus": strategy_mean(
            lambda item: select_model(
                item,
                lambda candidate: (candidate["consensus"], candidate["score"]),
            )
        ),
        "model_similarity_plus_consensus": strategy_mean(
            lambda item: select_model(
                item,
                lambda candidate: (
                    candidate["retrieval_similarity"]
                    + 0.25 * candidate["consensus"],
                    candidate["score"],
                ),
            )
        ),
    }

    metrics = {
        "prompts": len(prompts),
        "successful_prompts": len(predictions),
        "request_success_rate": len(predictions) / len(prompts),
        "mean_candidates_returned": fmean(
            len(item["candidates"]) for item in predictions
        )
        if predictions
        else 0.0,
        "top1_mean_middle_hold_f1": fmean(
            candidate["middle_hold_f1"] for candidate in top_candidates
        )
        if top_candidates
        else 0.0,
        "oracle_mean_middle_hold_f1": fmean(
            max(candidate["middle_hold_f1"] for candidate in item["candidates"])
            for item in predictions
        )
        if predictions
        else 0.0,
        "valid_candidate_rate": fmean(
            float(candidate["valid"]) for candidate in all_candidates
        )
        if all_candidates
        else 0.0,
        "novel_candidate_rate": fmean(
            float(candidate["novel_route"]) for candidate in all_candidates
        )
        if all_candidates
        else 0.0,
        "fallback_request_rate": fmean(
            float(item["used_fallback"]) for item in predictions
        )
        if predictions
        else 0.0,
        "mean_latency_ms": fmean(item["latency_ms"] for item in predictions)
        if predictions
        else 0.0,
        "nearest_route_baseline": nearest_metrics,
        "measured_retrieval_mean_middle_hold_f1": measured_retrieval_f1,
        "mean_f1_by_candidate_rank": mean_f1_by_rank,
        "ranking_strategy_mean_f1": strategy_f1,
        "top1_f1_delta_vs_nearest": (
            fmean(candidate["middle_hold_f1"] for candidate in top_candidates)
            - measured_retrieval_f1
        )
        if top_candidates
        else -measured_retrieval_f1,
        "failures": failures,
    }
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--board-holds", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--nearest-metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--model-version", default="structured-v1")
    args = parser.parse_args()

    service = Phase3RouteService.from_paths(
        model_dir=args.model_dir,
        train_path=args.train,
        board_holds_path=args.board_holds,
        manifest_path=args.manifest,
        model_version=args.model_version,
    )
    predictions, metrics = evaluate(
        service=service,
        prompts=_read_jsonl(args.prompts),
        nearest_metrics=json.loads(args.nearest_metrics.read_text(encoding="utf-8")),
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text(
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in predictions
        ),
        encoding="utf-8",
        newline="\n",
    )
    args.metrics.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
