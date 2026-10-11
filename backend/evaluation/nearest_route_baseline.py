"""Deterministic nearest-route retrieval baseline for the frozen evaluation set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

from backend.climb_path.schema import Hold, SUPPORTED_GRADES


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _hold(value: str) -> Hold:
    return Hold.from_label(value)


def _set_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Symmetric nearest-neighbour Manhattan distance between endpoint sets."""

    left_holds = [_hold(value) for value in left]
    right_holds = [_hold(value) for value in right]

    def directed(source: Iterable[Hold], target: Sequence[Hold]) -> int:
        return sum(
            min(abs(item.x - other.x) + abs(item.y - other.y) for other in target)
            for item in source
        )

    return directed(left_holds, right_holds) + directed(right_holds, left_holds)


def _grade_distance(left: str, right: str) -> int:
    positions = {grade: index for index, grade in enumerate(SUPPORTED_GRADES)}
    return abs(positions[left] - positions[right])


@dataclass(frozen=True)
class BaselinePrediction:
    source_route_id: str
    grade_distance: int
    endpoint_distance: int
    middle_holds: tuple[str, ...]


class NearestRouteBaseline:
    def __init__(self, training_records: Sequence[dict[str, Any]]) -> None:
        if not training_records:
            raise ValueError("training_records cannot be empty")
        self.training_records = tuple(training_records)

    @classmethod
    def from_jsonl(cls, path: Path) -> "NearestRouteBaseline":
        return cls(_read_jsonl(path))

    def predict(
        self,
        *,
        grade: str,
        start_holds: Sequence[str],
        end_holds: Sequence[str],
    ) -> BaselinePrediction:
        if grade not in SUPPORTED_GRADES:
            raise ValueError(f"Unsupported grade: {grade}")

        def score(record: dict[str, Any]) -> tuple[int, int, str]:
            return (
                _grade_distance(grade, record["grade"]),
                _set_distance(start_holds, record["start_holds"])
                + _set_distance(end_holds, record["end_holds"]),
                record["route_id"],
            )

        nearest = min(self.training_records, key=score)
        grade_distance, endpoint_distance, _ = score(nearest)
        return BaselinePrediction(
            source_route_id=nearest["route_id"],
            grade_distance=grade_distance,
            endpoint_distance=endpoint_distance,
            middle_holds=tuple(nearest["middle_holds"]),
        )


def _hold_f1(predicted: Sequence[str], reference: Sequence[str]) -> float:
    predicted_set = set(predicted)
    reference_set = set(reference)
    if not predicted_set and not reference_set:
        return 1.0
    if not predicted_set or not reference_set:
        return 0.0
    overlap = len(predicted_set & reference_set)
    precision = overlap / len(predicted_set)
    recall = overlap / len(reference_set)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def evaluate(
    baseline: NearestRouteBaseline,
    prompts: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not prompts:
        raise ValueError("prompts cannot be empty")
    predictions: list[dict[str, Any]] = []
    for prompt in prompts:
        prediction = baseline.predict(
            grade=prompt["grade"],
            start_holds=prompt["start_holds"],
            end_holds=prompt["end_holds"],
        )
        predictions.append(
            {
                "prompt_id": prompt["prompt_id"],
                "source_route_id": prediction.source_route_id,
                "grade_distance": prediction.grade_distance,
                "endpoint_distance": prediction.endpoint_distance,
                "predicted_middle_holds": list(prediction.middle_holds),
                "middle_hold_f1": _hold_f1(
                    prediction.middle_holds, prompt["reference_middle_holds"]
                ),
                "exact_middle_match": list(prediction.middle_holds)
                == prompt["reference_middle_holds"],
            }
        )

    summary = {
        "prompts": len(predictions),
        "exact_middle_match_rate": fmean(
            float(item["exact_middle_match"]) for item in predictions
        ),
        "mean_middle_hold_f1": fmean(item["middle_hold_f1"] for item in predictions),
        "mean_endpoint_distance": fmean(item["endpoint_distance"] for item in predictions),
        "same_grade_rate": fmean(float(item["grade_distance"] == 0) for item in predictions),
    }
    return predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data_contract/v1/train.jsonl"))
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("data_contract/v1/frozen_evaluation.jsonl"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data_contract/v1/nearest_route_predictions.jsonl"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("data_contract/v1/nearest_route_metrics.json"),
    )
    args = parser.parse_args()

    baseline = NearestRouteBaseline.from_jsonl(args.train)
    predictions, summary = evaluate(baseline, _read_jsonl(args.evaluation))
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
        for item in predictions
    )
    args.predictions.write_text(payload, encoding="utf-8", newline="\n")
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
