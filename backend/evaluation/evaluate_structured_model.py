"""Evaluate a Phase 2 structured model on the frozen v1 prompt set."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import fmean, median
import time
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.climb_path.tokens import EOS_TOKEN, hold_label, hold_token, prompt_tokens


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def route_fingerprint(
    start_holds: Sequence[str],
    middle_holds: Sequence[str],
    end_holds: Sequence[str],
) -> str:
    payload = {
        "start_holds": sorted(start_holds),
        "middle_holds": list(middle_holds),
        "end_holds": sorted(end_holds),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hold_f1(predicted: Sequence[str], reference: Sequence[str]) -> float:
    predicted_set = set(predicted)
    reference_set = set(reference)
    if not predicted_set and not reference_set:
        return 1.0
    if not predicted_set or not reference_set:
        return 0.0
    overlap = len(predicted_set & reference_set)
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_set)
    recall = overlap / len(reference_set)
    return 2 * precision * recall / (precision + recall)


def allowed_completion_token_ids(
    current_ids: Sequence[int],
    hold_ids: set[int],
    eos_id: int,
    *,
    prompt_length: int = 0,
    minimum_middle_holds: int = 0,
) -> list[int]:
    """Allow unused holds and EOS after the minimum route length is reached."""

    used_hold_ids = set(current_ids) & hold_ids
    allowed_ids = hold_ids - used_hold_ids
    generated_count = max(0, len(current_ids) - prompt_length)
    if generated_count >= minimum_middle_holds:
        allowed_ids.add(eos_id)
    return sorted(allowed_ids)


def evaluate(
    *,
    model_dir: Path,
    prompts_path: Path,
    train_path: Path,
    board_holds_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    nearest_metrics_path: Path | None = None,
    max_new_tokens: int = 24,
) -> dict[str, Any]:
    prompts = read_jsonl(prompts_path)
    train_records = read_jsonl(train_path)
    train_fingerprints = {record["fingerprint"] for record in train_records}
    grade_minimum_holds = {
        grade: int(median(lengths))
        for grade in sorted({record["grade"] for record in train_records})
        for lengths in [
            [
                len(record["middle_holds"])
                for record in train_records
                if record["grade"] == grade
            ]
        ]
    }
    board_holds = json.loads(board_holds_path.read_text(encoding="utf-8"))["holds"]
    board_hold_set = set(board_holds)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    eos_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    hold_ids = {
        tokenizer.convert_tokens_to_ids(hold_token(label)): label for label in board_holds
    }
    all_hold_ids = set(hold_ids)
    if len(hold_ids) != len(board_holds) or eos_id == tokenizer.unk_token_id:
        raise ValueError("Saved tokenizer does not contain the Phase 2 atomic vocabulary")

    predictions: list[dict[str, Any]] = []
    per_grade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        prompt_token_values = prompt_tokens(
            grade=prompt["grade"],
            start_holds=prompt["start_holds"],
            end_holds=prompt["end_holds"],
        )
        prompt_ids = [tokenizer.convert_tokens_to_ids(token) for token in prompt_token_values]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        def allowed_tokens(_batch_id, current_ids):
            return allowed_completion_token_ids(
                current_ids.tolist(),
                all_hold_ids,
                eos_id,
                prompt_length=len(prompt_ids),
                minimum_middle_holds=grade_minimum_holds[prompt["grade"]],
            )

        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
                pad_token_id=tokenizer.pad_token_id,
                prefix_allowed_tokens_fn=allowed_tokens,
            )[0]
        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000

        generated_ids = output[len(prompt_ids) :].tolist()
        saw_eos = eos_id in generated_ids
        if saw_eos:
            generated_ids = generated_ids[: generated_ids.index(eos_id)]
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        middle_holds = [label for token in generated_tokens if (label := hold_label(token))]
        all_generated_tokens_valid = len(middle_holds) == len(generated_ids)
        no_duplicates = len(set(middle_holds)) == len(middle_holds)
        no_role_overlap = not (
            set(middle_holds) & (set(prompt["start_holds"]) | set(prompt["end_holds"]))
        )
        fingerprint = route_fingerprint(
            prompt["start_holds"], middle_holds, prompt["end_holds"]
        )
        prediction = {
            "prompt_id": prompt["prompt_id"],
            "grade": prompt["grade"],
            "start_holds": prompt["start_holds"],
            "predicted_middle_holds": middle_holds,
            "end_holds": prompt["end_holds"],
            "reference_middle_holds": prompt["reference_middle_holds"],
            "middle_hold_f1": hold_f1(
                middle_holds, prompt["reference_middle_holds"]
            ),
            "exact_middle_match": middle_holds == prompt["reference_middle_holds"],
            "valid_hold_rate": (
                fmean(float(label in board_hold_set) for label in middle_holds)
                if middle_holds
                else 1.0
            ),
            "parseable": saw_eos and all_generated_tokens_valid,
            "duplicate_free": no_duplicates and no_role_overlap,
            "novel_route": fingerprint not in train_fingerprints,
            "latency_ms": latency_ms,
        }
        predictions.append(prediction)
        per_grade[prompt["grade"]].append(prediction)

    def summarize(items: Sequence[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "prompts": len(items),
            "mean_middle_hold_f1": fmean(item["middle_hold_f1"] for item in items),
            "exact_middle_match_rate": fmean(
                float(item["exact_middle_match"]) for item in items
            ),
            "valid_hold_rate": fmean(item["valid_hold_rate"] for item in items),
            "parseable_rate": fmean(float(item["parseable"]) for item in items),
            "duplicate_free_rate": fmean(float(item["duplicate_free"]) for item in items),
            "endpoint_preservation_rate": 1.0,
            "novel_route_rate": fmean(float(item["novel_route"]) for item in items),
            "mean_latency_ms": fmean(item["latency_ms"] for item in items),
            "mean_generated_middle_holds": fmean(
                len(item["predicted_middle_holds"]) for item in items
            ),
        }

    metrics: dict[str, Any] = {
        "model_dir": str(model_dir),
        "decoding": "greedy unique-hold allowlist with train-derived grade median minimum length",
        "grade_minimum_middle_holds": grade_minimum_holds,
        "overall": summarize(predictions),
        "by_grade": {grade: summarize(items) for grade, items in sorted(per_grade.items())},
    }
    if nearest_metrics_path and nearest_metrics_path.is_file():
        nearest = json.loads(nearest_metrics_path.read_text(encoding="utf-8"))
        metrics["nearest_route_baseline"] = nearest
        metrics["comparison"] = {
            "middle_hold_f1_delta": metrics["overall"]["mean_middle_hold_f1"]
            - nearest["mean_middle_hold_f1"],
            "exact_match_delta": metrics["overall"]["exact_middle_match_rate"]
            - nearest["exact_middle_match_rate"],
        }

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in predictions
        ),
        encoding="utf-8",
        newline="\n",
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--board-holds", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--nearest-metrics", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    evaluate(
        model_dir=args.model_dir,
        prompts_path=args.prompts,
        train_path=args.train,
        board_holds_path=args.board_holds,
        predictions_path=args.predictions,
        metrics_path=args.metrics,
        nearest_metrics_path=args.nearest_metrics,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
