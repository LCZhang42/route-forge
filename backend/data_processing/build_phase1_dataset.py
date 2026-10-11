"""Build the deterministic v1 MoonBoard data contract.

The builder deliberately ignores the legacy split assignment. It combines the
three quality CSVs, canonicalizes and groups routes, resolves duplicate labels,
and only then creates deterministic grade-stratified splits. This prevents the
same physical route from leaking across train, validation, and test.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from backend.climb_path.schema import Hold, SUPPORTED_GRADES


SCHEMA_VERSION = "climb-path-route/v1"
DATASET_VERSION = "v1"
DEFAULT_SEED = 42
EXPECTED_BOARD_HOLDS = 141
DEFAULT_INPUTS = (
    Path("data_reordered/moonboard_train_quality_filtered.csv"),
    Path("data_reordered/moonboard_val_quality.csv"),
    Path("data_reordered/moonboard_test_quality.csv"),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_holds(value: str, *, field: str, problem_id: str) -> list[Hold]:
    try:
        raw = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Problem {problem_id}: invalid {field}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"Problem {problem_id}: {field} must be a list")
    try:
        return [Hold.from_pair(pair) for pair in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Problem {problem_id}: invalid hold in {field}") from exc


def _ordered_unique(holds: Iterable[Hold]) -> list[Hold]:
    return list(dict.fromkeys(holds))


def _sorted_unique(holds: Iterable[Hold]) -> list[Hold]:
    return sorted(set(holds), key=lambda hold: (hold.y, hold.x))


def _route_payload(starts: Sequence[Hold], middle: Sequence[Hold], ends: Sequence[Hold]) -> dict[str, list[str]]:
    return {
        "start_holds": [hold.label for hold in starts],
        "middle_holds": [hold.label for hold in middle],
        "end_holds": [hold.label for hold in ends],
    }


def _route_fingerprint(starts: Sequence[Hold], middle: Sequence[Hold], ends: Sequence[Hold]) -> str:
    payload = _route_payload(starts, middle, ends)
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _source_priority(row: dict[str, Any]) -> tuple[float, float, str]:
    """Prefer the strongest community evidence, then a stable problem id."""

    return (
        -float(row["repeats"]),
        -float(row["quality_score"]),
        str(row["problem_id"]),
    )


def load_and_group(inputs: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any], set[Hold]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_stats: dict[str, Any] = {}
    board_holds: set[Hold] = set()
    input_rows = 0

    for path in inputs:
        rows_in_file = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                problem_id = str(raw["problem_id"])
                starts = _sorted_unique(
                    _parse_holds(raw["start_holds"], field="start_holds", problem_id=problem_id)
                )
                middle = _ordered_unique(
                    _parse_holds(raw["mid_holds"], field="mid_holds", problem_id=problem_id)
                )
                ends = _sorted_unique(
                    _parse_holds(raw["end_holds"], field="end_holds", problem_id=problem_id)
                )
                if not starts or not ends:
                    raise ValueError(f"Problem {problem_id}: start/end holds cannot be empty")

                role_overlap = (set(starts) & set(middle)) | (set(starts) & set(ends)) | (
                    set(middle) & set(ends)
                )
                if role_overlap:
                    labels = ", ".join(sorted(hold.label for hold in role_overlap))
                    raise ValueError(f"Problem {problem_id}: holds occur in multiple roles: {labels}")

                board_holds.update(starts)
                board_holds.update(middle)
                board_holds.update(ends)
                fingerprint = _route_fingerprint(starts, middle, ends)
                grouped[fingerprint].append(
                    {
                        "problem_id": problem_id,
                        "source_file": path.name,
                        "grade": str(raw["grade"]).strip(),
                        "starts": starts,
                        "middle": middle,
                        "ends": ends,
                        "repeats": int(float(raw.get("repeats") or 0)),
                        "quality_score": float(raw.get("quality_score") or 0.0),
                    }
                )
                rows_in_file += 1
                input_rows += 1

        input_stats[path.name] = {
            "rows": rows_in_file,
            "sha256": _sha256_file(path),
        }

    records: list[dict[str, Any]] = []
    grade_conflicts = 0
    duplicate_groups = 0
    duplicate_rows_removed = 0

    for fingerprint, candidates in sorted(grouped.items()):
        candidates.sort(key=_source_priority)
        selected = candidates[0]
        observed_grades = sorted({candidate["grade"] for candidate in candidates})
        if len(candidates) > 1:
            duplicate_groups += 1
            duplicate_rows_removed += len(candidates) - 1
        if len(observed_grades) > 1:
            grade_conflicts += 1

        payload = _route_payload(selected["starts"], selected["middle"], selected["ends"])
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "route_id": f"route_{fingerprint[:16]}",
                "fingerprint": fingerprint,
                "grade": selected["grade"],
                **payload,
                "hold_count": len(selected["starts"])
                + len(selected["middle"])
                + len(selected["ends"]),
                "metadata": {
                    "selected_problem_id": selected["problem_id"],
                    "source_problem_ids": sorted(
                        {candidate["problem_id"] for candidate in candidates}
                    ),
                    "source_files": sorted({candidate["source_file"] for candidate in candidates}),
                    "observed_grades": observed_grades,
                    "repeats_total": sum(candidate["repeats"] for candidate in candidates),
                    "selected_quality_score": selected["quality_score"],
                },
            }
        )

    stats = {
        "input_rows": input_rows,
        "canonical_routes": len(records),
        "duplicate_groups": duplicate_groups,
        "duplicate_rows_removed": duplicate_rows_removed,
        "grade_conflict_groups": grade_conflicts,
        "inputs": input_stats,
    }
    return records, stats, board_holds


def _stable_split_key(record: dict[str, Any], seed: int) -> str:
    return _sha256_bytes(f"{seed}:{record['fingerprint']}".encode("utf-8"))


def split_records(
    records: Sequence[dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[dict[str, Any]]]:
    by_grade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ood: list[dict[str, Any]] = []
    for record in records:
        if record["grade"] in SUPPORTED_GRADES:
            by_grade[record["grade"]].append(record)
        else:
            ood.append(record)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for grade in SUPPORTED_GRADES:
        grade_records = sorted(
            by_grade.get(grade, []),
            key=lambda record: (_stable_split_key(record, seed), record["fingerprint"]),
        )
        count = len(grade_records)
        if count == 0:
            continue
        validation_count = max(1, round(count * 0.10)) if count >= 3 else 0
        test_count = max(1, round(count * 0.10)) if count >= 3 else 0
        if validation_count + test_count >= count:
            validation_count = 0
            test_count = 1 if count > 1 else 0
        train_count = count - validation_count - test_count
        splits["train"].extend(grade_records[:train_count])
        splits["validation"].extend(
            grade_records[train_count : train_count + validation_count]
        )
        splits["test"].extend(grade_records[train_count + validation_count :])

    for name in splits:
        splits[name].sort(key=lambda record: record["route_id"])
    splits["ood"] = sorted(ood, key=lambda record: record["route_id"])
    return splits


def build_frozen_evaluation(
    test_records: Sequence[dict[str, Any]],
    *,
    limit: int = 128,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    by_grade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in test_records:
        by_grade[record["grade"]].append(record)
    for grade in by_grade:
        by_grade[grade].sort(
            key=lambda record: (_stable_split_key(record, seed + 1), record["route_id"])
        )

    selected: list[dict[str, Any]] = []
    positions = {grade: 0 for grade in SUPPORTED_GRADES}
    while len(selected) < min(limit, len(test_records)):
        added = False
        for grade in SUPPORTED_GRADES:
            position = positions[grade]
            candidates = by_grade.get(grade, [])
            if position >= len(candidates):
                continue
            selected.append(candidates[position])
            positions[grade] += 1
            added = True
            if len(selected) == min(limit, len(test_records)):
                break
        if not added:
            break

    return [
        {
            "schema_version": "climb-path-eval/v1",
            "prompt_id": f"prompt_{index:04d}",
            "grade": record["grade"],
            "start_holds": record["start_holds"],
            "end_holds": record["end_holds"],
            "reference_route_id": record["route_id"],
            "reference_fingerprint": record["fingerprint"],
            "reference_middle_holds": record["middle_holds"],
        }
        for index, record in enumerate(selected, start=1)
    ]


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    payload = "".join(f"{_canonical_json(record)}\n" for record in records)
    path.write_bytes(payload.encode("utf-8"))


def _grade_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["grade"] for record in records).items()))


def _assert_no_overlap(splits: dict[str, list[dict[str, Any]]]) -> None:
    names = ("train", "validation", "test", "ood")
    fingerprints = {
        name: {record["fingerprint"] for record in splits[name]} for name in names
    }
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = fingerprints[left] & fingerprints[right]
            if overlap:
                raise AssertionError(f"{left}/{right} route overlap: {len(overlap)}")


def build_dataset(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    evaluation_size: int = 128,
) -> dict[str, Any]:
    records, source_stats, board_holds = load_and_group(inputs)
    if len(board_holds) != EXPECTED_BOARD_HOLDS:
        raise ValueError(
            f"Expected {EXPECTED_BOARD_HOLDS} MoonBoard holds, found {len(board_holds)}"
        )
    splits = split_records(records, seed=seed)
    _assert_no_overlap(splits)
    evaluation = build_frozen_evaluation(
        splits["test"], limit=evaluation_size, seed=seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, Path] = {}
    for split_name, split_records_value in splits.items():
        filename = "validation.jsonl" if split_name == "validation" else f"{split_name}.jsonl"
        path = output_dir / filename
        _write_jsonl(path, split_records_value)
        output_files[filename] = path

    evaluation_path = output_dir / "frozen_evaluation.jsonl"
    _write_jsonl(evaluation_path, evaluation)
    output_files[evaluation_path.name] = evaluation_path

    hold_catalog = {
        "schema_version": "moonboard-2016-holds/v1",
        "source": "union of canonicalized Phase 1 input routes",
        "hold_count": len(board_holds),
        "holds": [hold.label for hold in sorted(board_holds, key=lambda hold: (hold.y, hold.x))],
    }
    catalog_path = output_dir / "board_holds.json"
    catalog_path.write_bytes((_canonical_json(hold_catalog) + "\n").encode("utf-8"))
    output_files[catalog_path.name] = catalog_path

    manifest = {
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "split_strategy": "deduplicate globally, then deterministic grade-stratified 80/10/10",
        "supported_grades": list(SUPPORTED_GRADES),
        "ood_grades": sorted({record["grade"] for record in splits["ood"]}),
        "source": source_stats,
        "board_hold_count": len(board_holds),
        "splits": {
            name: {
                "rows": len(split_records_value),
                "grade_counts": _grade_counts(split_records_value),
            }
            for name, split_records_value in splits.items()
        },
        "frozen_evaluation_rows": len(evaluation),
        "invariants": {
            "cross_split_fingerprint_overlap": 0,
            "unknown_hold_count": 0,
            "multi_endpoint_fields_preserved": True,
        },
        "files": {
            name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for name, path in sorted(output_files.items())
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--output-dir", type=Path, default=Path("data_contract/v1"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--evaluation-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.inputs,
        args.output_dir,
        seed=args.seed,
        evaluation_size=args.evaluation_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
