"""Data-derived route constraints and diagnostics for Phase 3 inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from statistics import median
from typing import Any, Iterable, Sequence

from .schema import Hold, Route, SUPPORTED_GRADES


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distance(left: Hold, right: Hold) -> float:
    return hypot(left.x - right.x, left.y - right.y)


def route_move_pairs(route: Route) -> tuple[tuple[Hold, Hold], ...]:
    """Return climbing transitions without treating simultaneous starts as moves."""

    pairs: list[tuple[Hold, Hold]] = []
    if route.middle_holds:
        first = route.middle_holds[0]
        start = min(route.start_holds, key=lambda hold: _distance(hold, first))
        pairs.append((start, first))
        pairs.extend(zip(route.middle_holds, route.middle_holds[1:]))
        last = route.middle_holds[-1]
        end = min(route.end_holds, key=lambda hold: _distance(last, hold))
        pairs.append((last, end))
    else:
        start, end = min(
            ((start, end) for start in route.start_holds for end in route.end_holds),
            key=lambda pair: _distance(*pair),
        )
        pairs.append((start, end))
    return tuple(pairs)


def route_move_distances(route: Route) -> tuple[float, ...]:
    return tuple(_distance(left, right) for left, right in route_move_pairs(route))


@dataclass(frozen=True)
class GradeProfile:
    grade: str
    minimum_middle_holds: int
    target_middle_holds: int
    maximum_middle_holds: int
    soft_move_limit: float
    hard_move_limit: float


@dataclass(frozen=True)
class RouteDiagnostics:
    hard_rejections: tuple[str, ...]
    warnings: tuple[str, ...]
    middle_hold_count: int
    target_middle_holds: int
    length_delta: int
    move_distances: tuple[float, ...]
    maximum_move_distance: float
    reachability_violations: int
    downward_moves: int
    novel_route: bool

    @property
    def valid(self) -> bool:
        return not self.hard_rejections

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["valid"] = self.valid
        return payload


def _record_route(record: dict[str, Any]) -> Route:
    return Route(
        grade=record["grade"],
        start_holds=tuple(Hold.from_label(label) for label in record["start_holds"]),
        middle_holds=tuple(Hold.from_label(label) for label in record["middle_holds"]),
        end_holds=tuple(Hold.from_label(label) for label in record["end_holds"]),
    )


def build_grade_profiles(records: Sequence[dict[str, Any]]) -> dict[str, GradeProfile]:
    """Build robust length and move limits using training data only."""

    profiles: dict[str, GradeProfile] = {}
    for grade in SUPPORTED_GRADES:
        grade_records = [record for record in records if record["grade"] == grade]
        if not grade_records:
            continue
        lengths = [len(record["middle_holds"]) for record in grade_records]
        distances = [
            distance
            for record in grade_records
            for distance in route_move_distances(_record_route(record))
        ]
        profiles[grade] = GradeProfile(
            grade=grade,
            minimum_middle_holds=max(1, int(_percentile(lengths, 0.05))),
            target_middle_holds=int(median(lengths)),
            maximum_middle_holds=max(1, int(_percentile(lengths, 0.95) + 0.999)),
            soft_move_limit=_percentile(distances, 0.95),
            hard_move_limit=_percentile(distances, 0.995),
        )
    return profiles


def route_fingerprint(route: Route) -> str:
    """Match the Phase 1 canonical fingerprint without importing build tooling."""

    import hashlib
    import json

    payload = {
        "start_holds": sorted(hold.label for hold in route.start_holds),
        "middle_holds": [hold.label for hold in route.middle_holds],
        "end_holds": sorted(hold.label for hold in route.end_holds),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def diagnose_route(
    route: Route,
    *,
    requested_grade: str,
    requested_starts: Sequence[Hold],
    requested_ends: Sequence[Hold],
    valid_hold_labels: set[str],
    profile: GradeProfile,
    training_fingerprints: set[str],
) -> RouteDiagnostics:
    hard_rejections: list[str] = []
    warnings: list[str] = []
    if route.grade != requested_grade:
        hard_rejections.append("grade_mismatch")
    if set(route.start_holds) != set(requested_starts):
        hard_rejections.append("start_endpoint_mismatch")
    if set(route.end_holds) != set(requested_ends):
        hard_rejections.append("end_endpoint_mismatch")
    if len(set(route.holds)) != len(route.holds):
        hard_rejections.append("duplicate_hold")
    if any(hold.label not in valid_hold_labels for hold in route.holds):
        hard_rejections.append("unknown_board_hold")

    middle_count = len(route.middle_holds)
    if middle_count < profile.minimum_middle_holds:
        hard_rejections.append("too_few_middle_holds")
    if middle_count > profile.maximum_middle_holds:
        hard_rejections.append("too_many_middle_holds")

    pairs = route_move_pairs(route)
    distances = tuple(_distance(left, right) for left, right in pairs)
    reachability_violations = sum(
        distance > profile.soft_move_limit for distance in distances
    )
    if any(distance > profile.hard_move_limit for distance in distances):
        hard_rejections.append("severe_reachability_violation")
    elif reachability_violations:
        warnings.append("one or more moves are unusually long for this grade")

    downward_moves = sum(right.y < left.y - 1 for left, right in pairs)
    if downward_moves:
        warnings.append("route contains a downward move")

    fingerprint = route_fingerprint(route)
    novel = fingerprint not in training_fingerprints
    if not novel:
        warnings.append("route exactly matches a training route")

    return RouteDiagnostics(
        hard_rejections=tuple(dict.fromkeys(hard_rejections)),
        warnings=tuple(warnings),
        middle_hold_count=middle_count,
        target_middle_holds=profile.target_middle_holds,
        length_delta=abs(middle_count - profile.target_middle_holds),
        move_distances=tuple(round(distance, 3) for distance in distances),
        maximum_move_distance=round(max(distances, default=0.0), 3),
        reachability_violations=reachability_violations,
        downward_moves=downward_moves,
        novel_route=novel,
    )


def unique_holds(labels: Iterable[str], excluded: set[str]) -> tuple[Hold, ...]:
    """Normalize a retrieved middle sequence while preserving its order."""

    result: list[Hold] = []
    seen = set(excluded)
    for label in labels:
        normalized = label.strip().upper()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(Hold.from_label(normalized))
    return tuple(result)
