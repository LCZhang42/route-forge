"""Data-derived route constraints and diagnostics for Phase 3 inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from math import hypot
from statistics import median
from typing import Any, Iterable, Sequence

from .schema import Hold, Route, SUPPORTED_GRADES


SUPPORTED_GENERATION_MODES = ("safer", "balanced", "dynamic")


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
    generation_mode: str
    feasibility_limit: float
    plausible_path: tuple[str, ...]

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


def generation_mode_limit(profile: GradeProfile, generation_mode: str) -> float:
    """Return the grade-relative graph edge limit for a movement style."""

    if generation_mode == "safer":
        return profile.soft_move_limit
    if generation_mode == "balanced":
        return min(profile.hard_move_limit, profile.soft_move_limit * 1.15)
    if generation_mode == "dynamic":
        return profile.hard_move_limit
    raise ValueError(
        f"Unsupported generation mode: {generation_mode}; expected one of "
        f"{', '.join(SUPPORTED_GENERATION_MODES)}"
    )


def _best_path_to_finish(route: Route, finish: Hold) -> tuple[Hold, ...]:
    """Find the upward-biased path whose largest move is as small as possible."""

    holds = tuple(dict.fromkeys(route.holds))
    starts = set(route.start_holds)
    queue: list[tuple[float, float, int, Hold, tuple[Hold, ...]]] = []
    best: dict[Hold, tuple[float, float, int]] = {}
    for start in route.start_holds:
        cost = (0.0, 0.0, 0)
        best[start] = cost
        heappush(queue, (*cost, start, (start,)))

    while queue:
        maximum, total, steps, current, path = heappop(queue)
        if best.get(current) != (maximum, total, steps):
            continue
        if current == finish:
            return path
        for neighbor in holds:
            if neighbor == current or neighbor in starts:
                continue
            # A one-row drop can represent a match, cross, or body reposition;
            # larger drops are not useful in a plausible upward beta.
            if neighbor.y < current.y - 1:
                continue
            distance = _distance(current, neighbor)
            next_cost = (max(maximum, distance), total + distance, steps + 1)
            if neighbor not in best or next_cost < best[neighbor]:
                best[neighbor] = next_cost
                heappush(queue, (*next_cost, neighbor, path + (neighbor,)))
    return ()


def plausible_beta_path(route: Route) -> tuple[Hold, ...]:
    """Return the gentlest graph path needed to reach the hardest finish hold."""

    paths = [
        path
        for finish in route.end_holds
        if (path := _best_path_to_finish(route, finish))
    ]
    if not paths:
        return ()
    return max(
        paths,
        key=lambda path: (
            max((_distance(left, right) for left, right in zip(path, path[1:])), default=0),
            sum(_distance(left, right) for left, right in zip(path, path[1:])),
        ),
    )


def diagnose_route(
    route: Route,
    *,
    requested_grade: str,
    requested_starts: Sequence[Hold],
    requested_ends: Sequence[Hold],
    valid_hold_labels: set[str],
    profile: GradeProfile,
    training_fingerprints: set[str],
    generation_mode: str = "balanced",
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

    path = plausible_beta_path(route)
    pairs = tuple(zip(path, path[1:]))
    distances = tuple(_distance(left, right) for left, right in pairs)
    feasibility_limit = generation_mode_limit(profile, generation_mode)
    reachability_violations = sum(
        distance > profile.soft_move_limit for distance in distances
    )
    if not path or any(distance > feasibility_limit for distance in distances):
        hard_rejections.append("no_plausible_beta_for_generation_mode")
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
        generation_mode=generation_mode,
        feasibility_limit=round(feasibility_limit, 3),
        plausible_path=tuple(hold.label for hold in path),
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
