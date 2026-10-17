"""Phase 3 constrained generation, ranking, and retrieval fallback."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from backend.evaluation.nearest_route_baseline import NearestRouteBaseline

from .constraints import (
    GradeProfile,
    RouteDiagnostics,
    build_grade_profiles,
    diagnose_route,
    route_fingerprint,
    unique_holds,
)
from .schema import Hold, Route, SUPPORTED_GRADES
from .tokens import EOS_TOKEN, hold_token, prompt_tokens


@dataclass(frozen=True)
class GeneratedMiddle:
    hold_labels: tuple[str, ...]
    average_log_probability: float


class MiddleCandidateGenerator(Protocol):
    def generate_middle_candidates(
        self,
        *,
        grade: str,
        start_holds: Sequence[str],
        end_holds: Sequence[str],
        attempts: int,
        temperature: float,
        top_p: float,
        seed: int,
        profile: GradeProfile,
    ) -> Sequence[GeneratedMiddle]: ...


@dataclass(frozen=True)
class RankedRouteCandidate:
    route: Route
    source: str
    score: float
    model_log_probability: float | None
    diagnostics: RouteDiagnostics
    source_route_id: str | None = None
    retrieval_similarity: float = 0.0


@dataclass(frozen=True)
class GenerationResult:
    candidates: tuple[RankedRouteCandidate, ...]
    seed: int
    generated_attempts: int
    rejected_attempts: int
    used_fallback: bool


class NoValidRouteError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TransformersStructuredGenerator:
    """Load the Phase 2 checkpoint and sample only legal atomic route tokens."""

    def __init__(self, model_dir: str | Path, board_holds: Sequence[str]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = Path(model_dir)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Structured model not found: {model_path}")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.eos_id = self.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        self.hold_id_to_label = {
            self.tokenizer.convert_tokens_to_ids(hold_token(label)): label
            for label in board_holds
        }
        if len(self.hold_id_to_label) != len(board_holds):
            raise ValueError("Structured tokenizer is missing board-hold tokens")

    def generate_middle_candidates(
        self,
        *,
        grade: str,
        start_holds: Sequence[str],
        end_holds: Sequence[str],
        attempts: int,
        temperature: float,
        top_p: float,
        seed: int,
        profile: GradeProfile,
    ) -> Sequence[GeneratedMiddle]:
        torch = self.torch
        prompt = prompt_tokens(
            grade=grade,
            start_holds=start_holds,
            end_holds=end_holds,
        )
        prompt_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in prompt]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        hold_ids = set(self.hold_id_to_label)

        def allowed_tokens(_batch_id, current_ids):
            used = set(current_ids.tolist()) & hold_ids
            generated_count = len(current_ids) - len(prompt_ids)
            if generated_count >= profile.maximum_middle_holds:
                return [self.eos_id]
            allowed = hold_ids - used
            if generated_count >= profile.target_middle_holds:
                allowed.add(self.eos_id)
            return sorted(allowed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=profile.maximum_middle_holds + 1,
                num_return_sequences=attempts,
                eos_token_id=self.eos_id,
                pad_token_id=self.tokenizer.pad_token_id,
                prefix_allowed_tokens_fn=allowed_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
            transition_scores = self.model.compute_transition_scores(
                generated.sequences,
                generated.scores,
                normalize_logits=True,
            )

        results: list[GeneratedMiddle] = []
        for sequence, scores in zip(generated.sequences, transition_scores):
            new_ids = sequence[len(prompt_ids) :].tolist()
            labels: list[str] = []
            hold_scores: list[float] = []
            for token_id, score in zip(new_ids, scores.tolist()):
                if token_id == self.eos_id:
                    break
                label = self.hold_id_to_label.get(token_id)
                if label is None:
                    continue
                labels.append(label)
                hold_scores.append(float(score))
            results.append(
                GeneratedMiddle(
                    hold_labels=tuple(labels),
                    average_log_probability=(
                        sum(hold_scores) / len(hold_scores) if hold_scores else -100.0
                    ),
                )
            )
        return results


class Phase3RouteService:
    """Combine model candidates, hard constraints, ranking, and retrieval."""

    def __init__(
        self,
        *,
        generator: MiddleCandidateGenerator,
        training_records: Sequence[dict[str, Any]],
        board_holds: Sequence[str],
        model_version: str,
        data_version: str,
    ) -> None:
        self.generator = generator
        self.training_records = tuple(training_records)
        self.valid_hold_labels = set(board_holds)
        self.valid_start_labels = {
            label
            for record in self.training_records
            for label in record["start_holds"]
        }
        self.valid_end_labels = {
            label
            for record in self.training_records
            for label in record["end_holds"]
        }
        self.profiles = build_grade_profiles(self.training_records)
        self.training_fingerprints = {
            record["fingerprint"] for record in self.training_records
        }
        self.retrieval = NearestRouteBaseline(self.training_records)
        self.model_version = model_version
        self.data_version = data_version

    @classmethod
    def from_paths(
        cls,
        *,
        model_dir: str | Path,
        train_path: str | Path,
        board_holds_path: str | Path,
        manifest_path: str | Path,
        model_version: str,
    ) -> "Phase3RouteService":
        records = _read_jsonl(Path(train_path))
        board_holds = json.loads(Path(board_holds_path).read_text(encoding="utf-8"))[
            "holds"
        ]
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        return cls(
            generator=TransformersStructuredGenerator(model_dir, board_holds),
            training_records=records,
            board_holds=board_holds,
            model_version=model_version,
            data_version=manifest["dataset_version"],
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "data_version": self.data_version,
            "supported_grades": list(SUPPORTED_GRADES),
            "board": "MoonBoard 2016",
            "board_holds": sorted(self.valid_hold_labels),
            "endpoint_holds": {
                "start": sorted(self.valid_start_labels),
                "end": sorted(self.valid_end_labels),
            },
            "display_row_offset": 1,
            "endpoint_cardinality": {"start": [1, 2], "end": [1, 2]},
            "candidate_limit": 3,
            "selection_policy": "validated_retrieval_first_with_model_alternatives",
            "profiles": {
                grade: {
                    "minimum_middle_holds": profile.minimum_middle_holds,
                    "target_middle_holds": profile.target_middle_holds,
                    "maximum_middle_holds": profile.maximum_middle_holds,
                    "soft_move_limit": round(profile.soft_move_limit, 3),
                    "hard_move_limit": round(profile.hard_move_limit, 3),
                }
                for grade, profile in self.profiles.items()
            },
        }

    def _diagnose(
        self,
        route: Route,
        starts: Sequence[Hold],
        ends: Sequence[Hold],
    ) -> RouteDiagnostics:
        return diagnose_route(
            route,
            requested_grade=route.grade,
            requested_starts=starts,
            requested_ends=ends,
            valid_hold_labels=self.valid_hold_labels,
            profile=self.profiles[route.grade],
            training_fingerprints=self.training_fingerprints,
        )

    @staticmethod
    def _score(log_probability: float, diagnostics: RouteDiagnostics) -> float:
        return (
            log_probability
            - 0.35 * diagnostics.reachability_violations
            - 0.08 * diagnostics.length_delta
            - 0.06 * diagnostics.downward_moves
            + (0.05 if diagnostics.novel_route else -0.25)
        )

    @staticmethod
    def _hold_f1(left: Sequence[str], right: Sequence[str]) -> float:
        left_set = set(left)
        right_set = set(right)
        if not left_set or not right_set:
            return float(left_set == right_set)
        overlap = len(left_set & right_set)
        if not overlap:
            return 0.0
        precision = overlap / len(left_set)
        recall = overlap / len(right_set)
        return 2 * precision * recall / (precision + recall)

    def generate(
        self,
        *,
        grade: str,
        start_holds: Sequence[str],
        end_holds: Sequence[str],
        candidate_count: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> GenerationResult:
        if grade not in self.profiles:
            raise ValueError(f"Unsupported grade: {grade}")
        normalized_starts = tuple(label.strip().upper() for label in start_holds)
        normalized_ends = tuple(label.strip().upper() for label in end_holds)
        if not 1 <= len(normalized_starts) <= 2 or not 1 <= len(normalized_ends) <= 2:
            raise ValueError("one or two start and end holds are required")
        if len(set(normalized_starts + normalized_ends)) != len(
            normalized_starts + normalized_ends
        ):
            raise ValueError("start and end holds must be unique")
        if any(
            label not in self.valid_hold_labels
            for label in normalized_starts + normalized_ends
        ):
            raise ValueError("all endpoints must exist on the MoonBoard 2016 layout")
        if any(label not in self.valid_start_labels for label in normalized_starts):
            raise ValueError("start holds must use observed MoonBoard 2016 start positions")
        if any(label not in self.valid_end_labels for label in normalized_ends):
            raise ValueError("finish holds must use MoonBoard 2016 top-row positions")

        starts = tuple(Hold.from_label(label) for label in normalized_starts)
        ends = tuple(Hold.from_label(label) for label in normalized_ends)
        attempts = min(max(candidate_count * 4, 4), 12)
        generated = self.generator.generate_middle_candidates(
            grade=grade,
            start_holds=normalized_starts,
            end_holds=normalized_ends,
            attempts=attempts,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            profile=self.profiles[grade],
        )

        ranked: list[RankedRouteCandidate] = []
        seen_fingerprints: set[str] = set()
        rejected = 0
        for candidate in generated:
            route = Route(
                grade=grade,
                start_holds=starts,
                middle_holds=tuple(
                    Hold.from_label(label) for label in candidate.hold_labels
                ),
                end_holds=ends,
            )
            diagnostics = self._diagnose(route, starts, ends)
            fingerprint = route_fingerprint(route)
            if not diagnostics.valid or fingerprint in seen_fingerprints:
                rejected += 1
                continue
            seen_fingerprints.add(fingerprint)
            ranked.append(
                RankedRouteCandidate(
                    route=route,
                    source="model",
                    score=self._score(
                        candidate.average_log_probability, diagnostics
                    ),
                    model_log_probability=candidate.average_log_probability,
                    diagnostics=diagnostics,
                )
            )

        ranked.sort(key=lambda item: (-item.score, route_fingerprint(item.route)))
        model_shortlist = ranked[:3]

        prediction = self.retrieval.predict(
            grade=grade,
            start_holds=normalized_starts,
            end_holds=normalized_ends,
        )
        retrieval_labels = tuple(prediction.middle_holds)
        middle = unique_holds(
            retrieval_labels,
            excluded=set(normalized_starts + normalized_ends),
        )
        fallback_route = Route(
            grade=grade,
            start_holds=starts,
            middle_holds=middle,
            end_holds=ends,
        )
        fallback_diagnostics = self._diagnose(fallback_route, starts, ends)
        fallback_candidate = (
            RankedRouteCandidate(
                route=fallback_route,
                source="retrieval_fallback",
                score=1.0,
                model_log_probability=None,
                diagnostics=fallback_diagnostics,
                source_route_id=prediction.source_route_id,
                retrieval_similarity=1.0,
            )
            if fallback_diagnostics.valid
            else None
        )

        retrieval_ranked: list[RankedRouteCandidate] = []
        for candidate in model_shortlist:
            candidate_labels = tuple(
                hold.label for hold in candidate.route.middle_holds
            )
            similarity = self._hold_f1(candidate_labels, retrieval_labels)
            retrieval_ranked.append(
                RankedRouteCandidate(
                    route=candidate.route,
                    source=candidate.source,
                    score=similarity + 0.01 * candidate.score,
                    model_log_probability=candidate.model_log_probability,
                    diagnostics=candidate.diagnostics,
                    source_route_id=candidate.source_route_id,
                    retrieval_similarity=similarity,
                )
            )
        retrieval_ranked.sort(
            key=lambda item: (
                -item.retrieval_similarity,
                -item.score,
                route_fingerprint(item.route),
            )
        )

        # The learned ranker did not beat nearest-route retrieval on the frozen
        # suite. Keep the validated retrieval route first until that promotion
        # gate passes, then expose the strongest distinct model routes as useful
        # alternatives rather than silently lowering top-1 quality.
        selected: list[RankedRouteCandidate] = []
        if fallback_candidate is not None:
            selected.append(fallback_candidate)
        selected_fingerprints = {
            route_fingerprint(candidate.route) for candidate in selected
        }
        for candidate in retrieval_ranked:
            fingerprint = route_fingerprint(candidate.route)
            if fingerprint in selected_fingerprints:
                continue
            selected.append(candidate)
            selected_fingerprints.add(fingerprint)
            if len(selected) >= candidate_count:
                break
        used_fallback = any(
            candidate.source == "retrieval_fallback" for candidate in selected
        )

        if not selected:
            raise NoValidRouteError("no candidate passed the Phase 3 constraints")
        return GenerationResult(
            candidates=tuple(selected[:candidate_count]),
            seed=seed,
            generated_attempts=len(generated),
            rejected_attempts=rejected,
            used_fallback=used_fallback,
        )
