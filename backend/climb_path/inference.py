"""DistilGPT-2 route inference with no implicit model downloads.

The model must already exist at ``model_dir``. On production this directory is
provided by a Modal Volume. Local imports and tests never contact Hugging Face.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from .schema import Hold, Route, SUPPORTED_GRADES


_GRADE_RE = re.compile(r"GRADE:\s*([^|]+)")
_START_RE = re.compile(r"START:\s*\((\d+),(\d+)\)")
_END_RE = re.compile(r"END:\s*\((\d+),(\d+)\)")
_MID_RE = re.compile(r"MID:\s*([^|]*?)(?:\||$)")
_HOLD_RE = re.compile(r"\((\d+),(\d+)\)")


def build_prompt(grade: str, start_hold: Hold, end_hold: Hold) -> str:
    if grade not in SUPPORTED_GRADES:
        raise ValueError(f"Unsupported grade: {grade}")
    return (
        f"GRADE: {grade} | START: ({start_hold.x},{start_hold.y}) | "
        f"END: ({end_hold.x},{end_hold.y}) | MID:"
    )


def parse_generated_text(text: str) -> Route | None:
    """Parse the Phase 0 text format into the canonical route schema."""

    grade_match = _GRADE_RE.search(text)
    start_match = _START_RE.search(text)
    end_match = _END_RE.search(text)
    if not grade_match or not start_match or not end_match:
        return None

    try:
        grade = grade_match.group(1).strip()
        start = Hold(int(start_match.group(1)), int(start_match.group(2)))
        end = Hold(int(end_match.group(1)), int(end_match.group(2)))
        middle: list[Hold] = []
        mid_match = _MID_RE.search(text)
        if mid_match:
            middle = [Hold(int(x), int(y)) for x, y in _HOLD_RE.findall(mid_match.group(1))]
        return Route(
            grade=grade,
            start_holds=(start,),
            middle_holds=tuple(middle),
            end_holds=(end,),
        )
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Candidate:
    route: Route
    warnings: tuple[str, ...]


def validate_route(
    route: Route,
    *,
    requested_grade: str,
    requested_start: Hold,
    requested_end: Hold,
    valid_holds: set[tuple[int, int]] | None = None,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if route.grade != requested_grade:
        warnings.append("generated grade did not match the request")
    if route.start_holds != (requested_start,):
        warnings.append("generated start hold did not match the request")
    if route.end_holds != (requested_end,):
        warnings.append("generated end hold did not match the request")
    if len(set(route.holds)) != len(route.holds):
        warnings.append("route contains duplicate holds")
    if valid_holds is not None:
        invalid = [hold.label for hold in route.holds if (hold.x, hold.y) not in valid_holds]
        if invalid:
            warnings.append(f"route contains invalid board holds: {', '.join(invalid)}")
    return tuple(warnings)


def normalize_generated_route(
    route: Route,
    *,
    valid_holds: set[tuple[int, int]] | None = None,
) -> tuple[Route, tuple[str, ...]]:
    """Turn noisy Phase 0 text output into a safe route for the API.

    DistilGPT-2 sometimes continues sampling after it emits the requested end
    hold. The API already owns the start/end contract, so anything after that
    first end marker is unusable. Repeated or nonexistent middle holds are also
    removed while retaining a warning so baseline quality remains observable.
    """

    seen = set(route.start_holds)
    middle: list[Hold] = []
    reached_end = False
    removed_duplicates = False
    removed_invalid: list[str] = []

    for hold in route.middle_holds:
        if hold in route.end_holds:
            reached_end = True
            break
        if hold in seen:
            removed_duplicates = True
            continue
        if valid_holds is not None and (hold.x, hold.y) not in valid_holds:
            removed_invalid.append(hold.label)
            continue
        seen.add(hold)
        middle.append(hold)

    warnings: list[str] = []
    if reached_end:
        warnings.append("generation was truncated at the requested end hold")
    if removed_duplicates:
        warnings.append("duplicate middle holds were removed")
    if removed_invalid:
        labels = ", ".join(dict.fromkeys(removed_invalid))
        warnings.append(f"invalid board holds were removed: {labels}")

    normalized = Route(
        grade=route.grade,
        start_holds=route.start_holds,
        middle_holds=tuple(middle),
        end_holds=route.end_holds,
    )
    return normalized, tuple(warnings)


class DistilGPT2Generator:
    """Load a fine-tuned model from local storage and generate route candidates."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        valid_holds: Iterable[tuple[int, int]] | None = None,
        device: str | None = None,
    ) -> None:
        model_path = Path(model_dir)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Fine-tuned model directory not found: {model_path}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.valid_holds = set(valid_holds) if valid_holds is not None else None

    def generate(
        self,
        *,
        grade: str,
        start_hold: Hold,
        end_hold: Hold,
        candidate_count: int = 1,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 80,
        seed: int | None = None,
    ) -> list[Candidate]:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        prompt = build_prompt(grade, start_hold, end_hold)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=candidate_count,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        candidates: list[Candidate] = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            route = parse_generated_text(text)
            if route is None:
                continue
            route, normalization_warnings = normalize_generated_route(
                route,
                valid_holds=self.valid_holds,
            )
            validation_warnings = validate_route(
                route,
                requested_grade=grade,
                requested_start=start_hold,
                requested_end=end_hold,
                valid_holds=self.valid_holds,
            )
            candidates.append(
                Candidate(
                    route=route,
                    warnings=normalization_warnings + validation_warnings,
                )
            )
        return candidates
