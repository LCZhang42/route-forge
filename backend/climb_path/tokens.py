"""Atomic Phase 2 token contract for grade-conditioned route completion."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Mapping, Sequence

from .schema import SUPPORTED_GRADES


PAD_TOKEN = "<|climb_pad|>"
EOS_TOKEN = "<|route_end|>"
STARTS_TOKEN = "<|starts|>"
ENDS_TOKEN = "<|ends|>"
MIDDLE_TOKEN = "<|middle|>"

_HOLD_LABEL_RE = re.compile(r"^[A-K](?:[1-9]|1[0-7])$")
_HOLD_TOKEN_RE = re.compile(r"^<\|hold:([A-K](?:[1-9]|1[0-7]))\|>$")


def grade_token(grade: str) -> str:
    if grade not in SUPPORTED_GRADES:
        raise ValueError(f"Unsupported grade: {grade}")
    return f"<|grade:{grade}|>"


def hold_token(label: str) -> str:
    normalized = label.strip().upper()
    if not _HOLD_LABEL_RE.fullmatch(normalized):
        raise ValueError(f"Invalid MoonBoard hold label: {label!r}")
    return f"<|hold:{normalized}|>"


def hold_label(token: str) -> str | None:
    match = _HOLD_TOKEN_RE.fullmatch(token)
    return match.group(1) if match else None


def additional_special_tokens(board_holds: Sequence[str]) -> list[str]:
    """Return a stable list of atomic tokens excluding PAD and EOS."""

    holds = sorted({label.strip().upper() for label in board_holds})
    if len(holds) != len(board_holds):
        raise ValueError("board_holds must be unique")
    return (
        [STARTS_TOKEN, ENDS_TOKEN, MIDDLE_TOKEN]
        + [grade_token(grade) for grade in SUPPORTED_GRADES]
        + [hold_token(label) for label in holds]
    )


def special_token_config(board_holds: Sequence[str]) -> dict[str, object]:
    return {
        "pad_token": PAD_TOKEN,
        "eos_token": EOS_TOKEN,
        "additional_special_tokens": additional_special_tokens(board_holds),
    }


def prompt_tokens(
    *,
    grade: str,
    start_holds: Sequence[str],
    end_holds: Sequence[str],
) -> list[str]:
    if not start_holds or not end_holds:
        raise ValueError("start_holds and end_holds cannot be empty")
    return (
        [grade_token(grade), STARTS_TOKEN]
        + [hold_token(label) for label in start_holds]
        + [ENDS_TOKEN]
        + [hold_token(label) for label in end_holds]
        + [MIDDLE_TOKEN]
    )


def completion_tokens(middle_holds: Sequence[str]) -> list[str]:
    return [hold_token(label) for label in middle_holds] + [EOS_TOKEN]


@dataclass(frozen=True)
class StructuredEncoding:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_length: int


def encode_route(
    record: Mapping[str, object],
    token_to_id: Callable[[str], int],
    *,
    max_length: int,
) -> StructuredEncoding:
    prompt = prompt_tokens(
        grade=str(record["grade"]),
        start_holds=record["start_holds"],  # type: ignore[arg-type]
        end_holds=record["end_holds"],  # type: ignore[arg-type]
    )
    completion = completion_tokens(record["middle_holds"])  # type: ignore[arg-type]
    prompt_ids = [token_to_id(token) for token in prompt]
    completion_ids = [token_to_id(token) for token in completion]
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_length:
        raise ValueError(
            f"Structured route needs {len(input_ids)} tokens, exceeding max_length={max_length}"
        )
    return StructuredEncoding(
        input_ids=tuple(input_ids),
        attention_mask=(1,) * len(input_ids),
        labels=(-100,) * len(prompt_ids) + tuple(completion_ids),
        prompt_length=len(prompt_ids),
    )
