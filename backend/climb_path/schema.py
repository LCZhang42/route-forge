"""Canonical MoonBoard route types.

These types intentionally do not depend on PyTorch or Transformers so they can
be imported by preprocessing, API, and test code without loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


SUPPORTED_GRADES = (
    "6B+",
    "6C",
    "6C+",
    "7A",
    "7A+",
    "7B",
    "7B+",
    "7C",
    "7C+",
)


@dataclass(frozen=True, order=True)
class Hold:
    """A hold on the 11-column by 17-row MoonBoard 2016 dataset."""

    x: int
    y: int

    def __post_init__(self) -> None:
        if not 0 <= self.x <= 10:
            raise ValueError(f"Hold x must be between 0 and 10, got {self.x}")
        if not 1 <= self.y <= 17:
            raise ValueError(f"Hold y must be between 1 and 17, got {self.y}")

    @classmethod
    def from_pair(cls, value: Sequence[int]) -> "Hold":
        if len(value) != 2:
            raise ValueError(f"A hold must contain exactly two coordinates: {value!r}")
        return cls(x=int(value[0]), y=int(value[1]))

    @classmethod
    def from_label(cls, value: str) -> "Hold":
        label = value.strip().upper()
        if len(label) < 2 or not label[0].isalpha() or not label[1:].isdigit():
            raise ValueError(f"Invalid hold label: {value!r}")
        return cls(x=ord(label[0]) - ord("A"), y=int(label[1:]))

    @property
    def label(self) -> str:
        return f"{chr(ord('A') + self.x)}{self.y}"

    def as_pair(self) -> list[int]:
        return [self.x, self.y]


@dataclass(frozen=True)
class Route:
    """A route with explicit start, middle, and end semantics."""

    grade: str
    start_holds: tuple[Hold, ...]
    middle_holds: tuple[Hold, ...]
    end_holds: tuple[Hold, ...]

    def __post_init__(self) -> None:
        if self.grade not in SUPPORTED_GRADES:
            raise ValueError(
                f"Unsupported grade {self.grade!r}; expected one of {SUPPORTED_GRADES}"
            )
        if not self.start_holds:
            raise ValueError("A route requires at least one start hold")
        if not self.end_holds:
            raise ValueError("A route requires at least one end hold")

    @classmethod
    def from_pairs(
        cls,
        *,
        grade: str,
        start_holds: Iterable[Sequence[int]],
        middle_holds: Iterable[Sequence[int]],
        end_holds: Iterable[Sequence[int]],
    ) -> "Route":
        return cls(
            grade=grade,
            start_holds=tuple(Hold.from_pair(hold) for hold in start_holds),
            middle_holds=tuple(Hold.from_pair(hold) for hold in middle_holds),
            end_holds=tuple(Hold.from_pair(hold) for hold in end_holds),
        )

    @property
    def holds(self) -> tuple[Hold, ...]:
        return self.start_holds + self.middle_holds + self.end_holds

    def as_pairs(self) -> list[list[int]]:
        return [hold.as_pair() for hold in self.holds]
