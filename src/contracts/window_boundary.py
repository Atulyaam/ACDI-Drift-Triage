from __future__ import annotations

from dataclasses import dataclass


def _is_strict_int(value: object) -> bool:
    """
    True only for real integers.

    bool is a subclass of int in Python, so it must be
    explicitly rejected.
    """
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
    )


@dataclass(frozen=True)
class WindowBoundary:
    """
    Pure index-based description of one deterministic
    external experiment window.

    This contract intentionally contains no timestamps.

    WindowManager knows sample indices only. Timestamp
    information is attached later by the data/stream layer.
    """

    run_id: str
    window_id: str

    start_index: int
    end_index: int
    sample_count: int

    internal_key: str
    is_partial: bool

    def __post_init__(self) -> None:

        if (
            not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if (
            not isinstance(self.window_id, str)
            or not self.window_id.strip()
        ):
            raise ValueError(
                "window_id must be a non-empty string."
            )

        if not _is_strict_int(self.start_index):
            raise TypeError(
                "start_index must be an integer (not a bool)."
            )

        if not _is_strict_int(self.end_index):
            raise TypeError(
                "end_index must be an integer (not a bool)."
            )

        if self.start_index < 0:
            raise ValueError(
                "start_index cannot be negative."
            )

        if self.end_index < self.start_index:
            raise ValueError(
                "end_index cannot be before start_index."
            )

        if not _is_strict_int(self.sample_count):
            raise TypeError(
                "sample_count must be an integer (not a bool)."
            )

        if self.sample_count <= 0:
            raise ValueError(
                "sample_count must be greater than zero."
            )

        expected_count = (
            self.end_index
            - self.start_index
            + 1
        )

        if self.sample_count != expected_count:
            raise ValueError(
                "sample_count is inconsistent with "
                "start_index/end_index: "
                f"expected {expected_count}, "
                f"got {self.sample_count}."
            )

        if (
            not isinstance(self.internal_key, str)
            or not self.internal_key.strip()
        ):
            raise ValueError(
                "internal_key must be a non-empty string."
            )

        expected_key = (
            f"{self.run_id}::{self.window_id}"
        )

        if self.internal_key != expected_key:
            raise ValueError(
                "internal_key must equal "
                "'run_id::window_id'. "
                f"Expected {expected_key!r}, "
                f"got {self.internal_key!r}."
            )

        if not isinstance(self.is_partial, bool):
            raise TypeError(
                "is_partial must be a boolean."
            )
