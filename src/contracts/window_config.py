from __future__ import annotations

from dataclasses import dataclass
import re


_WINDOW_ID_PREFIX_PATTERN = re.compile(
    r"^[A-Za-z0-9_]+$"
)


def _is_strict_int(value: object) -> bool:
    """
    True only for real integers.

    bool is a subclass of int in Python, so bool values
    are explicitly rejected from integer configuration fields.
    """
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
    )


@dataclass(frozen=True)
class WindowManagerConfig:
    """
    Immutable configuration for deterministic external windows.
    """

    run_id: str
    window_size: int
    total_samples: int
    start_index: int
    window_id_prefix: str
    drop_last_partial: bool = False

    def __post_init__(self) -> None:

        if (
            not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if not _is_strict_int(self.window_size):
            raise TypeError(
                "window_size must be an integer (not a bool)."
            )

        if self.window_size <= 0:
            raise ValueError(
                "window_size must be greater than zero."
            )

        if not _is_strict_int(self.total_samples):
            raise TypeError(
                "total_samples must be an integer (not a bool)."
            )

        if self.total_samples <= 0:
            raise ValueError(
                "total_samples must be greater than zero."
            )

        if not _is_strict_int(self.start_index):
            raise TypeError(
                "start_index must be an integer (not a bool)."
            )

        if self.start_index < 0:
            raise ValueError(
                "start_index cannot be negative."
            )

        if self.start_index >= self.total_samples:
            raise ValueError(
                "start_index must be less than total_samples."
            )

        if (
            not isinstance(self.window_id_prefix, str)
            or not self.window_id_prefix.strip()
        ):
            raise ValueError(
                "window_id_prefix must be a non-empty string."
            )

        if not _WINDOW_ID_PREFIX_PATTERN.fullmatch(
            self.window_id_prefix
        ):
            raise ValueError(
                "window_id_prefix may contain only "
                "letters, digits, and underscores."
            )

        if not isinstance(
            self.drop_last_partial,
            bool
        ):
            raise TypeError(
                "drop_last_partial must be a boolean."
            )
