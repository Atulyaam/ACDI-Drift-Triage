from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class WindowContext:
    """
    Canonical external experiment-window description.

    This represents the external reporting/traceability window.
    It is distinct from internal adaptive state maintained by
    streaming detectors such as ADWIN.
    """

    run_id: str
    window_id: str

    start_index: int
    end_index: int

    start_timestamp: datetime
    end_timestamp: datetime

    sample_count: int

    # metadata excluded from eq/hash: dicts are unhashable, and frozen
    # dataclasses generate __hash__ from compared fields by default.
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if not isinstance(self.window_id, str) or not self.window_id.strip():
            raise ValueError(
                "window_id must be a non-empty string."
            )

        # bool is a subclass of int in Python (isinstance(True, int) is
        # True), so we must explicitly exclude bool to avoid silently
        # accepting True/False as valid indices/counts.
        if not isinstance(self.start_index, int) or isinstance(self.start_index, bool):
            raise TypeError(
                "start_index must be an integer."
            )

        if not isinstance(self.end_index, int) or isinstance(self.end_index, bool):
            raise TypeError(
                "end_index must be an integer."
            )

        if self.start_index < 0:
            raise ValueError(
                "start_index cannot be negative."
            )

        if self.end_index < self.start_index:
            raise ValueError(
                "end_index must be greater than or equal to start_index."
            )

        if not isinstance(self.sample_count, int) or isinstance(self.sample_count, bool):
            raise TypeError(
                "sample_count must be an integer."
            )

        if self.sample_count < 0:
            raise ValueError(
                "sample_count cannot be negative."
            )

        expected_count = (
            self.end_index - self.start_index + 1
        )

        if self.sample_count != expected_count:
            raise ValueError(
                "sample_count must equal "
                "end_index - start_index + 1."
            )

        if not isinstance(self.start_timestamp, datetime):
            raise TypeError(
                "start_timestamp must be a datetime instance."
            )

        if not isinstance(self.end_timestamp, datetime):
            raise TypeError(
                "end_timestamp must be a datetime instance."
            )

        if self.start_timestamp.tzinfo is None:
            raise ValueError(
                "start_timestamp must be timezone-aware."
            )

        if self.end_timestamp.tzinfo is None:
            raise ValueError(
                "end_timestamp must be timezone-aware."
            )

        if self.end_timestamp < self.start_timestamp:
            raise ValueError(
                "end_timestamp cannot be earlier than start_timestamp."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        if not all(isinstance(k, str) for k in self.metadata.keys()):
            raise TypeError(
                "metadata keys must all be strings."
            )
