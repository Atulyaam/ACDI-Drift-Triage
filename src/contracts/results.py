from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DetectorResult:
    """
    Lightweight standardized detector result.

    Raw observations are intentionally not stored here.
    """

    run_id: str
    detector_name: str
    detector_instance_id: str

    reported_window_id: str

    detection_index: int | None

    drift_detected: bool

    score: float | None
    criterion: str | None

    observation_count: int

    timestamp: datetime

    # metadata excluded from eq/hash: dicts are unhashable, and frozen
    # dataclasses generate __hash__ from compared fields by default.
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if (
            not isinstance(self.detector_name, str)
            or not self.detector_name.strip()
        ):
            raise ValueError(
                "detector_name must be a non-empty string."
            )

        if (
            not isinstance(self.detector_instance_id, str)
            or not self.detector_instance_id.strip()
        ):
            raise ValueError(
                "detector_instance_id must be a non-empty string."
            )

        if (
            not isinstance(self.reported_window_id, str)
            or not self.reported_window_id.strip()
        ):
            raise ValueError(
                "reported_window_id must be a non-empty string."
            )

        if self.detection_index is not None:
            # bool is a subclass of int; exclude it explicitly so
            # True/False can never silently pass as a sample index.
            if (
                not isinstance(self.detection_index, int)
                or isinstance(self.detection_index, bool)
            ):
                raise TypeError(
                    "detection_index must be an integer or None."
                )

            if self.detection_index < 0:
                raise ValueError(
                    "detection_index cannot be negative."
                )

        if not isinstance(self.drift_detected, bool):
            raise TypeError(
                "drift_detected must be a boolean."
            )

        if self.score is not None:
            # bool is a subclass of int/float-compatible via int, so
            # isinstance(True, (int, float)) is True. Exclude it
            # explicitly to avoid a stray boolean silently becoming
            # a "valid" numeric score.
            if (
                not isinstance(self.score, (int, float))
                or isinstance(self.score, bool)
            ):
                raise TypeError(
                    "score must be numeric or None."
                )

        if self.criterion is not None:
            if not isinstance(self.criterion, str):
                raise TypeError(
                    "criterion must be a string or None."
                )

        if (
            not isinstance(self.observation_count, int)
            or isinstance(self.observation_count, bool)
        ):
            raise TypeError(
                "observation_count must be an integer."
            )

        if self.observation_count < 0:
            raise ValueError(
                "observation_count cannot be negative."
            )

        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                "timestamp must be a datetime instance."
            )

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        if not all(isinstance(k, str) for k in self.metadata.keys()):
            raise TypeError(
                "metadata keys must all be strings."
            )
