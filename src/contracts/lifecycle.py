from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


DetectorState = Literal[
    "ACTIVE",
    "DRIFT_PENDING",
]

ResolutionStatus = Literal[
    "UNRESOLVED",
    "RESOLVED",
    "UNRESOLVED_TIMEOUT",
]


@dataclass(frozen=True)
class StatefulDetectorLifecycle:
    """
    Lifecycle contract for stateful drift detectors.

    This contract governs detector lifecycle and auditability.
    The underlying River detector remains responsible for its
    algorithmic state.
    """

    run_id: str
    detector_name: str
    detector_instance_id: str

    state: DetectorState

    first_detection_index: int | None
    reported_window_id: str | None
    latch_window_id: str | None

    resolution_status: ResolutionStatus

    resolution_window_id: str | None

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

        if self.state not in (
            "ACTIVE",
            "DRIFT_PENDING",
        ):
            raise ValueError(
                "state must be 'ACTIVE' or 'DRIFT_PENDING'."
            )

        if self.resolution_status not in (
            "UNRESOLVED",
            "RESOLVED",
            "UNRESOLVED_TIMEOUT",
        ):
            raise ValueError(
                "Invalid resolution_status."
            )

        if self.first_detection_index is not None:
            # bool is a subclass of int; exclude it explicitly so
            # True/False can never silently pass as a sample index.
            if (
                not isinstance(self.first_detection_index, int)
                or isinstance(self.first_detection_index, bool)
            ):
                raise TypeError(
                    "first_detection_index must be an integer or None."
                )

            if self.first_detection_index < 0:
                raise ValueError(
                    "first_detection_index cannot be negative."
                )

        if self.state == "DRIFT_PENDING":
            if self.first_detection_index is None:
                raise ValueError(
                    "DRIFT_PENDING requires first_detection_index."
                )

            if not self.reported_window_id:
                raise ValueError(
                    "DRIFT_PENDING requires reported_window_id."
                )

            if not self.latch_window_id:
                raise ValueError(
                    "DRIFT_PENDING requires latch_window_id."
                )

            if self.resolution_status != "UNRESOLVED":
                raise ValueError(
                    "DRIFT_PENDING must have UNRESOLVED status."
                )

        if self.resolution_status == "RESOLVED":
            if self.state != "ACTIVE":
                raise ValueError(
                    "RESOLVED lifecycle must be ACTIVE."
                )

            if not self.resolution_window_id:
                raise ValueError(
                    "RESOLVED lifecycle requires resolution_window_id."
                )

        if (
            self.resolution_status == "UNRESOLVED_TIMEOUT"
            and self.state != "ACTIVE"
        ):
            raise ValueError(
                "UNRESOLVED_TIMEOUT lifecycle must be ACTIVE "
                "after forced reset."
            )

        # A RESOLVED or UNRESOLVED_TIMEOUT event must still carry the
        # historical record of WHICH drift event was resolved/timed
        # out. Without this, the audit trail loses traceability
        # (architecture Section 13: every result must remain
        # traceable to a window/detection event).
        if self.resolution_status in ("RESOLVED", "UNRESOLVED_TIMEOUT"):
            if self.first_detection_index is None:
                raise ValueError(
                    f"{self.resolution_status} requires "
                    "first_detection_index to preserve the historical "
                    "record of the resolved drift event."
                )

            if not self.reported_window_id:
                raise ValueError(
                    f"{self.resolution_status} requires "
                    "reported_window_id to preserve traceability."
                )

            if not self.latch_window_id:
                raise ValueError(
                    f"{self.resolution_status} requires "
                    "latch_window_id to preserve traceability."
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
