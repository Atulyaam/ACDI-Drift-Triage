
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


def _is_strict_int(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
    )


class DetectorState(str, Enum):
    """
    Canonical lifecycle state for one streaming detector.

    DetectorState is intentionally an Enum rather than a free-form
    string so downstream code cannot silently introduce typos.
    """

    ACTIVE = "active"
    DRIFT_DETECTED = "drift_detected"
    LATCHED = "latched"
    RESOLVED = "resolved"
    RESET = "reset"
    UNRESOLVED_TIMEOUT = "unresolved_timeout"


@dataclass(frozen=True)
class PredictionErrorObservation:
    """
    One per-sample prediction-error observation.

    error semantics:
        0 = correct prediction
        1 = incorrect prediction

    Timestamp is optional because detector semantics are based on
    sample_index and deterministic external windows. When timestamps
    are available from the upstream stream, they may be attached as
    metadata without changing detector semantics.

    This is an input observation only. It does not contain detector
    state or triage decisions.
    """

    run_id: str
    sample_index: int
    external_window_id: str
    error: int

    timestamp: datetime | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if not _is_strict_int(
            self.sample_index
        ):
            raise TypeError(
                "sample_index must be an integer "
                "(not bool)."
            )

        if self.sample_index < 0:
            raise ValueError(
                "sample_index cannot be negative."
            )

        if (
            not isinstance(
                self.external_window_id,
                str,
            )
            or not self.external_window_id.strip()
        ):
            raise ValueError(
                "external_window_id must be a "
                "non-empty string."
            )

        if not _is_strict_int(
            self.error
        ):
            raise TypeError(
                "error must be a strict integer 0 or 1."
            )

        if self.error not in (0, 1):
            raise ValueError(
                "error must be either 0 or 1."
            )

        if (
            self.timestamp is not None
            and not isinstance(
                self.timestamp,
                datetime,
            )
        ):
            raise TypeError(
                "timestamp must be a datetime or None."
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary."
            )

        if not all(
            isinstance(key, str)
            for key in self.metadata
        ):
            raise TypeError(
                "metadata keys must all be strings."
            )


@dataclass(frozen=True)
class DetectorUpdateResult:
    """
    Result of one detector update.

    This is adapter-level state/reporting data.

    detection:
        True only when this update constitutes a detector
        drift signal.

    state:
        Explicit DetectorState enum.

    The result does not contain raw River detector objects.
    """

    detector_name: str
    run_id: str

    sample_index: int
    reported_window_id: str

    detection: bool
    state: DetectorState

    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.detector_name,
                str,
            )
            or not self.detector_name.strip()
        ):
            raise ValueError(
                "detector_name must be a non-empty string."
            )

        if (
            not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if not _is_strict_int(
            self.sample_index
        ):
            raise TypeError(
                "sample_index must be an integer "
                "(not bool)."
            )

        if self.sample_index < 0:
            raise ValueError(
                "sample_index cannot be negative."
            )

        if (
            not isinstance(
                self.reported_window_id,
                str,
            )
            or not self.reported_window_id.strip()
        ):
            raise ValueError(
                "reported_window_id must be a "
                "non-empty string."
            )

        if not isinstance(
            self.detection,
            bool,
        ):
            raise TypeError(
                "detection must be a boolean."
            )

        if not isinstance(
            self.state,
            DetectorState,
        ):
            raise TypeError(
                "state must be a DetectorState."
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary."
            )

        if not all(
            isinstance(key, str)
            for key in self.metadata
        ):
            raise TypeError(
                "metadata keys must all be strings."
            )


class DetectorFactory:
    """
    Run-isolated factory for streaming detector adapters.

    The concrete detector implementation is intentionally deferred.
    This class currently defines the uniqueness/lifecycle contract.

    A run_id can be issued only once per factory instance.
    Reusing a run_id is rejected explicitly.
    """

    def __init__(self) -> None:
        self._issued_run_ids: set[str] = set()

    def reserve_run_id(
        self,
        run_id: str,
    ) -> str:
        """
        Reserve a run_id exactly once.

        Concrete adapter construction will be added when the
        ADWIN/DDM/Page-Hinkley adapters are implemented.
        """

        if (
            not isinstance(run_id, str)
            or not run_id.strip()
        ):
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if run_id in self._issued_run_ids:
            raise ValueError(
                f"run_id {run_id!r} has already "
                "been issued by this factory."
            )

        self._issued_run_ids.add(run_id)

        return run_id

    @property
    def issued_run_ids(self) -> frozenset[str]:
        return frozenset(
            self._issued_run_ids
        )
