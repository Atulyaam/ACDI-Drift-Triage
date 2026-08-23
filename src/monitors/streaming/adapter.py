
from __future__ import annotations

from abc import ABC, abstractmethod

from src.monitors.streaming.error_contracts import (
    DetectorState,
    DetectorUpdateResult,
    PredictionErrorObservation,
)


class StreamingDetectorAdapter(ABC):
    """
    Common contract for ADWIN, DDM, and Page-Hinkley adapters.

    The adapter owns:
    - one detector instance
    - one run_id
    - lifecycle state
    - last accepted sample index
    - latch window identity

    Concrete adapters own the River-specific implementation.

    Important lifecycle rules:
    - update() processes exactly one observation.
    - River continues receiving samples while LATCHED.
    - River drift flags are ignored for lifecycle transitions
      after the initial latch.
    - reset() clears detector/lifecycle state but NOT
      last_sample_index.
    - a new external window while LATCHED causes
      UNRESOLVED_TIMEOUT, forced reset, then the triggering
      observation is processed normally.
    """

    def __init__(
        self,
        run_id: str,
        detector_name: str,
    ) -> None:
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
        ):
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if (
            not isinstance(detector_name, str)
            or not detector_name.strip()
        ):
            raise ValueError(
                "detector_name must be a non-empty string."
            )

        self._run_id = run_id
        self._detector_name = detector_name

        self._state = DetectorState.ACTIVE
        self._last_sample_index: int | None = None
        self._latch_window_id: str | None = None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def detector_name(self) -> str:
        return self._detector_name

    @property
    def state(self) -> DetectorState:
        return self._state

    @property
    def last_sample_index(self) -> int | None:
        return self._last_sample_index

    @property
    def latch_window_id(self) -> str | None:
        return self._latch_window_id

    @abstractmethod
    def update(
        self,
        observation: PredictionErrorObservation,
    ) -> DetectorUpdateResult:
        """
        Process exactly one prediction-error observation.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Explicitly reset detector and lifecycle state.

        Implementations must preserve _last_sample_index.
        """

    def _validate_observation(
        self,
        observation: PredictionErrorObservation,
    ) -> None:
        if not isinstance(
            observation,
            PredictionErrorObservation,
        ):
            raise TypeError(
                "observation must be a "
                "PredictionErrorObservation."
            )

        if observation.run_id != self._run_id:
            raise ValueError(
                "observation.run_id must match "
                "adapter.run_id."
            )

        if (
            self._last_sample_index is not None
            and observation.sample_index
            <= self._last_sample_index
        ):
            raise ValueError(
                "sample_index must be strictly increasing."
            )

    def _record_sample_index(
        self,
        observation: PredictionErrorObservation,
    ) -> None:
        self._last_sample_index = (
            observation.sample_index
        )

    def _enter_latched(
        self,
        window_id: str,
    ) -> None:
        self._state = DetectorState.LATCHED
        self._latch_window_id = window_id

    def _mark_drift_detected(self) -> None:
        self._state = DetectorState.DRIFT_DETECTED

    def _mark_resolved(self) -> None:
        self._state = DetectorState.RESOLVED

    def _mark_reset(self) -> None:
        self._state = DetectorState.RESET
        self._latch_window_id = None

    def _mark_active(self) -> None:
        self._state = DetectorState.ACTIVE

    def _mark_unresolved_timeout(self) -> None:
        self._state = DetectorState.UNRESOLVED_TIMEOUT

    def _is_latched_into_new_window(
        self,
        window_id: str,
    ) -> bool:
        return (
            self._state
            is DetectorState.LATCHED
            and self._latch_window_id is not None
            and window_id != self._latch_window_id
        )
