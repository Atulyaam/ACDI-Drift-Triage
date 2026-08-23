
from __future__ import annotations

import river
from river import drift

from src.monitors.streaming.adapter import (
    StreamingDetectorAdapter,
)

from src.monitors.streaming.configs import (
    DDMConfig,
)

from src.monitors.streaming.error_contracts import (
    DetectorState,
    DetectorUpdateResult,
    PredictionErrorObservation,
)


_RIVER_VERSION = river.__version__


class DDMAdapter(StreamingDetectorAdapter):
    """
    River DDM adapter verified against River 0.23.0.

    DDM exposes two independent River signals:

        warning_detected
        drift_detected

    Only drift_detected participates in our lifecycle.
    warning_detected is diagnostic metadata only.

    Trigger sample:
        result.state = DRIFT_DETECTED
        adapter state = LATCHED (transition happens via
        _enter_latched() ONLY, mirroring the exact ADWIN
        structure -- no other lifecycle-state-mutating call
        happens on this path, so there is exactly one
        persistent state transition, not two).

    While LATCHED:
        - continue feeding DDM
        - capture warning/drift metadata on every update
        - ignore repeated drift alarms for lifecycle
        - detection=False
        - state=LATCHED

    Timeout:
        new external window while LATCHED
        -> unresolved_timeout
        -> fresh DDM instance
        -> same observation processed once
    """

    def __init__(
        self,
        run_id: str,
        config: DDMConfig | None = None,
    ) -> None:
        super().__init__(
            run_id=run_id,
            detector_name="DDM",
        )

        self._config = (
            config
            if config is not None
            else DDMConfig()
        )

        self._river_detector = (
            self._build_river_detector()
        )

        self._require_reset_compatibility()

    @property
    def config(self) -> DDMConfig:
        return self._config

    def _build_river_detector(self):
        """
        Explicit config -> River DDM mapping.
        """
        return drift.binary.DDM(
            warm_start=self._config.warm_start,
            warning_threshold=(
                self._config.warning_threshold
            ),
            drift_threshold=(
                self._config.drift_threshold
            ),
        )

    def _require_reset_compatibility(self) -> None:
        reset_method = getattr(
            self._river_detector,
            "_reset",
            None,
        )

        if not callable(reset_method):
            raise RuntimeError(
                "Installed River version "
                f"{_RIVER_VERSION!r} does not expose "
                "callable DDM._reset()."
            )

    def update(
        self,
        observation: PredictionErrorObservation,
    ) -> DetectorUpdateResult:

        self._validate_observation(
            observation
        )

        timeout_triggered = (
            self._is_latched_into_new_window(
                observation.external_window_id
            )
        )

        if timeout_triggered:
            self._mark_unresolved_timeout()

            # Fresh DDM before processing the triggering sample.
            self._river_detector = (
                self._build_river_detector()
            )

            self._mark_reset()
            self._mark_active()

        # DDM consumes every sample, including while LATCHED.
        self._river_detector.update(
            observation.error
        )

        raw_river_drift_detected = bool(
            self._river_detector.drift_detected
        )

        warning_detected = bool(
            self._river_detector.warning_detected
        )

        # ----------------------------------------------------
        # Metadata is deliberately built independently from
        # lifecycle branching so warning_detected can NEVER
        # accidentally disappear on one path.
        # ----------------------------------------------------

        metadata = {
            "raw_river_drift_detected":
                raw_river_drift_detected,
            "warning_detected":
                warning_detected,
            "river_version":
                _RIVER_VERSION,
        }

        if timeout_triggered:
            metadata[
                "unresolved_timeout"
            ] = True

        detection = False
        reported_state = self.state
        trigger_detected = False

        if self.state is DetectorState.ACTIVE:

            if raw_river_drift_detected:
                detection = True
                trigger_detected = True

                # IMPORTANT: this branch intentionally does NOT
                # call any state-mutating lifecycle helper here.
                # This exactly mirrors the fixed ADWIN structure.
                # reported_state is a hardcoded literal, and the
                # ONLY persistent state transition happens later
                # via _enter_latched(), once, after the result
                # object is already built.
                reported_state = (
                    DetectorState.DRIFT_DETECTED
                )

        elif self.state is DetectorState.LATCHED:

            # Repeated drift alarms are diagnostics only.
            detection = False
            reported_state = (
                DetectorState.LATCHED
            )

        result = DetectorUpdateResult(
            detector_name=self.detector_name,
            run_id=self.run_id,
            sample_index=(
                observation.sample_index
            ),
            reported_window_id=(
                observation.external_window_id
            ),
            detection=detection,
            state=reported_state,
            metadata=metadata,
        )

        if trigger_detected:
            self._enter_latched(
                observation.external_window_id
            )

        self._record_sample_index(
            observation
        )

        return result

    def reset(self) -> None:
        """
        Lifecycle reset = fresh DDM instance.

        last_sample_index is intentionally preserved.
        """
        self._river_detector = (
            self._build_river_detector()
        )

        self._mark_reset()
        self._mark_active()
