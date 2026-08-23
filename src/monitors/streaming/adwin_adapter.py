
from __future__ import annotations

import river
from river import drift

from src.monitors.streaming.adapter import (
    StreamingDetectorAdapter,
)

from src.monitors.streaming.configs import (
    ADWINConfig,
)

from src.monitors.streaming.error_contracts import (
    DetectorState,
    DetectorUpdateResult,
    PredictionErrorObservation,
)


_RIVER_VERSION = river.__version__


class ADWINAdapter(StreamingDetectorAdapter):
    """
    ADWIN adapter verified against River 0.23.0.

    Trigger sample:
        result.state = DRIFT_DETECTED
        adapter state -> LATCHED (applied AFTER the result
        object is built, so the trigger sample's own result
        always reports DRIFT_DETECTED, never LATCHED).

    While LATCHED:
        River keeps receiving samples.
        Repeated River alarms are ignored for lifecycle voting.

    New external window while LATCHED:
        unresolved timeout
        -> fresh River detector
        -> same triggering observation processed once

    reset():
        fresh River detector
        lifecycle -> ACTIVE
        last_sample_index preserved
    """

    def __init__(
        self,
        run_id: str,
        config: ADWINConfig | None = None,
    ) -> None:
        super().__init__(
            run_id=run_id,
            detector_name="ADWIN",
        )

        self._config = (
            config
            if config is not None
            else ADWINConfig()
        )

        self._river_detector = (
            self._build_river_detector()
        )

        self._require_reset_compatibility()

    @property
    def config(self) -> ADWINConfig:
        return self._config

    def _build_river_detector(self):
        return drift.ADWIN(
            delta=self._config.delta,
            clock=self._config.clock,
            max_buckets=self._config.max_buckets,
            min_window_length=self._config.min_window_length,
            grace_period=self._config.grace_period,
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
                "callable ADWIN._reset()."
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

            # IMPORTANT:
            # fresh detector BEFORE processing the triggering sample
            self._river_detector = (
                self._build_river_detector()
            )

            self._mark_reset()
            self._mark_active()

        # Always feed River.
        self._river_detector.update(
            observation.error
        )

        raw_river_drift_detected = bool(
            self._river_detector.drift_detected
        )

        detection = False
        reported_state = self.state
        trigger_detected = False

        if self.state is DetectorState.ACTIVE:

            if raw_river_drift_detected:
                detection = True
                trigger_detected = True

                # Transient result state — hardcoded literal,
                # NOT a re-read of self.state, so this can never
                # race with the persistent lifecycle transition.
                reported_state = (
                    DetectorState.DRIFT_DETECTED
                )

        elif self.state is DetectorState.LATCHED:

            # Ignore repeated River alarms for our lifecycle.
            detection = False
            reported_state = (
                DetectorState.LATCHED
            )

        metadata = {
            "raw_river_drift_detected": (
                raw_river_drift_detected
            ),
            "river_version": _RIVER_VERSION,
        }

        if timeout_triggered:
            metadata[
                "unresolved_timeout"
            ] = True

        result = DetectorUpdateResult(
            detector_name=self.detector_name,
            run_id=self.run_id,
            sample_index=observation.sample_index,
            reported_window_id=observation.external_window_id,
            detection=detection,
            state=reported_state,
            metadata=metadata,
        )

        # Persistent lifecycle transition happens AFTER the
        # transient result has already been captured/built.
        if trigger_detected:
            self._enter_latched(
                observation.external_window_id
            )

        self._record_sample_index(
            observation
        )

        return result

    def reset(self) -> None:
        # Lifecycle reset = brand-new River detector.
        self._river_detector = (
            self._build_river_detector()
        )

        self._mark_reset()
        self._mark_active()

        # DO NOT clear last_sample_index.
