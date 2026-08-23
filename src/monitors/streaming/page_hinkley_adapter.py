
from __future__ import annotations

import river
from river import drift

from src.monitors.streaming.adapter import (
    StreamingDetectorAdapter,
)

from src.monitors.streaming.configs import (
    PageHinkleyConfig,
)

from src.monitors.streaming.error_contracts import (
    DetectorState,
    DetectorUpdateResult,
    PredictionErrorObservation,
)


_RIVER_VERSION = river.__version__


class PageHinkleyAdapter(StreamingDetectorAdapter):
    """
    Page-Hinkley adapter verified against River 0.23.0.

    Page-Hinkley exposes only drift detection.
    It has no warning signal.

    Trigger sample:
        result.state = DRIFT_DETECTED
        adapter state = LATCHED (single persistent transition,
        applied via _enter_latched() only, after the result
        object is already built -- matching the exact structure
        proven correct for ADWIN and DDM. No other lifecycle-
        state-mutating call happens on this path.)

    While LATCHED:
        - continue feeding River
        - repeated River alarms are ignored
        - detection=False
        - state=LATCHED

    New external window while LATCHED:
        - unresolved timeout
        - create fresh River detector
        - process the same triggering observation once

    reset():
        - create fresh Page-Hinkley instance
        - lifecycle -> ACTIVE
        - preserve last_sample_index

    Important:
    River's own adaptive/statistical state is distinct from
    the adapter lifecycle state.
    """

    def __init__(
        self,
        run_id: str,
        config: PageHinkleyConfig | None = None,
    ) -> None:
        super().__init__(
            run_id=run_id,
            detector_name="PAGE_HINKLEY",
        )

        self._config = (
            config
            if config is not None
            else PageHinkleyConfig()
        )

        self._river_detector = (
            self._build_river_detector()
        )

        self._require_reset_compatibility()

    @property
    def config(self) -> PageHinkleyConfig:
        return self._config

    def _build_river_detector(self):
        """
        Explicit config -> River Page-Hinkley mapping.
        """
        return drift.PageHinkley(
            min_instances=(
                self._config.min_instances
            ),
            delta=self._config.delta,
            threshold=self._config.threshold,
            alpha=self._config.alpha,
            mode=self._config.mode,
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
                "callable PageHinkley._reset()."
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

            # Fresh detector BEFORE processing the
            # timeout-triggering observation.
            self._river_detector = (
                self._build_river_detector()
            )

            self._mark_reset()
            self._mark_active()

        # Page-Hinkley receives every valid observation.
        self._river_detector.update(
            observation.error
        )

        raw_river_drift_detected = bool(
            self._river_detector.drift_detected
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

        detection = False
        reported_state = self.state
        trigger_detected = False

        if self.state is DetectorState.ACTIVE:

            if raw_river_drift_detected:
                detection = True
                trigger_detected = True

                # No lifecycle-state-mutating call happens here.
                # reported_state is a hardcoded literal. The only
                # persistent state transition happens later, once,
                # via _enter_latched() -- exactly matching the
                # proven ADWIN/DDM structure.
                reported_state = (
                    DetectorState.DRIFT_DETECTED
                )

        elif self.state is DetectorState.LATCHED:

            # Repeated River alarms are diagnostic only.
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
        Lifecycle reset = fresh Page-Hinkley instance.

        last_sample_index is intentionally preserved.
        """
        self._river_detector = (
            self._build_river_detector()
        )

        self._mark_reset()
        self._mark_active()
