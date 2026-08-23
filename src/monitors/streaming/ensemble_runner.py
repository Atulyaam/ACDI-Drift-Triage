from __future__ import annotations

from src.monitors.streaming.adapter import (
    StreamingDetectorAdapter,
)

from src.monitors.streaming.error_contracts import (
    DetectorState,
    DetectorUpdateResult,
    PredictionErrorObservation,
)

from src.monitors.streaming.error_vote import (
    ErrorDriftVoteResult,
    compute_error_drift_vote,
)


_ADWIN_NAME = "ADWIN"
_DDM_NAME = "DDM"
_PAGE_HINKLEY_NAME = "PAGE_HINKLEY"


class StreamingEnsembleRunner:
    """
    Orchestrate the three streaming detectors at sample level,
    while producing the canonical error consensus at external
    window level.

    IMPORTANT ARCHITECTURAL BOUNDARY:

    Detector adapters:
        operate sample-by-sample.

    ErrorDriftVoteResult:
        represents one same-sample consensus input.

    This runner:
        accumulates whether each detector triggered AT LEAST
        ONCE during the current external window, then at window
        close creates synthetic, aligned DetectorUpdateResult
        objects using the window's final sample index and calls
        the existing compute_error_drift_vote() unchanged.

    Existing detector contracts are not modified.

    The runner NEVER automatically calls adapter.reset().
    Detector lifecycle/reset semantics remain owned by the
    individual adapters / future triage layer.
    """

    def __init__(
        self,
        run_id: str,
        adwin_adapter: StreamingDetectorAdapter,
        ddm_adapter: StreamingDetectorAdapter,
        page_hinkley_adapter: StreamingDetectorAdapter,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string.")

        self._validate_adapter(adwin_adapter, _ADWIN_NAME, "adwin_adapter")
        self._validate_adapter(ddm_adapter, _DDM_NAME, "ddm_adapter")
        self._validate_adapter(
            page_hinkley_adapter, _PAGE_HINKLEY_NAME, "page_hinkley_adapter"
        )

        adapter_run_ids = {
            adwin_adapter.run_id,
            ddm_adapter.run_id,
            page_hinkley_adapter.run_id,
        }

        if adapter_run_ids != {run_id}:
            raise ValueError(
                "All detector adapters must use the same run_id as "
                "the runner."
            )

        self._run_id = run_id

        self._adwin_adapter = adwin_adapter
        self._ddm_adapter = ddm_adapter
        self._page_hinkley_adapter = page_hinkley_adapter

        self._current_window_id: str | None = None
        self._last_sample_index: int | None = None

        self._adwin_triggered = False
        self._ddm_triggered = False
        self._page_hinkley_triggered = False

    @staticmethod
    def _validate_adapter(
        adapter: StreamingDetectorAdapter,
        expected_name: str,
        parameter_name: str,
    ) -> None:
        if not isinstance(adapter, StreamingDetectorAdapter):
            raise TypeError(
                f"{parameter_name} must be a StreamingDetectorAdapter."
            )

        if adapter.detector_name != expected_name:
            raise ValueError(
                f"{parameter_name} must wrap detector {expected_name!r}; "
                f"received {adapter.detector_name!r}."
            )

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def current_window_id(self) -> str | None:
        return self._current_window_id

    @property
    def last_sample_index(self) -> int | None:
        return self._last_sample_index

    @property
    def adwin_triggered(self) -> bool:
        return self._adwin_triggered

    @property
    def ddm_triggered(self) -> bool:
        return self._ddm_triggered

    @property
    def page_hinkley_triggered(self) -> bool:
        return self._page_hinkley_triggered

    def process_observation(
        self,
        observation: PredictionErrorObservation,
    ) -> tuple[
        DetectorUpdateResult,
        DetectorUpdateResult,
        DetectorUpdateResult,
    ]:
        """
        Feed one observation to all three adapters.

        All three adapters receive the exact same observation.

        The current external window is established by the first
        observation. A window change requires the caller to close
        the current window explicitly before processing the next
        window.
        """

        if not isinstance(observation, PredictionErrorObservation):
            raise TypeError(
                "observation must be a PredictionErrorObservation."
            )

        if observation.run_id != self._run_id:
            raise ValueError(
                "observation.run_id must match runner.run_id."
            )

        if self._current_window_id is None:
            self._current_window_id = observation.external_window_id

        elif observation.external_window_id != self._current_window_id:
            raise ValueError(
                "A new external window was received before the "
                "current window was closed."
            )

        adwin_result = self._adwin_adapter.update(observation)
        ddm_result = self._ddm_adapter.update(observation)
        page_hinkley_result = self._page_hinkley_adapter.update(observation)

        # Window-level trigger accumulation:
        # a detector is considered triggered for this window if
        # it produced detection=True at ANY sample in the window.
        self._adwin_triggered = (
            self._adwin_triggered or adwin_result.detection
        )

        self._ddm_triggered = (
            self._ddm_triggered or ddm_result.detection
        )

        self._page_hinkley_triggered = (
            self._page_hinkley_triggered or page_hinkley_result.detection
        )

        self._last_sample_index = observation.sample_index

        return (adwin_result, ddm_result, page_hinkley_result)

    def close_window(self) -> ErrorDriftVoteResult:
        """
        Close the current external window and produce exactly one
        canonical 2-of-3 consensus result.

        The detector-level results created here are intentionally
        synthetic window-level projections. They reuse the existing
        DetectorUpdateResult contract so compute_error_drift_vote()
        remains completely unchanged.

        No detector.reset() is called here.
        """

        if self._current_window_id is None:
            raise ValueError("No active external window to close.")

        if self._last_sample_index is None:
            raise ValueError(
                "Cannot close an external window before processing "
                "at least one observation."
            )

        window_id = self._current_window_id
        sample_index = self._last_sample_index

        adwin_result = self._build_window_result(
            detector_name=_ADWIN_NAME,
            detection=self._adwin_triggered,
            sample_index=sample_index,
            window_id=window_id,
        )

        ddm_result = self._build_window_result(
            detector_name=_DDM_NAME,
            detection=self._ddm_triggered,
            sample_index=sample_index,
            window_id=window_id,
        )

        page_hinkley_result = self._build_window_result(
            detector_name=_PAGE_HINKLEY_NAME,
            detection=self._page_hinkley_triggered,
            sample_index=sample_index,
            window_id=window_id,
        )

        vote = compute_error_drift_vote(
            adwin_result, ddm_result, page_hinkley_result
        )

        # Start a fresh window-level aggregation bucket.
        #
        # IMPORTANT:
        # detector adapter lifecycle is NOT reset here.
        self._current_window_id = None
        self._last_sample_index = None

        self._adwin_triggered = False
        self._ddm_triggered = False
        self._page_hinkley_triggered = False

        return vote

    def _build_window_result(
        self,
        detector_name: str,
        detection: bool,
        sample_index: int,
        window_id: str,
    ) -> DetectorUpdateResult:
        return DetectorUpdateResult(
            detector_name=detector_name,
            run_id=self._run_id,
            sample_index=sample_index,
            reported_window_id=window_id,
            detection=bool(detection),
            state=(
                DetectorState.DRIFT_DETECTED
                if detection
                else DetectorState.ACTIVE
            ),
            metadata={
                "source": "window_level_aggregation",
                "triggered_this_window": bool(detection),
            },
        )
