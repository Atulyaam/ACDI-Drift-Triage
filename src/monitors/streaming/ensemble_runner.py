
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

from src.windows.manager import WindowManager


_ADWIN_NAME = "ADWIN"
_DDM_NAME = "DDM"
_PAGE_HINKLEY_NAME = "PAGE_HINKLEY"


class StreamingEnsembleRunner:
    """
    Orchestrate the three streaming detectors at sample level,
    while producing the canonical error consensus at external
    window level.

    OPTIONAL WindowManager integration:
        When window_manager is provided, every observation is
        cross-validated against it BEFORE the existing
        same-window lifecycle check. window_manager is optional
        and fully backward-compatible: omitting it preserves the
        exact prior behavior.
    """

    def __init__(
        self,
        run_id: str,
        adwin_adapter: StreamingDetectorAdapter,
        ddm_adapter: StreamingDetectorAdapter,
        page_hinkley_adapter: StreamingDetectorAdapter,
        window_manager: WindowManager | None = None,
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

        if window_manager is not None:
            if not isinstance(window_manager, WindowManager):
                raise TypeError(
                    "window_manager must be a WindowManager instance."
                )

            if window_manager.config.run_id != run_id:
                raise ValueError(
                    "window_manager.config.run_id must match "
                    "runner.run_id."
                )

        self._run_id = run_id

        self._adwin_adapter = adwin_adapter
        self._ddm_adapter = ddm_adapter
        self._page_hinkley_adapter = page_hinkley_adapter

        self._window_manager = window_manager

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

    def _validate_against_window_manager(
        self,
        observation: PredictionErrorObservation,
    ) -> None:
        """
        Cross-check the observation against the external
        WindowManager, if one was supplied.

        WindowManager.window_for_index(sample_index) returns a
        WindowBoundary and raises IndexError when sample_index is
        out of range or falls inside a dropped partial window.
        """

        if self._window_manager is None:
            return

        try:
            boundary = self._window_manager.window_for_index(
                observation.sample_index
            )
        except IndexError as exc:
            raise ValueError(
                "sample_index is outside the WindowManager's "
                f"valid range: {observation.sample_index!r}."
            ) from exc

        if boundary.window_id != observation.external_window_id:
            raise ValueError(
                "observation.external_window_id does not match "
                "the WindowManager's expected window for "
                f"sample_index={observation.sample_index!r}: "
                f"expected {boundary.window_id!r}, got "
                f"{observation.external_window_id!r}."
            )

    def process_observation(
        self,
        observation: PredictionErrorObservation,
    ) -> tuple[
        DetectorUpdateResult,
        DetectorUpdateResult,
        DetectorUpdateResult,
    ]:
        if not isinstance(observation, PredictionErrorObservation):
            raise TypeError(
                "observation must be a PredictionErrorObservation."
            )

        if observation.run_id != self._run_id:
            raise ValueError(
                "observation.run_id must match runner.run_id."
            )

        self._validate_against_window_manager(observation)

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
