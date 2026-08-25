import pytest

from src.contracts.window_config import WindowManagerConfig
from src.windows.manager import WindowManager

from src.monitors.streaming.adapter import (
    StreamingDetectorAdapter,
)

from src.monitors.streaming.ensemble_runner import (
    StreamingEnsembleRunner,
)

from src.monitors.streaming.error_contracts import (
    DetectorState,
    DetectorUpdateResult,
    PredictionErrorObservation,
)


RUN_ID = "RUN_001"


def _observation(sample_index, window_id, error=0):
    return PredictionErrorObservation(
        run_id=RUN_ID,
        sample_index=sample_index,
        external_window_id=window_id,
        error=error,
    )


def _window_manager(
    total_samples=20,
    window_size=10,
    drop_last_partial=False,
):
    return WindowManager(
        WindowManagerConfig(
            run_id=RUN_ID,
            window_size=window_size,
            total_samples=total_samples,
            start_index=0,
            window_id_prefix="WIN_",
            drop_last_partial=drop_last_partial,
        )
    )


class DummyAdapter(StreamingDetectorAdapter):
    """
    Deterministic adapter used only for runner integration tests.

    It does not use River.
    """

    def __init__(self, run_id, detector_name):
        super().__init__(run_id=run_id, detector_name=detector_name)
        self.update_calls = 0

    def update(self, observation):
        self._validate_observation(observation)

        self.update_calls += 1

        result = DetectorUpdateResult(
            detector_name=self.detector_name,
            run_id=self.run_id,
            sample_index=observation.sample_index,
            reported_window_id=observation.external_window_id,
            detection=False,
            state=DetectorState.ACTIVE,
        )

        self._record_sample_index(observation)

        return result

    def reset(self):
        self._mark_reset()
        self._mark_active()


def _runner(window_manager):
    return StreamingEnsembleRunner(
        run_id=RUN_ID,
        adwin_adapter=DummyAdapter(RUN_ID, "ADWIN"),
        ddm_adapter=DummyAdapter(RUN_ID, "DDM"),
        page_hinkley_adapter=DummyAdapter(RUN_ID, "PAGE_HINKLEY"),
        window_manager=window_manager,
    )


# ============================================================
# RED-PHASE INTEGRATION TESTS
#
# At this point StreamingEnsembleRunner does not yet accept
# window_manager. These tests intentionally expose that missing
# integration capability.
#
# Once the production change is implemented, these same tests
# become the regression contract.
# ============================================================


def test_valid_sample_window_pair_is_accepted():
    manager = _window_manager()

    runner = _runner(manager)

    result = runner.process_observation(
        _observation(sample_index=5, window_id="WIN_000001")
    )

    assert result[0].sample_index == 5
    assert runner.current_window_id == "WIN_000001"


def test_wrong_window_id_for_sample_index_is_rejected():
    manager = _window_manager()

    runner = _runner(manager)

    with pytest.raises(ValueError):
        runner.process_observation(
            _observation(sample_index=5, window_id="WIN_000002")
        )


def test_sample_index_outside_window_manager_range_is_rejected():
    manager = _window_manager(total_samples=20)

    runner = _runner(manager)

    with pytest.raises(ValueError):
        runner.process_observation(
            _observation(sample_index=20, window_id="WIN_999999")
        )


def test_dropped_partial_window_is_rejected():
    manager = _window_manager(
        total_samples=15,
        window_size=10,
        drop_last_partial=True,
    )

    runner = _runner(manager)

    with pytest.raises(ValueError):
        runner.process_observation(
            _observation(sample_index=12, window_id="WIN_000002")
        )


def test_correct_window_transition_after_close_is_accepted():
    manager = _window_manager(total_samples=20, window_size=10)

    runner = _runner(manager)

    runner.process_observation(
        _observation(sample_index=5, window_id="WIN_000001")
    )

    runner.close_window()

    runner.process_observation(
        _observation(sample_index=10, window_id="WIN_000002")
    )

    assert runner.current_window_id == "WIN_000002"


def test_mid_window_window_id_mismatch_is_rejected():
    """
    Both observations refer to the same current runner window,
    so the existing 'new window before close' lifecycle check
    must NOT be the reason for rejection.

    The WindowManager cross-validation must detect the mismatch.
    This is why WindowManager validation must run BEFORE the
    existing lifecycle check in the locked validation order.
    """

    manager = _window_manager(total_samples=20, window_size=10)

    runner = _runner(manager)

    runner.process_observation(
        _observation(sample_index=5, window_id="WIN_000001")
    )

    with pytest.raises(ValueError):
        runner.process_observation(
            _observation(sample_index=6, window_id="WIN_000002")
        )


def test_window_manager_run_id_mismatch_is_rejected():
    manager = WindowManager(
        WindowManagerConfig(
            run_id="RUN_002",
            window_size=10,
            total_samples=20,
            start_index=0,
            window_id_prefix="WIN_",
            drop_last_partial=False,
        )
    )

    with pytest.raises(ValueError):
        StreamingEnsembleRunner(
            run_id=RUN_ID,
            adwin_adapter=DummyAdapter(RUN_ID, "ADWIN"),
            ddm_adapter=DummyAdapter(RUN_ID, "DDM"),
            page_hinkley_adapter=DummyAdapter(RUN_ID, "PAGE_HINKLEY"),
            window_manager=manager,
        )


def test_window_manager_is_optional_for_backward_compatibility():
    runner = StreamingEnsembleRunner(
        run_id=RUN_ID,
        adwin_adapter=DummyAdapter(RUN_ID, "ADWIN"),
        ddm_adapter=DummyAdapter(RUN_ID, "DDM"),
        page_hinkley_adapter=DummyAdapter(RUN_ID, "PAGE_HINKLEY"),
    )

    result = runner.process_observation(
        _observation(sample_index=5, window_id="ARBITRARY_WINDOW")
    )

    assert result[0].sample_index == 5
