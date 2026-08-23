import pytest

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


def _obs(sample_index, error=0, window="WIN_001", run_id="RUN_001"):
    return PredictionErrorObservation(
        run_id=run_id,
        sample_index=sample_index,
        external_window_id=window,
        error=error,
    )


class FakeAdapter(StreamingDetectorAdapter):
    """
    Deterministic adapter for runner/orchestration tests.

    Each fake can be configured with the sample indices at which
    it emits detection=True.
    """

    def __init__(self, run_id, detector_name, trigger_indices=None):
        super().__init__(run_id=run_id, detector_name=detector_name)

        self.trigger_indices = set(trigger_indices or [])

        self.update_calls = 0
        self.reset_calls = 0

    def update(self, observation):
        self._validate_observation(observation)

        self.update_calls += 1

        detection = observation.sample_index in self.trigger_indices

        result = DetectorUpdateResult(
            detector_name=self.detector_name,
            run_id=self.run_id,
            sample_index=observation.sample_index,
            reported_window_id=observation.external_window_id,
            detection=detection,
            state=(
                DetectorState.DRIFT_DETECTED
                if detection
                else DetectorState.ACTIVE
            ),
        )

        self._record_sample_index(observation)

        return result

    def reset(self):
        self.reset_calls += 1
        self._mark_reset()
        self._mark_active()


def _runner(adwin_triggers=None, ddm_triggers=None, ph_triggers=None):
    return StreamingEnsembleRunner(
        run_id="RUN_001",
        adwin_adapter=FakeAdapter("RUN_001", "ADWIN", adwin_triggers),
        ddm_adapter=FakeAdapter("RUN_001", "DDM", ddm_triggers),
        page_hinkley_adapter=FakeAdapter(
            "RUN_001", "PAGE_HINKLEY", ph_triggers
        ),
    )


# ============================================================
# Construction
# ============================================================

def test_runner_constructs():
    runner = _runner()

    assert runner.run_id == "RUN_001"
    assert runner.current_window_id is None
    assert runner.last_sample_index is None

    assert runner.adwin_triggered is False
    assert runner.ddm_triggered is False
    assert runner.page_hinkley_triggered is False


def test_runner_rejects_wrong_adapter_type():
    with pytest.raises(TypeError):
        StreamingEnsembleRunner(
            "RUN_001",
            object(),
            FakeAdapter("RUN_001", "DDM"),
            FakeAdapter("RUN_001", "PAGE_HINKLEY"),
        )


def test_runner_rejects_wrong_detector_identity():
    with pytest.raises(ValueError):
        StreamingEnsembleRunner(
            "RUN_001",
            FakeAdapter("RUN_001", "DDM"),
            FakeAdapter("RUN_001", "DDM"),
            FakeAdapter("RUN_001", "PAGE_HINKLEY"),
        )


def test_runner_rejects_mismatched_adapter_run_ids():
    with pytest.raises(ValueError):
        StreamingEnsembleRunner(
            "RUN_001",
            FakeAdapter("RUN_002", "ADWIN"),
            FakeAdapter("RUN_001", "DDM"),
            FakeAdapter("RUN_001", "PAGE_HINKLEY"),
        )


# ============================================================
# Observation processing
# ============================================================

def test_first_observation_establishes_current_window():
    runner = _runner()

    results = runner.process_observation(_obs(0))

    assert len(results) == 3
    assert runner.current_window_id == "WIN_001"
    assert runner.last_sample_index == 0


def test_all_three_adapters_receive_same_observation():
    runner = _runner()

    results = runner.process_observation(_obs(10, error=1))

    assert [result.sample_index for result in results] == [10, 10, 10]
    assert [result.reported_window_id for result in results] == [
        "WIN_001",
        "WIN_001",
        "WIN_001",
    ]


def test_runner_rejects_observation_from_wrong_run():
    runner = _runner()

    with pytest.raises(ValueError):
        runner.process_observation(_obs(0, run_id="RUN_002"))


def test_runner_rejects_new_window_before_close():
    runner = _runner()

    runner.process_observation(_obs(0, window="WIN_001"))

    with pytest.raises(ValueError):
        runner.process_observation(_obs(1, window="WIN_002"))


# ============================================================
# Window-level trigger accumulation
# ============================================================

def test_single_sample_trigger_is_accumulated_for_window():
    runner = _runner(adwin_triggers={5})

    runner.process_observation(_obs(0))
    runner.process_observation(_obs(5))

    assert runner.adwin_triggered is True
    assert runner.ddm_triggered is False
    assert runner.page_hinkley_triggered is False


def test_trigger_remains_true_after_later_non_trigger_samples():
    runner = _runner(adwin_triggers={5})

    runner.process_observation(_obs(0))
    runner.process_observation(_obs(5))
    runner.process_observation(_obs(6))
    runner.process_observation(_obs(7))

    assert runner.adwin_triggered is True


def test_each_detector_has_independent_window_trigger_flag():
    runner = _runner(
        adwin_triggers={3}, ddm_triggers={5}, ph_triggers={7}
    )

    for index in range(8):
        runner.process_observation(_obs(index))

    assert runner.adwin_triggered is True
    assert runner.ddm_triggered is True
    assert runner.page_hinkley_triggered is True


# ============================================================
# Window-close vote
# ============================================================

def test_all_stable_window_produces_zero_vote():
    runner = _runner()

    for index in range(10):
        runner.process_observation(_obs(index))

    result = runner.close_window()

    assert result.error_vote_count == 0
    assert result.error_drift is False

    assert result.adwin_drift is False
    assert result.ddm_drift is False
    assert result.page_hinkley_drift is False


def test_only_one_detector_triggering_produces_one_vote():
    runner = _runner(adwin_triggers={5})

    for index in range(10):
        runner.process_observation(_obs(index))

    result = runner.close_window()

    assert result.adwin_drift is True
    assert result.ddm_drift is False
    assert result.page_hinkley_drift is False

    assert result.error_vote_count == 1
    assert result.error_drift is False


def test_two_detectors_triggering_produces_drift():
    runner = _runner(adwin_triggers={5}, ddm_triggers={7})

    for index in range(10):
        runner.process_observation(_obs(index))

    result = runner.close_window()

    assert result.adwin_drift is True
    assert result.ddm_drift is True
    assert result.page_hinkley_drift is False

    assert result.error_vote_count == 2
    assert result.error_drift is True


def test_three_detectors_triggering_produces_drift():
    runner = _runner(
        adwin_triggers={2}, ddm_triggers={5}, ph_triggers={8}
    )

    for index in range(10):
        runner.process_observation(_obs(index))

    result = runner.close_window()

    assert result.error_vote_count == 3
    assert result.error_drift is True


# ============================================================
# Different trigger times are intentionally allowed
# ============================================================

def test_detectors_can_trigger_on_different_samples_in_same_window():
    runner = _runner(
        adwin_triggers={2}, ddm_triggers={5}, ph_triggers={8}
    )

    for index in range(9):
        runner.process_observation(_obs(index))

    result = runner.close_window()

    assert result.adwin_drift is True
    assert result.ddm_drift is True
    assert result.page_hinkley_drift is True

    assert result.error_vote_count == 3
    assert result.error_drift is True


# ============================================================
# Synthetic window-level result alignment
# ============================================================

def test_window_vote_uses_last_sample_index_for_all_detectors():
    runner = _runner(adwin_triggers={2}, ddm_triggers={5})

    for index in range(10):
        runner.process_observation(_obs(index))

    result = runner.close_window()

    # The underlying voter only receives synthetic aligned
    # DetectorUpdateResults from the runner.
    assert result.sample_index == 9
    assert result.reported_window_id == "WIN_001"


def test_window_vote_uses_same_run_id():
    runner = _runner(adwin_triggers={2}, ddm_triggers={5})

    runner.process_observation(_obs(0))

    result = runner.close_window()

    assert result.run_id == "RUN_001"


# ============================================================
# Window lifecycle
# ============================================================

def test_close_window_resets_only_runner_aggregation_state():
    runner = _runner(adwin_triggers={2}, ddm_triggers={5})

    for index in range(6):
        runner.process_observation(_obs(index))

    runner.close_window()

    assert runner.current_window_id is None
    assert runner.last_sample_index is None

    assert runner.adwin_triggered is False
    assert runner.ddm_triggered is False
    assert runner.page_hinkley_triggered is False


def test_close_window_does_not_reset_detectors():
    runner = _runner(adwin_triggers={2})

    for index in range(3):
        runner.process_observation(_obs(index))

    adwin = runner._adwin_adapter
    ddm = runner._ddm_adapter
    ph = runner._page_hinkley_adapter

    runner.close_window()

    assert adwin.reset_calls == 0
    assert ddm.reset_calls == 0
    assert ph.reset_calls == 0


def test_next_external_window_can_start_after_close():
    runner = _runner()

    runner.process_observation(_obs(0, window="WIN_001"))

    first_vote = runner.close_window()

    assert first_vote.reported_window_id == "WIN_001"

    runner.process_observation(_obs(1, window="WIN_002"))

    assert runner.current_window_id == "WIN_002"


def test_second_window_gets_fresh_aggregation_flags():
    runner = _runner(adwin_triggers={0, 2})

    runner.process_observation(_obs(0, window="WIN_001"))

    first_vote = runner.close_window()

    assert first_vote.error_vote_count == 1

    runner.process_observation(_obs(1, window="WIN_002"))

    second_vote = runner.close_window()

    assert second_vote.error_vote_count == 0
    assert second_vote.error_drift is False


# ============================================================
# Close-window errors
# ============================================================

def test_close_without_active_window_is_rejected():
    runner = _runner()

    with pytest.raises(ValueError):
        runner.close_window()


def test_new_window_after_close_is_required_to_reestablish_context():
    runner = _runner()

    runner.process_observation(_obs(0, window="WIN_001"))

    runner.close_window()

    assert runner.current_window_id is None

    runner.process_observation(_obs(1, window="WIN_002"))

    assert runner.current_window_id == "WIN_002"
