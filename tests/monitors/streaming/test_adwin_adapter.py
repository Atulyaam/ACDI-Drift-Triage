
import pytest

from src.monitors.streaming.adwin_adapter import (
    ADWINAdapter,
)

from src.monitors.streaming.configs import (
    ADWINConfig,
)

from src.monitors.streaming.error_contracts import (
    DetectorState,
    PredictionErrorObservation,
)


def _obs(
    sample_index,
    error=0,
    window="WIN_001",
    run_id="RUN_001",
):
    return PredictionErrorObservation(
        run_id=run_id,
        sample_index=sample_index,
        external_window_id=window,
        error=error,
    )


def test_adwin_adapter_constructs():
    adapter = ADWINAdapter("RUN_001")

    assert adapter.run_id == "RUN_001"
    assert adapter.detector_name == "ADWIN"
    assert adapter.state is DetectorState.ACTIVE
    assert adapter.last_sample_index is None


def test_adwin_detector_starts_without_drift():
    adapter = ADWINAdapter("RUN_001")

    assert adapter._river_detector.drift_detected is False


def test_adwin_config_is_explicitly_mapped_to_river():
    config = ADWINConfig(
        delta=0.01,
        clock=17,
        max_buckets=7,
        min_window_length=11,
        grace_period=13,
    )

    adapter = ADWINAdapter(
        "RUN_001",
        config=config,
    )

    detector = adapter._river_detector

    assert detector.delta == 0.01
    assert detector.clock == 17
    assert detector.max_buckets == 7
    assert detector.min_window_length == 11
    assert detector.grace_period == 13


def test_update_returns_detector_update_result():
    adapter = ADWINAdapter("RUN_001")

    result = adapter.update(
        _obs(0, 0)
    )

    assert result.detector_name == "ADWIN"
    assert result.run_id == "RUN_001"
    assert result.sample_index == 0
    assert result.reported_window_id == "WIN_001"
    assert result.detection is False


def test_update_advances_sample_index():
    adapter = ADWINAdapter("RUN_001")

    adapter.update(
        _obs(10)
    )

    assert adapter.last_sample_index == 10


def test_run_id_mismatch_is_rejected():
    adapter = ADWINAdapter("RUN_001")

    with pytest.raises(ValueError):
        adapter.update(
            _obs(
                0,
                run_id="RUN_002",
            )
        )


def test_duplicate_sample_index_is_rejected():
    adapter = ADWINAdapter("RUN_001")

    adapter.update(
        _obs(10)
    )

    with pytest.raises(ValueError):
        adapter.update(
            _obs(10)
        )


def test_out_of_order_sample_index_is_rejected():
    adapter = ADWINAdapter("RUN_001")

    adapter.update(
        _obs(10)
    )

    with pytest.raises(ValueError):
        adapter.update(
            _obs(9)
        )


def test_reset_creates_new_river_instance():
    adapter = ADWINAdapter("RUN_001")

    old_detector = adapter._river_detector

    adapter.update(
        _obs(10)
    )

    adapter.reset()

    assert adapter._river_detector is not old_detector
    assert adapter._river_detector.drift_detected is False


def test_reset_preserves_last_sample_index():
    adapter = ADWINAdapter("RUN_001")

    adapter.update(
        _obs(50)
    )

    adapter.reset()

    assert adapter.last_sample_index == 50
    assert adapter.state is DetectorState.ACTIVE


def test_reset_does_not_use_old_detector_state():
    adapter = ADWINAdapter("RUN_001")

    old_detector = adapter._river_detector

    adapter.reset()

    assert adapter._river_detector is not old_detector
    assert adapter._river_detector.drift_detected is False


def test_river_reset_compatibility_guard_passes():
    adapter = ADWINAdapter("RUN_001")

    adapter._require_reset_compatibility()

    assert hasattr(
        adapter._river_detector,
        "_reset",
    )


def test_drift_trigger_is_reported_as_drift_detected():
    class FakeADWIN:
        def __init__(self):
            self.drift_detected = False
            self.delta = 0.49
            self.clock = 1
            self.max_buckets = 5
            self.min_window_length = 1
            self.grace_period = 1

        def update(self, value):
            self.drift_detected = True

    adapter = ADWINAdapter("RUN_001")
    adapter._river_detector = FakeADWIN()

    result = adapter.update(
        _obs(0, error=1)
    )

    assert result.detection is True
    assert result.state is DetectorState.DRIFT_DETECTED
    assert adapter.state is DetectorState.LATCHED


def test_drift_detected_state_is_transient():
    class FakeADWIN:
        def __init__(self):
            self.drift_detected = False
            self.calls = 0
            self.delta = 0.01
            self.clock = 1
            self.max_buckets = 5
            self.min_window_length = 1
            self.grace_period = 1

        def update(self, value):
            self.calls += 1
            self.drift_detected = self.calls <= 2

    adapter = ADWINAdapter("RUN_001")
    adapter._river_detector = FakeADWIN()

    first = adapter.update(
        _obs(0, error=1)
    )

    second = adapter.update(
        _obs(1, error=1)
    )

    assert first.detection is True
    assert first.state is DetectorState.DRIFT_DETECTED

    assert second.detection is False
    assert second.state is DetectorState.LATCHED


def test_repeated_river_drift_alarms_are_ignored_while_latched():
    class AlwaysDriftingADWIN:
        def __init__(self):
            self.drift_detected = False
            self.calls = 0
            self.delta = 0.01
            self.clock = 32
            self.max_buckets = 5
            self.min_window_length = 5
            self.grace_period = 10

        def update(self, value):
            self.calls += 1
            self.drift_detected = True

    adapter = ADWINAdapter("RUN_001")

    fake = AlwaysDriftingADWIN()
    adapter._river_detector = fake

    first = adapter.update(
        _obs(0, error=1)
    )

    second = adapter.update(
        _obs(1, error=1)
    )

    third = adapter.update(
        _obs(2, error=1)
    )

    assert first.detection is True
    assert first.state is DetectorState.DRIFT_DETECTED

    assert second.detection is False
    assert second.state is DetectorState.LATCHED

    assert third.detection is False
    assert third.state is DetectorState.LATCHED

    assert fake.calls == 3


def test_raw_river_drift_flag_is_preserved_in_metadata():
    class AlwaysDriftingADWIN:
        def __init__(self):
            self.drift_detected = True
            self.delta = 0.01
            self.clock = 32
            self.max_buckets = 5
            self.min_window_length = 5
            self.grace_period = 10

        def update(self, value):
            self.drift_detected = True

    adapter = ADWINAdapter("RUN_001")
    adapter._river_detector = AlwaysDriftingADWIN()

    first = adapter.update(
        _obs(0, error=1)
    )

    second = adapter.update(
        _obs(1, error=1)
    )

    assert (
        first.metadata["raw_river_drift_detected"]
        is True
    )

    assert (
        second.metadata["raw_river_drift_detected"]
        is True
    )

    assert "warning_detected" not in second.metadata


def test_new_window_while_latched_triggers_timeout():
    class FakeADWIN:
        def __init__(self):
            self.drift_detected = True
            self.calls = 0
            self.delta = 0.01
            self.clock = 32
            self.max_buckets = 5
            self.min_window_length = 5
            self.grace_period = 10

        def update(self, value):
            self.calls += 1
            self.drift_detected = True

    adapter = ADWINAdapter("RUN_001")

    adapter._river_detector = FakeADWIN()

    # Fresh detector on timeout will also be fake.
    adapter._build_river_detector = (
        lambda: FakeADWIN()
    )

    first = adapter.update(
        _obs(
            0,
            error=1,
            window="WIN_001",
        )
    )

    assert first.state is DetectorState.DRIFT_DETECTED
    assert adapter.state is DetectorState.LATCHED

    second = adapter.update(
        _obs(
            1,
            error=0,
            window="WIN_002",
        )
    )

    assert (
        second.metadata["unresolved_timeout"]
        is True
    )

    assert second.sample_index == 1
    assert adapter.last_sample_index == 1


def test_timeout_replaces_river_detector():
    class FakeADWIN:
        def __init__(self):
            self.drift_detected = True
            self.calls = 0
            self.delta = 0.01
            self.clock = 32
            self.max_buckets = 5
            self.min_window_length = 5
            self.grace_period = 10

        def update(self, value):
            self.calls += 1

    adapter = ADWINAdapter("RUN_001")

    old_detector = FakeADWIN()
    adapter._river_detector = old_detector

    adapter._build_river_detector = (
        lambda: FakeADWIN()
    )

    first = adapter.update(
        _obs(
            0,
            error=1,
            window="WIN_001",
        )
    )

    second = adapter.update(
        _obs(
            1,
            error=0,
            window="WIN_002",
        )
    )

    new_detector = adapter._river_detector

    assert first.state is DetectorState.DRIFT_DETECTED
    assert new_detector is not old_detector
    assert new_detector.calls == 1
    assert (
        second.metadata["unresolved_timeout"]
        is True
    )


def test_timeout_triggering_observation_is_processed_by_new_detector():
    class CountingADWIN:
        def __init__(self):
            self.drift_detected = False
            self.calls = 0
            self.delta = 0.01
            self.clock = 32
            self.max_buckets = 5
            self.min_window_length = 5
            self.grace_period = 10

        def update(self, value):
            self.calls += 1

    adapter = ADWINAdapter("RUN_001")

    old_detector = CountingADWIN()
    adapter._river_detector = old_detector

    adapter._enter_latched("WIN_001")

    adapter._build_river_detector = (
        lambda: CountingADWIN()
    )

    result = adapter.update(
        _obs(
            10,
            error=1,
            window="WIN_002",
        )
    )

    new_detector = adapter._river_detector

    assert new_detector is not old_detector
    assert new_detector.calls == 1
    assert result.sample_index == 10
    assert (
        result.metadata["unresolved_timeout"]
        is True
    )


def test_two_adapters_have_independent_river_instances():
    adapter_a = ADWINAdapter("RUN_A")
    adapter_b = ADWINAdapter("RUN_B")

    assert (
        adapter_a._river_detector
        is not
        adapter_b._river_detector
    )


def test_one_adapter_reset_does_not_replace_other_adapter_detector():
    adapter_a = ADWINAdapter("RUN_A")
    adapter_b = ADWINAdapter("RUN_B")

    detector_b = adapter_b._river_detector

    adapter_a.reset()

    assert adapter_b._river_detector is detector_b


def test_explicit_reset_returns_adapter_to_active():
    adapter = ADWINAdapter("RUN_001")

    adapter._enter_latched("WIN_001")
    adapter.reset()

    assert adapter.state is DetectorState.ACTIVE
    assert adapter.latch_window_id is None


def test_reset_preserves_monotonic_sample_index():
    adapter = ADWINAdapter("RUN_001")

    adapter.update(
        _obs(100)
    )

    adapter.reset()

    assert adapter.last_sample_index == 100

    result = adapter.update(
        _obs(101)
    )

    assert result.sample_index == 101
