
import pytest

from src.monitors.streaming.page_hinkley_adapter import (
    PageHinkleyAdapter,
)

from src.monitors.streaming.configs import (
    PageHinkleyConfig,
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


def test_page_hinkley_adapter_constructs():
    adapter = PageHinkleyAdapter("RUN_001")

    assert adapter.run_id == "RUN_001"
    assert (
        adapter.detector_name
        == "PAGE_HINKLEY"
    )
    assert adapter.state is DetectorState.ACTIVE


def test_starts_without_drift():
    adapter = PageHinkleyAdapter("RUN_001")

    assert (
        adapter._river_detector.drift_detected
        is False
    )


def test_config_is_explicitly_mapped_to_river():
    config = PageHinkleyConfig(
        min_instances=40,
        delta=0.01,
        threshold=25.0,
        alpha=0.99,
        mode="up",
    )

    adapter = PageHinkleyAdapter(
        "RUN_001",
        config=config,
    )

    detector = adapter._river_detector

    assert detector.min_instances == 40
    assert detector.delta == 0.01
    assert detector.threshold == 25.0
    assert detector.alpha == 0.99
    assert detector.mode == "up"


def test_update_returns_result():
    adapter = PageHinkleyAdapter("RUN_001")

    result = adapter.update(
        _obs(0)
    )

    assert result.detector_name == "PAGE_HINKLEY"
    assert result.run_id == "RUN_001"
    assert result.sample_index == 0
    assert result.detection is False


def test_run_id_mismatch_is_rejected():
    adapter = PageHinkleyAdapter("RUN_001")

    with pytest.raises(ValueError):
        adapter.update(
            _obs(
                0,
                run_id="RUN_002",
            )
        )


def test_duplicate_index_is_rejected():
    adapter = PageHinkleyAdapter("RUN_001")

    adapter.update(
        _obs(10)
    )

    with pytest.raises(ValueError):
        adapter.update(
            _obs(10)
        )


def test_out_of_order_index_is_rejected():
    adapter = PageHinkleyAdapter("RUN_001")

    adapter.update(
        _obs(10)
    )

    with pytest.raises(ValueError):
        adapter.update(
            _obs(9)
        )


def test_warning_metadata_does_not_exist():
    adapter = PageHinkleyAdapter("RUN_001")

    result = adapter.update(
        _obs(0)
    )

    assert (
        "warning_detected"
        not in result.metadata
    )


def test_drift_trigger_reports_drift_detected():
    class DriftingPH:
        def __init__(self):
            self.drift_detected = False
            self.min_instances = 30
            self.delta = 0.005
            self.threshold = 50.0
            self.alpha = 0.9999
            self.mode = "both"

        def update(self, value):
            self.drift_detected = True

    adapter = PageHinkleyAdapter("RUN_001")

    adapter._river_detector = DriftingPH()

    result = adapter.update(
        _obs(
            0,
            error=1,
        )
    )

    assert result.detection is True
    assert (
        result.state
        is DetectorState.DRIFT_DETECTED
    )

    assert (
        adapter.state
        is DetectorState.LATCHED
    )


def test_drift_state_is_transient():
    class DriftingPH:
        def __init__(self):
            self.drift_detected = False
            self.calls = 0
            self.min_instances = 30
            self.delta = 0.005
            self.threshold = 50.0
            self.alpha = 0.9999
            self.mode = "both"

        def update(self, value):
            self.calls += 1
            self.drift_detected = True

    adapter = PageHinkleyAdapter("RUN_001")

    adapter._river_detector = DriftingPH()

    first = adapter.update(
        _obs(0, error=1)
    )

    second = adapter.update(
        _obs(1, error=1)
    )

    assert (
        first.state
        is DetectorState.DRIFT_DETECTED
    )

    assert first.detection is True

    assert (
        second.state
        is DetectorState.LATCHED
    )

    assert second.detection is False


def test_repeated_drift_is_ignored_while_latched():
    class DriftingPH:
        def __init__(self):
            self.drift_detected = False
            self.calls = 0
            self.min_instances = 30
            self.delta = 0.005
            self.threshold = 50.0
            self.alpha = 0.9999
            self.mode = "both"

        def update(self, value):
            self.calls += 1
            self.drift_detected = True

    adapter = PageHinkleyAdapter("RUN_001")

    fake = DriftingPH()
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
    assert second.detection is False
    assert third.detection is False

    assert (
        second.state
        is DetectorState.LATCHED
    )

    assert (
        third.state
        is DetectorState.LATCHED
    )

    assert fake.calls == 3


def test_reset_creates_new_page_hinkley():
    adapter = PageHinkleyAdapter("RUN_001")

    old_detector = adapter._river_detector

    adapter.update(
        _obs(10)
    )

    adapter.reset()

    new_detector = adapter._river_detector

    assert new_detector is not old_detector

    assert (
        new_detector.drift_detected
        is False
    )


def test_reset_preserves_last_sample_index():
    adapter = PageHinkleyAdapter("RUN_001")

    adapter.update(
        _obs(50)
    )

    adapter.reset()

    assert adapter.last_sample_index == 50
    assert adapter.state is DetectorState.ACTIVE


def test_reset_compatibility_guard_passes():
    adapter = PageHinkleyAdapter("RUN_001")

    adapter._require_reset_compatibility()

    assert hasattr(
        adapter._river_detector,
        "_reset",
    )


def test_timeout_processes_same_observation_on_fresh_detector():
    class CountingPH:
        def __init__(self):
            self.drift_detected = False
            self.calls = 0
            self.min_instances = 30
            self.delta = 0.005
            self.threshold = 50.0
            self.alpha = 0.9999
            self.mode = "both"

        def update(self, value):
            self.calls += 1

    adapter = PageHinkleyAdapter("RUN_001")

    old_detector = CountingPH()

    adapter._river_detector = old_detector

    adapter._enter_latched(
        "WIN_001"
    )

    adapter._build_river_detector = (
        lambda: CountingPH()
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

    assert (
        result.metadata[
            "unresolved_timeout"
        ]
        is True
    )


def test_two_adapters_have_independent_instances():
    adapter_a = PageHinkleyAdapter("RUN_A")
    adapter_b = PageHinkleyAdapter("RUN_B")

    assert (
        adapter_a._river_detector
        is not
        adapter_b._river_detector
    )


def test_reset_one_adapter_does_not_replace_other():
    adapter_a = PageHinkleyAdapter("RUN_A")
    adapter_b = PageHinkleyAdapter("RUN_B")

    detector_b = adapter_b._river_detector

    adapter_a.reset()

    assert (
        adapter_b._river_detector
        is detector_b
    )


def test_explicit_reset_returns_to_active():
    adapter = PageHinkleyAdapter("RUN_001")

    adapter._enter_latched(
        "WIN_001"
    )

    adapter.reset()

    assert (
        adapter.state
        is DetectorState.ACTIVE
    )

    assert adapter.latch_window_id is None
