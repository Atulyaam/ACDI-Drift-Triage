
import pytest

from src.monitors.streaming.ddm_adapter import (
    DDMAdapter,
)

from src.monitors.streaming.configs import (
    DDMConfig,
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


def test_ddm_adapter_constructs():
    adapter = DDMAdapter("RUN_001")

    assert adapter.run_id == "RUN_001"
    assert adapter.detector_name == "DDM"
    assert adapter.state is DetectorState.ACTIVE
    assert adapter.last_sample_index is None


def test_ddm_starts_without_warning_or_drift():
    adapter = DDMAdapter("RUN_001")

    assert adapter._river_detector.drift_detected is False
    assert adapter._river_detector.warning_detected is False


def test_ddm_config_is_explicitly_mapped_to_river():
    config = DDMConfig(
        warm_start=40,
        warning_threshold=2.5,
        drift_threshold=4.0,
    )

    adapter = DDMAdapter(
        "RUN_001",
        config=config,
    )

    detector = adapter._river_detector

    assert detector.warm_start == 40
    assert detector.warning_threshold == 2.5
    assert detector.drift_threshold == 4.0


def test_update_returns_detector_update_result():
    adapter = DDMAdapter("RUN_001")

    result = adapter.update(
        _obs(0, error=0)
    )

    assert result.detector_name == "DDM"
    assert result.run_id == "RUN_001"
    assert result.sample_index == 0
    assert result.reported_window_id == "WIN_001"
    assert result.detection is False


def test_update_advances_sample_index():
    adapter = DDMAdapter("RUN_001")

    adapter.update(
        _obs(10)
    )

    assert adapter.last_sample_index == 10


def test_run_id_mismatch_is_rejected():
    adapter = DDMAdapter("RUN_001")

    with pytest.raises(ValueError):
        adapter.update(
            _obs(
                0,
                run_id="RUN_002",
            )
        )


def test_duplicate_sample_index_is_rejected():
    adapter = DDMAdapter("RUN_001")

    adapter.update(
        _obs(10)
    )

    with pytest.raises(ValueError):
        adapter.update(
            _obs(10)
        )


def test_out_of_order_sample_index_is_rejected():
    adapter = DDMAdapter("RUN_001")

    adapter.update(
        _obs(10)
    )

    with pytest.raises(ValueError):
        adapter.update(
            _obs(9)
        )


def test_warning_does_not_trigger_drift_lifecycle():
    class WarningOnlyDDM:
        def __init__(self):
            self.warning_detected = True
            self.drift_detected = False
            self.warm_start = 30
            self.warning_threshold = 2.0
            self.drift_threshold = 3.0

        def update(self, value):
            self.warning_detected = True
            self.drift_detected = False

    adapter = DDMAdapter("RUN_001")

    fake = WarningOnlyDDM()
    adapter._river_detector = fake

    result = adapter.update(
        _obs(0, error=1)
    )

    assert result.detection is False
    assert result.state is DetectorState.ACTIVE

    assert (
        result.metadata["warning_detected"]
        is True
    )

    assert (
        result.metadata["raw_river_drift_detected"]
        is False
    )


def test_warning_is_captured_on_every_update():
    class WarningOscillatingDDM:
        def __init__(self):
            self.warning_detected = False
            self.drift_detected = False
            self.calls = 0
            self.warm_start = 30
            self.warning_threshold = 2.0
            self.drift_threshold = 3.0

        def update(self, value):
            self.calls += 1
            self.warning_detected = (
                self.calls in (1, 3)
            )
            self.drift_detected = False

    adapter = DDMAdapter("RUN_001")

    fake = WarningOscillatingDDM()
    adapter._river_detector = fake

    first = adapter.update(
        _obs(0)
    )

    second = adapter.update(
        _obs(1)
    )

    third = adapter.update(
        _obs(2)
    )

    assert first.metadata["warning_detected"] is True
    assert second.metadata["warning_detected"] is False
    assert third.metadata["warning_detected"] is True

    assert first.state is DetectorState.ACTIVE
    assert second.state is DetectorState.ACTIVE
    assert third.state is DetectorState.ACTIVE

    assert first.detection is False
    assert second.detection is False
    assert third.detection is False


def test_drift_trigger_reports_drift_detected():
    class DriftingDDM:
        def __init__(self):
            self.warning_detected = True
            self.drift_detected = False
            self.calls = 0
            self.warm_start = 30
            self.warning_threshold = 2.0
            self.drift_threshold = 3.0

        def update(self, value):
            self.calls += 1
            self.warning_detected = True
            self.drift_detected = True

    adapter = DDMAdapter("RUN_001")

    fake = DriftingDDM()
    adapter._river_detector = fake

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

    assert adapter.state is DetectorState.LATCHED

    assert (
        result.metadata["raw_river_drift_detected"]
        is True
    )

    assert (
        result.metadata["warning_detected"]
        is True
    )


def test_drift_detected_state_is_transient():
    class DriftingDDM:
        def __init__(self):
            self.warning_detected = False
            self.drift_detected = False
            self.calls = 0
            self.warm_start = 30
            self.warning_threshold = 2.0
            self.drift_threshold = 3.0

        def update(self, value):
            self.calls += 1
            self.drift_detected = True

    adapter = DDMAdapter("RUN_001")

    adapter._river_detector = DriftingDDM()

    first = adapter.update(
        _obs(0, error=1)
    )

    second = adapter.update(
        _obs(1, error=1)
    )

    assert first.detection is True
    assert (
        first.state
        is DetectorState.DRIFT_DETECTED
    )

    assert second.detection is False
    assert (
        second.state
        is DetectorState.LATCHED
    )


def test_repeated_drift_is_ignored_while_warning_varies():
    class MixedSignalDDM:
        def __init__(self):
            self.warning_detected = False
            self.drift_detected = False
            self.calls = 0
            self.warm_start = 30
            self.warning_threshold = 2.0
            self.drift_threshold = 3.0

        def update(self, value):
            self.calls += 1

            self.drift_detected = True

            self.warning_detected = (
                self.calls in (1, 3)
            )

    adapter = DDMAdapter("RUN_001")

    fake = MixedSignalDDM()
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
    assert (
        first.state
        is DetectorState.DRIFT_DETECTED
    )

    assert second.detection is False
    assert (
        second.state
        is DetectorState.LATCHED
    )

    assert third.detection is False
    assert (
        third.state
        is DetectorState.LATCHED
    )

    assert (
        first.metadata["warning_detected"]
        is True
    )

    assert (
        second.metadata["warning_detected"]
        is False
    )

    assert (
        third.metadata["warning_detected"]
        is True
    )


def test_warning_metadata_exists_on_active_update():
    adapter = DDMAdapter("RUN_001")

    result = adapter.update(
        _obs(0)
    )

    assert "warning_detected" in result.metadata
    assert "raw_river_drift_detected" in result.metadata
    assert "river_version" in result.metadata


def test_warning_metadata_exists_while_latched():
    class DriftingDDM:
        def __init__(self):
            self.warning_detected = True
            self.drift_detected = True
            self.warm_start = 30
            self.warning_threshold = 2.0
            self.drift_threshold = 3.0

        def update(self, value):
            self.warning_detected = True
            self.drift_detected = True

    adapter = DDMAdapter("RUN_001")
    adapter._river_detector = DriftingDDM()

    adapter.update(
        _obs(0, error=1)
    )

    result = adapter.update(
        _obs(1, error=1)
    )

    assert (
        result.metadata["warning_detected"]
        is True
    )

    assert (
        result.metadata["raw_river_drift_detected"]
        is True
    )


def test_reset_creates_new_ddm_instance():
    adapter = DDMAdapter("RUN_001")

    old_detector = adapter._river_detector

    adapter.update(
        _obs(10)
    )

    adapter.reset()

    new_detector = adapter._river_detector

    assert new_detector is not old_detector
    assert new_detector.drift_detected is False
    assert new_detector.warning_detected is False


def test_reset_preserves_last_sample_index():
    adapter = DDMAdapter("RUN_001")

    adapter.update(
        _obs(50)
    )

    adapter.reset()

    assert adapter.last_sample_index == 50
    assert adapter.state is DetectorState.ACTIVE


def test_reset_compatibility_guard_passes():
    adapter = DDMAdapter("RUN_001")

    adapter._require_reset_compatibility()

    assert hasattr(
        adapter._river_detector,
        "_reset",
    )


def test_timeout_uses_fresh_ddm_and_processes_same_observation():
    class CountingDDM:
        def __init__(self):
            self.warning_detected = False
            self.drift_detected = False
            self.calls = 0
            self.warm_start = 30
            self.warning_threshold = 2.0
            self.drift_threshold = 3.0

        def update(self, value):
            self.calls += 1

    adapter = DDMAdapter("RUN_001")

    old_detector = CountingDDM()
    adapter._river_detector = old_detector

    adapter._enter_latched(
        "WIN_001"
    )

    adapter._build_river_detector = (
        lambda: CountingDDM()
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
        result.metadata["unresolved_timeout"]
        is True
    )

    assert result.sample_index == 10


def test_two_ddm_adapters_have_independent_instances():
    adapter_a = DDMAdapter("RUN_A")
    adapter_b = DDMAdapter("RUN_B")

    assert (
        adapter_a._river_detector
        is not
        adapter_b._river_detector
    )


def test_reset_one_ddm_does_not_replace_other():
    adapter_a = DDMAdapter("RUN_A")
    adapter_b = DDMAdapter("RUN_B")

    detector_b = adapter_b._river_detector

    adapter_a.reset()

    assert (
        adapter_b._river_detector
        is detector_b
    )


def test_explicit_reset_returns_to_active():
    adapter = DDMAdapter("RUN_001")

    adapter._enter_latched(
        "WIN_001"
    )

    adapter.reset()

    assert (
        adapter.state
        is DetectorState.ACTIVE
    )

    assert adapter.latch_window_id is None


def test_reset_preserves_monotonic_sequence():
    adapter = DDMAdapter("RUN_001")

    adapter.update(
        _obs(100)
    )

    adapter.reset()

    result = adapter.update(
        _obs(101)
    )

    assert result.sample_index == 101
    assert adapter.last_sample_index == 101
