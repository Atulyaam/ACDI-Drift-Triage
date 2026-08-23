import pytest

from src.monitors.streaming.adapter import (
    StreamingDetectorAdapter,
)

from src.monitors.streaming.configs import (
    ADWINConfig,
    DDMConfig,
    PageHinkleyConfig,
)

from src.monitors.streaming.error_contracts import (
    DetectorState,
    DetectorUpdateResult,
    PredictionErrorObservation,
)


# ============================================================
# ADWINConfig
# ============================================================

def test_adwin_config_accepts_explicit_values():
    config = ADWINConfig(
        delta=0.01,
        clock=16,
        max_buckets=7,
        min_window_length=10,
        grace_period=20,
    )

    assert config.delta == 0.01
    assert config.clock == 16
    assert config.max_buckets == 7
    assert config.min_window_length == 10
    assert config.grace_period == 20


@pytest.mark.parametrize(
    "field",
    ["clock", "max_buckets", "min_window_length", "grace_period"],
)
def test_adwin_positive_integer_fields_reject_bool(field):
    values = {
        "delta": 0.01,
        "clock": 16,
        "max_buckets": 7,
        "min_window_length": 10,
        "grace_period": 20,
    }
    values[field] = True

    with pytest.raises(TypeError):
        ADWINConfig(**values)


def test_adwin_delta_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        ADWINConfig(delta=0.0)
    with pytest.raises(ValueError):
        ADWINConfig(delta=1.0)
    with pytest.raises(ValueError):
        ADWINConfig(delta=-0.1)
    with pytest.raises(ValueError):
        ADWINConfig(delta=1.1)


# ============================================================
# DDMConfig
# ============================================================

def test_ddm_config_accepts_explicit_values():
    config = DDMConfig(
        warm_start=40,
        warning_threshold=2.5,
        drift_threshold=4.0,
    )

    assert config.warm_start == 40
    assert config.warning_threshold == 2.5
    assert config.drift_threshold == 4.0


def test_ddm_warning_threshold_must_be_below_drift_threshold():
    with pytest.raises(ValueError):
        DDMConfig(warning_threshold=3.0, drift_threshold=3.0)
    with pytest.raises(ValueError):
        DDMConfig(warning_threshold=4.0, drift_threshold=3.0)


def test_ddm_positive_fields_reject_bool():
    with pytest.raises(TypeError):
        DDMConfig(warm_start=True)
    with pytest.raises(TypeError):
        DDMConfig(warning_threshold=True)
    with pytest.raises(TypeError):
        DDMConfig(drift_threshold=True)


def test_ddm_thresholds_reject_non_finite():
    with pytest.raises(ValueError):
        DDMConfig(warning_threshold=float("nan"))
    with pytest.raises(ValueError):
        DDMConfig(drift_threshold=float("inf"))


# ============================================================
# Page-HinkleyConfig
# ============================================================

def test_page_hinkley_config_accepts_explicit_values():
    config = PageHinkleyConfig(
        min_instances=40,
        delta=0.01,
        threshold=25.0,
        alpha=0.99,
        mode="up",
    )

    assert config.min_instances == 40
    assert config.delta == 0.01
    assert config.threshold == 25.0
    assert config.alpha == 0.99
    assert config.mode == "up"


@pytest.mark.parametrize("mode", ["up", "down", "both"])
def test_page_hinkley_accepts_valid_mode(mode):
    config = PageHinkleyConfig(mode=mode)
    assert config.mode == mode


def test_page_hinkley_rejects_invalid_mode():
    with pytest.raises(ValueError):
        PageHinkleyConfig(mode="invalid")


def test_page_hinkley_alpha_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        PageHinkleyConfig(alpha=0.0)
    with pytest.raises(ValueError):
        PageHinkleyConfig(alpha=1.0)


def test_page_hinkley_alpha_rejects_bool():
    with pytest.raises(TypeError):
        PageHinkleyConfig(alpha=True)


def test_page_hinkley_threshold_rejects_non_positive():
    with pytest.raises(ValueError):
        PageHinkleyConfig(threshold=0.0)
    with pytest.raises(ValueError):
        PageHinkleyConfig(threshold=-1.0)


# ============================================================
# Common adapter contract
# ============================================================

class DummyAdapter(StreamingDetectorAdapter):
    """
    Minimal fake implementation for contract testing.
    It deliberately does not use River.
    """

    def __init__(self, run_id="RUN_001"):
        super().__init__(run_id=run_id, detector_name="DUMMY")

    def update(self, observation):
        self._validate_observation(observation)
        self._record_sample_index(observation)

        result = DetectorUpdateResult(
            detector_name=self.detector_name,
            run_id=self.run_id,
            sample_index=observation.sample_index,
            reported_window_id=observation.external_window_id,
            detection=False,
            state=self.state,
        )

        return result

    def reset(self):
        self._mark_reset()
        self._mark_active()


def test_adapter_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        StreamingDetectorAdapter(run_id="RUN_001", detector_name="DUMMY")


def test_dummy_adapter_has_run_identity():
    adapter = DummyAdapter()

    assert adapter.run_id == "RUN_001"
    assert adapter.detector_name == "DUMMY"
    assert adapter.state is DetectorState.ACTIVE
    assert adapter.last_sample_index is None


def test_adapter_accepts_first_sample():
    adapter = DummyAdapter()

    observation = PredictionErrorObservation(
        run_id="RUN_001",
        sample_index=10,
        external_window_id="WIN_001",
        error=0,
    )

    result = adapter.update(observation)

    assert result.sample_index == 10
    assert adapter.last_sample_index == 10


def test_adapter_rejects_run_id_mismatch():
    adapter = DummyAdapter(run_id="RUN_001")

    observation = PredictionErrorObservation(
        run_id="RUN_002",
        sample_index=1,
        external_window_id="WIN_001",
        error=0,
    )

    with pytest.raises(ValueError):
        adapter.update(observation)


def test_adapter_rejects_duplicate_sample_index():
    adapter = DummyAdapter()

    first = PredictionErrorObservation(
        run_id="RUN_001", sample_index=10, external_window_id="WIN_001", error=0,
    )
    second = PredictionErrorObservation(
        run_id="RUN_001", sample_index=10, external_window_id="WIN_001", error=1,
    )

    adapter.update(first)

    with pytest.raises(ValueError):
        adapter.update(second)


def test_adapter_rejects_out_of_order_sample_index():
    adapter = DummyAdapter()

    first = PredictionErrorObservation(
        run_id="RUN_001", sample_index=10, external_window_id="WIN_001", error=0,
    )
    second = PredictionErrorObservation(
        run_id="RUN_001", sample_index=9, external_window_id="WIN_001", error=1,
    )

    adapter.update(first)

    with pytest.raises(ValueError):
        adapter.update(second)


def test_reset_does_not_clear_sample_index():
    adapter = DummyAdapter()

    observation = PredictionErrorObservation(
        run_id="RUN_001", sample_index=50, external_window_id="WIN_001", error=0,
    )

    adapter.update(observation)
    adapter.reset()

    assert adapter.last_sample_index == 50


def test_reset_returns_adapter_to_active_state():
    adapter = DummyAdapter()

    adapter._mark_drift_detected()
    adapter._enter_latched("WIN_001")

    assert adapter.state is DetectorState.LATCHED

    adapter.reset()

    assert adapter.state is DetectorState.ACTIVE
    assert adapter.latch_window_id is None


def test_latched_new_window_is_detected():
    adapter = DummyAdapter()
    adapter._enter_latched("WIN_001")

    assert adapter._is_latched_into_new_window("WIN_002") is True


def test_latched_same_window_is_not_timeout():
    adapter = DummyAdapter()
    adapter._enter_latched("WIN_001")

    assert adapter._is_latched_into_new_window("WIN_001") is False


def test_lifecycle_state_helpers():
    adapter = DummyAdapter()

    adapter._mark_drift_detected()
    assert adapter.state is DetectorState.DRIFT_DETECTED

    adapter._enter_latched("WIN_001")
    assert adapter.state is DetectorState.LATCHED

    adapter._mark_unresolved_timeout()
    assert adapter.state is DetectorState.UNRESOLVED_TIMEOUT


def test_update_result_contains_enum_state():
    adapter = DummyAdapter()

    observation = PredictionErrorObservation(
        run_id="RUN_001", sample_index=1, external_window_id="WIN_001", error=0,
    )

    result = adapter.update(observation)

    assert isinstance(result.state, DetectorState)
