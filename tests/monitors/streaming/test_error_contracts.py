
import pytest
from datetime import datetime, timezone

from src.monitors.streaming.error_contracts import (
    DetectorFactory,
    DetectorState,
    DetectorUpdateResult,
    PredictionErrorObservation,
)


# ============================================================
# PredictionErrorObservation
# ============================================================

def test_valid_prediction_error_observation():
    observation = PredictionErrorObservation(
        run_id="RUN_001",
        sample_index=42,
        external_window_id="WIN_000001",
        error=1,
    )

    assert observation.run_id == "RUN_001"
    assert observation.sample_index == 42
    assert observation.external_window_id == "WIN_000001"
    assert observation.error == 1
    assert observation.timestamp is None


def test_prediction_error_zero_is_valid():
    observation = PredictionErrorObservation(
        run_id="RUN_001",
        sample_index=0,
        external_window_id="WIN_000001",
        error=0,
    )

    assert observation.error == 0


def test_timestamp_is_optional():
    timestamp = datetime.now(timezone.utc)

    observation = PredictionErrorObservation(
        run_id="RUN_001",
        sample_index=1,
        external_window_id="WIN_000001",
        error=0,
        timestamp=timestamp,
    )

    assert observation.timestamp == timestamp


def test_invalid_error_float_zero_is_rejected():
    with pytest.raises(TypeError):
        PredictionErrorObservation(
            run_id="RUN_001",
            sample_index=1,
            external_window_id="WIN_000001",
            error=0.0,
        )


def test_invalid_error_float_one_is_rejected():
    with pytest.raises(TypeError):
        PredictionErrorObservation(
            run_id="RUN_001",
            sample_index=1,
            external_window_id="WIN_000001",
            error=1.0,
        )


def test_invalid_error_bool_is_rejected():
    with pytest.raises(TypeError):
        PredictionErrorObservation(
            run_id="RUN_001",
            sample_index=1,
            external_window_id="WIN_000001",
            error=True,
        )


def test_invalid_error_negative_is_rejected():
    with pytest.raises(ValueError):
        PredictionErrorObservation(
            run_id="RUN_001",
            sample_index=1,
            external_window_id="WIN_000001",
            error=-1,
        )


def test_invalid_error_two_is_rejected():
    with pytest.raises(ValueError):
        PredictionErrorObservation(
            run_id="RUN_001",
            sample_index=1,
            external_window_id="WIN_000001",
            error=2,
        )


def test_negative_sample_index_is_rejected():
    with pytest.raises(ValueError):
        PredictionErrorObservation(
            run_id="RUN_001",
            sample_index=-1,
            external_window_id="WIN_000001",
            error=0,
        )


def test_boolean_sample_index_is_rejected():
    with pytest.raises(TypeError):
        PredictionErrorObservation(
            run_id="RUN_001",
            sample_index=True,
            external_window_id="WIN_000001",
            error=0,
        )


def test_invalid_timestamp_type_is_rejected():
    with pytest.raises(TypeError):
        PredictionErrorObservation(
            run_id="RUN_001",
            sample_index=1,
            external_window_id="WIN_000001",
            error=0,
            timestamp="2026-01-01",
        )


def test_observation_metadata_is_hash_safe():
    observation = PredictionErrorObservation(
        run_id="RUN_001",
        sample_index=1,
        external_window_id="WIN_000001",
        error=0,
        metadata={"source": "model"},
    )

    hash(observation)


# ============================================================
# DetectorState
# ============================================================

def test_detector_state_is_enum():
    assert isinstance(
        DetectorState.ACTIVE,
        DetectorState,
    )


def test_detector_state_contains_all_locked_states():
    expected = {
        DetectorState.ACTIVE,
        DetectorState.DRIFT_DETECTED,
        DetectorState.LATCHED,
        DetectorState.RESOLVED,
        DetectorState.RESET,
        DetectorState.UNRESOLVED_TIMEOUT,
    }

    assert set(DetectorState) == expected


def test_detector_state_is_not_free_form_string():
    with pytest.raises(TypeError):
        DetectorUpdateResult(
            detector_name="ADWIN",
            run_id="RUN_001",
            sample_index=1,
            reported_window_id="WIN_000001",
            detection=False,
            state="active",
        )


# ============================================================
# DetectorUpdateResult
# ============================================================

def test_valid_detector_update_result():
    result = DetectorUpdateResult(
        detector_name="ADWIN",
        run_id="RUN_001",
        sample_index=500,
        reported_window_id="WIN_000001",
        detection=True,
        state=DetectorState.DRIFT_DETECTED,
    )

    assert result.detector_name == "ADWIN"
    assert result.detection is True
    assert result.state is DetectorState.DRIFT_DETECTED


def test_detector_update_result_rejects_bool_detection():
    with pytest.raises(TypeError):
        DetectorUpdateResult(
            detector_name="ADWIN",
            run_id="RUN_001",
            sample_index=500,
            reported_window_id="WIN_000001",
            detection=1,
            state=DetectorState.DRIFT_DETECTED,
        )


def test_detector_update_result_rejects_negative_index():
    with pytest.raises(ValueError):
        DetectorUpdateResult(
            detector_name="ADWIN",
            run_id="RUN_001",
            sample_index=-1,
            reported_window_id="WIN_000001",
            detection=False,
            state=DetectorState.ACTIVE,
        )


def test_detector_update_result_metadata_is_hash_safe():
    result = DetectorUpdateResult(
        detector_name="DDM",
        run_id="RUN_001",
        sample_index=10,
        reported_window_id="WIN_000001",
        detection=False,
        state=DetectorState.ACTIVE,
        metadata={"river_version": "0.23.0"},
    )

    hash(result)


# ============================================================
# DetectorFactory / run isolation
# ============================================================

def test_factory_can_reserve_new_run_id():
    factory = DetectorFactory()

    returned = factory.reserve_run_id(
        "RUN_001"
    )

    assert returned == "RUN_001"
    assert factory.issued_run_ids == frozenset(
        {"RUN_001"}
    )


def test_factory_rejects_run_id_reuse():
    factory = DetectorFactory()

    factory.reserve_run_id("RUN_001")

    with pytest.raises(ValueError):
        factory.reserve_run_id("RUN_001")


def test_factory_allows_different_run_ids():
    factory = DetectorFactory()

    factory.reserve_run_id("RUN_001")
    factory.reserve_run_id("RUN_002")

    assert factory.issued_run_ids == frozenset(
        {
            "RUN_001",
            "RUN_002",
        }
    )


def test_different_factory_can_start_same_run_id():
    """
    Run isolation is scoped to one factory lifecycle.

    A new factory represents a new detector-factory lifecycle,
    so it may independently reserve the same run_id.
    """
    factory_a = DetectorFactory()
    factory_b = DetectorFactory()

    assert (
        factory_a.reserve_run_id("RUN_001")
        == "RUN_001"
    )

    assert (
        factory_b.reserve_run_id("RUN_001")
        == "RUN_001"
    )


def test_factory_rejects_empty_run_id():
    factory = DetectorFactory()

    with pytest.raises(ValueError):
        factory.reserve_run_id("")


def test_factory_rejects_non_string_run_id():
    factory = DetectorFactory()

    with pytest.raises(ValueError):
        factory.reserve_run_id(123)
