from datetime import datetime, timezone

import pytest

from src.contracts.results import DetectorResult


def valid_result() -> DetectorResult:
    return DetectorResult(
        run_id="TEST_RUN_001",
        detector_name="ADWIN",
        detector_instance_id="ADWIN_0001",
        reported_window_id="WIN_042",
        detection_index=41650,
        drift_detected=True,
        score=0.91,
        criterion="drift_detected",
        observation_count=650,
        timestamp=datetime.now(timezone.utc),
    )


def test_valid_detector_result():
    result = valid_result()

    assert result.detector_name == "ADWIN"
    assert result.reported_window_id == "WIN_042"
    assert result.detection_index == 41650
    assert result.drift_detected is True


def test_detection_index_can_be_none():
    result = DetectorResult(
        run_id="TEST_RUN_001",
        detector_name="KS",
        detector_instance_id="KS_0001",
        reported_window_id="WIN_042",
        detection_index=None,
        drift_detected=False,
        score=0.12,
        criterion="p_value >= 0.05",
        observation_count=1000,
        timestamp=datetime.now(timezone.utc),
    )

    assert result.detection_index is None


def test_negative_detection_index_rejected():
    with pytest.raises(ValueError):
        DetectorResult(
            run_id="TEST_RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_0001",
            reported_window_id="WIN_042",
            detection_index=-1,
            drift_detected=True,
            score=0.9,
            criterion="drift_detected",
            observation_count=10,
            timestamp=datetime.now(timezone.utc),
        )


def test_negative_observation_count_rejected():
    with pytest.raises(ValueError):
        DetectorResult(
            run_id="TEST_RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_0001",
            reported_window_id="WIN_042",
            detection_index=10,
            drift_detected=True,
            score=0.9,
            criterion="drift_detected",
            observation_count=-1,
            timestamp=datetime.now(timezone.utc),
        )


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        DetectorResult(
            run_id="TEST_RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_0001",
            reported_window_id="WIN_042",
            detection_index=10,
            drift_detected=True,
            score=0.9,
            criterion="drift_detected",
            observation_count=10,
            timestamp=datetime.now(),
        )


def test_score_can_be_none():
    result = DetectorResult(
        run_id="TEST_RUN_001",
        detector_name="DDM",
        detector_instance_id="DDM_0001",
        reported_window_id="WIN_042",
        detection_index=100,
        drift_detected=True,
        score=None,
        criterion=None,
        observation_count=100,
        timestamp=datetime.now(timezone.utc),
    )

    assert result.score is None


def test_metadata_must_be_dict():
    with pytest.raises(TypeError):
        DetectorResult(
            run_id="TEST_RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_0001",
            reported_window_id="WIN_042",
            detection_index=10,
            drift_detected=True,
            score=0.9,
            criterion="drift_detected",
            observation_count=10,
            timestamp=datetime.now(timezone.utc),
            metadata="invalid",
        )


def test_bool_score_rejected():
    # bool is a subclass of int, so isinstance(True, (int, float)) is
    # True; without an explicit exclusion, a stray boolean could
    # silently pass as a "valid" numeric score.
    with pytest.raises(TypeError):
        DetectorResult(
            run_id="TEST_RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_0001",
            reported_window_id="WIN_042",
            detection_index=10,
            drift_detected=True,
            score=True,
            criterion="drift_detected",
            observation_count=10,
            timestamp=datetime.now(timezone.utc),
        )


def test_bool_detection_index_rejected():
    with pytest.raises(TypeError):
        DetectorResult(
            run_id="TEST_RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_0001",
            reported_window_id="WIN_042",
            detection_index=True,
            drift_detected=True,
            score=0.9,
            criterion="drift_detected",
            observation_count=10,
            timestamp=datetime.now(timezone.utc),
        )


def test_bool_observation_count_rejected():
    with pytest.raises(TypeError):
        DetectorResult(
            run_id="TEST_RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_0001",
            reported_window_id="WIN_042",
            detection_index=10,
            drift_detected=True,
            score=0.9,
            criterion="drift_detected",
            observation_count=True,
            timestamp=datetime.now(timezone.utc),
        )


def test_detector_result_is_hashable():
    result = valid_result()
    # Should not raise TypeError even though metadata is a dict
    hash(result)


def test_detector_result_metadata_non_string_keys_rejected():
    with pytest.raises(TypeError):
        DetectorResult(
            run_id="TEST_RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_0001",
            reported_window_id="WIN_042",
            detection_index=10,
            drift_detected=True,
            score=0.9,
            criterion="drift_detected",
            observation_count=10,
            timestamp=datetime.now(timezone.utc),
            metadata={17: "invalid_key"},
        )
