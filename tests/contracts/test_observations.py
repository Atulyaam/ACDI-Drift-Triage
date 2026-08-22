from datetime import datetime, timezone

import pytest

from src.contracts.observations import (
    FeatureDistributionObservation,
    ConfidenceDistributionObservation,
    PredictionErrorObservation,
)


def valid_observation() -> FeatureDistributionObservation:
    return FeatureDistributionObservation(
        run_id="TEST_RUN_001",
        feature_name="flow_duration",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
        timestamp=datetime.now(timezone.utc),
    )


def test_valid_feature_distribution_observation():
    observation = valid_observation()

    assert observation.run_id == "TEST_RUN_001"
    assert observation.feature_name == "flow_duration"
    assert observation.reference_window_id == "REF_001"
    assert observation.current_window_id == "WIN_001"


def test_empty_run_id_rejected():
    with pytest.raises(ValueError):
        FeatureDistributionObservation(
            run_id="",
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            timestamp=datetime.now(timezone.utc),
        )


def test_empty_feature_name_rejected():
    with pytest.raises(ValueError):
        FeatureDistributionObservation(
            run_id="TEST_RUN_001",
            feature_name="",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            timestamp=datetime.now(timezone.utc),
        )


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        FeatureDistributionObservation(
            run_id="TEST_RUN_001",
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            timestamp=datetime.now(),
        )


def test_metadata_must_be_dict():
    with pytest.raises(TypeError):
        FeatureDistributionObservation(
            run_id="TEST_RUN_001",
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            timestamp=datetime.now(timezone.utc),
            metadata="invalid",
        )


def test_same_reference_and_current_window_rejected():
    with pytest.raises(ValueError):
        FeatureDistributionObservation(
            run_id="TEST_RUN_001",
            feature_name="flow_duration",
            reference_window_id="WIN_042",
            current_window_id="WIN_042",
            timestamp=datetime.now(timezone.utc),
        )


def test_observation_is_hashable_despite_metadata():
    observation = FeatureDistributionObservation(
        run_id="TEST_RUN_001",
        feature_name="flow_duration",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
        timestamp=datetime.now(timezone.utc),
        metadata={"feature_index": 17},
    )
    # Should not raise TypeError even though metadata is a dict
    hash(observation)


def test_metadata_non_string_keys_rejected():
    with pytest.raises(TypeError):
        FeatureDistributionObservation(
            run_id="TEST_RUN_001",
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            timestamp=datetime.now(timezone.utc),
            metadata={17: "invalid_key"},
        )


def valid_confidence_observation() -> ConfidenceDistributionObservation:
    return ConfidenceDistributionObservation(
        run_id="TEST_RUN_001",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
        confidence_definition="max_class_prob",
        timestamp=datetime.now(timezone.utc),
    )


def test_valid_confidence_observation():
    observation = valid_confidence_observation()

    assert observation.run_id == "TEST_RUN_001"
    assert observation.confidence_definition == "max_class_prob"


def test_invalid_confidence_definition_rejected():
    with pytest.raises(ValueError):
        ConfidenceDistributionObservation(
            run_id="TEST_RUN_001",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            confidence_definition="invalid",
            timestamp=datetime.now(timezone.utc),
        )


def test_confidence_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        ConfidenceDistributionObservation(
            run_id="TEST_RUN_001",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            confidence_definition="max_class_prob",
            timestamp=datetime.now(),
        )


def test_confidence_same_reference_and_current_window_rejected():
    with pytest.raises(ValueError):
        ConfidenceDistributionObservation(
            run_id="TEST_RUN_001",
            reference_window_id="WIN_042",
            current_window_id="WIN_042",
            confidence_definition="max_class_prob",
            timestamp=datetime.now(timezone.utc),
        )


def test_confidence_observation_is_hashable_despite_metadata():
    observation = ConfidenceDistributionObservation(
        run_id="TEST_RUN_001",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
        confidence_definition="max_class_prob",
        timestamp=datetime.now(timezone.utc),
        metadata={"note": "baseline"},
    )
    hash(observation)


def test_confidence_metadata_non_string_keys_rejected():
    with pytest.raises(TypeError):
        ConfidenceDistributionObservation(
            run_id="TEST_RUN_001",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            confidence_definition="max_class_prob",
            timestamp=datetime.now(timezone.utc),
            metadata={17: "invalid_key"},
        )


def valid_prediction_error_observation():
    return PredictionErrorObservation(
        run_id="TEST_RUN_001",
        window_id="WIN_001",
        error_stream=(0, 1, 0, 0, 1),
        sample_count=5,
        start_index=100,
        end_index=104,
        timestamp=datetime.now(timezone.utc),
    )


def test_valid_prediction_error_observation():
    observation = valid_prediction_error_observation()

    assert observation.error_stream == (
        0, 1, 0, 0, 1
    )
    assert observation.sample_count == 5


def test_error_stream_rejects_invalid_values():
    with pytest.raises(ValueError):
        PredictionErrorObservation(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            error_stream=(0, 1, 2),
            sample_count=3,
            start_index=0,
            end_index=2,
            timestamp=datetime.now(timezone.utc),
        )


def test_error_stream_rejects_negative_values():
    with pytest.raises(ValueError):
        PredictionErrorObservation(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            error_stream=(-1, 0, 1),
            sample_count=3,
            start_index=0,
            end_index=2,
            timestamp=datetime.now(timezone.utc),
        )


def test_error_stream_rejects_bool_values():
    # bool is a subclass of int; True == 1 and False == 0, so a naive
    # "value not in (0, 1)" check would silently accept these.
    with pytest.raises(TypeError):
        PredictionErrorObservation(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            error_stream=(0, 1, True),
            sample_count=3,
            start_index=0,
            end_index=2,
            timestamp=datetime.now(timezone.utc),
        )


def test_error_stream_rejects_float_values():
    # 1.0 == 1 in Python, so a naive membership check would silently
    # accept floats too.
    with pytest.raises(TypeError):
        PredictionErrorObservation(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            error_stream=(0, 1.0, 0),
            sample_count=3,
            start_index=0,
            end_index=2,
            timestamp=datetime.now(timezone.utc),
        )


def test_sample_count_must_match_stream_length():
    with pytest.raises(ValueError):
        PredictionErrorObservation(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            error_stream=(0, 1, 0),
            sample_count=2,
            start_index=0,
            end_index=2,
            timestamp=datetime.now(timezone.utc),
        )


def test_end_index_must_match_sample_count():
    with pytest.raises(ValueError):
        PredictionErrorObservation(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            error_stream=(0, 1, 0),
            sample_count=3,
            start_index=10,
            end_index=99,
            timestamp=datetime.now(timezone.utc),
        )


def test_naive_timestamp_rejected_for_error_observation():
    with pytest.raises(ValueError):
        PredictionErrorObservation(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            error_stream=(0, 1),
            sample_count=2,
            start_index=0,
            end_index=1,
            timestamp=datetime.now(),
        )


def test_prediction_error_observation_is_hashable():
    observation = valid_prediction_error_observation()
    # Should not raise TypeError even though metadata is a dict
    hash(observation)


def test_prediction_error_metadata_non_string_keys_rejected():
    with pytest.raises(TypeError):
        PredictionErrorObservation(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            error_stream=(0, 1, 0),
            sample_count=3,
            start_index=0,
            end_index=2,
            timestamp=datetime.now(timezone.utc),
            metadata={17: "invalid_key"},
        )
