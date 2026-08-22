
import pytest

from src.monitors.batch.entropy import (
    EntropyComputationInput,
)

from src.monitors.batch.entropy_detector import (
    compute_entropy_drift,
)

from src.monitors.batch.ks import (
    KSResult,
)


def test_entropy_detector_returns_existing_ksresult():
    entropy_input = EntropyComputationInput(
        reference_probabilities=[
            0.01,
            0.02,
            0.03,
            0.04,
        ],
        current_probabilities=[
            0.45,
            0.50,
            0.55,
            0.60,
        ],
    )

    result = compute_entropy_drift(
        computation_input=entropy_input,
        feature_name="prediction_entropy",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    assert isinstance(
        result,
        KSResult,
    )


def test_entropy_detector_preserves_sample_counts():
    entropy_input = EntropyComputationInput(
        reference_probabilities=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ],
        current_probabilities=[
            0.2,
            0.4,
            0.6,
        ],
    )

    result = compute_entropy_drift(
        computation_input=entropy_input,
        feature_name="prediction_entropy",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    assert result.n_ref == 5
    assert result.n_cur == 3


def test_entropy_detector_preserves_window_ids():
    entropy_input = EntropyComputationInput(
        reference_probabilities=[
            0.1,
            0.2,
            0.3,
        ],
        current_probabilities=[
            0.7,
            0.8,
            0.9,
        ],
    )

    result = compute_entropy_drift(
        computation_input=entropy_input,
        feature_name="prediction_entropy",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    assert result.reference_window_id == "REF_001"
    assert result.current_window_id == "WIN_001"


def test_entropy_direction_flip_is_not_detected_as_entropy_shift():
    entropy_input = EntropyComputationInput(
        reference_probabilities=[
            0.1,
            0.1,
            0.9,
            0.9,
        ],
        current_probabilities=[
            0.9,
            0.9,
            0.1,
            0.1,
        ],
    )

    result = compute_entropy_drift(
        computation_input=entropy_input,
        feature_name="prediction_entropy",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    assert result.d_statistic == 0.0
    assert result.p_value == 1.0


def test_entropy_detector_rejects_same_window():
    entropy_input = EntropyComputationInput(
        reference_probabilities=[
            0.1,
            0.2,
        ],
        current_probabilities=[
            0.7,
            0.8,
        ],
    )

    with pytest.raises(ValueError):
        compute_entropy_drift(
            computation_input=entropy_input,
            feature_name="prediction_entropy",
            reference_window_id="WIN_001",
            current_window_id="WIN_001",
        )


def test_entropy_detector_rejects_invalid_input_type():
    with pytest.raises(TypeError):
        compute_entropy_drift(
            computation_input="invalid",
            feature_name="prediction_entropy",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
        )
