import pytest

from src.monitors.batch.confidence import (
    ConfidenceDriftResult,
)

from src.monitors.batch.confidence_detector import (
    compute_confidence_drift,
)

from src.monitors.batch.entropy import (
    EntropyComputationInput,
)

from src.monitors.batch.ks import (
    KSComputationInput,
    compute_ks,
)


def _input():
    return EntropyComputationInput(
        reference_probabilities=[0.01, 0.02, 0.03, 0.04, 0.05],
        current_probabilities=[0.45, 0.48, 0.50, 0.52, 0.55],
    )


def test_returns_confidence_drift_result():
    result = compute_confidence_drift(
        computation_input=_input(),
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )
    assert isinstance(result, ConfidenceDriftResult)


def test_preserves_ks_statistics():
    entropy_input = _input()

    result = compute_confidence_drift(
        computation_input=entropy_input,
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    direct_ks = compute_ks(
        computation_input=KSComputationInput(
            reference_values=entropy_input.reference_entropy,
            current_values=entropy_input.current_entropy,
            min_samples=entropy_input.min_samples,
        ),
        feature_name="prediction_entropy",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    assert result.d_statistic == direct_ks.d_statistic
    assert result.p_value == direct_ks.p_value


def test_preserves_sample_counts():
    result = compute_confidence_drift(
        computation_input=_input(),
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )
    assert result.n_ref == 5
    assert result.n_cur == 5


def test_preserves_window_ids():
    result = compute_confidence_drift(
        computation_input=_input(),
        reference_window_id="REF_123",
        current_window_id="WIN_456",
    )
    assert result.reference_window_id == "REF_123"
    assert result.current_window_id == "WIN_456"


def test_significance_is_derived_using_alpha():
    result = compute_confidence_drift(
        computation_input=_input(),
        reference_window_id="REF_001",
        current_window_id="WIN_001",
        alpha=0.05,
    )
    assert result.significant == (result.p_value <= 0.05)


def test_custom_alpha_is_respected():
    result = compute_confidence_drift(
        computation_input=_input(),
        reference_window_id="REF_001",
        current_window_id="WIN_001",
        alpha=0.50,
    )
    assert result.alpha == 0.50
    assert result.significant == (result.p_value <= 0.50)


def test_constant_mapping_is_explicit():
    entropy_input = EntropyComputationInput(
        reference_probabilities=[0.5, 0.5, 0.5, 0.5],
        current_probabilities=[0.1, 0.2, 0.8, 0.9],
    )

    result = compute_confidence_drift(
        computation_input=entropy_input,
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    assert result.entropy_constant_reference is True
    assert result.entropy_constant_current is False


def test_metadata_is_automatically_set():
    result = compute_confidence_drift(
        computation_input=_input(),
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )
    assert result.metadata == {"signal": "predictive_entropy"}


def test_invalid_input_type_is_rejected():
    with pytest.raises(TypeError):
        compute_confidence_drift(
            computation_input="invalid",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
        )


def test_same_window_is_rejected_by_ks_layer():
    with pytest.raises(ValueError):
        compute_confidence_drift(
            computation_input=_input(),
            reference_window_id="WIN_001",
            current_window_id="WIN_001",
        )


def test_custom_alpha_invalid_value_propagates():
    with pytest.raises(ValueError):
        compute_confidence_drift(
            computation_input=_input(),
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            alpha=1.0,
        )


def test_custom_alpha_bool_is_rejected():
    with pytest.raises(TypeError):
        compute_confidence_drift(
            computation_input=_input(),
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            alpha=True,
        )
