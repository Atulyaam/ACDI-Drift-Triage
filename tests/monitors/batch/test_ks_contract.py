import numpy as np
import pytest

from src.monitors.batch.ks import (
    KSComputationInput,
    KSResult,
    compute_ks,
)


# ============================================================
# KSComputationInput tests
# ============================================================

def test_default_min_samples_is_technical_floor():
    obj = KSComputationInput(
        reference_values=[1.0, 2.0],
        current_values=[3.0, 4.0],
    )

    assert obj.min_samples == 2


def test_lists_are_converted_to_float64_arrays():
    obj = KSComputationInput(
        reference_values=[1, 2, 3],
        current_values=[4, 5, 6],
    )

    assert isinstance(
        obj.reference_values,
        np.ndarray,
    )
    assert isinstance(
        obj.current_values,
        np.ndarray,
    )
    assert obj.reference_values.dtype == np.float64
    assert obj.current_values.dtype == np.float64


def test_ndarrays_are_accepted():
    obj = KSComputationInput(
        reference_values=np.array([1, 2, 3]),
        current_values=np.array([4, 5, 6]),
    )

    assert obj.reference_values.shape == (3,)
    assert obj.current_values.shape == (3,)


def test_non_numeric_reference_rejected_with_type_error():
    with pytest.raises(TypeError):
        KSComputationInput(
            reference_values=["a", "b"],
            current_values=[1.0, 2.0],
        )


def test_non_numeric_current_rejected_with_type_error():
    with pytest.raises(TypeError):
        KSComputationInput(
            reference_values=[1.0, 2.0],
            current_values=["a", "b"],
        )


def test_nan_reference_rejected():
    with pytest.raises(ValueError):
        KSComputationInput(
            reference_values=[1.0, np.nan],
            current_values=[1.0, 2.0],
        )


def test_nan_current_rejected():
    with pytest.raises(ValueError):
        KSComputationInput(
            reference_values=[1.0, 2.0],
            current_values=[1.0, np.nan],
        )


def test_positive_infinity_rejected():
    with pytest.raises(ValueError):
        KSComputationInput(
            reference_values=[1.0, np.inf],
            current_values=[1.0, 2.0],
        )


def test_negative_infinity_rejected():
    with pytest.raises(ValueError):
        KSComputationInput(
            reference_values=[1.0, 2.0],
            current_values=[1.0, -np.inf],
        )


def test_empty_reference_rejected():
    with pytest.raises(ValueError):
        KSComputationInput(
            reference_values=[],
            current_values=[1.0, 2.0],
        )


def test_empty_current_rejected():
    with pytest.raises(ValueError):
        KSComputationInput(
            reference_values=[1.0, 2.0],
            current_values=[],
        )


def test_min_samples_enforced():
    with pytest.raises(ValueError):
        KSComputationInput(
            reference_values=[1.0],
            current_values=[2.0, 3.0],
            min_samples=2,
        )


def test_min_samples_rejects_bool():
    with pytest.raises(TypeError):
        KSComputationInput(
            reference_values=[1.0, 2.0],
            current_values=[2.0, 3.0],
            min_samples=True,
        )


def test_min_samples_below_technical_floor_rejected():
    with pytest.raises(ValueError):
        KSComputationInput(
            reference_values=[1.0, 2.0],
            current_values=[2.0, 3.0],
            min_samples=1,
        )


def test_multidimensional_reference_rejected():
    with pytest.raises(ValueError):
        KSComputationInput(
            reference_values=[
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            current_values=[1.0, 2.0],
        )


def test_multidimensional_current_rejected():
    with pytest.raises(ValueError):
        KSComputationInput(
            reference_values=[1.0, 2.0],
            current_values=[
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )


def test_constant_reference_flag():
    obj = KSComputationInput(
        reference_values=[5.0, 5.0, 5.0],
        current_values=[1.0, 2.0, 3.0],
    )

    assert obj.is_constant_reference is True
    assert obj.is_constant_current is False


def test_constant_current_flag():
    obj = KSComputationInput(
        reference_values=[1.0, 2.0, 3.0],
        current_values=[5.0, 5.0, 5.0],
    )

    assert obj.is_constant_reference is False
    assert obj.is_constant_current is True


def test_both_constant_flags():
    obj = KSComputationInput(
        reference_values=[5.0, 5.0, 5.0],
        current_values=[7.0, 7.0, 7.0],
    )

    assert obj.is_constant_reference is True
    assert obj.is_constant_current is True


def test_input_arrays_are_read_only():
    obj = KSComputationInput(
        reference_values=[1.0, 2.0],
        current_values=[3.0, 4.0],
    )

    with pytest.raises(ValueError):
        obj.reference_values[0] = 99.0


def test_eq_false_avoids_numpy_array_equality_issue():
    first = KSComputationInput(
        reference_values=[1.0, 2.0],
        current_values=[3.0, 4.0],
    )

    second = KSComputationInput(
        reference_values=[1.0, 2.0],
        current_values=[3.0, 4.0],
    )

    assert first is not second
    assert first != second


# ============================================================
# KSResult tests
# ============================================================

def test_valid_ks_result():
    result = KSResult(
        feature_name="flow_duration",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
        d_statistic=0.25,
        p_value=0.03,
        n_ref=100,
        n_cur=100,
        is_constant_reference=False,
        is_constant_current=False,
    )

    assert result.feature_name == "flow_duration"
    assert result.d_statistic == 0.25
    assert result.p_value == 0.03


def test_ks_result_rejects_invalid_counts():
    with pytest.raises(ValueError):
        KSResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            d_statistic=0.25,
            p_value=0.03,
            n_ref=0,
            n_cur=100,
            is_constant_reference=False,
            is_constant_current=False,
        )


def test_ks_result_rejects_same_reference_and_current_window():
    with pytest.raises(ValueError):
        KSResult(
            feature_name="flow_duration",
            reference_window_id="WIN_001",
            current_window_id="WIN_001",
            d_statistic=0.25,
            p_value=0.03,
            n_ref=100,
            n_cur=100,
            is_constant_reference=False,
            is_constant_current=False,
        )


def test_ks_result_rejects_d_statistic_out_of_range():
    with pytest.raises(ValueError):
        KSResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            d_statistic=1.5,
            p_value=0.03,
            n_ref=100,
            n_cur=100,
            is_constant_reference=False,
            is_constant_current=False,
        )


def test_ks_result_rejects_negative_p_value():
    with pytest.raises(ValueError):
        KSResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            d_statistic=0.25,
            p_value=-0.01,
            n_ref=100,
            n_cur=100,
            is_constant_reference=False,
            is_constant_current=False,
        )


def test_ks_result_rejects_p_value_above_one():
    with pytest.raises(ValueError):
        KSResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            d_statistic=0.25,
            p_value=1.01,
            n_ref=100,
            n_cur=100,
            is_constant_reference=False,
            is_constant_current=False,
        )


def test_ks_result_rejects_d_statistic_bool():
    with pytest.raises(TypeError):
        KSResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            d_statistic=True,
            p_value=0.03,
            n_ref=100,
            n_cur=100,
            is_constant_reference=False,
            is_constant_current=False,
        )


def test_ks_result_rejects_p_value_bool():
    with pytest.raises(TypeError):
        KSResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            d_statistic=0.25,
            p_value=False,
            n_ref=100,
            n_cur=100,
            is_constant_reference=False,
            is_constant_current=False,
        )


def test_ks_result_with_metadata_is_hashable():
    result = KSResult(
        feature_name="flow_duration",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
        d_statistic=0.25,
        p_value=0.03,
        n_ref=100,
        n_cur=100,
        is_constant_reference=False,
        is_constant_current=False,
        metadata={"method": "ks_2samp"},
    )

    hash(result)


def test_ks_result_metadata_rejects_non_string_keys():
    with pytest.raises(TypeError):
        KSResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            d_statistic=0.25,
            p_value=0.03,
            n_ref=100,
            n_cur=100,
            is_constant_reference=False,
            is_constant_current=False,
            metadata={1: "bad_key"},
        )


# ============================================================
# compute_ks() implementation tests
# ============================================================

def test_compute_ks_identical_distributions():
    result = compute_ks(
        KSComputationInput(
            reference_values=[1.0, 2.0, 3.0, 4.0],
            current_values=[1.0, 2.0, 3.0, 4.0],
        ),
        feature_name="flow_duration",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    assert result.d_statistic == 0.0
    assert 0.0 <= result.p_value <= 1.0
    assert result.n_ref == 4
    assert result.n_cur == 4


def test_compute_ks_shifted_distributions():
    result = compute_ks(
        KSComputationInput(
            reference_values=[
                0.0, 0.1, 0.2, 0.3,
                0.4, 0.5, 0.6, 0.7,
            ],
            current_values=[
                10.0, 10.1, 10.2, 10.3,
                10.4, 10.5, 10.6, 10.7,
            ],
        ),
        feature_name="flow_duration",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    assert result.d_statistic > 0.0
    assert 0.0 <= result.p_value <= 1.0


def test_compute_ks_constant_reference_allowed():
    result = compute_ks(
        KSComputationInput(
            reference_values=[5.0, 5.0, 5.0, 5.0],
            current_values=[1.0, 2.0, 3.0, 4.0],
        ),
        feature_name="flow_duration",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    assert result.is_constant_reference is True
    assert result.is_constant_current is False
    assert 0.0 <= result.d_statistic <= 1.0
    assert 0.0 <= result.p_value <= 1.0


def test_compute_ks_constant_both_allowed():
    result = compute_ks(
        KSComputationInput(
            reference_values=[5.0, 5.0, 5.0, 5.0],
            current_values=[5.0, 5.0, 5.0, 5.0],
        ),
        feature_name="flow_duration",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    assert result.is_constant_reference is True
    assert result.is_constant_current is True
    assert result.d_statistic == 0.0


def test_compute_ks_rejects_same_window():
    with pytest.raises(ValueError):
        compute_ks(
            KSComputationInput(
                reference_values=[1.0, 2.0],
                current_values=[3.0, 4.0],
            ),
            feature_name="flow_duration",
            reference_window_id="WIN_001",
            current_window_id="WIN_001",
        )


def test_compute_ks_rejects_invalid_input_type():
    with pytest.raises(TypeError):
        compute_ks(
            computation_input="invalid",
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
        )


def test_compute_ks_stores_locked_method_metadata():
    result = compute_ks(
        KSComputationInput(
            reference_values=[1.0, 2.0, 3.0],
            current_values=[1.0, 2.0, 3.0],
        ),
        feature_name="flow_duration",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    assert result.metadata["alternative"] == "two-sided"
    assert result.metadata["method"] == "asymp"


def test_compute_ks_output_within_valid_range_for_small_samples():
    result = compute_ks(
        KSComputationInput(
            reference_values=[1.0, 2.0],
            current_values=[3.0, 4.0],
            min_samples=2,
        ),
        feature_name="flow_duration",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
    )

    assert 0.0 <= result.d_statistic <= 1.0
    assert 0.0 <= result.p_value <= 1.0
