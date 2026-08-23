import pytest

from src.monitors.batch.confidence import (
    ConfidenceDriftResult,
)


def _valid_result(**overrides):
    data = {
        "reference_window_id": "REF_001",
        "current_window_id": "WIN_001",
        "d_statistic": 0.25,
        "p_value": 0.01,
        "n_ref": 100,
        "n_cur": 100,
        "significant": True,
        "alpha": 0.05,
        "entropy_constant_reference": False,
        "entropy_constant_current": False,
    }
    data.update(overrides)
    return ConfidenceDriftResult(**data)


def test_valid_confidence_result():
    result = _valid_result()
    assert result.reference_window_id == "REF_001"
    assert result.current_window_id == "WIN_001"
    assert result.d_statistic == 0.25
    assert result.p_value == 0.01
    assert result.significant is True


def test_default_alpha_is_point_zero_five():
    result = ConfidenceDriftResult(
        reference_window_id="REF_001",
        current_window_id="WIN_001",
        d_statistic=0.2,
        p_value=0.03,
        n_ref=50,
        n_cur=50,
        significant=True,
    )
    assert result.alpha == 0.05


def test_alpha_accepts_valid_value():
    result = _valid_result(alpha=0.10, significant=True, p_value=0.05)
    assert result.alpha == 0.10


def test_alpha_rejects_zero():
    with pytest.raises(ValueError):
        _valid_result(alpha=0.0)


def test_alpha_rejects_one():
    with pytest.raises(ValueError):
        _valid_result(alpha=1.0)


def test_alpha_rejects_bool():
    with pytest.raises(TypeError):
        _valid_result(alpha=True)


def test_alpha_rejects_nan():
    with pytest.raises(ValueError):
        _valid_result(alpha=float("nan"))


def test_alpha_rejects_infinity():
    with pytest.raises(ValueError):
        _valid_result(alpha=float("inf"))


def test_rejects_same_window_pair():
    with pytest.raises(ValueError):
        _valid_result(current_window_id="REF_001")


def test_rejects_d_out_of_range():
    with pytest.raises(ValueError):
        _valid_result(d_statistic=1.1)


def test_rejects_negative_d():
    with pytest.raises(ValueError):
        _valid_result(d_statistic=-0.1)


def test_rejects_invalid_p_value():
    with pytest.raises(ValueError):
        _valid_result(p_value=1.1, significant=False)


def test_rejects_negative_p_value():
    with pytest.raises(ValueError):
        _valid_result(p_value=-0.1, significant=False)


def test_rejects_bool_p_value():
    with pytest.raises(TypeError):
        _valid_result(p_value=True, significant=False)


def test_rejects_bool_d_statistic():
    with pytest.raises(TypeError):
        _valid_result(d_statistic=True, significant=False)


def test_rejects_invalid_n_ref():
    with pytest.raises(ValueError):
        _valid_result(n_ref=0)


def test_rejects_invalid_n_cur():
    with pytest.raises(ValueError):
        _valid_result(n_cur=0)


def test_rejects_bool_significant():
    with pytest.raises(TypeError):
        _valid_result(significant=1)


def test_rejects_bool_constant_reference():
    with pytest.raises(TypeError):
        _valid_result(entropy_constant_reference=1)


def test_rejects_bool_constant_current():
    with pytest.raises(TypeError):
        _valid_result(entropy_constant_current=1)


def test_significant_must_match_p_and_alpha():
    with pytest.raises(ValueError):
        _valid_result(p_value=0.20, alpha=0.05, significant=True)


def test_non_significant_result_is_valid():
    result = _valid_result(p_value=0.20, alpha=0.05, significant=False)
    assert result.significant is False


def test_metadata_is_hash_safe():
    result = _valid_result(metadata={"signal": "predictive_entropy"})
    hash(result)


def test_raw_arrays_are_not_part_of_result():
    result = _valid_result()
    assert not hasattr(result, "reference_probabilities")
    assert not hasattr(result, "current_probabilities")
    assert not hasattr(result, "reference_entropy")
    assert not hasattr(result, "current_entropy")
