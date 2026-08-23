
import pytest

from src.monitors.batch.confidence_detector import (
    compute_confidence_drift,
)

from src.monitors.batch.entropy import (
    EntropyComputationInput,
)


REF_WIN = "REF_001"
CUR_WIN = "WIN_001"


def _run(
    reference,
    current,
    alpha=0.05,
):
    entropy_input = EntropyComputationInput(
        reference_probabilities=reference,
        current_probabilities=current,
    )

    return compute_confidence_drift(
        computation_input=entropy_input,
        reference_window_id=REF_WIN,
        current_window_id=CUR_WIN,
        alpha=alpha,
    )


# ============================================================
# 1. Same uncertainty distribution
# Exact scenario reused from STEP 10.4.4
# ============================================================

def test_same_uncertainty_is_not_significant():
    reference = [
        0.1,
        0.2,
        0.3,
        0.7,
        0.8,
        0.9,
    ]

    current = [
        0.9,
        0.8,
        0.7,
        0.3,
        0.2,
        0.1,
    ]

    result = _run(
        reference,
        current,
        alpha=0.05,
    )

    assert result.significant is False


# ============================================================
# 2. Strong uncertainty increase
# Exact scenario reused from STEP 10.4.4
# ============================================================

def test_uncertainty_increase_is_significant():
    reference = [
        0.01,
        0.02,
        0.03,
        0.97,
        0.98,
        0.99,
    ]

    current = [
        0.45,
        0.47,
        0.49,
        0.51,
        0.53,
        0.55,
    ]

    result = _run(
        reference,
        current,
        alpha=0.05,
    )

    assert result.significant is True
    assert result.p_value <= result.alpha


# ============================================================
# 3. Strong uncertainty decrease
# Exact scenario reused from STEP 10.4.4
# ============================================================

def test_uncertainty_decrease_is_significant():
    reference = [
        0.45,
        0.47,
        0.49,
        0.51,
        0.53,
        0.55,
    ]

    current = [
        0.01,
        0.02,
        0.03,
        0.97,
        0.98,
        0.99,
    ]

    result = _run(
        reference,
        current,
        alpha=0.05,
    )

    assert result.significant is True
    assert result.p_value <= result.alpha


# ============================================================
# 4. Constant entropy — full semantic result
# Exact scenario reused from STEP 10.4.4
# ============================================================

def test_identical_constant_entropy_is_not_significant():
    reference = [
        0.5,
        0.5,
        0.5,
        0.5,
    ]

    current = [
        0.5,
        0.5,
        0.5,
        0.5,
    ]

    result = _run(
        reference,
        current,
        alpha=0.05,
    )

    assert result.significant is False
    assert result.p_value > result.alpha

    assert (
        result.entropy_constant_reference
        is True
    )

    assert (
        result.entropy_constant_current
        is True
    )


# ============================================================
# 5. Boundary probabilities
# Exact scenario reused from STEP 10.4.4
# ============================================================

def test_boundary_probabilities_produce_valid_semantic_result():
    reference = [
        0.0,
        1.0,
        0.0,
        1.0,
    ]

    current = [
        0.5,
        0.5,
        0.5,
        0.5,
    ]

    result = _run(
        reference,
        current,
        alpha=0.05,
    )

    assert 0.0 <= result.d_statistic <= 1.0
    assert 0.0 <= result.p_value <= 1.0

    assert result.n_ref == 4
    assert result.n_cur == 4


# ============================================================
# 6. Direction-only flip — known scope limitation
# Exact scenario reused from STEP 10.4.4
# ============================================================

def test_direction_only_flip_is_not_significant():
    reference = [
        0.05,
        0.05,
        0.10,
        0.90,
        0.95,
        0.95,
    ]

    current = [
        0.95,
        0.95,
        0.90,
        0.10,
        0.05,
        0.05,
    ]

    result = _run(
        reference,
        current,
        alpha=0.50,
    )

    assert result.d_statistic == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert result.p_value == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert result.significant is False


# ============================================================
# 7. Alpha sensitivity — same underlying data
#
# First get a deterministic p-value, then place the two
# alpha values around that observed p-value.
# ============================================================

def test_alpha_sensitivity_uses_same_underlying_evidence():
    reference = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    current = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.45,
        0.55,
        0.60,
        0.70,
    ]

    baseline = _run(
        reference,
        current,
        alpha=0.05,
    )

    p = baseline.p_value

    # This test requires a non-degenerate p-value.
    assert 0.0 < p < 1.0

    alpha_low = p / 2.0
    alpha_high = (p + 1.0) / 2.0

    low = _run(
        reference,
        current,
        alpha=alpha_low,
    )

    high = _run(
        reference,
        current,
        alpha=alpha_high,
    )

    assert low.significant is False
    assert high.significant is True

    assert low.p_value == high.p_value
    assert low.d_statistic == high.d_statistic


# ============================================================
# 8. Semantic significance invariant
# ============================================================

def test_significance_always_matches_p_value_and_alpha():
    reference = [
        0.01,
        0.02,
        0.03,
        0.97,
        0.98,
        0.99,
    ]

    current = [
        0.45,
        0.47,
        0.49,
        0.51,
        0.53,
        0.55,
    ]

    for alpha in [
        0.01,
        0.05,
        0.10,
        0.50,
    ]:
        result = _run(
            reference,
            current,
            alpha=alpha,
        )

        assert result.significant == (
            result.p_value <= alpha
        )


# ============================================================
# 9. Metadata remains semantic and automatic
# ============================================================

def test_confidence_result_metadata_identifies_signal():
    result = _run(
        [
            0.01,
            0.02,
            0.98,
            0.99,
        ],
        [
            0.45,
            0.50,
            0.55,
            0.60,
        ],
    )

    assert result.metadata == {
        "signal": "predictive_entropy",
    }
