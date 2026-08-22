import numpy as np
import pytest

from src.monitors.batch.entropy import (
    EntropyComputationInput,
)

from src.monitors.batch.entropy_detector import (
    compute_entropy_drift,
)


REF_WIN = "REF_001"
CUR_WIN = "WIN_001"


def _run_entropy(reference, current):
    entropy_input = EntropyComputationInput(
        reference_probabilities=reference,
        current_probabilities=current,
    )

    return compute_entropy_drift(
        computation_input=entropy_input,
        feature_name="prediction_entropy",
        reference_window_id=REF_WIN,
        current_window_id=CUR_WIN,
    )


# ============================================================
# 1. Same uncertainty distribution -> no entropy drift
# ============================================================

def test_same_entropy_distribution_has_no_drift():
    reference = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    current = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]

    result = _run_entropy(reference, current)

    assert result.d_statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0, abs=1e-12)


# ============================================================
# 2. Strong uncertainty increase
# ============================================================

def test_entropy_increase_produces_distribution_drift():
    reference = [0.01, 0.02, 0.03, 0.97, 0.98, 0.99]
    current = [0.45, 0.47, 0.49, 0.51, 0.53, 0.55]

    result = _run_entropy(reference, current)

    assert result.d_statistic > 0.0
    assert result.p_value < 0.01


# ============================================================
# 3. Strong uncertainty decrease
# ============================================================

def test_entropy_decrease_produces_distribution_drift():
    reference = [0.45, 0.47, 0.49, 0.51, 0.53, 0.55]
    current = [0.01, 0.02, 0.03, 0.97, 0.98, 0.99]

    result = _run_entropy(reference, current)

    assert result.d_statistic > 0.0
    assert result.p_value < 0.01


# ============================================================
# 4. Both entropy distributions constant and identical
# ============================================================

def test_identical_constant_entropy_distributions_have_no_drift():
    reference = [0.5, 0.5, 0.5, 0.5]
    current = [0.5, 0.5, 0.5, 0.5]

    result = _run_entropy(reference, current)

    assert result.d_statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0, abs=1e-12)
    assert result.is_constant_reference is True
    assert result.is_constant_current is True


# ============================================================
# 5. Boundary probabilities must remain finite
# ============================================================

def test_boundary_probabilities_produce_finite_entropy_results():
    reference = [0.0, 1.0, 0.0, 1.0]
    current = [0.5, 0.5, 0.5, 0.5]

    result = _run_entropy(reference, current)

    assert np.isfinite(result.d_statistic)
    assert np.isfinite(result.p_value)
    assert 0.0 <= result.d_statistic <= 1.0
    assert 0.0 <= result.p_value <= 1.0


# ============================================================
# 6. Different sample sizes
# ============================================================

def test_entropy_supports_different_sample_sizes():
    reference = [0.01, 0.02, 0.05, 0.95, 0.98, 0.99, 0.10, 0.90, 0.20, 0.80]
    current = [0.45, 0.48, 0.50, 0.52, 0.55]

    result = _run_entropy(reference, current)

    assert result.n_ref == 10
    assert result.n_cur == 5
    assert 0.0 <= result.d_statistic <= 1.0
    assert 0.0 <= result.p_value <= 1.0


# ============================================================
# 7. Direction-only flip is intentionally invisible
# ============================================================

def test_direction_only_probability_flip_is_not_entropy_drift():
    reference = [0.05, 0.05, 0.10, 0.90, 0.95, 0.95]
    current = [0.95, 0.95, 0.90, 0.10, 0.05, 0.05]

    result = _run_entropy(reference, current)

    assert result.d_statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0, abs=1e-12)


# ============================================================
# 8. Result bounds over deterministic scenarios
# ============================================================

@pytest.mark.parametrize(
    "reference,current",
    [
        ([0.01, 0.02, 0.03, 0.97, 0.98, 0.99], [0.10, 0.20, 0.30, 0.70, 0.80, 0.90]),
        ([0.20, 0.30, 0.40, 0.60, 0.70, 0.80], [0.01, 0.05, 0.10, 0.90, 0.95, 0.99]),
        ([0.25, 0.35, 0.45, 0.55, 0.65, 0.75], [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]),
    ],
)
def test_entropy_result_bounds(reference, current):
    result = _run_entropy(reference, current)

    assert 0.0 <= result.d_statistic <= 1.0
    assert 0.0 <= result.p_value <= 1.0


# ============================================================
# 9. Entropy output remains based on the existing KS engine
# ============================================================

def test_entropy_result_contains_standard_ks_fields():
    result = _run_entropy(
        [0.01, 0.02, 0.98, 0.99],
        [0.45, 0.50, 0.55, 0.60],
    )

    assert hasattr(result, "d_statistic")
    assert hasattr(result, "p_value")
    assert hasattr(result, "n_ref")
    assert hasattr(result, "n_cur")
    assert result.feature_name == "prediction_entropy"
