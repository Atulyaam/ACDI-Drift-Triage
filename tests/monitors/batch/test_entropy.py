
import numpy as np
import pytest

from src.monitors.batch.entropy import (
    EntropyComputationInput,
)


# ============================================================
# 1. Basic input validation
# ============================================================

def test_valid_probability_input():
    obj = EntropyComputationInput(
        reference_probabilities=[
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ],
        current_probabilities=[
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
        ],
    )

    assert (
        obj.reference_probabilities.dtype
        == np.float64
    )

    assert (
        obj.current_probabilities.dtype
        == np.float64
    )


def test_default_min_samples_is_two():
    obj = EntropyComputationInput(
        reference_probabilities=[
            0.0,
            1.0,
        ],
        current_probabilities=[
            0.5,
            0.5,
        ],
    )

    assert obj.min_samples == 2


def test_min_samples_rejects_bool():
    with pytest.raises(TypeError):
        EntropyComputationInput(
            reference_probabilities=[
                0.0,
                1.0,
            ],
            current_probabilities=[
                0.5,
                0.5,
            ],
            min_samples=True,
        )


def test_min_samples_below_two_rejected():
    with pytest.raises(ValueError):
        EntropyComputationInput(
            reference_probabilities=[
                0.0,
                1.0,
            ],
            current_probabilities=[
                0.5,
                0.5,
            ],
            min_samples=1,
        )


# ============================================================
# 2. Empty / invalid probability input
# ============================================================

def test_empty_reference_rejected():
    with pytest.raises(ValueError):
        EntropyComputationInput(
            reference_probabilities=[],
            current_probabilities=[
                0.5,
                0.5,
            ],
        )


def test_empty_current_rejected():
    with pytest.raises(ValueError):
        EntropyComputationInput(
            reference_probabilities=[
                0.5,
                0.5,
            ],
            current_probabilities=[],
        )


def test_nan_reference_rejected():
    with pytest.raises(ValueError):
        EntropyComputationInput(
            reference_probabilities=[
                0.1,
                np.nan,
                0.9,
            ],
            current_probabilities=[
                0.1,
                0.5,
                0.9,
            ],
        )


def test_nan_current_rejected():
    with pytest.raises(ValueError):
        EntropyComputationInput(
            reference_probabilities=[
                0.1,
                0.5,
                0.9,
            ],
            current_probabilities=[
                0.1,
                np.nan,
                0.9,
            ],
        )


def test_positive_infinity_rejected():
    with pytest.raises(ValueError):
        EntropyComputationInput(
            reference_probabilities=[
                0.1,
                np.inf,
                0.9,
            ],
            current_probabilities=[
                0.1,
                0.5,
                0.9,
            ],
        )


def test_negative_infinity_rejected():
    with pytest.raises(ValueError):
        EntropyComputationInput(
            reference_probabilities=[
                0.1,
                -np.inf,
                0.9,
            ],
            current_probabilities=[
                0.1,
                0.5,
                0.9,
            ],
        )


def test_probability_below_zero_rejected():
    with pytest.raises(ValueError):
        EntropyComputationInput(
            reference_probabilities=[
                -0.01,
                0.5,
            ],
            current_probabilities=[
                0.1,
                0.5,
            ],
        )


def test_probability_above_one_rejected():
    with pytest.raises(ValueError):
        EntropyComputationInput(
            reference_probabilities=[
                1.01,
                0.5,
            ],
            current_probabilities=[
                0.1,
                0.5,
            ],
        )


def test_non_numeric_reference_rejected():
    with pytest.raises(TypeError):
        EntropyComputationInput(
            reference_probabilities=[
                "bad",
                "data",
            ],
            current_probabilities=[
                0.1,
                0.5,
            ],
        )


def test_non_numeric_current_rejected():
    with pytest.raises(TypeError):
        EntropyComputationInput(
            reference_probabilities=[
                0.1,
                0.5,
            ],
            current_probabilities=[
                "bad",
                "data",
            ],
        )


def test_multidimensional_reference_rejected():
    with pytest.raises(ValueError):
        EntropyComputationInput(
            reference_probabilities=[
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            current_probabilities=[
                0.1,
                0.5,
            ],
        )


def test_multidimensional_current_rejected():
    with pytest.raises(ValueError):
        EntropyComputationInput(
            reference_probabilities=[
                0.1,
                0.5,
            ],
            current_probabilities=[
                [0.1, 0.2],
                [0.3, 0.4],
            ],
        )


# ============================================================
# 3. Immutability
# ============================================================

def test_reference_and_current_arrays_are_read_only():
    obj = EntropyComputationInput(
        reference_probabilities=[
            0.1,
            0.5,
        ],
        current_probabilities=[
            0.2,
            0.8,
        ],
    )

    with pytest.raises(ValueError):
        obj.reference_probabilities[0] = 0.9

    with pytest.raises(ValueError):
        obj.current_probabilities[0] = 0.9


# ============================================================
# 4. Binary entropy boundary behavior
# ============================================================

def test_entropy_zero_at_probability_zero():
    obj = EntropyComputationInput(
        reference_probabilities=[
            0.0,
            0.0,
        ],
        current_probabilities=[
            0.5,
            0.5,
        ],
    )

    assert np.all(
        obj.reference_entropy == 0.0
    )


def test_entropy_zero_at_probability_one():
    obj = EntropyComputationInput(
        reference_probabilities=[
            1.0,
            1.0,
        ],
        current_probabilities=[
            0.5,
            0.5,
        ],
    )

    assert np.all(
        obj.reference_entropy == 0.0
    )


def test_entropy_one_at_probability_half():
    obj = EntropyComputationInput(
        reference_probabilities=[
            0.5,
            0.5,
        ],
        current_probabilities=[
            0.5,
            0.5,
        ],
    )

    assert np.allclose(
        obj.reference_entropy,
        1.0,
    )

    assert np.allclose(
        obj.current_entropy,
        1.0,
    )


def test_entropy_is_finite_and_bounded():
    obj = EntropyComputationInput(
        reference_probabilities=[
            0.0,
            0.1,
            0.5,
            0.9,
            1.0,
        ],
        current_probabilities=[
            0.01,
            0.25,
            0.5,
            0.75,
            0.99,
        ],
    )

    assert np.isfinite(
        obj.reference_entropy
    ).all()

    assert np.isfinite(
        obj.current_entropy
    ).all()

    assert (
        obj.reference_entropy >= 0.0
    ).all()

    assert (
        obj.reference_entropy <= 1.0
    ).all()

    assert (
        obj.current_entropy >= 0.0
    ).all()

    assert (
        obj.current_entropy <= 1.0
    ).all()


# ============================================================
# 5. Entropy symmetry / scope limitation
# ============================================================

def test_entropy_is_symmetric():
    """
    H(p) == H(1-p).

    This explicitly documents the known limitation:
    entropy does not preserve probability direction.
    """

    obj = EntropyComputationInput(
        reference_probabilities=[
            0.1,
            0.2,
            0.3,
        ],
        current_probabilities=[
            0.9,
            0.8,
            0.7,
        ],
    )

    assert np.allclose(
        obj.reference_entropy,
        obj.current_entropy,
    )


def test_entropy_symmetry_scope_limitation():
    """
    A pure confidence-direction flip is intentionally not
    detected by the entropy signal.
    """

    obj = EntropyComputationInput(
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

    assert np.allclose(
        obj.reference_entropy,
        obj.current_entropy,
    )


# ============================================================
# 6. Entropy arrays are read-only
# ============================================================

def test_entropy_arrays_are_read_only():
    obj = EntropyComputationInput(
        reference_probabilities=[
            0.1,
            0.5,
        ],
        current_probabilities=[
            0.2,
            0.8,
        ],
    )

    with pytest.raises(ValueError):
        obj.reference_entropy[0] = 999.0

    with pytest.raises(ValueError):
        obj.current_entropy[0] = 999.0
