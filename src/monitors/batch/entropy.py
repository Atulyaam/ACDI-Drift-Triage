from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_TECHNICAL_MIN_SAMPLES = 2


def _is_strict_int(value: object) -> bool:
    """
    True only for real integers.

    bool is a subclass of int in Python, so bool is
    explicitly rejected.
    """
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
    )


@dataclass(frozen=True, eq=False)
class EntropyComputationInput:
    """
    Transient runtime input for binary predictive entropy.

    This object exists only to validate and carry probability
    streams into the entropy transformation layer.

    IMPORTANT:
    - Never serialize.
    - Never log.
    - Never persist.
    - Never store in DetectorResult.
    - Never write to CSV/Parquet.

    Scope limitation:
    Binary entropy measures predictive uncertainty only.

    Because H(p) == H(1-p), entropy does NOT preserve
    directional class-probability information.

    For example, confident-negative predictions (p ~= 0.1)
    and confident-positive predictions (p ~= 0.9) produce
    the same entropy.

    Therefore this core entropy signal can detect changes
    in predictive uncertainty but cannot detect a pure
    confidence-direction flip.

    Directional probability drift, if required in future,
    must be treated as a separate signal.

    Constant-value detection is intentionally NOT implemented
    in this contract. Constancy of entropy distributions is
    determined by the existing compute_ks() path, which
    remains the single source of truth.
    """

    reference_probabilities: np.ndarray
    current_probabilities: np.ndarray
    min_samples: int = _TECHNICAL_MIN_SAMPLES

    def __post_init__(self) -> None:
        self._validate_min_samples()

        reference = self._to_probability_array(
            self.reference_probabilities,
            "reference_probabilities",
        )

        current = self._to_probability_array(
            self.current_probabilities,
            "current_probabilities",
        )

        if reference.size == 0:
            raise ValueError(
                "reference_probabilities cannot be empty."
            )

        if current.size == 0:
            raise ValueError(
                "current_probabilities cannot be empty."
            )

        if reference.size < self.min_samples:
            raise ValueError(
                "reference_probabilities contains "
                f"{reference.size} samples, but "
                f"min_samples={self.min_samples}."
            )

        if current.size < self.min_samples:
            raise ValueError(
                "current_probabilities contains "
                f"{current.size} samples, but "
                f"min_samples={self.min_samples}."
            )

        # Frozen dataclass does not freeze ndarray contents.
        reference.setflags(write=False)
        current.setflags(write=False)

        object.__setattr__(
            self,
            "reference_probabilities",
            reference,
        )

        object.__setattr__(
            self,
            "current_probabilities",
            current,
        )

    def _validate_min_samples(self) -> None:
        if not _is_strict_int(self.min_samples):
            raise TypeError(
                "min_samples must be an integer (not a bool)."
            )

        if self.min_samples < _TECHNICAL_MIN_SAMPLES:
            raise ValueError(
                "min_samples must be >= "
                f"{_TECHNICAL_MIN_SAMPLES}."
            )

    @staticmethod
    def _to_probability_array(
        values: Any,
        field_name: str,
    ) -> np.ndarray:
        """
        Convert values to float64 and enforce strict
        probability semantics.

        Valid:
            0 <= p <= 1

        Rejected:
            NaN
            +/-inf
            p < 0
            p > 1
            non-numeric values
            multidimensional arrays
        """

        try:
            array = np.asarray(
                values,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{field_name} must contain only numeric "
                "probability values."
            ) from exc

        if array.ndim != 1:
            raise ValueError(
                f"{field_name} must be a 1-dimensional array."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f"{field_name} must contain only finite values."
            )

        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError(
                f"{field_name} must contain values in "
                "the closed interval [0, 1]."
            )

        return array

    @staticmethod
    def _binary_entropy(
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """
        Convert probabilities to binary predictive entropy
        using log base 2.

        Boundary handling:
            H(0) = 0
            H(1) = 0

        p=0 and p=1 are handled explicitly so log2(0)
        is never evaluated.
        """

        entropy = np.zeros_like(
            probabilities,
            dtype=np.float64,
        )

        interior = (
            (probabilities > 0.0)
            & (probabilities < 1.0)
        )

        p = probabilities[interior]

        entropy[interior] = (
            -p * np.log2(p)
            - (1.0 - p) * np.log2(1.0 - p)
        )

        entropy.setflags(write=False)

        return entropy

    @property
    def reference_entropy(self) -> np.ndarray:
        """
        Per-sample entropy for the reference probability stream.

        Returns a read-only array.
        """
        return self._binary_entropy(
            self.reference_probabilities
        )

    @property
    def current_entropy(self) -> np.ndarray:
        """
        Per-sample entropy for the current probability stream.

        Returns a read-only array.
        """
        return self._binary_entropy(
            self.current_probabilities
        )
