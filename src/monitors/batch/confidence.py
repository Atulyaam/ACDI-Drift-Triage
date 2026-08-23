from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math


_DEFAULT_ALPHA = 0.05


def _is_strict_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def _validate_alpha(value: object) -> float:
    if not _is_strict_number(value):
        raise TypeError(
            "alpha must be numeric (not bool)."
        )

    alpha = float(value)

    if not math.isfinite(alpha):
        raise ValueError(
            "alpha must be finite."
        )

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "alpha must satisfy 0 < alpha < 1."
        )

    return alpha


def _validate_probability(
    value: object,
    field_name: str,
) -> float:
    if not _is_strict_number(value):
        raise TypeError(
            f"{field_name} must be numeric (not bool)."
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1."
        )

    return value


@dataclass(frozen=True)
class ConfidenceDriftResult:
    """
    Semantic result for the confidence/entropy drift signal.

    The underlying statistical comparison is the existing KS
    computation applied to per-sample predictive entropy.

    This class does NOT implement another statistical test.

    Scope:
    - detects distributional change in predictive uncertainty
    - does NOT detect pure class-probability direction flips
      because H(p) == H(1-p)

    Raw probability and entropy arrays are intentionally absent.
    """

    reference_window_id: str
    current_window_id: str

    d_statistic: float
    p_value: float

    n_ref: int
    n_cur: int

    significant: bool
    alpha: float = _DEFAULT_ALPHA

    entropy_constant_reference: bool = False
    entropy_constant_current: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reference_window_id, str)
            or not self.reference_window_id.strip()
        ):
            raise ValueError(
                "reference_window_id must be a non-empty string."
            )

        if (
            not isinstance(self.current_window_id, str)
            or not self.current_window_id.strip()
        ):
            raise ValueError(
                "current_window_id must be a non-empty string."
            )

        if (
            self.reference_window_id
            == self.current_window_id
        ):
            raise ValueError(
                "reference_window_id and current_window_id "
                "must be different."
            )

        if not _is_strict_number(self.d_statistic):
            raise TypeError(
                "d_statistic must be numeric (not bool)."
            )

        d_statistic = float(self.d_statistic)

        if not math.isfinite(d_statistic):
            raise ValueError(
                "d_statistic must be finite."
            )

        if not 0.0 <= d_statistic <= 1.0:
            raise ValueError(
                "d_statistic must be between 0 and 1."
            )

        p_value = _validate_probability(
            self.p_value,
            "p_value",
        )

        if (
            not isinstance(self.n_ref, int)
            or isinstance(self.n_ref, bool)
        ):
            raise TypeError(
                "n_ref must be an integer (not bool)."
            )

        if (
            not isinstance(self.n_cur, int)
            or isinstance(self.n_cur, bool)
        ):
            raise TypeError(
                "n_cur must be an integer (not bool)."
            )

        if self.n_ref <= 0:
            raise ValueError(
                "n_ref must be greater than zero."
            )

        if self.n_cur <= 0:
            raise ValueError(
                "n_cur must be greater than zero."
            )

        if not isinstance(self.significant, bool):
            raise TypeError(
                "significant must be a boolean."
            )

        alpha = _validate_alpha(self.alpha)

        if not isinstance(self.entropy_constant_reference, bool):
            raise TypeError(
                "entropy_constant_reference must be a boolean."
            )

        if not isinstance(self.entropy_constant_current, bool):
            raise TypeError(
                "entropy_constant_current must be a boolean."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        if not all(
            isinstance(key, str)
            for key in self.metadata
        ):
            raise TypeError(
                "metadata keys must all be strings."
            )

        # Explicitly evaluate derived significance consistency,
        # using the already-validated float values.
        expected_significant = p_value <= alpha

        if self.significant != expected_significant:
            raise ValueError(
                "significant must equal p_value <= alpha."
            )

        # Canonicalize validated numeric fields onto the frozen
        # instance. Without this, a caller passing d_statistic=1
        # or p_value=0 (both valid, inclusive boundary values)
        # would leave self.d_statistic / self.p_value as int
        # instead of float, causing two logically-identical
        # results to differ in repr()/type depending on caller
        # input style. Same reasoning as the q-canonicalization
        # fix applied to FeatureDriftSummary.
        object.__setattr__(self, "d_statistic", d_statistic)
        object.__setattr__(self, "p_value", p_value)
        object.__setattr__(self, "alpha", alpha)
