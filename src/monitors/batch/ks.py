
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import ks_2samp


_TECHNICAL_MIN_SAMPLES = 2


def _is_strict_int(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
    )


def _is_strict_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


@dataclass(frozen=True, eq=False)
class KSComputationInput:
    """
    Transient runtime input for one KS computation.

    This object is NOT persistent.

    Never:
    - serialize it
    - log it
    - persist it
    - place it inside DetectorResult
    - write it to CSV/Parquet

    It only carries validated reference/current numeric arrays.
    """

    reference_values: np.ndarray
    current_values: np.ndarray
    min_samples: int = _TECHNICAL_MIN_SAMPLES

    def __post_init__(self) -> None:
        self._validate_min_samples()

        reference = self._to_float64_array(
            self.reference_values,
            "reference_values",
        )

        current = self._to_float64_array(
            self.current_values,
            "current_values",
        )

        if reference.size == 0:
            raise ValueError(
                "reference_values cannot be empty."
            )

        if current.size == 0:
            raise ValueError(
                "current_values cannot be empty."
            )

        if reference.size < self.min_samples:
            raise ValueError(
                "reference_values contains "
                f"{reference.size} samples, but "
                f"min_samples={self.min_samples}."
            )

        if current.size < self.min_samples:
            raise ValueError(
                "current_values contains "
                f"{current.size} samples, but "
                f"min_samples={self.min_samples}."
            )

        if not np.isfinite(reference).all():
            raise ValueError(
                "reference_values must not contain "
                "NaN or infinite values."
            )

        if not np.isfinite(current).all():
            raise ValueError(
                "current_values must not contain "
                "NaN or infinite values."
            )

        # Frozen dataclass does not freeze ndarray contents.
        reference.setflags(write=False)
        current.setflags(write=False)

        object.__setattr__(
            self,
            "reference_values",
            reference,
        )

        object.__setattr__(
            self,
            "current_values",
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
    def _to_float64_array(
        values: Any,
        field_name: str,
    ) -> np.ndarray:
        try:
            array = np.asarray(
                values,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{field_name} must contain only numeric values."
            ) from exc

        if array.ndim != 1:
            raise ValueError(
                f"{field_name} must be a 1-dimensional array."
            )

        return array

    @property
    def is_constant_reference(self) -> bool:
        return bool(
            np.all(
                self.reference_values
                == self.reference_values[0]
            )
        )

    @property
    def is_constant_current(self) -> bool:
        return bool(
            np.all(
                self.current_values
                == self.current_values[0]
            )
        )


@dataclass(frozen=True)
class KSResult:
    """
    Serializable summary of one feature-level KS result.

    Raw arrays are intentionally excluded.
    """

    feature_name: str
    reference_window_id: str
    current_window_id: str

    d_statistic: float
    p_value: float

    n_ref: int
    n_cur: int

    is_constant_reference: bool
    is_constant_current: bool

    metadata: dict[str, Any] | None = field(
        default=None,
        compare=False,
    )

    def __post_init__(self) -> None:

        if (
            not isinstance(self.feature_name, str)
            or not self.feature_name.strip()
        ):
            raise ValueError(
                "feature_name must be a non-empty string."
            )

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

        if not _is_strict_number(
            self.d_statistic
        ):
            raise TypeError(
                "d_statistic must be numeric (not bool)."
            )

        if not 0.0 <= float(
            self.d_statistic
        ) <= 1.0:
            raise ValueError(
                "d_statistic must be between 0 and 1."
            )

        if not _is_strict_number(
            self.p_value
        ):
            raise TypeError(
                "p_value must be numeric (not bool)."
            )

        if not 0.0 <= float(
            self.p_value
        ) <= 1.0:
            raise ValueError(
                "p_value must be between 0 and 1."
            )

        if (
            not isinstance(self.n_ref, int)
            or isinstance(self.n_ref, bool)
        ):
            raise TypeError(
                "n_ref must be an integer (not a bool)."
            )

        if (
            not isinstance(self.n_cur, int)
            or isinstance(self.n_cur, bool)
        ):
            raise TypeError(
                "n_cur must be an integer (not a bool)."
            )

        if self.n_ref <= 0:
            raise ValueError(
                "n_ref must be greater than zero."
            )

        if self.n_cur <= 0:
            raise ValueError(
                "n_cur must be greater than zero."
            )

        if not isinstance(
            self.is_constant_reference,
            bool,
        ):
            raise TypeError(
                "is_constant_reference must be a boolean."
            )

        if not isinstance(
            self.is_constant_current,
            bool,
        ):
            raise TypeError(
                "is_constant_current must be a boolean."
            )

        if self.metadata is not None:

            if not isinstance(
                self.metadata,
                dict,
            ):
                raise TypeError(
                    "metadata must be a dictionary or None."
                )

            if not all(
                isinstance(key, str)
                for key in self.metadata
            ):
                raise TypeError(
                    "metadata keys must all be strings."
                )


def compute_ks(
    computation_input: KSComputationInput,
    feature_name: str,
    reference_window_id: str,
    current_window_id: str,
) -> KSResult:
    """
    Compute a two-sided asymptotic two-sample KS test.

    Locked settings:
        alternative = "two-sided"
        method = "asymp"
    """

    if not isinstance(
        computation_input,
        KSComputationInput,
    ):
        raise TypeError(
            "computation_input must be a KSComputationInput."
        )

    if (
        not isinstance(feature_name, str)
        or not feature_name.strip()
    ):
        raise ValueError(
            "feature_name must be a non-empty string."
        )

    if (
        not isinstance(reference_window_id, str)
        or not reference_window_id.strip()
    ):
        raise ValueError(
            "reference_window_id must be a non-empty string."
        )

    if (
        not isinstance(current_window_id, str)
        or not current_window_id.strip()
    ):
        raise ValueError(
            "current_window_id must be a non-empty string."
        )

    if reference_window_id == current_window_id:
        raise ValueError(
            "reference_window_id and current_window_id "
            "must be different."
        )

    raw_result = ks_2samp(
        computation_input.reference_values,
        computation_input.current_values,
        alternative="two-sided",
        method="asymp",
    )

    d_statistic = float(
        np.clip(
            float(raw_result.statistic),
            0.0,
            1.0,
        )
    )

    p_value = float(
        np.clip(
            float(raw_result.pvalue),
            0.0,
            1.0,
        )
    )

    return KSResult(
        feature_name=feature_name,
        reference_window_id=reference_window_id,
        current_window_id=current_window_id,
        d_statistic=d_statistic,
        p_value=p_value,
        n_ref=int(
            computation_input.reference_values.size
        ),
        n_cur=int(
            computation_input.current_values.size
        ),
        is_constant_reference=(
            computation_input.is_constant_reference
        ),
        is_constant_current=(
            computation_input.is_constant_current
        ),
        metadata={
            "alternative": "two-sided",
            "method": "asymp",
        },
    )
