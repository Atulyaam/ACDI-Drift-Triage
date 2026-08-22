
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math


def _is_strict_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def _validate_probability(
    value: object,
    field_name: str,
) -> float:
    """
    Single shared probability validator.

    Used by both FDRResult and apply_bh_fdr so the
    validation rules cannot silently diverge.
    """

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


def _validate_q(value: object) -> float:
    """
    Shared FDR q validator.
    """

    if not _is_strict_number(value):
        raise TypeError(
            "q must be numeric (not bool)."
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            "q must be finite."
        )

    if not 0.0 < value < 1.0:
        raise ValueError(
            "q must satisfy 0 < q < 1."
        )

    return value


@dataclass(frozen=True)
class FDRConfig:
    """
    Configuration for Benjamini-Hochberg FDR control.
    """

    q: float = 0.05

    def __post_init__(self) -> None:
        _validate_q(self.q)


@dataclass(frozen=True)
class FDRResult:
    """
    Serializable result of one feature's BH/FDR assessment.
    """

    feature_name: str

    reference_window_id: str
    current_window_id: str

    raw_p_value: float
    adjusted_p_value: float

    significant: bool

    q: float

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

        _validate_probability(
            self.raw_p_value,
            "raw_p_value",
        )

        _validate_probability(
            self.adjusted_p_value,
            "adjusted_p_value",
        )

        if not isinstance(
            self.significant,
            bool,
        ):
            raise TypeError(
                "significant must be a boolean."
            )

        _validate_q(self.q)

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


def apply_bh_fdr(
    ks_results,
    expected_feature_names,
    config: FDRConfig,
) -> tuple[FDRResult, ...]:
    """
    Apply the Benjamini-Hochberg procedure to a COMPLETE
    feature-level KS result set.

    Guarantees:
    - fail-fast
    - no partial output
    - complete expected feature set required
    - duplicate features rejected
    - all results must share one window pair
    - p-values revalidated at this boundary
    - BH ranking happens internally
    - output order is restored to original ks_results order
    """

    # --------------------------------------------------------
    # 1. Validate config
    # --------------------------------------------------------

    if not isinstance(config, FDRConfig):
        raise TypeError(
            "config must be an FDRConfig."
        )

    q = _validate_q(config.q)

    # --------------------------------------------------------
    # 2. Validate expected feature names
    #    (materialize BEFORE emptiness check so generators
    #    can't slip past a truthiness test)
    # --------------------------------------------------------

    expected_names = list(
        expected_feature_names
    )

    if len(expected_names) == 0:
        raise ValueError(
            "expected_feature_names must be non-empty."
        )

    seen_expected: set[str] = set()

    for feature_name in expected_names:

        if not isinstance(feature_name, str):
            raise TypeError(
                "Every expected feature name must be a string."
            )

        if not feature_name.strip():
            raise ValueError(
                "Expected feature names must be non-empty strings."
            )

        if feature_name in seen_expected:
            raise ValueError(
                f"Duplicate expected feature name: "
                f"{feature_name!r}"
            )

        seen_expected.add(feature_name)

    # --------------------------------------------------------
    # 3. Validate KS result sequence
    #    (same materialize-before-check fix applied here)
    # --------------------------------------------------------

    results = list(ks_results)

    if len(results) == 0:
        raise ValueError(
            "ks_results must be non-empty."
        )

    expected_count = len(
        expected_names
    )

    actual_count = len(results)

    if actual_count != expected_count:
        raise ValueError(
            "KS result count does not match expected "
            f"feature count: expected {expected_count}, "
            f"got {actual_count}."
        )

    # --------------------------------------------------------
    # 4. Validate result types and duplicate features
    # --------------------------------------------------------

    actual_names: list[str] = []
    seen_actual: set[str] = set()

    from src.monitors.batch.ks import KSResult

    for result in results:

        if not isinstance(result, KSResult):
            raise TypeError(
                "ks_results must contain only KSResult objects."
            )

        if result.feature_name in seen_actual:
            raise ValueError(
                f"Duplicate KSResult feature name: "
                f"{result.feature_name!r}"
            )

        seen_actual.add(
            result.feature_name
        )

        actual_names.append(
            result.feature_name
        )

        # ----------------------------------------------------
        # Defensive p-value validation
        # ----------------------------------------------------

        _validate_probability(
            result.p_value,
            (
                f"p_value for feature "
                f"{result.feature_name!r}"
            ),
        )

    # --------------------------------------------------------
    # 5. Exact expected feature-set validation
    # --------------------------------------------------------

    if set(actual_names) != set(expected_names):
        missing = (
            set(expected_names)
            - set(actual_names)
        )

        extra = (
            set(actual_names)
            - set(expected_names)
        )

        raise ValueError(
            "KS feature set does not match expected "
            f"feature set. Missing={sorted(missing)!r}, "
            f"Extra={sorted(extra)!r}."
        )

    # --------------------------------------------------------
    # 6. Validate common window pair
    # --------------------------------------------------------

    reference_window_id = (
        results[0].reference_window_id
    )

    current_window_id = (
        results[0].current_window_id
    )

    for result in results:

        if (
            result.reference_window_id
            != reference_window_id
        ):
            raise ValueError(
                "All KSResults must share the same "
                "reference_window_id."
            )

        if (
            result.current_window_id
            != current_window_id
        ):
            raise ValueError(
                "All KSResults must share the same "
                "current_window_id."
            )

    if (
        reference_window_id
        == current_window_id
    ):
        raise ValueError(
            "reference_window_id and current_window_id "
            "must be different."
        )

    # --------------------------------------------------------
    # 7. Internal BH ranking
    # --------------------------------------------------------

    m = len(results)

    ranked = sorted(
        enumerate(results),
        key=lambda item: (
            float(item[1].p_value),
            item[0],
        ),
    )

    # --------------------------------------------------------
    # 8. Raw BH scaled values
    # --------------------------------------------------------

    scaled_values: list[tuple[int, float]] = []

    for rank, (
        original_index,
        result,
    ) in enumerate(
        ranked,
        start=1,
    ):
        p_value = float(
            result.p_value
        )

        scaled = (
            p_value
            * m
            / rank
        )

        scaled_values.append(
            (
                original_index,
                scaled,
            )
        )

    # --------------------------------------------------------
    # 9. Reverse cumulative minimum
    #
    # This enforces monotonic adjusted p-values.
    # --------------------------------------------------------

    adjusted_by_index: dict[int, float] = {}

    running_min = 1.0

    for original_index, scaled in reversed(
        scaled_values
    ):
        running_min = min(
            running_min,
            scaled,
        )

        adjusted_by_index[
            original_index
        ] = min(
            running_min,
            1.0,
        )

    # --------------------------------------------------------
    # 10. Restore ORIGINAL ks_results order
    # --------------------------------------------------------

    final_results: list[FDRResult] = []

    for original_index, result in enumerate(
        results
    ):

        adjusted_p = float(
            adjusted_by_index[
                original_index
            ]
        )

        significant = (
            adjusted_p <= q
        )

        rank = next(
            rank
            for rank, (
                ranked_original_index,
                _,
            ) in enumerate(
                ranked,
                start=1,
            )
            if ranked_original_index
            == original_index
        )

        final_results.append(
            FDRResult(
                feature_name=result.feature_name,
                reference_window_id=(
                    result.reference_window_id
                ),
                current_window_id=(
                    result.current_window_id
                ),
                raw_p_value=float(
                    result.p_value
                ),
                adjusted_p_value=adjusted_p,
                significant=significant,
                q=q,
                metadata={
                    "method": "benjamini_hochberg",
                    "m": m,
                    "rank": rank,
                },
            )
        )

    # --------------------------------------------------------
    # 11. Completeness invariant
    # --------------------------------------------------------

    if len(final_results) != expected_count:
        raise RuntimeError(
            "Internal FDR completeness invariant failed."
        )

    return tuple(final_results)
