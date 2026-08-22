from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.monitors.batch.fdr import FDRResult


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_strict_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def _validate_q(value: object) -> float:
    if not _is_strict_number(value):
        raise TypeError(
            "q must be numeric (not bool)."
        )

    q = float(value)

    if not (q == q) or q in (float("inf"), float("-inf")):
        raise ValueError(
            "q must be finite."
        )

    if not 0.0 < q <= 1.0:
        raise ValueError(
            "q must satisfy 0 < q <= 1."
        )

    return q


@dataclass(frozen=True)
class FeatureDriftSummary:
    """
    Aggregated feature-drift summary for exactly one
    reference/current window comparison and one FDR q.

    This class performs no statistical computation.
    """

    reference_window_id: str
    current_window_id: str
    q: float

    n_features_total: int
    n_features_significant: int

    drifted_feature_names: tuple[str, ...]

    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
    )

    @property
    def proportion_drifted(self) -> float:
        """
        Derived proportion of significant features.

        Never caller-supplied, so it cannot become inconsistent
        with the feature counts.
        """
        return (
            self.n_features_significant
            / self.n_features_total
        )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reference_window_id, str)
            or not self.reference_window_id.strip()
        ):
            raise ValueError(
                "reference_window_id must be a "
                "non-empty string."
            )

        if (
            not isinstance(self.current_window_id, str)
            or not self.current_window_id.strip()
        ):
            raise ValueError(
                "current_window_id must be a "
                "non-empty string."
            )

        if (
            self.reference_window_id
            == self.current_window_id
        ):
            raise ValueError(
                "reference_window_id and current_window_id "
                "must be different."
            )

        q = _validate_q(self.q)
        object.__setattr__(self, "q", q)

        if not _is_strict_int(
            self.n_features_total
        ):
            raise TypeError(
                "n_features_total must be an integer "
                "(not bool)."
            )

        if self.n_features_total <= 0:
            raise ValueError(
                "n_features_total must be greater than zero."
            )

        if not _is_strict_int(
            self.n_features_significant
        ):
            raise TypeError(
                "n_features_significant must be an integer "
                "(not bool)."
            )

        if not (
            0
            <= self.n_features_significant
            <= self.n_features_total
        ):
            raise ValueError(
                "n_features_significant must satisfy "
                "0 <= n_features_significant "
                "<= n_features_total."
            )

        if not isinstance(
            self.drifted_feature_names,
            tuple,
        ):
            raise TypeError(
                "drifted_feature_names must be a tuple."
            )

        if len(
            self.drifted_feature_names
        ) != self.n_features_significant:
            raise ValueError(
                "len(drifted_feature_names) must equal "
                "n_features_significant."
            )

        seen: set[str] = set()

        for feature_name in self.drifted_feature_names:
            if not isinstance(
                feature_name,
                str,
            ):
                raise TypeError(
                    "Every drifted feature name must "
                    "be a string."
                )

            if not feature_name.strip():
                raise ValueError(
                    "Drifted feature names must be "
                    "non-empty strings."
                )

            if feature_name in seen:
                raise ValueError(
                    f"Duplicate drifted feature name: "
                    f"{feature_name!r}"
                )

            seen.add(feature_name)

        proportion = self.proportion_drifted

        if not 0.0 <= proportion <= 1.0:
            raise ValueError(
                "proportion_drifted must be between 0 and 1."
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
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


def build_feature_drift_summary(
    fdr_results: tuple[FDRResult, ...],
) -> FeatureDriftSummary:
    """
    Build a complete feature-drift summary from FDR results.

    Validation is independent of the FDR layer:
    - non-empty input
    - no duplicate feature names
    - common reference window
    - common current window
    - common q
    """

    if not fdr_results:
        raise ValueError(
            "fdr_results must be non-empty."
        )

    results = tuple(fdr_results)

    if not all(
        isinstance(result, FDRResult)
        for result in results
    ):
        raise TypeError(
            "fdr_results must contain only FDRResult objects."
        )

    reference_window_id = (
        results[0].reference_window_id
    )

    current_window_id = (
        results[0].current_window_id
    )

    q = float(results[0].q)

    seen: set[str] = set()

    for result in results:
        if result.feature_name in seen:
            raise ValueError(
                f"Duplicate feature name: "
                f"{result.feature_name!r}"
            )

        seen.add(result.feature_name)

        if (
            result.reference_window_id
            != reference_window_id
        ):
            raise ValueError(
                "All FDRResults must share the same "
                "reference_window_id."
            )

        if (
            result.current_window_id
            != current_window_id
        ):
            raise ValueError(
                "All FDRResults must share the same "
                "current_window_id."
            )

        if float(result.q) != q:
            raise ValueError(
                "All FDRResults must use the same FDR q."
            )

    drifted_feature_names = tuple(
        result.feature_name
        for result in results
        if result.significant
    )

    n_features_total = len(results)

    n_features_significant = len(
        drifted_feature_names
    )

    return FeatureDriftSummary(
        reference_window_id=reference_window_id,
        current_window_id=current_window_id,
        q=q,
        n_features_total=n_features_total,
        n_features_significant=n_features_significant,
        drifted_feature_names=drifted_feature_names,
        metadata={
            "source": "fdr_results",
        },
    )
