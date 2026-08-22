from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.monitors.batch.ks import (
    KSComputationInput,
    KSResult,
    compute_ks,
)


def run_ks_for_features(
    reference_data: Mapping[str, Any],
    current_data: Mapping[str, Any],
    feature_names: Sequence[str],
    reference_window_id: str,
    current_window_id: str,
) -> tuple[KSResult, ...]:

    if not feature_names:
        raise ValueError(
            "feature_names must be non-empty."
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

    seen: set[str] = set()

    for feature_name in feature_names:

        if not isinstance(feature_name, str):
            raise TypeError(
                "Every feature name must be a string."
            )

        if not feature_name.strip():
            raise ValueError(
                "Feature names must be non-empty strings."
            )

        if feature_name in seen:
            raise ValueError(
                f"Duplicate feature name: {feature_name!r}"
            )

        seen.add(feature_name)

    missing_reference = [
        feature_name
        for feature_name in feature_names
        if feature_name not in reference_data
    ]

    if missing_reference:
        raise ValueError(
            "Missing feature(s) from reference_data: "
            f"{missing_reference!r}"
        )

    missing_current = [
        feature_name
        for feature_name in feature_names
        if feature_name not in current_data
    ]

    if missing_current:
        raise ValueError(
            "Missing feature(s) from current_data: "
            f"{missing_current!r}"
        )

    results: list[KSResult] = []

    for feature_name in feature_names:

        try:
            computation_input = KSComputationInput(
                reference_values=reference_data[feature_name],
                current_values=current_data[feature_name],
            )

            result = compute_ks(
                computation_input=computation_input,
                feature_name=feature_name,
                reference_window_id=reference_window_id,
                current_window_id=current_window_id,
            )

        except ValueError as exc:
            raise ValueError(
                f"KS computation failed for feature "
                f"{feature_name!r}: {exc}"
            ) from exc

        except TypeError as exc:
            raise TypeError(
                f"KS computation failed for feature "
                f"{feature_name!r}: {exc}"
            ) from exc

        results.append(result)

    return tuple(results)
