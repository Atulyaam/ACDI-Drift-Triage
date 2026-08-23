
from __future__ import annotations

from dataclasses import dataclass
import math


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


def _validate_positive_int(
    value: object,
    field_name: str,
) -> int:
    if not _is_strict_int(value):
        raise TypeError(
            f"{field_name} must be an integer (not bool)."
        )

    if value <= 0:
        raise ValueError(
            f"{field_name} must be greater than zero."
        )

    return value


def _validate_finite_number(
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

    return value


def _validate_open_unit_interval(
    value: object,
    field_name: str,
) -> float:
    value = _validate_finite_number(
        value,
        field_name,
    )

    if not 0.0 < value < 1.0:
        raise ValueError(
            f"{field_name} must satisfy 0 < {field_name} < 1."
        )

    return value


@dataclass(frozen=True)
class ADWINConfig:
    """
    Explicit configuration for River ADWIN 0.23.0.

    These values are passed explicitly to River rather than
    relying on River's defaults.
    """

    delta: float = 0.002
    clock: int = 32
    max_buckets: int = 5
    min_window_length: int = 5
    grace_period: int = 10

    def __post_init__(self) -> None:
        delta = _validate_open_unit_interval(
            self.delta,
            "delta",
        )

        _validate_positive_int(
            self.clock,
            "clock",
        )

        _validate_positive_int(
            self.max_buckets,
            "max_buckets",
        )

        _validate_positive_int(
            self.min_window_length,
            "min_window_length",
        )

        _validate_positive_int(
            self.grace_period,
            "grace_period",
        )


@dataclass(frozen=True)
class DDMConfig:
    """
    Explicit configuration for River DDM 0.23.0.
    """

    warm_start: int = 30
    warning_threshold: float = 2.0
    drift_threshold: float = 3.0

    def __post_init__(self) -> None:
        _validate_positive_int(
            self.warm_start,
            "warm_start",
        )

        warning = _validate_finite_number(
            self.warning_threshold,
            "warning_threshold",
        )

        drift = _validate_finite_number(
            self.drift_threshold,
            "drift_threshold",
        )

        if warning <= 0.0:
            raise ValueError(
                "warning_threshold must be greater than zero."
            )

        if drift <= 0.0:
            raise ValueError(
                "drift_threshold must be greater than zero."
            )

        if warning >= drift:
            raise ValueError(
                "warning_threshold must be strictly less "
                "than drift_threshold."
            )


@dataclass(frozen=True)
class PageHinkleyConfig:
    """
    Explicit configuration for River Page-Hinkley 0.23.0.

    River's mode accepts exactly:
        up
        down
        both
    """

    min_instances: int = 30
    delta: float = 0.005
    threshold: float = 50.0
    alpha: float = 0.9999
    mode: str = "both"

    def __post_init__(self) -> None:
        _validate_positive_int(
            self.min_instances,
            "min_instances",
        )

        delta = _validate_finite_number(
            self.delta,
            "delta",
        )

        if delta < 0.0:
            raise ValueError(
                "delta must be greater than or equal to zero."
            )

        threshold = _validate_finite_number(
            self.threshold,
            "threshold",
        )

        if threshold <= 0.0:
            raise ValueError(
                "threshold must be greater than zero."
            )

        _validate_open_unit_interval(
            self.alpha,
            "alpha",
        )

        if self.mode not in {
            "up",
            "down",
            "both",
        }:
            raise ValueError(
                "mode must be one of: 'up', 'down', 'both'."
            )
