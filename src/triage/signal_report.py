
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.monitors.batch.feature_summary import (
    FeatureDriftSummary,
)

from src.monitors.batch.confidence import (
    ConfidenceDriftResult,
)

from src.monitors.streaming.error_vote import (
    ErrorDriftVoteResult,
)


def _is_strict_int(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
    )


@dataclass(frozen=True)
class SignalReport:
    """
    Canonical monitor-to-triage evidence contract.

    SignalReport is evidence only.
    It does not make a defense decision.

    Canonical fields:

        run_id
        reference_window_id
        current_window_id
        feature_drift
        confidence_drift
        error_drift
        error_vote_count
        metadata

    The report is constructed through build_signal_report()
    from the canonical upstream monitor results.

    Important traceability semantics:

        reference_window_id:
            FeatureDriftSummary.reference_window_id
            == ConfidenceDriftResult.reference_window_id

        current_window_id:
            FeatureDriftSummary.current_window_id
            == ConfidenceDriftResult.current_window_id
            == ErrorDriftVoteResult.reported_window_id

        run_id:
            sourced from ErrorDriftVoteResult.run_id because
            the upstream batch result contracts do not expose
            run_id directly.

    feature_drift is a normalized boolean derived by the
    builder from FeatureDriftSummary.n_features_significant > 0.

    error_vote_count is defensively validated even though the
    upstream ErrorDriftVoteResult already derives it.
    """

    run_id: str

    reference_window_id: str
    current_window_id: str

    feature_drift: bool
    confidence_drift: bool
    error_drift: bool

    error_vote_count: int

    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if (
            not isinstance(
                self.reference_window_id,
                str,
            )
            or not self.reference_window_id.strip()
        ):
            raise ValueError(
                "reference_window_id must be a "
                "non-empty string."
            )

        if (
            not isinstance(
                self.current_window_id,
                str,
            )
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

        for field_name, value in (
            ("feature_drift", self.feature_drift),
            (
                "confidence_drift",
                self.confidence_drift,
            ),
            ("error_drift", self.error_drift),
        ):
            if not isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be a boolean."
                )

        if not _is_strict_int(
            self.error_vote_count
        ):
            raise TypeError(
                "error_vote_count must be an integer "
                "(not bool)."
            )

        if not (
            0 <= self.error_vote_count <= 3
        ):
            raise ValueError(
                "error_vote_count must satisfy "
                "0 <= error_vote_count <= 3."
            )

        if (
            not isinstance(
                self.metadata,
                dict,
            )
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


def build_signal_report(
    feature_summary: FeatureDriftSummary,
    confidence_result: ConfidenceDriftResult,
    error_vote_result: ErrorDriftVoteResult,
) -> SignalReport:
    """
    Normalize the three canonical monitor outputs into one
    monitor-to-triage evidence object.

    This function performs independent boundary validation.

    It does NOT:
        - perform statistical tests
        - choose a triage action
        - trigger retraining
        - recalibrate
        - freeze learning
    """

    if not isinstance(
        feature_summary,
        FeatureDriftSummary,
    ):
        raise TypeError(
            "feature_summary must be a "
            "FeatureDriftSummary."
        )

    if not isinstance(
        confidence_result,
        ConfidenceDriftResult,
    ):
        raise TypeError(
            "confidence_result must be a "
            "ConfidenceDriftResult."
        )

    if not isinstance(
        error_vote_result,
        ErrorDriftVoteResult,
    ):
        raise TypeError(
            "error_vote_result must be a "
            "ErrorDriftVoteResult."
        )

    # --------------------------------------------------------
    # Defensive validation of the upstream-derived vote count.
    # Do not blindly trust ErrorDriftVoteResult at this boundary.
    # --------------------------------------------------------

    vote_count = (
        error_vote_result.error_vote_count
    )

    if not _is_strict_int(vote_count):
        raise TypeError(
            "error_vote_count must be an integer "
            "(not bool)."
        )

    if not 0 <= vote_count <= 3:
        raise ValueError(
            "error_vote_count must satisfy "
            "0 <= error_vote_count <= 3."
        )

    # --------------------------------------------------------
    # Reference-window traceability.
    # --------------------------------------------------------

    if (
        feature_summary.reference_window_id
        != confidence_result.reference_window_id
    ):
        raise ValueError(
            "Feature and confidence results must have "
            "the same reference_window_id."
        )

    # --------------------------------------------------------
    # Current-window traceability.
    #
    # ErrorDriftVoteResult exposes one current-window identity
    # through reported_window_id.
    # --------------------------------------------------------

    if (
        feature_summary.current_window_id
        != confidence_result.current_window_id
    ):
        raise ValueError(
            "Feature and confidence results must have "
            "the same current_window_id."
        )

    if (
        feature_summary.current_window_id
        != error_vote_result.reported_window_id
    ):
        raise ValueError(
            "Error vote result must refer to the same "
            "current window as feature and confidence results."
        )

    # --------------------------------------------------------
    # Feature signal normalization.
    #
    # FeatureDriftSummary intentionally does not own a separate
    # feature_drift boolean. The current canonical mapping is:
    #
    #     n_features_significant > 0
    #
    # The upstream summary remains responsible for future
    # threshold-policy changes.
    # --------------------------------------------------------

    feature_drift = (
        feature_summary.n_features_significant > 0
    )

    # --------------------------------------------------------
    # Confidence and error signals are copied from their
    # canonical result objects.
    # --------------------------------------------------------

    confidence_drift = (
        confidence_result.significant
    )

    error_drift = (
        error_vote_result.error_drift
    )

    # --------------------------------------------------------
    # Provenance metadata.
    # --------------------------------------------------------

    metadata = {
        "source_signal_names": [
            "feature_drift_summary",
            "confidence_drift_result",
            "error_drift_vote_result",
        ],
    }

    return SignalReport(
        run_id=error_vote_result.run_id,
        reference_window_id=(
            feature_summary.reference_window_id
        ),
        current_window_id=(
            feature_summary.current_window_id
        ),
        feature_drift=feature_drift,
        confidence_drift=confidence_drift,
        error_drift=error_drift,
        error_vote_count=vote_count,
        metadata=metadata,
    )
