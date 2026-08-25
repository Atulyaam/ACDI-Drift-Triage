
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.triage.signal_report import SignalReport


class TriageAction(str, Enum):
    """
    Canonical five-way defense action.

    These are decisions only.
    Actual defense execution belongs to downstream layers.
    """

    NO_ACTION = "no_action"
    MONITOR = "monitor"
    RECALIBRATE = "recalibrate"
    RETRAIN = "retrain"
    FREEZE = "freeze"


@dataclass(frozen=True)
class TriageResult:
    """
    Canonical deterministic triage decision.

    Evidence comes from SignalReport.
    This object records the decision but performs no action.

    hold_retrain is an orthogonal control flag.

    Invariant:
        hold_retrain=True  <=> action == RECALIBRATE
        action != RECALIBRATE => hold_retrain=False
    """

    run_id: str
    reference_window_id: str
    current_window_id: str

    action: TriageAction
    hold_retrain: bool

    reason: str

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

        if not isinstance(
            self.action,
            TriageAction,
        ):
            raise TypeError(
                "action must be a TriageAction."
            )

        if not isinstance(
            self.hold_retrain,
            bool,
        ):
            raise TypeError(
                "hold_retrain must be a boolean."
            )

        if (
            self.hold_retrain
            and self.action
            is not TriageAction.RECALIBRATE
        ):
            raise ValueError(
                "hold_retrain=True is valid only when "
                "action is RECALIBRATE."
            )

        if (
            self.action
            is not TriageAction.RECALIBRATE
            and self.hold_retrain
        ):
            raise ValueError(
                "Non-RECALIBRATE actions cannot hold retraining."
            )

        if (
            not isinstance(
                self.reason,
                str,
            )
            or not self.reason.strip()
        ):
            raise ValueError(
                "reason must be a non-empty string."
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


def compute_triage_decision(
    signal_report: SignalReport,
) -> TriageResult:
    """
    Compute the deterministic five-way triage decision.

    Truth table:

        feature confidence error
            0       0        0   -> NO_ACTION
            1       0        0   -> MONITOR
            0       1        0   -> MONITOR
            0       0        1   -> RECALIBRATE + HOLD
            1       0        1   -> RETRAIN
            0       1        1   -> RETRAIN
            1       1        0   -> RECALIBRATE
            1       1        1   -> FREEZE

    No downstream action is executed here.
    """

    if not isinstance(
        signal_report,
        SignalReport,
    ):
        raise TypeError(
            "signal_report must be a SignalReport."
        )

    feature = signal_report.feature_drift
    confidence = signal_report.confidence_drift
    error = signal_report.error_drift

    # Three independent signal booleans are already validated
    # by SignalReport. This function treats them as canonical
    # evidence and applies only the frozen policy.

    active_count = (
        int(feature)
        + int(confidence)
        + int(error)
    )

    if active_count == 0:
        action = TriageAction.NO_ACTION
        hold_retrain = False
        reason = (
            "No feature, confidence, or error drift signal "
            "was detected."
        )

    elif active_count == 1:
        if error:
            action = TriageAction.RECALIBRATE
            hold_retrain = True
            reason = (
                "Error drift occurred without feature or "
                "confidence corroboration; recalibration is "
                "required while retraining remains on hold."
            )
        else:
            action = TriageAction.MONITOR
            hold_retrain = False

            if feature:
                reason = (
                    "Feature drift occurred without confidence "
                    "or error corroboration."
                )
            else:
                reason = (
                    "Confidence drift occurred without feature "
                    "or error corroboration."
                )

    elif active_count == 2:
        if error:
            action = TriageAction.RETRAIN
            hold_retrain = False
            reason = (
                "Error drift is corroborated by a second "
                "independent drift signal."
            )
        else:
            # Only feature + confidence are active.
            action = TriageAction.RECALIBRATE
            hold_retrain = False
            reason = (
                "Feature and confidence drift agree while "
                "error remains stable; use lightweight "
                "recalibration rather than retraining."
            )

    else:
        action = TriageAction.FREEZE
        hold_retrain = False
        reason = (
            "Feature, confidence, and error drift all fired "
            "in the same reporting window."
        )

    return TriageResult(
        run_id=signal_report.run_id,
        reference_window_id=(
            signal_report.reference_window_id
        ),
        current_window_id=(
            signal_report.current_window_id
        ),
        action=action,
        hold_retrain=hold_retrain,
        reason=reason,
        metadata={
            "source": "signal_report",
            "feature_drift": feature,
            "confidence_drift": confidence,
            "error_drift": error,
            "error_vote_count": (
                signal_report.error_vote_count
            ),
            "policy_version": "triage_v1",
        },
    )
