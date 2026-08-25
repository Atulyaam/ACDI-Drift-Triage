
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.defense.actions import (
    DefenseActionRequest,
)

from src.triage.triage import (
    TriageAction,
)


class ValidationStatus(str, Enum):
    """
    Outcome of one retraining-validation stage.
    """

    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class ShadowValidationResult:
    """
    Evidence from candidate-vs-current shadow validation.

    This object records evidence only.
    It performs no model execution.
    """

    status: ValidationStatus

    baseline_metric: float | None = None
    candidate_metric: float | None = None
    metric_delta: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.status,
            ValidationStatus,
        ):
            raise TypeError(
                "status must be a ValidationStatus."
            )

        self._validate_optional_metric(
            self.baseline_metric,
            "baseline_metric",
        )

        self._validate_optional_metric(
            self.candidate_metric,
            "candidate_metric",
        )

        self._validate_optional_metric(
            self.metric_delta,
            "metric_delta",
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

    @staticmethod
    def _validate_optional_metric(
        value: float | None,
        field_name: str,
    ) -> None:
        if value is None:
            return

        if isinstance(value, bool):
            raise TypeError(
                f"{field_name} must be numeric or None."
            )

        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{field_name} must be numeric or None."
            )


@dataclass(frozen=True)
class HistoricalReplayResult:
    """
    Evidence from replaying the candidate model on trusted
    historical data.

    This object records evidence only.
    It performs no model execution.
    """

    status: ValidationStatus

    baseline_metric: float | None = None
    candidate_metric: float | None = None
    metric_delta: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.status,
            ValidationStatus,
        ):
            raise TypeError(
                "status must be a ValidationStatus."
            )

        self._validate_optional_metric(
            self.baseline_metric,
            "baseline_metric",
        )

        self._validate_optional_metric(
            self.candidate_metric,
            "candidate_metric",
        )

        self._validate_optional_metric(
            self.metric_delta,
            "metric_delta",
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

    @staticmethod
    def _validate_optional_metric(
        value: float | None,
        field_name: str,
    ) -> None:
        if value is None:
            return

        if isinstance(value, bool):
            raise TypeError(
                f"{field_name} must be numeric or None."
            )

        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{field_name} must be numeric or None."
            )


@dataclass(frozen=True)
class RetrainGateResult:
    """
    Final promotion decision for a candidate retrained model.

    Promotion is allowed ONLY when:

        shadow.status == PASS
        AND
        replay.status == PASS

    FAIL and NOT_RUN are both non-promoting outcomes.

    This contract does not execute promotion or retraining.
    """

    request_id: str
    run_id: str

    reference_window_id: str
    current_window_id: str

    shadow_status: ValidationStatus
    replay_status: ValidationStatus

    promote: bool

    reason: str

    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.request_id,
                str,
            )
            or not self.request_id.strip()
        ):
            raise ValueError(
                "request_id must be a non-empty string."
            )

        if (
            not isinstance(
                self.run_id,
                str,
            )
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
            self.shadow_status,
            ValidationStatus,
        ):
            raise TypeError(
                "shadow_status must be a ValidationStatus."
            )

        if not isinstance(
            self.replay_status,
            ValidationStatus,
        ):
            raise TypeError(
                "replay_status must be a ValidationStatus."
            )

        if not isinstance(
            self.promote,
            bool,
        ):
            raise TypeError(
                "promote must be a boolean."
            )

        expected_promote = (
            self.shadow_status
            is ValidationStatus.PASS
            and
            self.replay_status
            is ValidationStatus.PASS
        )

        if self.promote != expected_promote:
            raise ValueError(
                "promote must equal "
                "(shadow_status == PASS and "
                "replay_status == PASS)."
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


def evaluate_retrain_gate(
    request: DefenseActionRequest,
    shadow: ShadowValidationResult,
    replay: HistoricalReplayResult,
) -> RetrainGateResult:
    """
    Evaluate whether a RETRAIN defense request may promote
    its candidate model.

    This function performs no training, shadow execution,
    historical replay, or model promotion.

    It consumes already-produced validation evidence.
    """

    if not isinstance(
        request,
        DefenseActionRequest,
    ):
        raise TypeError(
            "request must be a DefenseActionRequest."
        )

    if request.action is not TriageAction.RETRAIN:
        raise ValueError(
            "Retrain gate requires a RETRAIN "
            "DefenseActionRequest."
        )

    if request.hold_retrain:
        raise ValueError(
            "RETRAIN requests cannot have "
            "hold_retrain=True."
        )

    if not isinstance(
        shadow,
        ShadowValidationResult,
    ):
        raise TypeError(
            "shadow must be a ShadowValidationResult."
        )

    if not isinstance(
        replay,
        HistoricalReplayResult,
    ):
        raise TypeError(
            "replay must be a HistoricalReplayResult."
        )

    promote = (
        shadow.status
        is ValidationStatus.PASS
        and
        replay.status
        is ValidationStatus.PASS
    )

    if promote:
        reason = (
            "Candidate retraining passed both shadow "
            "validation and historical replay."
        )
    else:
        failed_stages = []

        if shadow.status is not ValidationStatus.PASS:
            failed_stages.append("shadow validation")

        if replay.status is not ValidationStatus.PASS:
            failed_stages.append("historical replay")

        reason = (
            "Candidate retraining was not promoted because "
            + " and ".join(failed_stages)
            + " did not pass."
        )

    return RetrainGateResult(
        request_id=request.request_id,
        run_id=request.run_id,
        reference_window_id=(
            request.reference_window_id
        ),
        current_window_id=(
            request.current_window_id
        ),
        shadow_status=shadow.status,
        replay_status=replay.status,
        promote=promote,
        reason=reason,
        metadata={
            "source": "defense_action_request",
            "gate_policy_version": "retrain_gate_v1",
        },
    )
