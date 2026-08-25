from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from src.triage.triage import (
    TriageAction,
    TriageResult,
)


def _is_uuid4(value: object) -> bool:
    if not isinstance(value, str):
        return False

    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return False

    return (
        parsed.version == 4
        and str(parsed) == value
    )


def _validate_non_empty_string(
    value: object,
    field_name: str,
) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a non-empty string."
        )


@dataclass(frozen=True)
class DefensePayload(ABC):
    """
    Common immutable base for action-specific defense payloads.

    Payloads describe requested behavior only.
    They never execute defense actions.
    """

    # kw_only=True is required here: subclasses (MonitorPayload,
    # RecalibrationPayload, FreezePayload) add required fields with
    # no default. Without kw_only, Python's dataclass field ordering
    # rule ("non-default argument follows default argument") raises
    # TypeError at class-definition time for every subclass, since
    # this defaulted field is inherited first in MRO order. kw_only
    # fields are excluded from that ordering constraint.
    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
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


@dataclass(frozen=True)
class MonitorPayload(DefensePayload):
    """
    Payload for MONITOR.

    observation_required has no default because the caller
    must explicitly state the monitoring intent.
    """

    observation_required: bool

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.observation_required, bool):
            raise TypeError(
                "observation_required must be a boolean."
            )


@dataclass(frozen=True)
class RecalibrationPayload(DefensePayload):
    """
    Payload for RECALIBRATE.

    hold_retrain is advisory only at this layer. It does not
    directly manipulate RetrainGate, cancel a pending retrain
    request, or execute anything. It is intent-only metadata
    for now; enforcement, if any, belongs to a future executor.
    """

    reason: str
    hold_retrain: bool

    def __post_init__(self) -> None:
        super().__post_init__()

        _validate_non_empty_string(self.reason, "reason")

        if not isinstance(self.hold_retrain, bool):
            raise TypeError(
                "hold_retrain must be a boolean."
            )


@dataclass(frozen=True)
class FreezePayload(DefensePayload):
    """
    Payload for FREEZE.

    This is a request description only.
    No actual model/learning freeze is performed here.
    """

    reason: str

    def __post_init__(self) -> None:
        super().__post_init__()

        _validate_non_empty_string(self.reason, "reason")


_ACTION_PAYLOAD_TYPES = {
    TriageAction.NO_ACTION: type(None),
    TriageAction.MONITOR: MonitorPayload,
    TriageAction.RECALIBRATE: RecalibrationPayload,
    TriageAction.RETRAIN: type(None),
    TriageAction.FREEZE: FreezePayload,
}


@dataclass(frozen=True)
class DefenseActionRequest:
    """
    Immutable command description for the defense layer.

    Architecture:

        Triage decides.
        DefenseActionRequest describes.
        Future executor performs.

    Action/payload mapping is enforced structurally:

        NO_ACTION    -> None
        MONITOR      -> MonitorPayload
        RECALIBRATE  -> RecalibrationPayload
        RETRAIN      -> None
        FREEZE       -> FreezePayload

    No defense action is executed by this class.
    """

    request_id: str

    run_id: str
    reference_window_id: str
    current_window_id: str

    action: TriageAction
    hold_retrain: bool

    reason: str

    payload: DefensePayload | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError(
                "request_id must be a string."
            )

        if not _is_uuid4(self.request_id):
            raise ValueError(
                "request_id must be a valid UUID4 string."
            )

        _validate_non_empty_string(self.run_id, "run_id")
        _validate_non_empty_string(
            self.reference_window_id, "reference_window_id"
        )
        _validate_non_empty_string(
            self.current_window_id, "current_window_id"
        )

        if self.reference_window_id == self.current_window_id:
            raise ValueError(
                "reference_window_id and current_window_id "
                "must be different."
            )

        if not isinstance(self.action, TriageAction):
            raise TypeError(
                "action must be a TriageAction."
            )

        if self.action not in _ACTION_PAYLOAD_TYPES:
            raise ValueError(
                f"Unrecognized TriageAction: {self.action!r}."
            )

        if not isinstance(self.hold_retrain, bool):
            raise TypeError(
                "hold_retrain must be a boolean."
            )

        if (
            self.hold_retrain
            and self.action is not TriageAction.RECALIBRATE
        ):
            raise ValueError(
                "hold_retrain=True is valid only when "
                "action is RECALIBRATE."
            )

        _validate_non_empty_string(self.reason, "reason")

        if (
            self.payload is not None
            and not isinstance(self.payload, DefensePayload)
        ):
            raise TypeError(
                "payload must be a DefensePayload or None."
            )

        expected_payload_type = _ACTION_PAYLOAD_TYPES[self.action]

        if expected_payload_type is type(None):
            if self.payload is not None:
                raise ValueError(
                    f"{self.action.value} does not accept "
                    "an action-specific payload."
                )
        elif not isinstance(self.payload, expected_payload_type):
            raise TypeError(
                f"{self.action.value} requires payload type "
                f"{expected_payload_type.__name__}."
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


def build_defense_action_request(
    triage_result: TriageResult,
    request_id: str | None = None,
    payload: DefensePayload | None = None,
) -> DefenseActionRequest:
    """
    Convert one TriageResult into one DefenseActionRequest.

    No triage logic is recomputed.
    No defense action is executed.

    Payload must match the TriageAction.
    """

    if not isinstance(triage_result, TriageResult):
        raise TypeError(
            "triage_result must be a TriageResult."
        )

    if request_id is None:
        request_id = str(uuid.uuid4())

    if not isinstance(request_id, str):
        raise TypeError(
            "request_id must be a string."
        )

    if not _is_uuid4(request_id):
        raise ValueError(
            "request_id must be a valid UUID4 string."
        )

    if not isinstance(triage_result.hold_retrain, bool):
        raise TypeError(
            "triage_result.hold_retrain must be a boolean."
        )

    if (
        triage_result.hold_retrain
        and triage_result.action is not TriageAction.RECALIBRATE
    ):
        raise ValueError(
            "triage_result has invalid "
            "hold_retrain/action combination."
        )

    return DefenseActionRequest(
        request_id=request_id,
        run_id=triage_result.run_id,
        reference_window_id=triage_result.reference_window_id,
        current_window_id=triage_result.current_window_id,
        action=triage_result.action,
        hold_retrain=triage_result.hold_retrain,
        reason=triage_result.reason,
        payload=payload,
        metadata={
            "source": "triage_result",
            "triage_policy_version": (
                triage_result.metadata.get("policy_version")
            ),
        },
    )
