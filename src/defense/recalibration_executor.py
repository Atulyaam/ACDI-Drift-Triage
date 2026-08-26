
from __future__ import annotations

"""
Concrete RECALIBRATE executor — V1 execution-contract layer only.

Scope:
    - accepts RECALIBRATE + RecalibrationPayload only
    - does not implement a concrete calibration algorithm
    - EXECUTED means the request was accepted as a valid instruction
    - hold_retrain is advisory only
    - request-level and payload-level hold_retrain are both preserved
      in result metadata
    - idempotency is in-memory and scoped to this executor instance

Import boundary:
    src.defense.* and Python standard library only.
"""

from src.defense.actions import (
    DefenseActionRequest,
    RecalibrationPayload,
    TriageAction,
)

from src.defense.executor import (
    DefenseExecutionResult,
    DefenseExecutor,
    ExecutionStatus,
)


_MSG_REJECTED_WRONG_ACTION = (
    "rejected: wrong action"
)

_MSG_REJECTED_WRONG_PAYLOAD = (
    "rejected: wrong payload type"
)

_MSG_EXECUTED = (
    "executed: recalibration request accepted"
)

_MSG_ALREADY_EXECUTED = (
    "already_executed: request was already processed"
)


_IDENTITY_FIELDS = (
    "run_id",
    "reference_window_id",
    "current_window_id",
    "action",
    "hold_retrain",
    "reason",
    "payload",
    "metadata",
)


class RecalibrationExecutor(
    DefenseExecutor
):
    """
    Executes RECALIBRATE requests only.

    Idempotency:
        - per executor instance
        - in-memory only
        - EXECUTED requests are remembered
        - REJECTED requests are not remembered
        - same request_id + changed request contents raises ValueError

    There is no SKIPPED outcome for recalibration in V1.
    """

    def __init__(self) -> None:
        self._completed: dict[
            str,
            tuple[
                DefenseActionRequest,
                DefenseExecutionResult,
            ],
        ] = {}

    @staticmethod
    def _request_identity_matches(
        stored: DefenseActionRequest,
        incoming: DefenseActionRequest,
    ) -> bool:
        for field_name in _IDENTITY_FIELDS:
            if getattr(
                stored,
                field_name,
                None,
            ) != getattr(
                incoming,
                field_name,
                None,
            ):
                return False

        return True

    @staticmethod
    def _result_from_request(
        request: DefenseActionRequest,
        status: ExecutionStatus,
        message: str,
    ) -> DefenseExecutionResult:
        payload = request.payload

        payload_hold_retrain = None

        if isinstance(
            payload,
            RecalibrationPayload,
        ):
            payload_hold_retrain = (
                payload.hold_retrain
            )

        return DefenseExecutionResult(
            request_id=request.request_id,
            status=status,
            action=request.action.value,
            run_id=request.run_id,
            reference_window_id=(
                request.reference_window_id
            ),
            current_window_id=(
                request.current_window_id
            ),
            message=message,
            metadata={
                "executor": (
                    "RecalibrationExecutor"
                ),
                "idempotency_scope": (
                    "executor_instance"
                ),
                "request_hold_retrain": (
                    request.hold_retrain
                ),
                "payload_hold_retrain": (
                    payload_hold_retrain
                ),
                "execution_scope": (
                    "request_acceptance_only"
                ),
            },
        )

    def execute(
        self,
        request: DefenseActionRequest,
    ) -> DefenseExecutionResult:

        if not isinstance(
            request,
            DefenseActionRequest,
        ):
            raise TypeError(
                "request must be a DefenseActionRequest."
            )

        # ----------------------------------------------------
        # Wrong action
        # ----------------------------------------------------
        if request.action is not TriageAction.RECALIBRATE:
            return self._result_from_request(
                request,
                ExecutionStatus.REJECTED,
                _MSG_REJECTED_WRONG_ACTION,
            )

        # ----------------------------------------------------
        # Wrong payload
        # ----------------------------------------------------
        if not isinstance(
            request.payload,
            RecalibrationPayload,
        ):
            return self._result_from_request(
                request,
                ExecutionStatus.REJECTED,
                _MSG_REJECTED_WRONG_PAYLOAD,
            )

        # ----------------------------------------------------
        # Idempotency
        # ----------------------------------------------------
        previous = self._completed.get(
            request.request_id
        )

        if previous is not None:
            stored_request, _stored_result = previous

            if not self._request_identity_matches(
                stored_request,
                request,
            ):
                raise ValueError(
                    "request_id was reused with "
                    "different request contents."
                )

            return self._result_from_request(
                request,
                ExecutionStatus.ALREADY_EXECUTED,
                _MSG_ALREADY_EXECUTED,
            )

        # ----------------------------------------------------
        # V1 execution semantics:
        # valid recalibration request accepted.
        #
        # No calibration library, model, checkpoint, or
        # RetrainGate is invoked here.
        # ----------------------------------------------------
        result = self._result_from_request(
            request,
            ExecutionStatus.EXECUTED,
            _MSG_EXECUTED,
        )

        # Only an accepted execution is remembered.
        self._completed[
            request.request_id
        ] = (
            request,
            result,
        )

        return result
