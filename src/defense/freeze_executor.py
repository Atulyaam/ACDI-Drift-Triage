
from __future__ import annotations

"""
Concrete FREEZE executor — V1 execution-contract layer only.

Scope:
    - accepts FREEZE + FreezePayload only
    - does NOT mutate model or learning state
    - EXECUTED means the freeze request was accepted as a
      valid, well-formed instruction
    - idempotency is in-memory and scoped to this executor instance

Import boundary:
    src.defense.* and Python standard library only.
"""

from src.defense.actions import (
    DefenseActionRequest,
    FreezePayload,
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
    "executed: freeze request accepted"
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


class FreezeExecutor(DefenseExecutor):
    """
    Executes FREEZE requests only.

    V1 is a request-acceptance layer:
        no model mutation
        no learning-state mutation
        no optimizer mutation
        no checkpoint mutation

    Idempotency:
        per executor instance, in-memory only.
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
                "executor": "FreezeExecutor",
                "idempotency_scope": (
                    "executor_instance"
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
        if request.action is not TriageAction.FREEZE:
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
            FreezePayload,
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
        # V1 semantics:
        # accept a valid freeze instruction.
        #
        # No actual system/model freeze occurs here.
        # ----------------------------------------------------
        result = self._result_from_request(
            request,
            ExecutionStatus.EXECUTED,
            _MSG_EXECUTED,
        )

        self._completed[
            request.request_id
        ] = (
            request,
            result,
        )

        return result
