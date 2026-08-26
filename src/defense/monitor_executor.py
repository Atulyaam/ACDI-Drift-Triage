from __future__ import annotations

"""
src/defense/monitor_executor.py

Concrete zero-risk MONITOR executor.

Scope:
    - accepts MONITOR only
    - never mutates model, calibration, retraining,
      learning, or detector state
    - records EXECUTED / SKIPPED
    - supports per-instance idempotency
"""

from src.defense.actions import (
    DefenseActionRequest,
    MonitorPayload,
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
    "executed: monitoring request accepted"
)

_MSG_SKIPPED = (
    "skipped: monitoring not required"
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


class MonitorExecutor(DefenseExecutor):
    """
    Executes MONITOR requests only.

    Idempotency scope:
        Per executor instance, in-memory only.

    A request that was successfully processed
    (EXECUTED or SKIPPED) is remembered.

    REJECTED requests are not remembered because they
    never constituted a valid execution.

    Reusing the same request_id with different request
    contents is treated as an upstream identity error
    and raises ValueError.
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
            stored_value = getattr(
                stored,
                field_name,
                None,
            )

            incoming_value = getattr(
                incoming,
                field_name,
                None,
            )

            if stored_value != incoming_value:
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
                "executor": "MonitorExecutor",
                "idempotency_scope": (
                    "executor_instance"
                ),
            },
        )

    def execute(
        self,
        request: DefenseActionRequest,
    ) -> DefenseExecutionResult:
        # ----------------------------------------------------
        # Boundary validation
        # ----------------------------------------------------
        if not isinstance(
            request,
            DefenseActionRequest,
        ):
            raise TypeError(
                "request must be a DefenseActionRequest."
            )

        # ----------------------------------------------------
        # Wrong-action rejection
        # ----------------------------------------------------
        if request.action is not TriageAction.MONITOR:
            return self._result_from_request(
                request=request,
                status=ExecutionStatus.REJECTED,
                message=_MSG_REJECTED_WRONG_ACTION,
            )

        # ----------------------------------------------------
        # Wrong-payload rejection
        # ----------------------------------------------------
        if not isinstance(
            request.payload,
            MonitorPayload,
        ):
            return self._result_from_request(
                request=request,
                status=ExecutionStatus.REJECTED,
                message=_MSG_REJECTED_WRONG_PAYLOAD,
            )

        # ----------------------------------------------------
        # Idempotency lookup
        # ----------------------------------------------------
        previous = self._completed.get(
            request.request_id
        )

        if previous is not None:
            stored_request, stored_result = previous

            if not self._request_identity_matches(
                stored_request,
                request,
            ):
                raise ValueError(
                    "request_id was reused with "
                    "different request contents."
                )

            return self._result_from_request(
                request=request,
                status=ExecutionStatus.ALREADY_EXECUTED,
                message=_MSG_ALREADY_EXECUTED,
            )

        # ----------------------------------------------------
        # MONITOR execution semantics
        # ----------------------------------------------------
        if request.payload.observation_required:
            result = self._result_from_request(
                request=request,
                status=ExecutionStatus.EXECUTED,
                message=_MSG_EXECUTED,
            )
        else:
            result = self._result_from_request(
                request=request,
                status=ExecutionStatus.SKIPPED,
                message=_MSG_SKIPPED,
            )

        # ----------------------------------------------------
        # Only successful/accepted outcomes are remembered.
        # REJECTED requests never reach this point.
        # ----------------------------------------------------
        self._completed[
            request.request_id
        ] = (
            request,
            result,
        )

        return result
