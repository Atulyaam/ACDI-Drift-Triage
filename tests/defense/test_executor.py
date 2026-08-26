import pytest

from src.triage.triage import (
    TriageAction,
)

from src.defense.actions import (
    DefenseActionRequest,
    MonitorPayload,
)

from src.defense.executor import (
    DefenseExecutionResult,
    DefenseExecutor,
    ExecutionStatus,
)


RUN_ID = "RUN_001"
REFERENCE = "WIN_000001"
CURRENT = "WIN_000002"
REQUEST_ID = "7d8f5c4e-7f72-4a90-9d2f-8f44f2cbf9e1"


def _request(action=TriageAction.MONITOR):
    # Only MONITOR is exercised by this test file's helper.
    # If other actions are ever passed here, this will need
    # the matching payload type (RecalibrationPayload /
    # FreezePayload) added explicitly, same as here.
    payload = MonitorPayload(observation_required=True)

    return DefenseActionRequest(
        request_id=REQUEST_ID,
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        action=action,
        hold_retrain=False,
        reason="executor test",
        payload=payload,
    )


# ============================================================
# Result contract
# ============================================================

def test_execution_status_enum_values():
    assert ExecutionStatus.EXECUTED.value == "executed"
    assert ExecutionStatus.SKIPPED.value == "skipped"
    assert ExecutionStatus.REJECTED.value == "rejected"
    assert ExecutionStatus.ALREADY_EXECUTED.value == "already_executed"


def test_execution_result_constructs():
    result = DefenseExecutionResult(
        request_id=REQUEST_ID,
        status=ExecutionStatus.EXECUTED,
        action="monitor",
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        message="monitor request executed",
    )

    assert result.request_id == REQUEST_ID
    assert result.status is ExecutionStatus.EXECUTED
    assert result.action == "monitor"


def test_execution_result_is_frozen():
    result = DefenseExecutionResult(
        request_id=REQUEST_ID,
        status=ExecutionStatus.SKIPPED,
        action="monitor",
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        message="skipped",
    )

    with pytest.raises(Exception):
        result.status = ExecutionStatus.EXECUTED


def test_invalid_status_type_rejected():
    with pytest.raises(TypeError):
        DefenseExecutionResult(
            request_id=REQUEST_ID,
            status="executed",
            action="monitor",
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            message="bad",
        )


def test_empty_message_rejected():
    with pytest.raises(ValueError):
        DefenseExecutionResult(
            request_id=REQUEST_ID,
            status=ExecutionStatus.REJECTED,
            action="monitor",
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            message=" ",
        )


def test_same_reference_and_current_window_rejected():
    with pytest.raises(ValueError):
        DefenseExecutionResult(
            request_id=REQUEST_ID,
            status=ExecutionStatus.REJECTED,
            action="monitor",
            run_id=RUN_ID,
            reference_window_id=CURRENT,
            current_window_id=CURRENT,
            message="invalid",
        )


def test_metadata_defaults_to_dict():
    result = DefenseExecutionResult(
        request_id=REQUEST_ID,
        status=ExecutionStatus.SKIPPED,
        action="monitor",
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        message="skipped",
    )

    assert isinstance(result.metadata, dict)


# ============================================================
# Executor interface
# ============================================================

def test_executor_is_abstract():
    with pytest.raises(TypeError):
        DefenseExecutor()


def test_concrete_executor_can_implement_interface():
    class FakeExecutor(DefenseExecutor):
        def execute(self, request):
            return DefenseExecutionResult(
                request_id=request.request_id,
                status=ExecutionStatus.EXECUTED,
                action=request.action.value,
                run_id=request.run_id,
                reference_window_id=request.reference_window_id,
                current_window_id=request.current_window_id,
                message="fake execution",
            )

    executor = FakeExecutor()
    result = executor.execute(_request())

    assert result.status is ExecutionStatus.EXECUTED
    assert result.request_id == REQUEST_ID


def test_executor_result_preserves_request_identity():
    class FakeExecutor(DefenseExecutor):
        def execute(self, request):
            return DefenseExecutionResult(
                request_id=request.request_id,
                status=ExecutionStatus.SKIPPED,
                action=request.action.value,
                run_id=request.run_id,
                reference_window_id=request.reference_window_id,
                current_window_id=request.current_window_id,
                message="skipped",
            )

    result = FakeExecutor().execute(_request())

    assert result.request_id == REQUEST_ID
    assert result.run_id == RUN_ID
    assert result.reference_window_id == REFERENCE
    assert result.current_window_id == CURRENT


# ============================================================
# Idempotency contract
# ============================================================

def test_same_request_id_can_be_detected_as_already_executed():
    class IdempotentFakeExecutor(DefenseExecutor):
        def __init__(self):
            self._completed = set()

        def execute(self, request):
            if request.request_id in self._completed:
                return DefenseExecutionResult(
                    request_id=request.request_id,
                    status=ExecutionStatus.ALREADY_EXECUTED,
                    action=request.action.value,
                    run_id=request.run_id,
                    reference_window_id=request.reference_window_id,
                    current_window_id=request.current_window_id,
                    message="request was already executed",
                )

            self._completed.add(request.request_id)

            return DefenseExecutionResult(
                request_id=request.request_id,
                status=ExecutionStatus.EXECUTED,
                action=request.action.value,
                run_id=request.run_id,
                reference_window_id=request.reference_window_id,
                current_window_id=request.current_window_id,
                message="first execution",
            )

    executor = IdempotentFakeExecutor()

    first = executor.execute(_request())
    second = executor.execute(_request())

    assert first.status is ExecutionStatus.EXECUTED
    assert second.status is ExecutionStatus.ALREADY_EXECUTED


def test_different_request_ids_are_independent():
    class IdempotentFakeExecutor(DefenseExecutor):
        def __init__(self):
            self._completed = set()

        def execute(self, request):
            if request.request_id in self._completed:
                return DefenseExecutionResult(
                    request_id=request.request_id,
                    status=ExecutionStatus.ALREADY_EXECUTED,
                    action=request.action.value,
                    run_id=request.run_id,
                    reference_window_id=request.reference_window_id,
                    current_window_id=request.current_window_id,
                    message="already executed",
                )

            self._completed.add(request.request_id)

            return DefenseExecutionResult(
                request_id=request.request_id,
                status=ExecutionStatus.EXECUTED,
                action=request.action.value,
                run_id=request.run_id,
                reference_window_id=request.reference_window_id,
                current_window_id=request.current_window_id,
                message="executed",
            )

    request_a = _request()

    request_b = DefenseActionRequest(
        request_id="9e17a8fd-87cb-4bc1-ae43-5c8b0c15f2a1",
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        action=TriageAction.MONITOR,
        hold_retrain=False,
        reason="executor test",
        payload=MonitorPayload(observation_required=True),
    )

    executor = IdempotentFakeExecutor()

    first = executor.execute(request_a)
    second = executor.execute(request_b)

    assert first.status is ExecutionStatus.EXECUTED
    assert second.status is ExecutionStatus.EXECUTED
