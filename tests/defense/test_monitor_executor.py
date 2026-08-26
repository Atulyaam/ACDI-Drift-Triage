
import ast
import uuid
from pathlib import Path

import pytest

from src.triage.triage import (
    TriageAction,
)

from src.defense.actions import (
    DefenseActionRequest,
    MonitorPayload,
)

from src.defense.executor import (
    ExecutionStatus,
)

from src.defense.monitor_executor import (
    MonitorExecutor,
)


RUN_ID = "RUN_001"
REFERENCE = "WIN_000001"
CURRENT = "WIN_000002"


def _uuid4():
    return str(uuid.uuid4())


def _request(
    *,
    request_id=None,
    observation_required=True,
    reason="monitor request",
):
    if request_id is None:
        request_id = _uuid4()

    return DefenseActionRequest(
        request_id=request_id,
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        action=TriageAction.MONITOR,
        hold_retrain=False,
        reason=reason,
        payload=MonitorPayload(
            observation_required=observation_required
        ),
    )


def test_monitor_true_returns_executed():
    executor = MonitorExecutor()

    result = executor.execute(
        _request(
            observation_required=True
        )
    )

    assert result.status is ExecutionStatus.EXECUTED
    assert result.action == "monitor"
    assert (
        result.message
        == "executed: monitoring request accepted"
    )


def test_monitor_false_returns_skipped():
    executor = MonitorExecutor()

    result = executor.execute(
        _request(
            observation_required=False
        )
    )

    assert result.status is ExecutionStatus.SKIPPED
    assert result.action == "monitor"
    assert (
        result.message
        == "skipped: monitoring not required"
    )


def test_wrong_action_returns_rejected():
    request = object.__new__(
        DefenseActionRequest
    )

    object.__setattr__(
        request,
        "request_id",
        _uuid4(),
    )
    object.__setattr__(
        request,
        "run_id",
        RUN_ID,
    )
    object.__setattr__(
        request,
        "reference_window_id",
        REFERENCE,
    )
    object.__setattr__(
        request,
        "current_window_id",
        CURRENT,
    )
    object.__setattr__(
        request,
        "action",
        TriageAction.FREEZE,
    )
    object.__setattr__(
        request,
        "hold_retrain",
        False,
    )
    object.__setattr__(
        request,
        "reason",
        "corrupted request",
    )
    object.__setattr__(
        request,
        "payload",
        None,
    )
    object.__setattr__(
        request,
        "metadata",
        {},
    )

    result = MonitorExecutor().execute(
        request
    )

    assert result.status is ExecutionStatus.REJECTED
    assert (
        result.message
        == "rejected: wrong action"
    )


def test_wrong_payload_returns_rejected():
    request = object.__new__(
        DefenseActionRequest
    )

    object.__setattr__(
        request,
        "request_id",
        _uuid4(),
    )
    object.__setattr__(
        request,
        "run_id",
        RUN_ID,
    )
    object.__setattr__(
        request,
        "reference_window_id",
        REFERENCE,
    )
    object.__setattr__(
        request,
        "current_window_id",
        CURRENT,
    )
    object.__setattr__(
        request,
        "action",
        TriageAction.MONITOR,
    )
    object.__setattr__(
        request,
        "hold_retrain",
        False,
    )
    object.__setattr__(
        request,
        "reason",
        "corrupted request",
    )
    object.__setattr__(
        request,
        "payload",
        None,
    )
    object.__setattr__(
        request,
        "metadata",
        {},
    )

    result = MonitorExecutor().execute(
        request
    )

    assert result.status is ExecutionStatus.REJECTED
    assert (
        result.message
        == "rejected: wrong payload type"
    )


def test_rejected_requests_are_not_idempotency_tracked():
    request = object.__new__(
        DefenseActionRequest
    )

    object.__setattr__(
        request,
        "request_id",
        _uuid4(),
    )
    object.__setattr__(
        request,
        "run_id",
        RUN_ID,
    )
    object.__setattr__(
        request,
        "reference_window_id",
        REFERENCE,
    )
    object.__setattr__(
        request,
        "current_window_id",
        CURRENT,
    )
    object.__setattr__(
        request,
        "action",
        TriageAction.FREEZE,
    )
    object.__setattr__(
        request,
        "hold_retrain",
        False,
    )
    object.__setattr__(
        request,
        "reason",
        "wrong action",
    )
    object.__setattr__(
        request,
        "payload",
        None,
    )
    object.__setattr__(
        request,
        "metadata",
        {},
    )

    executor = MonitorExecutor()

    first = executor.execute(request)
    second = executor.execute(request)

    assert (
        first.status
        is ExecutionStatus.REJECTED
    )
    assert (
        second.status
        is ExecutionStatus.REJECTED
    )


def test_same_request_returns_already_executed():
    executor = MonitorExecutor()

    request = _request()

    first = executor.execute(
        request
    )

    second = executor.execute(
        request
    )

    assert (
        first.status
        is ExecutionStatus.EXECUTED
    )

    assert (
        second.status
        is ExecutionStatus.ALREADY_EXECUTED
    )

    assert (
        second.request_id
        == first.request_id
    )


def test_same_request_with_false_observation_returns_already_executed():
    executor = MonitorExecutor()

    request = _request(
        observation_required=False
    )

    first = executor.execute(
        request
    )

    second = executor.execute(
        request
    )

    assert (
        first.status
        is ExecutionStatus.SKIPPED
    )

    assert (
        second.status
        is ExecutionStatus.ALREADY_EXECUTED
    )


def test_same_request_id_with_different_payload_is_rejected_loudly():
    executor = MonitorExecutor()

    request_id = _uuid4()

    first = _request(
        request_id=request_id,
        observation_required=True,
    )

    second = _request(
        request_id=request_id,
        observation_required=False,
    )

    executor.execute(first)

    with pytest.raises(ValueError):
        executor.execute(second)


def test_same_request_id_with_different_reason_is_rejected_loudly():
    executor = MonitorExecutor()

    request_id = _uuid4()

    first = _request(
        request_id=request_id,
        reason="original reason",
    )

    second = _request(
        request_id=request_id,
        reason="different reason",
    )

    executor.execute(first)

    with pytest.raises(ValueError):
        executor.execute(second)


def test_same_request_id_with_different_window_is_rejected_loudly():
    executor = MonitorExecutor()

    request_id = _uuid4()

    first = _request(
        request_id=request_id
    )

    second = DefenseActionRequest(
        request_id=request_id,
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id="WIN_000003",
        action=TriageAction.MONITOR,
        hold_retrain=False,
        reason="monitor request",
        payload=MonitorPayload(
            observation_required=True
        ),
    )

    executor.execute(first)

    with pytest.raises(ValueError):
        executor.execute(second)


def test_different_request_ids_execute_independently():
    executor = MonitorExecutor()

    first = executor.execute(
        _request(
            request_id=_uuid4()
        )
    )

    second = executor.execute(
        _request(
            request_id=_uuid4()
        )
    )

    assert (
        first.status
        is ExecutionStatus.EXECUTED
    )

    assert (
        second.status
        is ExecutionStatus.EXECUTED
    )


def test_idempotency_is_scoped_to_executor_instance():
    request = _request(
        request_id=_uuid4()
    )

    first_executor = MonitorExecutor()
    second_executor = MonitorExecutor()

    first = first_executor.execute(
        request
    )

    second = second_executor.execute(
        request
    )

    assert (
        first.status
        is ExecutionStatus.EXECUTED
    )

    assert (
        second.status
        is ExecutionStatus.EXECUTED
    )


def test_execution_result_preserves_traceability():
    request = _request(
        request_id=_uuid4()
    )

    result = MonitorExecutor().execute(
        request
    )

    assert result.request_id == request.request_id
    assert result.run_id == request.run_id

    assert (
        result.reference_window_id
        == request.reference_window_id
    )

    assert (
        result.current_window_id
        == request.current_window_id
    )


def test_execution_result_records_executor_name():
    result = MonitorExecutor().execute(
        _request()
    )

    assert (
        result.metadata["executor"]
        == "MonitorExecutor"
    )


def test_non_request_input_is_rejected():
    with pytest.raises(TypeError):
        MonitorExecutor().execute(
            object()
        )


def test_module_import_boundary_is_defense_only_plus_stdlib():
    # Anchored to this test file's own location rather than a bare
    # relative path, so it is not dependent on pytest's cwd matching
    # CODE_ROOT (this project has previously hit exactly that class
    # of failure with a 3-byte placeholder file going undetected
    # because a relative path silently resolved to nothing).
    #
    # tests/defense/test_monitor_executor.py
    #   parents[0] = tests/defense
    #   parents[1] = tests
    #   parents[2] = <repo root>
    repo_root = Path(__file__).resolve().parents[2]

    source_path = (
        repo_root
        / "src"
        / "defense"
        / "monitor_executor.py"
    )

    assert source_path.is_file(), (
        f"monitor_executor.py not found at {source_path}"
    )

    source = source_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Import,
        ):
            for alias in node.names:
                root = alias.name.split(".")[0]

                if root not in {
                    "__future__",
                    "typing",
                }:
                    raise AssertionError(
                        f"Disallowed import: {alias.name}"
                    )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module is None:
                continue

            root = node.module.split(".")[0]

            if node.level == 0:
                if root not in {
                    "__future__",
                    "typing",
                    "src",
                }:
                    raise AssertionError(
                        f"Disallowed import: {node.module}"
                    )

                if (
                    root == "src"
                    and not node.module.startswith(
                        "src.defense."
                    )
                ):
                    raise AssertionError(
                        f"Direct non-defense src import: "
                        f"{node.module}"
                    )

    assert True
