
import ast
import uuid
from pathlib import Path

import pytest

from src.triage.triage import (
    TriageAction,
)

from src.defense.actions import (
    DefenseActionRequest,
    RecalibrationPayload,
)

from src.defense.executor import (
    ExecutionStatus,
)

from src.defense.recalibration_executor import (
    RecalibrationExecutor,
)


RUN_ID = "RUN_001"
REFERENCE = "WIN_000001"
CURRENT = "WIN_000002"


def _uuid4():
    return str(
        uuid.uuid4()
    )


def _request(
    *,
    request_id=None,
    request_hold=False,
    payload_hold=False,
    reason="recalibration requested",
):
    if request_id is None:
        request_id = _uuid4()

    return DefenseActionRequest(
        request_id=request_id,
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        action=TriageAction.RECALIBRATE,
        hold_retrain=request_hold,
        reason=reason,
        payload=RecalibrationPayload(
            reason=reason,
            hold_retrain=payload_hold,
        ),
    )


def test_valid_recalibration_returns_executed():
    result = RecalibrationExecutor().execute(
        _request()
    )

    assert (
        result.status
        is ExecutionStatus.EXECUTED
    )

    assert result.action == "recalibrate"

    assert (
        result.message
        == "executed: recalibration request accepted"
    )


def test_no_skipped_outcome_is_used():
    result = RecalibrationExecutor().execute(
        _request()
    )

    assert (
        result.status
        is not ExecutionStatus.SKIPPED
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

    result = RecalibrationExecutor().execute(
        request
    )

    assert (
        result.status
        is ExecutionStatus.REJECTED
    )

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
        TriageAction.RECALIBRATE,
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

    result = RecalibrationExecutor().execute(
        request
    )

    assert (
        result.status
        is ExecutionStatus.REJECTED
    )

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

    executor = RecalibrationExecutor()

    first = executor.execute(
        request
    )

    second = executor.execute(
        request
    )

    assert (
        first.status
        is ExecutionStatus.REJECTED
    )

    assert (
        second.status
        is ExecutionStatus.REJECTED
    )


def test_duplicate_request_returns_already_executed():
    executor = RecalibrationExecutor()

    request = _request(
        request_id=_uuid4()
    )

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


def test_same_request_id_with_different_payload_is_rejected_loudly():
    executor = RecalibrationExecutor()

    request_id = _uuid4()

    first = _request(
        request_id=request_id,
        payload_hold=False,
    )

    second = _request(
        request_id=request_id,
        payload_hold=True,
    )

    executor.execute(first)

    with pytest.raises(ValueError):
        executor.execute(second)


def test_same_request_id_with_different_reason_is_rejected_loudly():
    executor = RecalibrationExecutor()

    request_id = _uuid4()

    first = _request(
        request_id=request_id,
        reason="first reason",
    )

    second = _request(
        request_id=request_id,
        reason="second reason",
    )

    executor.execute(first)

    with pytest.raises(ValueError):
        executor.execute(second)


def test_different_request_ids_are_independent():
    executor = RecalibrationExecutor()

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

    executor_a = RecalibrationExecutor()
    executor_b = RecalibrationExecutor()

    first = executor_a.execute(
        request
    )

    second = executor_b.execute(
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


def test_request_hold_retrain_is_recorded():
    result = RecalibrationExecutor().execute(
        _request(
            request_hold=True,
            payload_hold=False,
        )
    )

    assert (
        result.metadata[
            "request_hold_retrain"
        ]
        is True
    )


def test_payload_hold_retrain_is_recorded():
    result = RecalibrationExecutor().execute(
        _request(
            request_hold=False,
            payload_hold=True,
        )
    )

    assert (
        result.metadata[
            "payload_hold_retrain"
        ]
        is True
    )


def test_both_hold_values_are_preserved_separately():
    result = RecalibrationExecutor().execute(
        _request(
            request_hold=True,
            payload_hold=False,
        )
    )

    assert (
        result.metadata[
            "request_hold_retrain"
        ]
        is True
    )

    assert (
        result.metadata[
            "payload_hold_retrain"
        ]
        is False
    )


def test_hold_value_divergence_is_not_silently_resolved():
    result = RecalibrationExecutor().execute(
        _request(
            request_hold=True,
            payload_hold=False,
        )
    )

    assert (
        result.metadata[
            "request_hold_retrain"
        ]
        != result.metadata[
            "payload_hold_retrain"
        ]
    )


def test_traceability_is_preserved():
    result = RecalibrationExecutor().execute(
        _request()
    )

    assert result.run_id == RUN_ID

    assert (
        result.reference_window_id
        == REFERENCE
    )

    assert (
        result.current_window_id
        == CURRENT
    )


def test_executor_name_is_recorded():
    result = RecalibrationExecutor().execute(
        _request()
    )

    assert (
        result.metadata["executor"]
        == "RecalibrationExecutor"
    )


def test_execution_scope_is_request_acceptance_only():
    result = RecalibrationExecutor().execute(
        _request()
    )

    assert (
        result.metadata["execution_scope"]
        == "request_acceptance_only"
    )


def test_non_request_input_is_rejected():
    with pytest.raises(TypeError):
        RecalibrationExecutor().execute(
            object()
        )


def test_module_import_boundary_is_defense_only_plus_stdlib():
    # Anchored to this test file's own location rather than a bare
    # relative path — a bare relative path only resolves correctly
    # when pytest's cwd happens to equal CODE_ROOT. This project has
    # already hit that exact failure mode once (a 3-byte placeholder
    # file going undetected because a relative path silently
    # resolved), so this test is made cwd-independent on principle.
    #
    # tests/defense/test_recalibration_executor.py
    #   parents[0] = tests/defense
    #   parents[1] = tests
    #   parents[2] = <repo root>
    repo_root = Path(__file__).resolve().parents[2]

    source_path = (
        repo_root
        / "src"
        / "defense"
        / "recalibration_executor.py"
    )

    assert source_path.is_file(), (
        f"recalibration_executor.py not found at {source_path}"
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
