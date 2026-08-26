
from __future__ import annotations

import ast
from pathlib import Path
import uuid

import pytest

from src.triage.triage import (
    TriageAction,
)

from src.defense.actions import (
    DefenseActionRequest,
    FreezePayload,
)

from src.defense.executor import (
    ExecutionStatus,
)

from src.defense.freeze_executor import (
    FreezeExecutor,
)


# ============================================================
# Repository root for THIS TEST FILE
#
# This is cwd-independent because __file__ refers to the actual
# test_freeze_executor.py file when pytest executes it.
# ============================================================

CODE_ROOT = (
    Path(__file__).resolve().parents[2]
)

RUN_ID = "RUN_001"
REFERENCE = "WIN_000001"
CURRENT = "WIN_000002"


def _uuid4() -> str:
    return str(uuid.uuid4())


def _request(
    *,
    request_id: str | None = None,
    reason: str = "freeze requested",
) -> DefenseActionRequest:
    if request_id is None:
        request_id = _uuid4()

    return DefenseActionRequest(
        request_id=request_id,
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        action=TriageAction.FREEZE,
        hold_retrain=False,
        reason=reason,
        payload=FreezePayload(
            reason=reason,
        ),
    )


# ============================================================
# Basic execution
# ============================================================

def test_valid_freeze_returns_executed():
    result = FreezeExecutor().execute(
        _request()
    )

    assert (
        result.status
        is ExecutionStatus.EXECUTED
    )

    assert result.action == "freeze"

    assert (
        result.message
        == "executed: freeze request accepted"
    )


# ============================================================
# Wrong action
# ============================================================

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

    result = FreezeExecutor().execute(
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


# ============================================================
# Wrong payload
# ============================================================

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

    result = FreezeExecutor().execute(
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


# ============================================================
# Rejected requests are NOT stored for idempotency
# ============================================================

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

    executor = FreezeExecutor()

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


# ============================================================
# Idempotency
# ============================================================

def test_duplicate_request_returns_already_executed():
    executor = FreezeExecutor()

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


def test_same_request_id_with_different_reason_is_rejected_loudly():
    executor = FreezeExecutor()

    request_id = _uuid4()

    first = _request(
        request_id=request_id,
        reason="original",
    )

    second = _request(
        request_id=request_id,
        reason="different",
    )

    executor.execute(first)

    with pytest.raises(ValueError):
        executor.execute(second)


def test_same_request_id_with_different_payload_is_rejected_loudly():
    executor = FreezeExecutor()

    request_id = _uuid4()

    first = _request(
        request_id=request_id,
        reason="original",
    )

    second = object.__new__(
        DefenseActionRequest
    )

    object.__setattr__(
        second,
        "request_id",
        request_id,
    )
    object.__setattr__(
        second,
        "run_id",
        RUN_ID,
    )
    object.__setattr__(
        second,
        "reference_window_id",
        REFERENCE,
    )
    object.__setattr__(
        second,
        "current_window_id",
        CURRENT,
    )
    object.__setattr__(
        second,
        "action",
        TriageAction.FREEZE,
    )
    object.__setattr__(
        second,
        "hold_retrain",
        False,
    )
    object.__setattr__(
        second,
        "reason",
        "original",
    )
    object.__setattr__(
        second,
        "payload",
        FreezePayload(
            reason="changed payload",
        ),
    )
    object.__setattr__(
        second,
        "metadata",
        {},
    )

    executor.execute(first)

    with pytest.raises(ValueError):
        executor.execute(second)


def test_different_request_ids_are_independent():
    executor = FreezeExecutor()

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

    executor_a = FreezeExecutor()
    executor_b = FreezeExecutor()

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


# ============================================================
# Traceability
# ============================================================

def test_traceability_is_preserved():
    result = FreezeExecutor().execute(
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


# ============================================================
# Metadata
# ============================================================

def test_executor_name_is_recorded():
    result = FreezeExecutor().execute(
        _request()
    )

    assert (
        result.metadata["executor"]
        == "FreezeExecutor"
    )


def test_execution_scope_is_request_acceptance_only():
    result = FreezeExecutor().execute(
        _request()
    )

    assert (
        result.metadata["execution_scope"]
        == "request_acceptance_only"
    )


# ============================================================
# Input validation
# ============================================================

def test_non_request_input_is_rejected():
    with pytest.raises(TypeError):
        FreezeExecutor().execute(
            object()
        )


# ============================================================
# Import boundary
#
# This test reads the SOURCE FILE using CODE_ROOT, never a
# cwd-dependent relative path.
# ============================================================

def test_module_import_boundary_is_defense_only_plus_stdlib():

    source_path = (
        CODE_ROOT
        / "src"
        / "defense"
        / "freeze_executor.py"
    )

    assert source_path.exists(), (
        f"FreezeExecutor source not found: "
        f"{source_path}"
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

                root = alias.name.split(
                    "."
                )[0]

                if root not in {
                    "__future__",
                    "typing",
                }:
                    raise AssertionError(
                        f"Disallowed import: "
                        f"{alias.name}"
                    )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):

            if node.module is None:
                continue

            root = node.module.split(
                "."
            )[0]

            if node.level == 0:

                if root not in {
                    "__future__",
                    "typing",
                    "src",
                }:
                    raise AssertionError(
                        f"Disallowed import: "
                        f"{node.module}"
                    )

                if (
                    root == "src"
                    and not node.module.startswith(
                        "src.defense."
                    )
                ):
                    raise AssertionError(
                        "Direct non-defense "
                        "src import: "
                        f"{node.module}"
                    )
