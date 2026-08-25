
import pytest

from src.triage.triage import (
    TriageAction,
)

from src.defense.actions import (
    DefenseActionRequest,
    MonitorPayload,   # add this import if not already present
)

from src.defense.retrain_gate import (
    HistoricalReplayResult,
    RetrainGateResult,
    ShadowValidationResult,
    ValidationStatus,
    evaluate_retrain_gate,
)


RUN_ID = "RUN_001"
REFERENCE = "WIN_000001"
CURRENT = "WIN_000002"
REQUEST_ID = (
    "7d8f5c4e-7f72-4a90-9d2f-8f44f2cbf9e1"
)


def _request():
    return DefenseActionRequest(
        request_id=REQUEST_ID,
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        action=TriageAction.RETRAIN,
        hold_retrain=False,
        reason="corroborated drift",
    )


def _shadow(
    status=ValidationStatus.PASS,
):
    return ShadowValidationResult(
        status=status,
        baseline_metric=0.90,
        candidate_metric=0.92,
        metric_delta=0.02,
    )


def _replay(
    status=ValidationStatus.PASS,
):
    return HistoricalReplayResult(
        status=status,
        baseline_metric=0.89,
        candidate_metric=0.91,
        metric_delta=0.02,
    )


# ============================================================
# Status contracts
# ============================================================

def test_shadow_result_accepts_pass():
    result = _shadow()

    assert result.status is ValidationStatus.PASS


def test_replay_result_accepts_pass():
    result = _replay()

    assert result.status is ValidationStatus.PASS


def test_shadow_result_accepts_fail():
    result = _shadow(
        ValidationStatus.FAIL
    )

    assert result.status is ValidationStatus.FAIL


def test_replay_result_accepts_not_run():
    result = _replay(
        ValidationStatus.NOT_RUN
    )

    assert result.status is ValidationStatus.NOT_RUN


# ============================================================
# Gate truth table
# ============================================================

@pytest.mark.parametrize(
    (
        "shadow_status",
        "replay_status",
        "expected_promote",
    ),
    [
        (
            ValidationStatus.PASS,
            ValidationStatus.PASS,
            True,
        ),
        (
            ValidationStatus.PASS,
            ValidationStatus.FAIL,
            False,
        ),
        (
            ValidationStatus.PASS,
            ValidationStatus.NOT_RUN,
            False,
        ),
        (
            ValidationStatus.FAIL,
            ValidationStatus.PASS,
            False,
        ),
        (
            ValidationStatus.FAIL,
            ValidationStatus.FAIL,
            False,
        ),
        (
            ValidationStatus.FAIL,
            ValidationStatus.NOT_RUN,
            False,
        ),
        (
            ValidationStatus.NOT_RUN,
            ValidationStatus.PASS,
            False,
        ),
        (
            ValidationStatus.NOT_RUN,
            ValidationStatus.FAIL,
            False,
        ),
        (
            ValidationStatus.NOT_RUN,
            ValidationStatus.NOT_RUN,
            False,
        ),
    ],
)
def test_retrain_gate_truth_table(
    shadow_status,
    replay_status,
    expected_promote,
):
    result = evaluate_retrain_gate(
        _request(),
        _shadow(shadow_status),
        _replay(replay_status),
    )

    assert result.promote is expected_promote


def test_both_pass_promotes():
    result = evaluate_retrain_gate(
        _request(),
        _shadow(ValidationStatus.PASS),
        _replay(ValidationStatus.PASS),
    )

    assert result.promote is True
    assert result.shadow_status is ValidationStatus.PASS
    assert result.replay_status is ValidationStatus.PASS


def test_shadow_failure_blocks_promotion():
    result = evaluate_retrain_gate(
        _request(),
        _shadow(ValidationStatus.FAIL),
        _replay(ValidationStatus.PASS),
    )

    assert result.promote is False


def test_replay_failure_blocks_promotion():
    result = evaluate_retrain_gate(
        _request(),
        _shadow(ValidationStatus.PASS),
        _replay(ValidationStatus.FAIL),
    )

    assert result.promote is False


def test_not_run_never_counts_as_pass():
    result = evaluate_retrain_gate(
        _request(),
        _shadow(ValidationStatus.NOT_RUN),
        _replay(ValidationStatus.PASS),
    )

    assert result.promote is False


# ============================================================
# Request validation
# ============================================================

def test_non_retrain_request_is_rejected():
    request = DefenseActionRequest(
        request_id=REQUEST_ID,
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        action=TriageAction.MONITOR,
        hold_retrain=False,
        reason="monitor",
        payload=MonitorPayload(
            observation_required=True
        ),
    )

    with pytest.raises(ValueError):
        evaluate_retrain_gate(
            request,
            _shadow(),
            _replay(),
        )


def test_retrain_request_with_hold_is_rejected():
    # DefenseActionRequest itself normally prevents this.
    # This test verifies the gate boundary as well.
    request = object.__new__(
        DefenseActionRequest
    )

    object.__setattr__(
        request,
        "request_id",
        REQUEST_ID,
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
        TriageAction.RETRAIN,
    )
    object.__setattr__(
        request,
        "hold_retrain",
        True,
    )
    object.__setattr__(
        request,
        "reason",
        "malformed",
    )
    object.__setattr__(
        request,
        "metadata",
        {},
    )

    with pytest.raises(ValueError):
        evaluate_retrain_gate(
            request,
            _shadow(),
            _replay(),
        )


# ============================================================
# Input types
# ============================================================

def test_request_type_is_required():
    with pytest.raises(TypeError):
        evaluate_retrain_gate(
            object(),
            _shadow(),
            _replay(),
        )


def test_shadow_type_is_required():
    with pytest.raises(TypeError):
        evaluate_retrain_gate(
            _request(),
            object(),
            _replay(),
        )


def test_replay_type_is_required():
    with pytest.raises(TypeError):
        evaluate_retrain_gate(
            _request(),
            _shadow(),
            object(),
        )


# ============================================================
# Result validation
# ============================================================

def test_gate_result_requires_boolean_promote():
    with pytest.raises(TypeError):
        RetrainGateResult(
            request_id=REQUEST_ID,
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            shadow_status=ValidationStatus.PASS,
            replay_status=ValidationStatus.PASS,
            promote=1,
            reason="invalid",
        )


def test_gate_result_rejects_inconsistent_promote():
    with pytest.raises(ValueError):
        RetrainGateResult(
            request_id=REQUEST_ID,
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            shadow_status=ValidationStatus.PASS,
            replay_status=ValidationStatus.PASS,
            promote=False,
            reason="inconsistent",
        )


def test_gate_result_rejects_same_windows():
    with pytest.raises(ValueError):
        RetrainGateResult(
            request_id=REQUEST_ID,
            run_id=RUN_ID,
            reference_window_id=CURRENT,
            current_window_id=CURRENT,
            shadow_status=ValidationStatus.FAIL,
            replay_status=ValidationStatus.FAIL,
            promote=False,
            reason="invalid",
        )


# ============================================================
# Traceability
# ============================================================

def test_gate_result_preserves_request_traceability():
    result = evaluate_retrain_gate(
        _request(),
        _shadow(),
        _replay(),
    )

    assert result.request_id == REQUEST_ID
    assert result.run_id == RUN_ID
    assert result.reference_window_id == REFERENCE
    assert result.current_window_id == CURRENT


# ============================================================
# Reason and metadata
# ============================================================

def test_promotion_reason_is_non_empty():
    result = evaluate_retrain_gate(
        _request(),
        _shadow(ValidationStatus.PASS),
        _replay(ValidationStatus.PASS),
    )

    assert result.reason.strip()


def test_rejection_reason_identifies_failed_stage():
    result = evaluate_retrain_gate(
        _request(),
        _shadow(ValidationStatus.FAIL),
        _replay(ValidationStatus.PASS),
    )

    assert "shadow validation" in result.reason


def test_gate_policy_version_is_recorded():
    result = evaluate_retrain_gate(
        _request(),
        _shadow(),
        _replay(),
    )

    assert (
        result.metadata["gate_policy_version"]
        == "retrain_gate_v1"
    )


# ============================================================
# Immutability
# ============================================================

def test_shadow_result_is_frozen():
    result = _shadow()

    with pytest.raises(Exception):
        result.status = ValidationStatus.FAIL


def test_replay_result_is_frozen():
    result = _replay()

    with pytest.raises(Exception):
        result.status = ValidationStatus.FAIL


def test_gate_result_is_frozen():
    result = evaluate_retrain_gate(
        _request(),
        _shadow(),
        _replay(),
    )

    with pytest.raises(Exception):
        result.promote = False


def test_metadata_uses_dict_default():
    result = _shadow()

    assert isinstance(
        result.metadata,
        dict,
    )
