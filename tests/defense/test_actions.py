
import uuid

import pytest

from src.triage.triage import (
    TriageAction,
    TriageResult,
)

from src.defense.actions import (
    DefenseActionRequest,
    DefensePayload,
    FreezePayload,
    MonitorPayload,
    RecalibrationPayload,
    build_defense_action_request,
)


RUN_ID = "RUN_001"
REFERENCE = "WIN_000001"
CURRENT = "WIN_000002"


def _uuid4():
    return str(uuid.uuid4())


def _triage_result(
    action=TriageAction.MONITOR,
    hold_retrain=False,
):
    return TriageResult(
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        action=action,
        hold_retrain=hold_retrain,
        reason="test decision",
    )


# ============================================================
# Payload base / concrete contracts
# ============================================================

def test_monitor_payload_requires_explicit_boolean():
    payload = MonitorPayload(
        observation_required=True
    )

    assert payload.observation_required is True


def test_monitor_payload_rejects_integer():
    with pytest.raises(TypeError):
        MonitorPayload(
            observation_required=1
        )


def test_monitor_payload_requires_no_default():
    with pytest.raises(TypeError):
        MonitorPayload()


def test_recalibration_payload_constructs():
    payload = RecalibrationPayload(
        reason="confidence calibration requested",
        hold_retrain=True,
    )

    assert (
        payload.reason
        == "confidence calibration requested"
    )

    assert payload.hold_retrain is True


def test_recalibration_reason_must_be_non_empty():
    with pytest.raises(ValueError):
        RecalibrationPayload(
            reason="   ",
            hold_retrain=False,
        )


def test_recalibration_reason_must_be_string():
    with pytest.raises(ValueError):
        RecalibrationPayload(
            reason=123,
            hold_retrain=False,
        )


def test_recalibration_hold_must_be_boolean():
    with pytest.raises(TypeError):
        RecalibrationPayload(
            reason="recalibrate",
            hold_retrain=1,
        )


def test_freeze_payload_constructs():
    payload = FreezePayload(
        reason="all-signal burst"
    )

    assert payload.reason == "all-signal burst"


def test_freeze_reason_must_be_non_empty():
    with pytest.raises(ValueError):
        FreezePayload(
            reason=" "
        )


def test_freeze_reason_must_be_string():
    with pytest.raises(ValueError):
        FreezePayload(
            reason=123
        )


def test_payloads_are_defense_payloads():
    assert isinstance(
        MonitorPayload(
            observation_required=True
        ),
        DefensePayload,
    )

    assert isinstance(
        RecalibrationPayload(
            reason="recalibrate",
            hold_retrain=False,
        ),
        DefensePayload,
    )

    assert isinstance(
        FreezePayload(
            reason="freeze",
        ),
        DefensePayload,
    )


def test_payload_metadata_defaults_to_dict():
    payload = MonitorPayload(
        observation_required=True
    )

    assert isinstance(
        payload.metadata,
        dict,
    )


# ============================================================
# Valid action ↔ payload mappings
# ============================================================

def test_monitor_accepts_monitor_payload():
    request = DefenseActionRequest(
        request_id=_uuid4(),
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

    assert isinstance(
        request.payload,
        MonitorPayload,
    )


def test_recalibrate_accepts_recalibration_payload():
    request = DefenseActionRequest(
        request_id=_uuid4(),
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        action=TriageAction.RECALIBRATE,
        hold_retrain=False,
        reason="recalibrate",
        payload=RecalibrationPayload(
            reason="calibration update",
            hold_retrain=False,
        ),
    )

    assert isinstance(
        request.payload,
        RecalibrationPayload,
    )


def test_freeze_accepts_freeze_payload():
    request = DefenseActionRequest(
        request_id=_uuid4(),
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        action=TriageAction.FREEZE,
        hold_retrain=False,
        reason="freeze",
        payload=FreezePayload(
            reason="all-signal burst",
        ),
    )

    assert isinstance(
        request.payload,
        FreezePayload,
    )


def test_no_action_requires_no_payload():
    request = DefenseActionRequest(
        request_id=_uuid4(),
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        action=TriageAction.NO_ACTION,
        hold_retrain=False,
        reason="none",
        payload=None,
    )

    assert request.payload is None


def test_retrain_requires_no_action_specific_payload():
    request = DefenseActionRequest(
        request_id=_uuid4(),
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        action=TriageAction.RETRAIN,
        hold_retrain=False,
        reason="retrain",
        payload=None,
    )

    assert request.payload is None


# ============================================================
# Invalid action ↔ payload combinations
# ============================================================

def test_monitor_rejects_freeze_payload():
    with pytest.raises((TypeError, ValueError)):
        DefenseActionRequest(
            request_id=_uuid4(),
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            action=TriageAction.MONITOR,
            hold_retrain=False,
            reason="monitor",
            payload=FreezePayload(
                reason="wrong payload",
            ),
        )


def test_monitor_rejects_recalibration_payload():
    with pytest.raises((TypeError, ValueError)):
        DefenseActionRequest(
            request_id=_uuid4(),
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            action=TriageAction.MONITOR,
            hold_retrain=False,
            reason="monitor",
            payload=RecalibrationPayload(
                reason="wrong payload",
                hold_retrain=False,
            ),
        )


def test_recalibrate_rejects_monitor_payload():
    with pytest.raises((TypeError, ValueError)):
        DefenseActionRequest(
            request_id=_uuid4(),
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            action=TriageAction.RECALIBRATE,
            hold_retrain=False,
            reason="recalibrate",
            payload=MonitorPayload(
                observation_required=True
            ),
        )


def test_recalibrate_rejects_freeze_payload():
    with pytest.raises((TypeError, ValueError)):
        DefenseActionRequest(
            request_id=_uuid4(),
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            action=TriageAction.RECALIBRATE,
            hold_retrain=False,
            reason="recalibrate",
            payload=FreezePayload(
                reason="wrong payload",
            ),
        )


def test_freeze_rejects_monitor_payload():
    with pytest.raises((TypeError, ValueError)):
        DefenseActionRequest(
            request_id=_uuid4(),
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            action=TriageAction.FREEZE,
            hold_retrain=False,
            reason="freeze",
            payload=MonitorPayload(
                observation_required=True
            ),
        )


def test_retrain_rejects_action_specific_payload():
    with pytest.raises(ValueError):
        DefenseActionRequest(
            request_id=_uuid4(),
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            action=TriageAction.RETRAIN,
            hold_retrain=False,
            reason="retrain",
            payload=MonitorPayload(
                observation_required=True
            ),
        )


def test_no_action_rejects_action_specific_payload():
    with pytest.raises(ValueError):
        DefenseActionRequest(
            request_id=_uuid4(),
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            action=TriageAction.NO_ACTION,
            hold_retrain=False,
            reason="none",
            payload=FreezePayload(
                reason="wrong payload",
            ),
        )


# ============================================================
# Request-level validation
# ============================================================

def test_request_id_must_be_uuid4():
    with pytest.raises(ValueError):
        DefenseActionRequest(
            request_id="not-a-uuid",
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


def test_action_must_be_triage_action():
    with pytest.raises(TypeError):
        DefenseActionRequest(
            request_id=_uuid4(),
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            action="monitor",
            hold_retrain=False,
            reason="monitor",
            payload=MonitorPayload(
                observation_required=True
            ),
        )


def test_hold_retrain_must_be_boolean():
    with pytest.raises(TypeError):
        DefenseActionRequest(
            request_id=_uuid4(),
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            action=TriageAction.RECALIBRATE,
            hold_retrain=1,
            reason="recalibrate",
            payload=RecalibrationPayload(
                reason="calibration",
                hold_retrain=True,
            ),
        )


def test_request_is_frozen():
    request = DefenseActionRequest(
        request_id=_uuid4(),
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

    with pytest.raises(Exception):
        request.action = TriageAction.FREEZE


# ============================================================
# Builder isolation
# ============================================================

def test_builder_can_create_monitor_request():
    result = _triage_result(
        action=TriageAction.MONITOR
    )

    request = build_defense_action_request(
        result,
        payload=MonitorPayload(
            observation_required=True
        ),
    )

    assert request.action is TriageAction.MONITOR
    assert isinstance(
        request.payload,
        MonitorPayload,
    )


def test_builder_can_create_recalibration_request():
    result = _triage_result(
        action=TriageAction.RECALIBRATE,
        hold_retrain=True,
    )

    request = build_defense_action_request(
        result,
        payload=RecalibrationPayload(
            reason="confidence shift",
            hold_retrain=True,
        ),
    )

    assert (
        request.action
        is TriageAction.RECALIBRATE
    )


def test_builder_can_create_freeze_request():
    result = _triage_result(
        action=TriageAction.FREEZE
    )

    request = build_defense_action_request(
        result,
        payload=FreezePayload(
            reason="all signals",
        ),
    )

    assert (
        request.action
        is TriageAction.FREEZE
    )


def test_builder_rejects_payload_mismatch():
    result = _triage_result(
        action=TriageAction.MONITOR
    )

    with pytest.raises((TypeError, ValueError)):
        build_defense_action_request(
            result,
            payload=FreezePayload(
                reason="wrong",
            ),
        )


def test_builder_preserves_request_identity():
    request_id = _uuid4()

    request = build_defense_action_request(
        _triage_result(),
        request_id=request_id,
        payload=MonitorPayload(
            observation_required=True
        ),
    )

    assert request.request_id == request_id


def test_builder_does_not_execute_any_action():
    request = build_defense_action_request(
        _triage_result(
            action=TriageAction.FREEZE
        ),
        payload=FreezePayload(
            reason="all signals",
        ),
    )

    # Pure command description:
    assert request.action is TriageAction.FREEZE
    assert isinstance(
        request.payload,
        FreezePayload,
    )
