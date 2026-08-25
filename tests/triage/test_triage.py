
import pytest

from src.triage.signal_report import (
    SignalReport,
)

from src.triage.triage import (
    TriageAction,
    TriageResult,
    compute_triage_decision,
)


RUN_ID = "RUN_001"
REFERENCE = "WIN_000001"
CURRENT = "WIN_000002"


def _report(
    feature=False,
    confidence=False,
    error=False,
    vote_count=None,
):
    if vote_count is None:
        vote_count = int(error) * 2

    return SignalReport(
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        feature_drift=feature,
        confidence_drift=confidence,
        error_drift=error,
        error_vote_count=vote_count,
    )


# ============================================================
# Complete eight-case truth table
# ============================================================

@pytest.mark.parametrize(
    (
        "feature",
        "confidence",
        "error",
        "expected_action",
        "expected_hold",
    ),
    [
        (False, False, False, TriageAction.NO_ACTION, False),
        (True, False, False, TriageAction.MONITOR, False),
        (False, True, False, TriageAction.MONITOR, False),
        (False, False, True, TriageAction.RECALIBRATE, True),
        (True, False, True, TriageAction.RETRAIN, False),
        (False, True, True, TriageAction.RETRAIN, False),
        (True, True, False, TriageAction.RECALIBRATE, False),
        (True, True, True, TriageAction.FREEZE, False),
    ],
)
def test_complete_truth_table(
    feature,
    confidence,
    error,
    expected_action,
    expected_hold,
):
    result = compute_triage_decision(
        _report(
            feature=feature,
            confidence=confidence,
            error=error,
            vote_count=(int(error) * 2),
        )
    )

    assert result.action is expected_action
    assert result.hold_retrain is expected_hold


def test_error_alone_is_recalibrate_with_hold():
    result = compute_triage_decision(_report(error=True, vote_count=1))
    assert result.action is TriageAction.RECALIBRATE
    assert result.hold_retrain is True


def test_feature_plus_error_is_retrain():
    result = compute_triage_decision(_report(feature=True, error=True, vote_count=2))
    assert result.action is TriageAction.RETRAIN
    assert result.hold_retrain is False


def test_confidence_plus_error_is_retrain():
    result = compute_triage_decision(_report(confidence=True, error=True, vote_count=2))
    assert result.action is TriageAction.RETRAIN
    assert result.hold_retrain is False


def test_feature_plus_confidence_without_error_is_recalibrate():
    result = compute_triage_decision(
        _report(feature=True, confidence=True, error=False, vote_count=0)
    )
    assert result.action is TriageAction.RECALIBRATE
    assert result.hold_retrain is False


# ============================================================
# Evidence traceability
# ============================================================

def test_result_preserves_signal_report_traceability():
    result = compute_triage_decision(_report(feature=True, confidence=True))
    assert result.run_id == RUN_ID
    assert result.reference_window_id == REFERENCE
    assert result.current_window_id == CURRENT


# ============================================================
# Result validation
# ============================================================

def test_action_must_be_triage_action():
    with pytest.raises(TypeError):
        TriageResult(
            run_id=RUN_ID, reference_window_id=REFERENCE, current_window_id=CURRENT,
            action="retrain", hold_retrain=False, reason="invalid action type",
        )


def test_hold_retrain_must_be_bool():
    with pytest.raises(TypeError):
        TriageResult(
            run_id=RUN_ID, reference_window_id=REFERENCE, current_window_id=CURRENT,
            action=TriageAction.RECALIBRATE, hold_retrain=1, reason="invalid hold type",
        )


def test_hold_retrain_requires_recalibrate():
    with pytest.raises(ValueError):
        TriageResult(
            run_id=RUN_ID, reference_window_id=REFERENCE, current_window_id=CURRENT,
            action=TriageAction.RETRAIN, hold_retrain=True, reason="invalid state",
        )


def test_non_recalibrate_cannot_hold_retrain():
    with pytest.raises(ValueError):
        TriageResult(
            run_id=RUN_ID, reference_window_id=REFERENCE, current_window_id=CURRENT,
            action=TriageAction.FREEZE, hold_retrain=True, reason="invalid state",
        )


def test_same_reference_and_current_window_rejected():
    with pytest.raises(ValueError):
        TriageResult(
            run_id=RUN_ID, reference_window_id=CURRENT, current_window_id=CURRENT,
            action=TriageAction.NO_ACTION, hold_retrain=False, reason="invalid windows",
        )


def test_empty_reason_rejected():
    with pytest.raises(ValueError):
        TriageResult(
            run_id=RUN_ID, reference_window_id=REFERENCE, current_window_id=CURRENT,
            action=TriageAction.MONITOR, hold_retrain=False, reason="",
        )


def test_metadata_must_be_dict():
    with pytest.raises(TypeError):
        TriageResult(
            run_id=RUN_ID, reference_window_id=REFERENCE, current_window_id=CURRENT,
            action=TriageAction.MONITOR, hold_retrain=False, reason="valid reason",
            metadata="invalid",
        )


def test_metadata_keys_must_be_strings():
    with pytest.raises(TypeError):
        TriageResult(
            run_id=RUN_ID, reference_window_id=REFERENCE, current_window_id=CURRENT,
            action=TriageAction.MONITOR, hold_retrain=False, reason="valid reason",
            metadata={1: "invalid"},
        )


def test_triage_result_is_frozen():
    result = compute_triage_decision(_report())
    with pytest.raises(Exception):
        result.action = TriageAction.FREEZE


# ============================================================
# Input validation
# ============================================================

def test_signal_report_type_is_required():
    with pytest.raises(TypeError):
        compute_triage_decision(object())


# ============================================================
# Policy metadata
# ============================================================

def test_policy_version_is_recorded():
    result = compute_triage_decision(_report(feature=True, error=True))
    assert result.metadata["policy_version"] == "triage_v1"


def test_source_is_signal_report():
    result = compute_triage_decision(_report())
    assert result.metadata["source"] == "signal_report"


def test_signal_values_are_recorded_in_metadata():
    result = compute_triage_decision(
        _report(feature=True, confidence=False, error=True, vote_count=2)
    )
    assert result.metadata["feature_drift"] is True
    assert result.metadata["confidence_drift"] is False
    assert result.metadata["error_drift"] is True
    assert result.metadata["error_vote_count"] == 2


# ============================================================
# Reason is deterministic and non-empty
# ============================================================

def test_every_truth_table_result_has_reason():
    combinations = [
        (False, False, False, 0), (True, False, False, 0), (False, True, False, 0),
        (False, False, True, 1), (True, False, True, 2), (False, True, True, 2),
        (True, True, False, 0), (True, True, True, 3),
    ]
    for feature, confidence, error, vote_count in combinations:
        result = compute_triage_decision(
            _report(feature=feature, confidence=confidence, error=error, vote_count=vote_count)
        )
        assert isinstance(result.reason, str)
        assert result.reason.strip()


def test_no_action_has_no_hold():
    result = compute_triage_decision(_report())
    assert result.action is TriageAction.NO_ACTION
    assert result.hold_retrain is False


def test_freeze_has_no_hold():
    result = compute_triage_decision(
        _report(feature=True, confidence=True, error=True, vote_count=3)
    )
    assert result.action is TriageAction.FREEZE
    assert result.hold_retrain is False
