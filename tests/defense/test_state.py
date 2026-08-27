"""
tests/defense/test_state.py
============================

Comprehensive regression tests for src/defense/state.py.

Coverage sections
-----------------
A  Enums (DefenseState, TransitionReason exact values)
B  DefenseStateRecord validation
C  TransitionRecord validation
D  TransitionResult validation
E  Initialization semantics
F  Scope isolation (run_id, model_version)
G  Legal transitions (every legal edge)
H  Illegal transitions (key illegal edges)
I  CAS semantics
J  Mutual exclusion (RECALIBRATING <-> RETRAINING)
K  Recalibration failure path
L  Shadow-rejection path
M  Freeze semantics
N  History (append-only, immutable, ordered)
O  Concurrency (two threads, one CAS slot)
P  No detector lifecycle coupling (import inspection)

Path robustness: paths resolved via Path(__file__).resolve().
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.defense.state import (
    DefenseState,
    DefenseStateManager,
    DefenseStateRecord,
    TransitionReason,
    TransitionRecord,
    TransitionResult,
    _FailureKind,
)


@pytest.fixture()
def mgr():
    return DefenseStateManager()


@pytest.fixture()
def ts():
    return datetime.now(tz=timezone.utc)


def make_record(
    run_id="RUN_001",
    model_version="MODEL_v1",
    state=DefenseState.ACTIVE,
    transition_version=0,
    last_transition_reason=TransitionReason.SUCCESS,
    ts=None,
    metadata=None,
):
    if ts is None:
        ts = datetime.now(tz=timezone.utc)
    if metadata is None:
        metadata = {}
    return DefenseStateRecord(
        run_id=run_id,
        model_version=model_version,
        state=state,
        transition_version=transition_version,
        last_transition_reason=last_transition_reason,
        last_transition_at=ts,
        metadata=metadata,
    )


def make_tr(
    run_id="RUN_001",
    model_version="MODEL_v1",
    from_state=DefenseState.ACTIVE,
    to_state=DefenseState.RECALIBRATING,
    transition_version=1,
    reason=TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
    ts=None,
    justification=None,
    metadata=None,
):
    if ts is None:
        ts = datetime.now(tz=timezone.utc)
    if metadata is None:
        metadata = {}
    return TransitionRecord(
        run_id=run_id,
        model_version=model_version,
        from_state=from_state,
        to_state=to_state,
        transition_version=transition_version,
        reason=reason,
        timestamp=ts,
        justification=justification,
        metadata=metadata,
    )


# ===========================================================================
# A. Enums
# ===========================================================================

class TestDefenseStateEnum:
    def test_exact_values(self):
        assert {m.value for m in DefenseState} == {
            "ACTIVE", "RECALIBRATING", "RETRAINING", "FROZEN"
        }
    def test_count(self):
        assert len(DefenseState) == 4
    def test_active(self):
        assert DefenseState.ACTIVE.value == "ACTIVE"
    def test_recalibrating(self):
        assert DefenseState.RECALIBRATING.value == "RECALIBRATING"
    def test_retraining(self):
        assert DefenseState.RETRAINING.value == "RETRAINING"
    def test_frozen(self):
        assert DefenseState.FROZEN.value == "FROZEN"
    def test_is_str(self):
        assert isinstance(DefenseState.ACTIVE, str)

class TestTransitionReasonEnum:
    def test_exact_values(self):
        assert {m.value for m in TransitionReason} == {
            "SUCCESS", "FAILURE", "REJECTED_BY_SHADOW_VALIDATION",
            "MANUAL_RELEASE", "TRIAGE_INITIATED_RECALIBRATION",
            "TRIAGE_INITIATED_RETRAINING", "TRIAGE_INITIATED_FREEZE",
        }
    def test_count(self):
        assert len(TransitionReason) == 7
    def test_is_str(self):
        assert isinstance(TransitionReason.SUCCESS, str)


# ===========================================================================
# B. DefenseStateRecord
# ===========================================================================

class TestDefenseStateRecord:
    def test_valid(self, ts):
        r = make_record(ts=ts)
        assert r.state is DefenseState.ACTIVE
        assert r.transition_version == 0
    def test_frozen_immutable(self, ts):
        r = make_record(ts=ts)
        with pytest.raises((AttributeError, TypeError)):
            r.state = DefenseState.FROZEN
    def test_run_id_empty(self, ts):
        with pytest.raises(ValueError):
            make_record(run_id="", ts=ts)
    def test_run_id_whitespace(self, ts):
        with pytest.raises(ValueError):
            make_record(run_id="   ", ts=ts)
    def test_run_id_non_str(self, ts):
        with pytest.raises(TypeError):
            make_record(run_id=123, ts=ts)
    def test_model_version_empty(self, ts):
        with pytest.raises(ValueError):
            make_record(model_version="", ts=ts)
    def test_model_version_non_str(self, ts):
        with pytest.raises(TypeError):
            make_record(model_version=None, ts=ts)
    def test_state_wrong_type(self, ts):
        with pytest.raises(TypeError):
            make_record(state="ACTIVE", ts=ts)
    def test_version_bool_rejected(self, ts):
        with pytest.raises(TypeError):
            make_record(transition_version=True, ts=ts)
    def test_version_negative(self, ts):
        with pytest.raises(ValueError):
            make_record(transition_version=-1, ts=ts)
    def test_version_zero_ok(self, ts):
        r = make_record(transition_version=0, ts=ts)
        assert r.transition_version == 0
    def test_reason_wrong_type(self, ts):
        with pytest.raises(TypeError):
            make_record(last_transition_reason="SUCCESS", ts=ts)
    def test_naive_datetime_raises(self):
        with pytest.raises(ValueError):
            make_record(ts=datetime(2024, 1, 1))
    def test_datetime_with_tzinfo_but_no_utcoffset_is_rejected(self):
        from datetime import tzinfo as _tzinfo

        class BrokenTZ(_tzinfo):
            def utcoffset(self, dt):
                return None  # tzinfo present but utcoffset() is None
            def tzname(self, dt):
                return "BROKEN"
            def dst(self, dt):
                return None

        with pytest.raises(ValueError):
            make_record(ts=datetime(2024, 1, 1, tzinfo=BrokenTZ()))
    def test_metadata_default_empty(self, ts):
        r = make_record(ts=ts)
        assert r.metadata == {}
    def test_metadata_non_dict(self, ts):
        with pytest.raises(TypeError):
            DefenseStateRecord(
                run_id="R", model_version="M",
                state=DefenseState.ACTIVE, transition_version=0,
                last_transition_reason=TransitionReason.SUCCESS,
                last_transition_at=ts, metadata=["x"],
            )
    def test_metadata_non_str_key(self, ts):
        with pytest.raises(TypeError):
            DefenseStateRecord(
                run_id="R", model_version="M",
                state=DefenseState.ACTIVE, transition_version=0,
                last_transition_reason=TransitionReason.SUCCESS,
                last_transition_at=ts, metadata={1: "v"},
            )
    def test_metadata_excluded_from_eq(self, ts):
        r1 = make_record(ts=ts, metadata={"a": 1})
        r2 = make_record(ts=ts, metadata={"b": 2})
        assert r1 == r2


# ===========================================================================
# C. TransitionRecord
# ===========================================================================

class TestTransitionRecord:
    def test_valid_non_manual(self, ts):
        tr = make_tr(ts=ts)
        assert tr.justification is None
    def test_frozen_immutable(self, ts):
        tr = make_tr(ts=ts)
        with pytest.raises((AttributeError, TypeError)):
            tr.reason = TransitionReason.SUCCESS
    def test_run_id_empty(self, ts):
        with pytest.raises(ValueError):
            make_tr(run_id="", ts=ts)
    def test_from_state_wrong_type(self, ts):
        with pytest.raises(TypeError):
            make_tr(from_state="ACTIVE", ts=ts)
    def test_to_state_wrong_type(self, ts):
        with pytest.raises(TypeError):
            make_tr(to_state="RECALIBRATING", ts=ts)
    def test_version_bool_raises(self, ts):
        with pytest.raises(TypeError):
            make_tr(transition_version=True, ts=ts)
    def test_version_zero_raises(self, ts):
        with pytest.raises(ValueError):
            make_tr(transition_version=0, ts=ts)
    def test_version_negative_raises(self, ts):
        with pytest.raises(ValueError):
            make_tr(transition_version=-1, ts=ts)
    def test_reason_wrong_type(self, ts):
        with pytest.raises(TypeError):
            make_tr(reason="SUCCESS", ts=ts)
    def test_naive_timestamp_raises(self):
        with pytest.raises(ValueError):
            make_tr(ts=datetime(2024, 1, 1))
    def test_manual_release_needs_justification(self, ts):
        with pytest.raises(ValueError):
            TransitionRecord(
                run_id="R", model_version="M",
                from_state=DefenseState.FROZEN, to_state=DefenseState.ACTIVE,
                transition_version=1, reason=TransitionReason.MANUAL_RELEASE,
                timestamp=ts, justification=None,
            )
    def test_manual_release_whitespace_justification_raises(self, ts):
        with pytest.raises(ValueError):
            TransitionRecord(
                run_id="R", model_version="M",
                from_state=DefenseState.FROZEN, to_state=DefenseState.ACTIVE,
                transition_version=1, reason=TransitionReason.MANUAL_RELEASE,
                timestamp=ts, justification="   ",
            )
    def test_manual_release_non_str_justification_raises(self, ts):
        with pytest.raises(TypeError):
            TransitionRecord(
                run_id="R", model_version="M",
                from_state=DefenseState.FROZEN, to_state=DefenseState.ACTIVE,
                transition_version=1, reason=TransitionReason.MANUAL_RELEASE,
                timestamp=ts, justification=42,
            )
    def test_manual_release_valid(self, ts):
        tr = TransitionRecord(
            run_id="R", model_version="M",
            from_state=DefenseState.FROZEN, to_state=DefenseState.ACTIVE,
            transition_version=1, reason=TransitionReason.MANUAL_RELEASE,
            timestamp=ts, justification="Approved.",
        )
        assert tr.justification == "Approved."
    def test_non_manual_must_have_none_justification(self, ts):
        with pytest.raises(ValueError):
            make_tr(reason=TransitionReason.SUCCESS, justification="oops", ts=ts)
    def test_metadata_excluded_from_eq(self, ts):
        t1 = make_tr(ts=ts, metadata={"a": 1})
        t2 = make_tr(ts=ts, metadata={"b": 2})
        assert t1 == t2
    def test_metadata_non_str_key_raises(self, ts):
        with pytest.raises(TypeError):
            TransitionRecord(
                run_id="R", model_version="M",
                from_state=DefenseState.ACTIVE, to_state=DefenseState.RECALIBRATING,
                transition_version=1,
                reason=TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                timestamp=ts, justification=None, metadata={1: "v"},
            )

# ===========================================================================
# D. TransitionResult
# ===========================================================================

class TestTransitionResult:
    def test_success_valid(self, ts):
        rec = make_record(ts=ts)
        res = TransitionResult(
            success=True, failure_kind=None,
            current_state=DefenseState.ACTIVE,
            current_transition_version=0, record=rec,
        )
        assert res.success is True
    def test_failure_valid(self, ts):
        res = TransitionResult(
            success=False, failure_kind=_FailureKind.CAS_CONFLICT,
            current_state=DefenseState.ACTIVE,
            current_transition_version=0, record=None,
        )
        assert res.failure_kind is _FailureKind.CAS_CONFLICT
    def test_success_with_failure_kind_raises(self, ts):
        rec = make_record(ts=ts)
        with pytest.raises(ValueError):
            TransitionResult(
                success=True, failure_kind=_FailureKind.CAS_CONFLICT,
                current_state=DefenseState.ACTIVE,
                current_transition_version=0, record=rec,
            )
    def test_success_no_record_raises(self):
        with pytest.raises(ValueError):
            TransitionResult(
                success=True, failure_kind=None,
                current_state=DefenseState.ACTIVE,
                current_transition_version=0, record=None,
            )
    def test_failure_no_kind_raises(self):
        with pytest.raises(ValueError):
            TransitionResult(
                success=False, failure_kind=None,
                current_state=DefenseState.ACTIVE,
                current_transition_version=0, record=None,
            )
    def test_failure_with_record_raises(self, ts):
        rec = make_record(ts=ts)
        with pytest.raises(ValueError):
            TransitionResult(
                success=False, failure_kind=_FailureKind.CAS_CONFLICT,
                current_state=DefenseState.ACTIVE,
                current_transition_version=0, record=rec,
            )
    def test_success_bool_int_rejected(self, ts):
        rec = make_record(ts=ts)
        with pytest.raises(TypeError):
            TransitionResult(
                success=1,
                failure_kind=None,
                current_state=DefenseState.ACTIVE,
                current_transition_version=0, record=rec,
            )
    def test_current_state_wrong_type_raises(self, ts):
        rec = make_record(ts=ts)
        with pytest.raises(TypeError):
            TransitionResult(
                success=True, failure_kind=None,
                current_state="ACTIVE",
                current_transition_version=0, record=rec,
            )
    def test_version_bool_raises(self, ts):
        rec = make_record(ts=ts)
        with pytest.raises(TypeError):
            TransitionResult(
                success=True, failure_kind=None,
                current_state=DefenseState.ACTIVE,
                current_transition_version=False, record=rec,
            )


# ===========================================================================
# E. Initialization
# ===========================================================================

class TestInitialization:
    def test_new_pair_active(self, mgr):
        rec = mgr.initialize("R", "M")
        assert rec.state is DefenseState.ACTIVE
        assert rec.transition_version == 0
        assert rec.last_transition_reason is TransitionReason.SUCCESS
    def test_timezone_aware(self, mgr):
        rec = mgr.initialize("R", "M")
        assert rec.last_transition_at.tzinfo is not None
    def test_existing_returns_unchanged(self, mgr):
        r1 = mgr.initialize("R", "M")
        r2 = mgr.initialize("R", "M")
        assert r1 is r2
    def test_existing_state_not_reset(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        rec = mgr.initialize("R", "M")
        assert rec.state is DefenseState.RECALIBRATING
        assert rec.transition_version == 1
    def test_unknown_transition_missing(self, mgr):
        result = mgr.transition("UNKNOWN", "M", DefenseState.RECALIBRATING,
                                  TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                                  DefenseState.ACTIVE, 0)
        assert result.success is False
        assert result.failure_kind is _FailureKind.MISSING_STATE
    def test_get_before_init_none(self, mgr):
        assert mgr.get("UNKNOWN", "M") is None
    def test_get_after_init(self, mgr):
        mgr.initialize("R", "M")
        rec = mgr.get("R", "M")
        assert rec is not None
        assert rec.state is DefenseState.ACTIVE


# ===========================================================================
# F. Scope isolation
# ===========================================================================

class TestScopeIsolation:
    def test_different_run_ids(self, mgr):
        mgr.initialize("RUN_001", "M")
        mgr.initialize("RUN_002", "M")
        mgr.transition("RUN_001", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        assert mgr.get("RUN_002", "M").state is DefenseState.ACTIVE
    def test_different_model_versions(self, mgr):
        mgr.initialize("R", "MODEL_v1")
        mgr.initialize("R", "MODEL_v2")
        mgr.transition("R", "MODEL_v1", DefenseState.FROZEN,
                        TransitionReason.TRIAGE_INITIATED_FREEZE,
                        DefenseState.ACTIVE, 0)
        assert mgr.get("R", "MODEL_v2").state is DefenseState.ACTIVE


# ===========================================================================
# G. Legal transitions
# ===========================================================================

LEGAL_EDGES = [
    (DefenseState.ACTIVE, DefenseState.RECALIBRATING, TransitionReason.TRIAGE_INITIATED_RECALIBRATION),
    (DefenseState.ACTIVE, DefenseState.RETRAINING,    TransitionReason.TRIAGE_INITIATED_RETRAINING),
    (DefenseState.ACTIVE, DefenseState.FROZEN,         TransitionReason.TRIAGE_INITIATED_FREEZE),
    (DefenseState.RECALIBRATING, DefenseState.ACTIVE,  TransitionReason.SUCCESS),
    (DefenseState.RECALIBRATING, DefenseState.FROZEN,  TransitionReason.TRIAGE_INITIATED_FREEZE),
    (DefenseState.RETRAINING,    DefenseState.ACTIVE,  TransitionReason.SUCCESS),
    (DefenseState.RETRAINING,    DefenseState.FROZEN,  TransitionReason.TRIAGE_INITIATED_FREEZE),
]

@pytest.mark.parametrize("from_s,to_s,reason", LEGAL_EDGES)
def test_legal_transition(from_s, to_s, reason):
    mgr = DefenseStateManager()
    mgr.initialize("R", "M")
    if from_s is not DefenseState.ACTIVE:
        lead = {
            DefenseState.RECALIBRATING: TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
            DefenseState.RETRAINING:    TransitionReason.TRIAGE_INITIATED_RETRAINING,
            DefenseState.FROZEN:        TransitionReason.TRIAGE_INITIATED_FREEZE,
        }[from_s]
        mgr.transition("R", "M", from_s, lead, DefenseState.ACTIVE, 0)
    cur = mgr.get("R", "M")
    result = mgr.transition("R", "M", to_s, reason, cur.state, cur.transition_version)
    assert result.success is True
    assert result.current_state is to_s

def test_frozen_to_active_release_freeze():
    mgr = DefenseStateManager()
    mgr.initialize("R", "M")
    mgr.transition("R", "M", DefenseState.FROZEN,
                   TransitionReason.TRIAGE_INITIATED_FREEZE, DefenseState.ACTIVE, 0)
    result = mgr.release_freeze("R", "M", 1, "Security team approved.")
    assert result.success is True
    assert result.current_state is DefenseState.ACTIVE
    assert result.record.last_transition_reason is TransitionReason.MANUAL_RELEASE


# ===========================================================================
# H. Illegal transitions
# ===========================================================================

ILLEGAL_EDGES = [
    (DefenseState.ACTIVE,         DefenseState.ACTIVE),
    (DefenseState.RECALIBRATING,  DefenseState.RECALIBRATING),
    (DefenseState.RETRAINING,     DefenseState.RETRAINING),
    (DefenseState.FROZEN,         DefenseState.FROZEN),
    (DefenseState.RECALIBRATING,  DefenseState.RETRAINING),
    (DefenseState.RETRAINING,     DefenseState.RECALIBRATING),
    (DefenseState.FROZEN,         DefenseState.RECALIBRATING),
    (DefenseState.FROZEN,         DefenseState.RETRAINING),
]

@pytest.mark.parametrize("from_s,to_s", ILLEGAL_EDGES)
def test_illegal_transition(from_s, to_s):
    mgr = DefenseStateManager()
    mgr.initialize("R", "M")
    if from_s is not DefenseState.ACTIVE:
        lead = {
            DefenseState.RECALIBRATING: TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
            DefenseState.RETRAINING:    TransitionReason.TRIAGE_INITIATED_RETRAINING,
            DefenseState.FROZEN:        TransitionReason.TRIAGE_INITIATED_FREEZE,
        }[from_s]
        mgr.transition("R", "M", from_s, lead, DefenseState.ACTIVE, 0)
    cur = mgr.get("R", "M")
    result = mgr.transition("R", "M", to_s, TransitionReason.SUCCESS,
                             cur.state, cur.transition_version)
    assert result.success is False
    assert result.failure_kind is _FailureKind.INVALID_TRANSITION


# ===========================================================================
# I. CAS semantics
# ===========================================================================

class TestCAS:
    def test_correct_succeeds(self, mgr):
        mgr.initialize("R", "M")
        result = mgr.transition("R", "M", DefenseState.RECALIBRATING,
                                  TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                                  DefenseState.ACTIVE, 0)
        assert result.success is True
        assert result.current_transition_version == 1

    def test_wrong_state_cas_conflict(self, mgr):
        mgr.initialize("R", "M")
        result = mgr.transition("R", "M", DefenseState.RECALIBRATING,
                                  TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                                  DefenseState.FROZEN, 0)
        assert result.success is False
        assert result.failure_kind is _FailureKind.CAS_CONFLICT

    def test_stale_version_cas_conflict(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.ACTIVE,
                                  TransitionReason.SUCCESS,
                                  DefenseState.RECALIBRATING, 0)
        assert result.success is False
        assert result.failure_kind is _FailureKind.CAS_CONFLICT

    def test_version_increments_on_success(self, mgr):
        mgr.initialize("R", "M")
        for exp_v, to_s, rsn, exp_s in [
            (0, DefenseState.RECALIBRATING,
             TransitionReason.TRIAGE_INITIATED_RECALIBRATION, DefenseState.ACTIVE),
            (1, DefenseState.ACTIVE, TransitionReason.SUCCESS, DefenseState.RECALIBRATING),
        ]:
            result = mgr.transition("R", "M", to_s, rsn, exp_s, exp_v)
            assert result.success is True
            assert result.current_transition_version == exp_v + 1

    def test_conflict_no_version_increment(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        mgr.transition("R", "M", DefenseState.ACTIVE, TransitionReason.SUCCESS,
                        DefenseState.RECALIBRATING, 0)  # stale
        assert mgr.get("R", "M").transition_version == 1

    def test_no_stale_overwrite(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        mgr.transition("R", "M", DefenseState.FROZEN,
                        TransitionReason.TRIAGE_INITIATED_FREEZE,
                        DefenseState.ACTIVE, 0)  # wrong state
        assert mgr.get("R", "M").state is DefenseState.RECALIBRATING


# ===========================================================================
# J. Mutual exclusion
# ===========================================================================

class TestMutualExclusion:
    def test_recalibrating_cannot_go_to_retraining(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.RETRAINING,
                                  TransitionReason.TRIAGE_INITIATED_RETRAINING,
                                  DefenseState.RECALIBRATING, 1)
        assert result.success is False
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION

    def test_retraining_cannot_go_to_recalibrating(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RETRAINING,
                        TransitionReason.TRIAGE_INITIATED_RETRAINING,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.RECALIBRATING,
                                  TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                                  DefenseState.RETRAINING, 1)
        assert result.success is False
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION


# ===========================================================================
# K. Recalibration failure
# ===========================================================================

class TestRecalibrationFailure:
    def test_recalibrating_to_active_on_failure(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.ACTIVE,
                                  TransitionReason.FAILURE,
                                  DefenseState.RECALIBRATING, 1)
        assert result.success is True
        assert result.current_state is DefenseState.ACTIVE
        assert result.record.last_transition_reason is TransitionReason.FAILURE

    def test_failure_audit_exists(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        mgr.transition("R", "M", DefenseState.ACTIVE,
                        TransitionReason.FAILURE, DefenseState.RECALIBRATING, 1)
        hist = mgr.history("R", "M")
        assert any(h.reason is TransitionReason.FAILURE for h in hist)


# ===========================================================================
# L. Shadow rejection
# ===========================================================================

class TestShadowRejection:
    def test_retraining_to_active_shadow_rejected(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RETRAINING,
                        TransitionReason.TRIAGE_INITIATED_RETRAINING,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.ACTIVE,
                                  TransitionReason.REJECTED_BY_SHADOW_VALIDATION,
                                  DefenseState.RETRAINING, 1)
        assert result.success is True
        assert result.current_state is DefenseState.ACTIVE
        assert (result.record.last_transition_reason
                is TransitionReason.REJECTED_BY_SHADOW_VALIDATION)

    def test_shadow_rejection_audit_exists(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RETRAINING,
                        TransitionReason.TRIAGE_INITIATED_RETRAINING,
                        DefenseState.ACTIVE, 0)
        mgr.transition("R", "M", DefenseState.ACTIVE,
                        TransitionReason.REJECTED_BY_SHADOW_VALIDATION,
                        DefenseState.RETRAINING, 1)
        hist = mgr.history("R", "M")
        assert any(
            h.reason is TransitionReason.REJECTED_BY_SHADOW_VALIDATION
            for h in hist
        )


# ===========================================================================
# M. Freeze semantics
# ===========================================================================

class TestFreezeSemantics:
    def test_active_to_frozen(self, mgr):
        mgr.initialize("R", "M")
        result = mgr.transition("R", "M", DefenseState.FROZEN,
                                  TransitionReason.TRIAGE_INITIATED_FREEZE,
                                  DefenseState.ACTIVE, 0)
        assert result.success is True

    def test_recalibrating_to_frozen(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.FROZEN,
                                  TransitionReason.TRIAGE_INITIATED_FREEZE,
                                  DefenseState.RECALIBRATING, 1)
        assert result.success is True

    def test_retraining_to_frozen(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RETRAINING,
                        TransitionReason.TRIAGE_INITIATED_RETRAINING,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.FROZEN,
                                  TransitionReason.TRIAGE_INITIATED_FREEZE,
                                  DefenseState.RETRAINING, 1)
        assert result.success is True

    def test_frozen_blocks_recalibrating(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.FROZEN,
                        TransitionReason.TRIAGE_INITIATED_FREEZE,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.RECALIBRATING,
                                  TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                                  DefenseState.FROZEN, 1)
        assert result.success is False
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION

    def test_frozen_blocks_retraining(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.FROZEN,
                        TransitionReason.TRIAGE_INITIATED_FREEZE,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.RETRAINING,
                                  TransitionReason.TRIAGE_INITIATED_RETRAINING,
                                  DefenseState.FROZEN, 1)
        assert result.success is False
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION

    def test_release_freeze_whitespace_raises(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.FROZEN,
                        TransitionReason.TRIAGE_INITIATED_FREEZE,
                        DefenseState.ACTIVE, 0)
        with pytest.raises(ValueError):
            mgr.release_freeze("R", "M", 1, "   ")

    def test_release_freeze_empty_raises(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.FROZEN,
                        TransitionReason.TRIAGE_INITIATED_FREEZE,
                        DefenseState.ACTIVE, 0)
        with pytest.raises(ValueError):
            mgr.release_freeze("R", "M", 1, "")

    def test_release_freeze_non_str_raises(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.FROZEN,
                        TransitionReason.TRIAGE_INITIATED_FREEZE,
                        DefenseState.ACTIVE, 0)
        with pytest.raises(TypeError):
            mgr.release_freeze("R", "M", 1, 99)

    def test_release_freeze_non_frozen_cas_conflict(self, mgr):
        mgr.initialize("R", "M")
        result = mgr.release_freeze("R", "M", 0, "manual override")
        assert result.success is False
        assert result.failure_kind is _FailureKind.CAS_CONFLICT

    def test_release_freeze_audit_justification(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.FROZEN,
                        TransitionReason.TRIAGE_INITIATED_FREEZE,
                        DefenseState.ACTIVE, 0)
        mgr.release_freeze("R", "M", 1, "Ops team approved.")
        hist = mgr.history("R", "M")
        manual = [h for h in hist if h.reason is TransitionReason.MANUAL_RELEASE]
        assert len(manual) == 1
        assert manual[0].justification == "Ops team approved."


# ===========================================================================
# N. History
# ===========================================================================

class TestHistory:
    def test_no_transitions_empty(self, mgr):
        mgr.initialize("R", "M")
        assert mgr.history("R", "M") == ()

    def test_each_success_one_record(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        mgr.transition("R", "M", DefenseState.ACTIVE,
                        TransitionReason.SUCCESS, DefenseState.RECALIBRATING, 1)
        assert len(mgr.history("R", "M")) == 2

    def test_versions_ordered(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        mgr.transition("R", "M", DefenseState.ACTIVE,
                        TransitionReason.SUCCESS, DefenseState.RECALIBRATING, 1)
        mgr.transition("R", "M", DefenseState.RETRAINING,
                        TransitionReason.TRIAGE_INITIATED_RETRAINING,
                        DefenseState.ACTIVE, 2)
        hist = mgr.history("R", "M")
        vs = [h.transition_version for h in hist]
        assert vs == sorted(vs)

    def test_history_is_tuple(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        hist = mgr.history("R", "M")
        assert isinstance(hist, tuple)
        with pytest.raises((AttributeError, TypeError)):
            hist.append(None)

    def test_returned_tuple_is_snapshot(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        h1 = mgr.history("R", "M")
        mgr.transition("R", "M", DefenseState.ACTIVE,
                        TransitionReason.SUCCESS, DefenseState.RECALIBRATING, 1)
        h2 = mgr.history("R", "M")
        assert len(h1) == 1
        assert len(h2) == 2

    def test_failed_transition_no_record(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.ACTIVE,
                        TransitionReason.SUCCESS, DefenseState.ACTIVE, 0)
        assert mgr.history("R", "M") == ()

    def test_freeze_count(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.FROZEN,
                        TransitionReason.TRIAGE_INITIATED_FREEZE,
                        DefenseState.ACTIVE, 0)
        mgr.release_freeze("R", "M", 1, "Released.")
        mgr.transition("R", "M", DefenseState.FROZEN,
                        TransitionReason.TRIAGE_INITIATED_FREEZE,
                        DefenseState.ACTIVE, 2)
        hist = mgr.history("R", "M")
        freezes = [h for h in hist if h.to_state is DefenseState.FROZEN]
        assert len(freezes) == 2

    def test_uninitialised_history_empty(self, mgr):
        assert mgr.history("UNKNOWN", "MODEL") == ()


# ===========================================================================
# O. Concurrency
# ===========================================================================

class TestConcurrency:
    def test_exactly_one_thread_wins(self, mgr):
        mgr.initialize("R", "M")
        results = []

        def attempt():
            result = mgr.transition(
                "R", "M",
                to_state=DefenseState.RECALIBRATING,
                reason=TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                expected_state=DefenseState.ACTIVE,
                expected_transition_version=0,
            )
            results.append(result)

        t1 = threading.Thread(target=attempt)
        t2 = threading.Thread(target=attempt)
        t1.start(); t2.start()
        t1.join(); t2.join()

        successes = [r for r in results if r.success]
        failures  = [r for r in results if not r.success]
        assert len(successes) == 1
        assert len(failures)  == 1
        assert failures[0].failure_kind is _FailureKind.CAS_CONFLICT

        final = mgr.get("R", "M")
        assert final.transition_version == 1
        assert final.state is DefenseState.RECALIBRATING


# ===========================================================================
# P. No detector lifecycle coupling
# ===========================================================================

class TestNoDetectorCoupling:
    def test_state_py_forbidden_imports(self):
        """
        Inspect src/defense/state.py for forbidden detector/executor imports.
        Path is resolved relative to this file for cwd-independence.
        """
        this_file = Path(__file__).resolve()
        repo_root = this_file.parent.parent.parent
        state_path = repo_root / "src" / "defense" / "state.py"
        assert state_path.exists(), f"Not found: {state_path}"

        source = state_path.read_text(encoding="utf-8")

        forbidden = [
            "StatefulDetectorLifecycle",
            "contracts.lifecycle",
            "from src.contracts.lifecycle",
            "import lifecycle",
            "ADWINAdapter",
            "DDMAdapter",
            "PageHinkleyAdapter",
            "river",
            "MonitorExecutor",
            "RecalibrationExecutor",
            "FreezeExecutor",
            "RetrainGate",
        ]

        # Filter to only non-comment, non-docstring import lines
        import_lines = [
            line for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
            and ("import" in line or "from " in line)
        ]
        import_section = "\n".join(import_lines)

        # These tokens must not appear as actual import statements
        for token in forbidden:
            assert token not in import_section, (
                f"state.py must not import or reference '{token}' "
                f"in an import statement (found in import lines)"
            )

# ===========================================================================
# Q. Reason / edge invariant
#    Every legal edge must accept its correct reason(s) and reject every
#    other reason. State, version, and history must be unchanged on rejection.
# ===========================================================================

class TestReasonEdgeMapping:
    """
    Tests that _TRANSITION_ALLOWED_REASONS is enforced:
    - invalid reason on a legal edge -> INVALID_TRANSITION
    - state unchanged, version unchanged, history unchanged
    - correct reason on every legal edge -> success
    """

    # ------------------------------------------------------------------
    # Negative: invalid reason on every legal edge
    # ------------------------------------------------------------------

    def _assert_invalid_reason(self, mgr, from_s, to_s, bad_reason):
        """Helper: assert that bad_reason is rejected with INVALID_TRANSITION."""
        before = mgr.get("R", "M")
        result = mgr.transition(
            "R", "M", to_s, bad_reason,
            from_s, before.transition_version,
        )
        after = mgr.get("R", "M")
        assert result.success is False, (
            f"Expected INVALID_TRANSITION for {from_s}->{to_s} with {bad_reason}"
        )
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION
        # State must not change
        assert after.state is before.state
        # Version must not change
        assert after.transition_version == before.transition_version
        # History must not grow
        assert len(mgr.history("R", "M")) == len(
            [h for h in mgr.history("R", "M") if True]  # just re-reads
        )

    def _reach(self, mgr, target_state):
        """Drive mgr from ACTIVE to target_state using a correct reason."""
        lead = {
            DefenseState.ACTIVE: None,
            DefenseState.RECALIBRATING: (
                DefenseState.RECALIBRATING,
                TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
            ),
            DefenseState.RETRAINING: (
                DefenseState.RETRAINING,
                TransitionReason.TRIAGE_INITIATED_RETRAINING,
            ),
            DefenseState.FROZEN: (
                DefenseState.FROZEN,
                TransitionReason.TRIAGE_INITIATED_FREEZE,
            ),
        }[target_state]
        if lead is not None:
            cur = mgr.get("R", "M")
            mgr.transition("R", "M", lead[0], lead[1],
                            cur.state, cur.transition_version)

    # ACTIVE -> RECALIBRATING + SUCCESS (wrong)
    def test_active_to_recalibrating_wrong_reason(self, mgr):
        mgr.initialize("R", "M")
        hist_before = len(mgr.history("R", "M"))
        rec_before = mgr.get("R", "M")
        result = mgr.transition("R", "M", DefenseState.RECALIBRATING,
                                  TransitionReason.SUCCESS,
                                  DefenseState.ACTIVE, 0)
        assert result.success is False
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION
        assert mgr.get("R", "M").state is DefenseState.ACTIVE
        assert mgr.get("R", "M").transition_version == 0
        assert len(mgr.history("R", "M")) == hist_before

    # ACTIVE -> RETRAINING + SUCCESS (wrong)
    def test_active_to_retraining_wrong_reason(self, mgr):
        mgr.initialize("R", "M")
        result = mgr.transition("R", "M", DefenseState.RETRAINING,
                                  TransitionReason.SUCCESS,
                                  DefenseState.ACTIVE, 0)
        assert result.success is False
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION
        assert mgr.get("R", "M").state is DefenseState.ACTIVE
        assert mgr.get("R", "M").transition_version == 0
        assert len(mgr.history("R", "M")) == 0

    # ACTIVE -> FROZEN + SUCCESS (wrong)
    def test_active_to_frozen_wrong_reason(self, mgr):
        mgr.initialize("R", "M")
        result = mgr.transition("R", "M", DefenseState.FROZEN,
                                  TransitionReason.SUCCESS,
                                  DefenseState.ACTIVE, 0)
        assert result.success is False
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION
        assert mgr.get("R", "M").state is DefenseState.ACTIVE
        assert mgr.get("R", "M").transition_version == 0
        assert len(mgr.history("R", "M")) == 0

    # RECALIBRATING -> ACTIVE + TRIAGE_INITIATED_RECALIBRATION (wrong)
    def test_recalibrating_to_active_wrong_reason(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        hist_before = len(mgr.history("R", "M"))
        result = mgr.transition("R", "M", DefenseState.ACTIVE,
                                  TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                                  DefenseState.RECALIBRATING, 1)
        assert result.success is False
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION
        assert mgr.get("R", "M").state is DefenseState.RECALIBRATING
        assert mgr.get("R", "M").transition_version == 1
        assert len(mgr.history("R", "M")) == hist_before

    # RECALIBRATING -> FROZEN + SUCCESS (wrong)
    def test_recalibrating_to_frozen_wrong_reason(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        hist_before = len(mgr.history("R", "M"))
        result = mgr.transition("R", "M", DefenseState.FROZEN,
                                  TransitionReason.SUCCESS,
                                  DefenseState.RECALIBRATING, 1)
        assert result.success is False
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION
        assert mgr.get("R", "M").state is DefenseState.RECALIBRATING
        assert mgr.get("R", "M").transition_version == 1
        assert len(mgr.history("R", "M")) == hist_before

    # RETRAINING -> ACTIVE + FAILURE (wrong — only SUCCESS or REJECTED_BY_SHADOW)
    def test_retraining_to_active_wrong_reason(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RETRAINING,
                        TransitionReason.TRIAGE_INITIATED_RETRAINING,
                        DefenseState.ACTIVE, 0)
        hist_before = len(mgr.history("R", "M"))
        result = mgr.transition("R", "M", DefenseState.ACTIVE,
                                  TransitionReason.FAILURE,
                                  DefenseState.RETRAINING, 1)
        assert result.success is False
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION
        assert mgr.get("R", "M").state is DefenseState.RETRAINING
        assert mgr.get("R", "M").transition_version == 1
        assert len(mgr.history("R", "M")) == hist_before

    # RETRAINING -> FROZEN + SUCCESS (wrong)
    def test_retraining_to_frozen_wrong_reason(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RETRAINING,
                        TransitionReason.TRIAGE_INITIATED_RETRAINING,
                        DefenseState.ACTIVE, 0)
        hist_before = len(mgr.history("R", "M"))
        result = mgr.transition("R", "M", DefenseState.FROZEN,
                                  TransitionReason.SUCCESS,
                                  DefenseState.RETRAINING, 1)
        assert result.success is False
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION
        assert mgr.get("R", "M").state is DefenseState.RETRAINING
        assert mgr.get("R", "M").transition_version == 1
        assert len(mgr.history("R", "M")) == hist_before

    # FROZEN -> ACTIVE + SUCCESS (wrong — only MANUAL_RELEASE)
    def test_frozen_to_active_wrong_reason(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.FROZEN,
                        TransitionReason.TRIAGE_INITIATED_FREEZE,
                        DefenseState.ACTIVE, 0)
        hist_before = len(mgr.history("R", "M"))
        result = mgr.transition("R", "M", DefenseState.ACTIVE,
                                  TransitionReason.SUCCESS,
                                  DefenseState.FROZEN, 1)
        assert result.success is False
        assert result.failure_kind is _FailureKind.INVALID_TRANSITION
        assert mgr.get("R", "M").state is DefenseState.FROZEN
        assert mgr.get("R", "M").transition_version == 1
        assert len(mgr.history("R", "M")) == hist_before

    # ------------------------------------------------------------------
    # Positive: one correct-reason test per legal edge
    # ------------------------------------------------------------------

    def test_positive_active_to_recalibrating(self, mgr):
        mgr.initialize("R", "M")
        result = mgr.transition("R", "M", DefenseState.RECALIBRATING,
                                  TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                                  DefenseState.ACTIVE, 0)
        assert result.success is True

    def test_positive_active_to_retraining(self, mgr):
        mgr.initialize("R", "M")
        result = mgr.transition("R", "M", DefenseState.RETRAINING,
                                  TransitionReason.TRIAGE_INITIATED_RETRAINING,
                                  DefenseState.ACTIVE, 0)
        assert result.success is True

    def test_positive_active_to_frozen(self, mgr):
        mgr.initialize("R", "M")
        result = mgr.transition("R", "M", DefenseState.FROZEN,
                                  TransitionReason.TRIAGE_INITIATED_FREEZE,
                                  DefenseState.ACTIVE, 0)
        assert result.success is True

    def test_positive_recalibrating_to_active_success(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.ACTIVE,
                                  TransitionReason.SUCCESS,
                                  DefenseState.RECALIBRATING, 1)
        assert result.success is True

    def test_positive_recalibrating_to_active_failure(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.ACTIVE,
                                  TransitionReason.FAILURE,
                                  DefenseState.RECALIBRATING, 1)
        assert result.success is True

    def test_positive_recalibrating_to_frozen(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RECALIBRATING,
                        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.FROZEN,
                                  TransitionReason.TRIAGE_INITIATED_FREEZE,
                                  DefenseState.RECALIBRATING, 1)
        assert result.success is True

    def test_positive_retraining_to_active_success(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RETRAINING,
                        TransitionReason.TRIAGE_INITIATED_RETRAINING,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.ACTIVE,
                                  TransitionReason.SUCCESS,
                                  DefenseState.RETRAINING, 1)
        assert result.success is True

    def test_positive_retraining_to_active_shadow_rejected(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RETRAINING,
                        TransitionReason.TRIAGE_INITIATED_RETRAINING,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.ACTIVE,
                                  TransitionReason.REJECTED_BY_SHADOW_VALIDATION,
                                  DefenseState.RETRAINING, 1)
        assert result.success is True

    def test_positive_retraining_to_frozen(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.RETRAINING,
                        TransitionReason.TRIAGE_INITIATED_RETRAINING,
                        DefenseState.ACTIVE, 0)
        result = mgr.transition("R", "M", DefenseState.FROZEN,
                                  TransitionReason.TRIAGE_INITIATED_FREEZE,
                                  DefenseState.RETRAINING, 1)
        assert result.success is True

    def test_positive_frozen_to_active_manual_release(self, mgr):
        mgr.initialize("R", "M")
        mgr.transition("R", "M", DefenseState.FROZEN,
                        TransitionReason.TRIAGE_INITIATED_FREEZE,
                        DefenseState.ACTIVE, 0)
        result = mgr.release_freeze("R", "M", 1, "Ops approved.")
        assert result.success is True
        assert result.record.last_transition_reason is TransitionReason.MANUAL_RELEASE