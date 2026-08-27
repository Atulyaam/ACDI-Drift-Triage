"""
src/defense/state.py
====================

Defense lifecycle state management for the ACDI + Drift-Triage system.

SCOPE OF THIS MODULE
--------------------
This module owns ONLY:

    DefenseState         - legal defense lifecycle states
    TransitionReason     - auditable reasons for state transitions
    DefenseStateRecord   - immutable point-in-time state snapshot
    TransitionRecord     - immutable append-only audit entry
    TransitionResult     - structured outcome of every transition attempt
    DefenseStateManager  - CAS-guarded, thread-safe in-memory state manager

OUT OF SCOPE (future milestones)
---------------------------------
    DefenseOrchestrator
    Real recalibration algorithm
    Real freeze mechanism
    Real retraining pipeline
    Model-version registry
    Distributed locking / Redis / database persistence

ARCHITECTURAL SEPARATION
------------------------
DefenseState is INDEPENDENT of detector lifecycle.
This module must NOT import:

    StatefulDetectorLifecycle
    contracts.lifecycle.DetectorState
    streaming DetectorState
    ADWIN / DDM / PageHinkley
    River
    MonitorExecutor / RecalibrationExecutor / FreezeExecutor
    RetrainGate

State scope: (run_id, model_version)
Each unique pair has an independent, isolated DefenseState.

V1 LIMITATION
-------------
Storage is in-memory, per DefenseStateManager instance, single Python process.
Thread safety is enforced via threading.RLock. The compare-validate-write
operation is atomic within that process. This is NOT distributed concurrency
control. No Redis, no database, no file locking.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_non_empty_string(value, field_name):
    """Raise ValueError/TypeError if value is not a non-empty str."""
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a str, got "
            f"{type(value).__name__}."
        )
    if not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string."
        )


def _require_strict_int(value, field_name):
    """Reject bool masquerading as int."""
    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be an int (bool not accepted)."
        )
    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be an int, got "
            f"{type(value).__name__}."
        )


def _require_strict_bool(value, field_name):
    """Require exactly bool (not int sub-type)."""
    if not isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be a bool, got "
            f"{type(value).__name__}."
        )


def _require_timezone_aware_datetime(value, field_name):
    if not isinstance(value, datetime):
        raise TypeError(
            f"{field_name} must be a datetime, got "
            f"{type(value).__name__}."
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{field_name} must be timezone-aware."
        )


def _require_metadata_dict(value, field_name="metadata"):
    if not isinstance(value, dict):
        raise TypeError(
            f"{field_name} must be a dict, got "
            f"{type(value).__name__}."
        )
    if not all(isinstance(k, str) for k in value):
        raise TypeError(
            f"All {field_name} keys must be str."
        )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DefenseState(str, Enum):
    """
    Legal lifecycle states for a defense scope.

    ACTIVE         - normal operating state; adaptive operations are allowed.
    RECALIBRATING  - a recalibration operation is in flight.
    RETRAINING     - a retraining pipeline is in flight.
    FROZEN         - all adaptive model mutation blocked;
                     inference and monitoring continue.

    Mutual exclusion: RECALIBRATING and RETRAINING may never be active
    concurrently for the same (run_id, model_version).

    FROZEN -> ACTIVE requires explicit MANUAL_RELEASE only.
    """

    ACTIVE = "ACTIVE"
    RECALIBRATING = "RECALIBRATING"
    RETRAINING = "RETRAINING"
    FROZEN = "FROZEN"


class TransitionReason(str, Enum):
    """
    Auditable reasons for state transitions.

    Reason semantics by transition:

        ACTIVE -> RECALIBRATING         TRIAGE_INITIATED_RECALIBRATION
        ACTIVE -> RETRAINING            TRIAGE_INITIATED_RETRAINING
        ACTIVE/RECALIBRATING/
          RETRAINING -> FROZEN          TRIAGE_INITIATED_FREEZE
        RECALIBRATING -> ACTIVE (ok)    SUCCESS
        RECALIBRATING -> ACTIVE (bad)   FAILURE
        RETRAINING -> ACTIVE (ok)       SUCCESS
        RETRAINING -> ACTIVE (rejected) REJECTED_BY_SHADOW_VALIDATION
        FROZEN -> ACTIVE                MANUAL_RELEASE

    MANUAL_RELEASE requires a non-empty justification string.
    All other reasons require justification == None.
    """

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    REJECTED_BY_SHADOW_VALIDATION = "REJECTED_BY_SHADOW_VALIDATION"
    MANUAL_RELEASE = "MANUAL_RELEASE"
    TRIAGE_INITIATED_RECALIBRATION = "TRIAGE_INITIATED_RECALIBRATION"
    TRIAGE_INITIATED_RETRAINING = "TRIAGE_INITIATED_RETRAINING"
    TRIAGE_INITIATED_FREEZE = "TRIAGE_INITIATED_FREEZE"


# ---------------------------------------------------------------------------
# Legal transition table
# ---------------------------------------------------------------------------

_LEGAL_TRANSITIONS = frozenset(
    {
        # From ACTIVE
        (DefenseState.ACTIVE, DefenseState.RECALIBRATING),
        (DefenseState.ACTIVE, DefenseState.RETRAINING),
        (DefenseState.ACTIVE, DefenseState.FROZEN),
        # From RECALIBRATING
        (DefenseState.RECALIBRATING, DefenseState.ACTIVE),
        (DefenseState.RECALIBRATING, DefenseState.FROZEN),
        # From RETRAINING
        (DefenseState.RETRAINING, DefenseState.ACTIVE),
        (DefenseState.RETRAINING, DefenseState.FROZEN),
        # From FROZEN
        (DefenseState.FROZEN, DefenseState.ACTIVE),
    }
)


# ---------------------------------------------------------------------------
# Allowed reasons per (from_state, to_state) edge
# ---------------------------------------------------------------------------
# Single source of truth.  Every legal edge must appear here.
# Supplying a reason that is not in the allowed set is a semantic violation
# and returns INVALID_TRANSITION (not CAS_CONFLICT).

_TRANSITION_ALLOWED_REASONS = {
    (DefenseState.ACTIVE, DefenseState.RECALIBRATING): frozenset({
        TransitionReason.TRIAGE_INITIATED_RECALIBRATION,
    }),
    (DefenseState.ACTIVE, DefenseState.RETRAINING): frozenset({
        TransitionReason.TRIAGE_INITIATED_RETRAINING,
    }),
    (DefenseState.ACTIVE, DefenseState.FROZEN): frozenset({
        TransitionReason.TRIAGE_INITIATED_FREEZE,
    }),
    (DefenseState.RECALIBRATING, DefenseState.ACTIVE): frozenset({
        TransitionReason.SUCCESS,
        TransitionReason.FAILURE,
    }),
    (DefenseState.RECALIBRATING, DefenseState.FROZEN): frozenset({
        TransitionReason.TRIAGE_INITIATED_FREEZE,
    }),
    (DefenseState.RETRAINING, DefenseState.ACTIVE): frozenset({
        TransitionReason.SUCCESS,
        TransitionReason.REJECTED_BY_SHADOW_VALIDATION,
    }),
    (DefenseState.RETRAINING, DefenseState.FROZEN): frozenset({
        TransitionReason.TRIAGE_INITIATED_FREEZE,
    }),
    (DefenseState.FROZEN, DefenseState.ACTIVE): frozenset({
        TransitionReason.MANUAL_RELEASE,
    }),
}


# ---------------------------------------------------------------------------
# TransitionResult failure-kind classifier
# ---------------------------------------------------------------------------


class _FailureKind(str, Enum):
    """
    Internal classifier for structured transition failures.
    Distinct from TransitionReason (which records WHY a transition happened).
    This classifies WHY a transition was REJECTED.
    """

    CAS_CONFLICT = "CAS_CONFLICT"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    MISSING_STATE = "MISSING_STATE"


# ---------------------------------------------------------------------------
# Immutable contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefenseStateRecord:
    """
    Immutable point-in-time snapshot of the defense state
    for one (run_id, model_version) scope.
    """

    run_id: str
    model_version: str
    state: DefenseState
    transition_version: int
    last_transition_reason: TransitionReason
    last_transition_at: datetime
    metadata: dict[str, Any] = field(
    default_factory=dict,
    compare=False,
)

    def __post_init__(self):
        _require_non_empty_string(self.run_id, "run_id")
        _require_non_empty_string(self.model_version, "model_version")

        if not isinstance(self.state, DefenseState):
            raise TypeError(
                "state must be a DefenseState, got "
                f"{type(self.state).__name__}."
            )

        _require_strict_int(self.transition_version, "transition_version")
        if self.transition_version < 0:
            raise ValueError("transition_version must be >= 0.")

        if not isinstance(self.last_transition_reason, TransitionReason):
            raise TypeError(
                "last_transition_reason must be a TransitionReason, got "
                f"{type(self.last_transition_reason).__name__}."
            )

        _require_timezone_aware_datetime(
            self.last_transition_at, "last_transition_at"
        )

        _require_metadata_dict(self.metadata)


@dataclass(frozen=True)
class TransitionRecord:
    """
    Immutable append-only audit entry recording one completed state transition.
    """

    run_id: str
    model_version: str
    from_state: DefenseState
    to_state: DefenseState
    transition_version: int
    reason: TransitionReason
    timestamp: datetime
    justification: str | None # str | None
    metadata: dict[str, Any] = field(
    default_factory=dict,
    compare=False,
)

    def __post_init__(self):
        _require_non_empty_string(self.run_id, "run_id")
        _require_non_empty_string(self.model_version, "model_version")

        if not isinstance(self.from_state, DefenseState):
            raise TypeError(
                "from_state must be a DefenseState, got "
                f"{type(self.from_state).__name__}."
            )

        if not isinstance(self.to_state, DefenseState):
            raise TypeError(
                "to_state must be a DefenseState, got "
                f"{type(self.to_state).__name__}."
            )

        _require_strict_int(self.transition_version, "transition_version")
        if self.transition_version < 1:
            raise ValueError(
                "transition_version in a TransitionRecord must be >= 1."
            )

        if not isinstance(self.reason, TransitionReason):
            raise TypeError(
                "reason must be a TransitionReason, got "
                f"{type(self.reason).__name__}."
            )

        _require_timezone_aware_datetime(self.timestamp, "timestamp")

        # Justification rules
        if self.reason is TransitionReason.MANUAL_RELEASE:
            if self.justification is None:
                raise ValueError(
                    "justification is required (non-empty str) "
                    "when reason is MANUAL_RELEASE."
                )
            if not isinstance(self.justification, str):
                raise TypeError(
                    "justification must be a str when reason is MANUAL_RELEASE."
                )
            if not self.justification.strip():
                raise ValueError(
                    "justification must be a non-empty string "
                    "when reason is MANUAL_RELEASE."
                )
        else:
            if self.justification is not None:
                raise ValueError(
                    "justification must be None for all "
                    "TransitionReasons except MANUAL_RELEASE."
                )

        _require_metadata_dict(self.metadata)


@dataclass(frozen=True)
class TransitionResult:
    """
    Structured, immutable outcome of every transition attempt.

    success == True   -> transition applied; record is new DefenseStateRecord.
    success == False  -> transition rejected:
        CAS_CONFLICT       stale expected state or version.
        INVALID_TRANSITION requested edge is illegal.
        MISSING_STATE      pair not initialized; call initialize() first.
    """

   
    success: bool
    failure_kind: _FailureKind | None
    current_state: DefenseState
    current_transition_version: int
    record: DefenseStateRecord | None

    def __post_init__(self):
        _require_strict_bool(self.success, "success")

        if self.success:
            if self.failure_kind is not None:
                raise ValueError("failure_kind must be None on success.")
            if self.record is None:
                raise ValueError(
                    "record must be a DefenseStateRecord on success."
                )
        else:
            if self.failure_kind is None:
                raise ValueError("failure_kind must be set on failure.")
            if not isinstance(self.failure_kind, _FailureKind):
                raise TypeError(
                    "failure_kind must be a _FailureKind, got "
                    f"{type(self.failure_kind).__name__}."
                )
            if self.record is not None:
                raise ValueError("record must be None on failure.")

        if not isinstance(self.current_state, DefenseState):
            raise TypeError(
                "current_state must be a DefenseState, got "
                f"{type(self.current_state).__name__}."
            )

        _require_strict_int(
            self.current_transition_version, "current_transition_version"
        )
        if self.current_transition_version < 0:
            raise ValueError("current_transition_version must be >= 0.")

        if self.record is not None and not isinstance(
            self.record, DefenseStateRecord
        ):
            raise TypeError(
                "record must be a DefenseStateRecord or None."
            )


# ---------------------------------------------------------------------------
# Internal helpers to build TransitionResult
# ---------------------------------------------------------------------------


def _make_success_result(record):
    return TransitionResult(
        success=True,
        failure_kind=None,
        current_state=record.state,
        current_transition_version=record.transition_version,
        record=record,
    )


def _make_failure_result(kind, current_record):
    return TransitionResult(
        success=False,
        failure_kind=kind,
        current_state=current_record.state,
        current_transition_version=current_record.transition_version,
        record=None,
    )


def _make_missing_result():
    """
    Return a structured failure when the scope is not initialized.
    Uses sentinel ACTIVE/0 values since there is no current record.
    """
    return TransitionResult(
        success=False,
        failure_kind=_FailureKind.MISSING_STATE,
        current_state=DefenseState.ACTIVE,
        current_transition_version=0,
        record=None,
    )


# ---------------------------------------------------------------------------
# DefenseStateManager
# ---------------------------------------------------------------------------


class DefenseStateManager:
    """
    CAS-guarded, thread-safe, in-memory defense state manager.

    Each (run_id, model_version) pair has an independent DefenseState.
    State transitions are atomically guarded by compare-and-swap semantics:
    callers supply expected_state and expected_transition_version.

    V1 LIMITATION
    -------------
    Storage is in-memory, per instance, single Python process.
    Thread safety is via threading.RLock. The compare-validate-write block
    is atomic within that process. This is NOT distributed concurrency.

    EXPLICIT INITIALIZATION REQUIRED
    ---------------------------------
    Calling transition() or release_freeze() on an unknown
    (run_id, model_version) returns a MISSING_STATE failure.
    Auto-creation is not permitted.

    HISTORY
    -------
    Every successful transition appends one TransitionRecord.
    Initialization does NOT append a TransitionRecord.
    History is read-only to callers; internal mutability is hidden.
    """

    def __init__(self):
        # _states: dict[(run_id, model_version), DefenseStateRecord]
        self._states = {}
        # _history: dict[(run_id, model_version), list[TransitionRecord]]
        self._history = {}
        # Single re-entrant lock guards all state + history mutations.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, run_id, model_version, metadata=None):
        """
        Ensure (run_id, model_version) exists with ACTIVE/version 0 state.

        If the pair already exists, the existing record is returned
        unchanged. No reset, no version increment, no new history entry.

        Explicit initialization is REQUIRED before any transition.
        Unknown pairs are NOT auto-created during transition.
        """
        _require_non_empty_string(run_id, "run_id")
        _require_non_empty_string(model_version, "model_version")

        if metadata is None:
            metadata = {}
        _require_metadata_dict(metadata, "metadata")

        key = (run_id, model_version)

        with self._lock:
            if key in self._states:
                return self._states[key]

            record = DefenseStateRecord(
                run_id=run_id,
                model_version=model_version,
                state=DefenseState.ACTIVE,
                transition_version=0,
                last_transition_reason=TransitionReason.SUCCESS,
                last_transition_at=datetime.now(tz=timezone.utc),
                metadata=dict(metadata),
            )
            self._states[key] = record
            self._history[key] = []
            return record

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, run_id, model_version):
        """Return the current DefenseStateRecord or None if not initialized."""
        _require_non_empty_string(run_id, "run_id")
        _require_non_empty_string(model_version, "model_version")
        with self._lock:
            return self._states.get((run_id, model_version))

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def history(self, run_id, model_version):
        """
        Return an immutable copy of the transition history, ordered by
        transition_version ascending.

        Returns empty tuple if not initialized or no transitions yet.
        Callers cannot mutate manager history through the returned object.
        """
        _require_non_empty_string(run_id, "run_id")
        _require_non_empty_string(model_version, "model_version")
        with self._lock:
            raw = self._history.get((run_id, model_version), [])
            return tuple(raw)

    # ------------------------------------------------------------------
    # Internal CAS transition core
    # ------------------------------------------------------------------

    def _do_transition(
        self,
        run_id,
        model_version,
        to_state,
        reason,
        expected_state,
        expected_transition_version,
        justification,
        metadata,
    ):
        """
        Internal CAS-guarded state transition.

        The compare + validate + write block executes atomically
        under self._lock.
        """
        key = (run_id, model_version)

        with self._lock:
            current = self._states.get(key)

            if current is None:
                return _make_missing_result()

            # CAS check: both expected_state AND expected_version must match.
            if (
                current.state is not expected_state
                or current.transition_version != expected_transition_version
            ):
                return _make_failure_result(_FailureKind.CAS_CONFLICT, current)

            # Legality check — topology
            if (current.state, to_state) not in _LEGAL_TRANSITIONS:
                return _make_failure_result(
                    _FailureKind.INVALID_TRANSITION, current
                )

            # Legality check — reason must be valid for this specific edge
            allowed_reasons = _TRANSITION_ALLOWED_REASONS.get(
                (current.state, to_state), frozenset()
            )
            if reason not in allowed_reasons:
                return _make_failure_result(
                    _FailureKind.INVALID_TRANSITION, current
                )

            new_version = current.transition_version + 1
            ts = datetime.now(tz=timezone.utc)

            new_record = DefenseStateRecord(
                run_id=run_id,
                model_version=model_version,
                state=to_state,
                transition_version=new_version,
                last_transition_reason=reason,
                last_transition_at=ts,
                metadata=dict(metadata),
            )

            audit = TransitionRecord(
                run_id=run_id,
                model_version=model_version,
                from_state=current.state,
                to_state=to_state,
                transition_version=new_version,
                reason=reason,
                timestamp=ts,
                justification=justification,
                metadata=dict(metadata),
            )

            self._states[key] = new_record
            self._history[key].append(audit)

            return _make_success_result(new_record)

    # ------------------------------------------------------------------
    # Public transition API
    # ------------------------------------------------------------------

    def transition(
        self,
        run_id,
        model_version,
        to_state,
        reason,
        expected_state,
        expected_transition_version,
        metadata=None,
    ):
        """
        Request a CAS-guarded state transition.

        FROZEN -> ACTIVE must use release_freeze() instead of this method.

        Returns
        -------
        TransitionResult
        """
        _require_non_empty_string(run_id, "run_id")
        _require_non_empty_string(model_version, "model_version")

        if not isinstance(to_state, DefenseState):
            raise TypeError("to_state must be a DefenseState.")
        if not isinstance(reason, TransitionReason):
            raise TypeError("reason must be a TransitionReason.")
        if not isinstance(expected_state, DefenseState):
            raise TypeError("expected_state must be a DefenseState.")
        _require_strict_int(
            expected_transition_version, "expected_transition_version"
        )
        if metadata is None:
            metadata = {}
        _require_metadata_dict(metadata, "metadata")

        return self._do_transition(
            run_id=run_id,
            model_version=model_version,
            to_state=to_state,
            reason=reason,
            expected_state=expected_state,
            expected_transition_version=expected_transition_version,
            justification=None,
            metadata=metadata,
        )

    def release_freeze(
        self,
        run_id,
        model_version,
        expected_transition_version,
        justification,
        metadata=None,
    ):
        """
        Manually release a FROZEN state, transitioning to ACTIVE.

        This is the ONLY permitted path for FROZEN -> ACTIVE.
        Automatic thaw is not supported.

        Parameters
        ----------
        justification  non-empty str documenting the release reason (required)

        Returns
        -------
        TransitionResult
        """
        _require_non_empty_string(run_id, "run_id")
        _require_non_empty_string(model_version, "model_version")
        _require_strict_int(
            expected_transition_version, "expected_transition_version"
        )

        if not isinstance(justification, str):
            raise TypeError("justification must be a str.")
        if not justification.strip():
            raise ValueError("justification must be a non-empty string.")

        if metadata is None:
            metadata = {}
        _require_metadata_dict(metadata, "metadata")

        return self._do_transition(
            run_id=run_id,
            model_version=model_version,
            to_state=DefenseState.ACTIVE,
            reason=TransitionReason.MANUAL_RELEASE,
            expected_state=DefenseState.FROZEN,
            expected_transition_version=expected_transition_version,
            justification=justification,
            metadata=metadata,
        )