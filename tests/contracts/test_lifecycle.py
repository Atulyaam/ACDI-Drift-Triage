from datetime import datetime, timezone

import pytest

from src.contracts.lifecycle import (
    StatefulDetectorLifecycle,
)


def active_lifecycle():
    return StatefulDetectorLifecycle(
        run_id="RUN_001",
        detector_name="ADWIN",
        detector_instance_id="ADWIN_001",
        state="ACTIVE",
        first_detection_index=None,
        reported_window_id=None,
        latch_window_id=None,
        resolution_status="UNRESOLVED",
        resolution_window_id=None,
        timestamp=datetime.now(timezone.utc),
    )


def pending_lifecycle():
    return StatefulDetectorLifecycle(
        run_id="RUN_001",
        detector_name="ADWIN",
        detector_instance_id="ADWIN_001",
        state="DRIFT_PENDING",
        first_detection_index=41650,
        reported_window_id="WIN_042",
        latch_window_id="WIN_042",
        resolution_status="UNRESOLVED",
        resolution_window_id=None,
        timestamp=datetime.now(timezone.utc),
    )


def resolved_lifecycle():
    # RESOLVED must preserve the historical drift-event record.
    return StatefulDetectorLifecycle(
        run_id="RUN_001",
        detector_name="ADWIN",
        detector_instance_id="ADWIN_001",
        state="ACTIVE",
        first_detection_index=41650,
        reported_window_id="WIN_042",
        latch_window_id="WIN_042",
        resolution_status="RESOLVED",
        resolution_window_id="WIN_043",
        timestamp=datetime.now(timezone.utc),
    )


def test_active_lifecycle():
    lifecycle = active_lifecycle()

    assert lifecycle.state == "ACTIVE"
    assert lifecycle.resolution_status == "UNRESOLVED"


def test_pending_lifecycle():
    lifecycle = pending_lifecycle()

    assert lifecycle.state == "DRIFT_PENDING"
    assert lifecycle.first_detection_index == 41650
    assert lifecycle.reported_window_id == "WIN_042"
    assert lifecycle.latch_window_id == "WIN_042"


def test_resolved_lifecycle_preserves_history():
    lifecycle = resolved_lifecycle()

    assert lifecycle.resolution_status == "RESOLVED"
    assert lifecycle.first_detection_index == 41650
    assert lifecycle.reported_window_id == "WIN_042"
    assert lifecycle.resolution_window_id == "WIN_043"


def test_pending_requires_detection_index():
    with pytest.raises(ValueError):
        StatefulDetectorLifecycle(
            run_id="RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_001",
            state="DRIFT_PENDING",
            first_detection_index=None,
            reported_window_id="WIN_042",
            latch_window_id="WIN_042",
            resolution_status="UNRESOLVED",
            resolution_window_id=None,
            timestamp=datetime.now(timezone.utc),
        )


def test_pending_requires_window_ids():
    with pytest.raises(ValueError):
        StatefulDetectorLifecycle(
            run_id="RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_001",
            state="DRIFT_PENDING",
            first_detection_index=41650,
            reported_window_id=None,
            latch_window_id=None,
            resolution_status="UNRESOLVED",
            resolution_window_id=None,
            timestamp=datetime.now(timezone.utc),
        )


def test_resolved_requires_active_state():
    with pytest.raises(ValueError):
        StatefulDetectorLifecycle(
            run_id="RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_001",
            state="DRIFT_PENDING",
            first_detection_index=41650,
            reported_window_id="WIN_042",
            latch_window_id="WIN_042",
            resolution_status="RESOLVED",
            resolution_window_id="WIN_043",
            timestamp=datetime.now(timezone.utc),
        )


def test_resolved_requires_resolution_window():
    with pytest.raises(ValueError):
        StatefulDetectorLifecycle(
            run_id="RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_001",
            state="ACTIVE",
            first_detection_index=41650,
            reported_window_id="WIN_042",
            latch_window_id="WIN_042",
            resolution_status="RESOLVED",
            resolution_window_id=None,
            timestamp=datetime.now(timezone.utc),
        )


def test_resolved_requires_historical_detection_index():
    # RESOLVED without a first_detection_index loses traceability.
    with pytest.raises(ValueError):
        StatefulDetectorLifecycle(
            run_id="RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_001",
            state="ACTIVE",
            first_detection_index=None,
            reported_window_id=None,
            latch_window_id=None,
            resolution_status="RESOLVED",
            resolution_window_id="WIN_043",
            timestamp=datetime.now(timezone.utc),
        )


def test_unresolved_timeout_requires_historical_record():
    with pytest.raises(ValueError):
        StatefulDetectorLifecycle(
            run_id="RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_001",
            state="ACTIVE",
            first_detection_index=None,
            reported_window_id=None,
            latch_window_id=None,
            resolution_status="UNRESOLVED_TIMEOUT",
            resolution_window_id=None,
            timestamp=datetime.now(timezone.utc),
        )


def test_timeout_requires_active_state_after_reset():
    with pytest.raises(ValueError):
        StatefulDetectorLifecycle(
            run_id="RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_001",
            state="DRIFT_PENDING",
            first_detection_index=41650,
            reported_window_id="WIN_042",
            latch_window_id="WIN_042",
            resolution_status="UNRESOLVED_TIMEOUT",
            resolution_window_id=None,
            timestamp=datetime.now(timezone.utc),
        )


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        StatefulDetectorLifecycle(
            run_id="RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_001",
            state="ACTIVE",
            first_detection_index=None,
            reported_window_id=None,
            latch_window_id=None,
            resolution_status="UNRESOLVED",
            resolution_window_id=None,
            timestamp=datetime.now(),
        )


def test_invalid_state_rejected():
    with pytest.raises(ValueError):
        StatefulDetectorLifecycle(
            run_id="RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_001",
            state="INVALID",
            first_detection_index=None,
            reported_window_id=None,
            latch_window_id=None,
            resolution_status="UNRESOLVED",
            resolution_window_id=None,
            timestamp=datetime.now(timezone.utc),
        )


def test_bool_first_detection_index_rejected():
    # bool is a subclass of int; must be explicitly excluded.
    with pytest.raises(TypeError):
        StatefulDetectorLifecycle(
            run_id="RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_001",
            state="DRIFT_PENDING",
            first_detection_index=True,
            reported_window_id="WIN_042",
            latch_window_id="WIN_042",
            resolution_status="UNRESOLVED",
            resolution_window_id=None,
            timestamp=datetime.now(timezone.utc),
        )


def test_lifecycle_is_hashable():
    lifecycle = active_lifecycle()
    # Should not raise TypeError even though metadata is a dict
    hash(lifecycle)


def test_lifecycle_metadata_non_string_keys_rejected():
    with pytest.raises(TypeError):
        StatefulDetectorLifecycle(
            run_id="RUN_001",
            detector_name="ADWIN",
            detector_instance_id="ADWIN_001",
            state="ACTIVE",
            first_detection_index=None,
            reported_window_id=None,
            latch_window_id=None,
            resolution_status="UNRESOLVED",
            resolution_window_id=None,
            timestamp=datetime.now(timezone.utc),
            metadata={17: "invalid_key"},
        )
