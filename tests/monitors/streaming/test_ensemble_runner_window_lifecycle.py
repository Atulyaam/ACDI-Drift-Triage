
import pytest

from src.contracts.window_config import (
    WindowManagerConfig,
)

from src.windows.manager import (
    WindowManager,
)

from src.monitors.streaming.adwin_adapter import (
    ADWINAdapter,
)

from src.monitors.streaming.configs import (
    ADWINConfig,
    DDMConfig,
    PageHinkleyConfig,
)

from src.monitors.streaming.ddm_adapter import (
    DDMAdapter,
)

from src.monitors.streaming.ensemble_runner import (
    StreamingEnsembleRunner,
)

from src.monitors.streaming.error_contracts import (
    DetectorState,
    PredictionErrorObservation,
)

from src.monitors.streaming.page_hinkley_adapter import (
    PageHinkleyAdapter,
)


RUN_ID = "RUN_001"


def _window_manager(
    total_samples=20,
    window_size=10,
):
    return WindowManager(
        WindowManagerConfig(
            run_id=RUN_ID,
            window_size=window_size,
            total_samples=total_samples,
            start_index=0,
            window_id_prefix="WIN_",
            drop_last_partial=False,
        )
    )


def _obs(
    sample_index,
    window_id,
    error=0,
):
    return PredictionErrorObservation(
        run_id=RUN_ID,
        sample_index=sample_index,
        external_window_id=window_id,
        error=error,
    )


class AlwaysDriftingFakeRiver:
    """
    Deterministic River substitute.

    Every update reports drift=True.
    """

    def __init__(self):
        self.drift_detected = False
        self.warning_detected = False

        self.calls = 0

        # ADWIN attributes
        self.delta = 0.002
        self.clock = 32
        self.max_buckets = 5
        self.min_window_length = 5
        self.grace_period = 10

        # DDM attributes
        self.warm_start = 30
        self.warning_threshold = 2.0
        self.drift_threshold = 3.0

        # Page-Hinkley attributes
        self.min_instances = 30
        self.threshold = 50.0
        self.alpha = 0.9999
        self.mode = "both"

    def update(self, value):
        self.calls += 1
        self.drift_detected = True
        self.warning_detected = True

    def _reset(self):
        self.calls = 0
        self.drift_detected = False
        self.warning_detected = False


class NeverDriftingFakeRiver:
    """
    Deterministic River substitute with no drift.
    """

    def __init__(self):
        self.drift_detected = False
        self.warning_detected = False

        self.calls = 0

        self.delta = 0.002
        self.clock = 32
        self.max_buckets = 5
        self.min_window_length = 5
        self.grace_period = 10

        self.warm_start = 30
        self.warning_threshold = 2.0
        self.drift_threshold = 3.0

        self.min_instances = 30
        self.threshold = 50.0
        self.alpha = 0.9999
        self.mode = "both"

    def update(self, value):
        self.calls += 1
        self.drift_detected = False
        self.warning_detected = False

    def _reset(self):
        self.calls = 0
        self.drift_detected = False
        self.warning_detected = False


def _make_runner(
    *,
    drifting=True,
):
    manager = _window_manager(
        total_samples=20,
        window_size=10,
    )

    adwin = ADWINAdapter(
        run_id=RUN_ID,
        config=ADWINConfig(),
    )

    ddm = DDMAdapter(
        run_id=RUN_ID,
        config=DDMConfig(),
    )

    ph = PageHinkleyAdapter(
        run_id=RUN_ID,
        config=PageHinkleyConfig(),
    )

    Fake = (
        AlwaysDriftingFakeRiver
        if drifting
        else NeverDriftingFakeRiver
    )

    # Test-only deterministic River substitution.
    adwin._river_detector = Fake()
    ddm._river_detector = Fake()
    ph._river_detector = Fake()

    # Timeout/reset must construct fresh deterministic
    # River instances too.
    adwin._build_river_detector = Fake
    ddm._build_river_detector = Fake
    ph._build_river_detector = Fake

    runner = StreamingEnsembleRunner(
        run_id=RUN_ID,
        adwin_adapter=adwin,
        ddm_adapter=ddm,
        page_hinkley_adapter=ph,
        window_manager=manager,
    )

    return (
        runner,
        adwin,
        ddm,
        ph,
        manager,
    )


# ============================================================
# External boundary alignment
# ============================================================

def test_first_window_uses_manager_boundary():
    runner, *_ = _make_runner(
        drifting=False,
    )

    runner.process_observation(
        _obs(
            sample_index=0,
            window_id="WIN_000001",
        )
    )

    assert (
        runner.current_window_id
        == "WIN_000001"
    )

    assert runner.last_sample_index == 0


def test_second_window_uses_manager_boundary():
    runner, *_ = _make_runner(
        drifting=False,
    )

    runner.process_observation(
        _obs(
            sample_index=9,
            window_id="WIN_000001",
        )
    )

    first_vote = runner.close_window()

    assert (
        first_vote.reported_window_id
        == "WIN_000001"
    )

    runner.process_observation(
        _obs(
            sample_index=10,
            window_id="WIN_000002",
        )
    )

    assert (
        runner.current_window_id
        == "WIN_000002"
    )


# ============================================================
# Empty / double close behavior
# ============================================================

def test_empty_window_cannot_be_closed():
    runner, *_ = _make_runner(
        drifting=False,
    )

    with pytest.raises(ValueError):
        runner.close_window()


def test_same_window_cannot_be_closed_twice():
    runner, *_ = _make_runner(
        drifting=False,
    )

    runner.process_observation(
        _obs(
            sample_index=0,
            window_id="WIN_000001",
        )
    )

    runner.close_window()

    with pytest.raises(ValueError):
        runner.close_window()


# ============================================================
# Detector lifecycle across external window boundary
# ============================================================

def test_latched_detectors_timeout_on_next_external_window():
    runner, adwin, ddm, ph, _ = _make_runner(
        drifting=True,
    )

    # First sample of first window causes all three detectors
    # to trigger and latch.
    results = runner.process_observation(
        _obs(
            sample_index=0,
            window_id="WIN_000001",
            error=1,
        )
    )

    assert all(
        result.detection is True
        for result in results
    )

    assert adwin.state is DetectorState.LATCHED
    assert ddm.state is DetectorState.LATCHED
    assert ph.state is DetectorState.LATCHED

    first_vote = runner.close_window()

    assert first_vote.error_vote_count == 3
    assert first_vote.error_drift is True

    # First observation of the next external window causes
    # each latched adapter to resolve its unresolved timeout.
    results_next = runner.process_observation(
        _obs(
            sample_index=10,
            window_id="WIN_000002",
            error=0,
        )
    )

    assert all(
        result.metadata.get(
            "unresolved_timeout",
            False,
        )
        is True
        for result in results_next
    )


def test_timeout_triggering_observation_is_processed_once():
    runner, adwin, ddm, ph, _ = _make_runner(
        drifting=True,
    )

    runner.process_observation(
        _obs(
            sample_index=0,
            window_id="WIN_000001",
            error=1,
        )
    )

    runner.close_window()

    old_adwin = adwin._river_detector
    old_ddm = ddm._river_detector
    old_ph = ph._river_detector

    results = runner.process_observation(
        _obs(
            sample_index=10,
            window_id="WIN_000002",
            error=0,
        )
    )

    new_adwin = adwin._river_detector
    new_ddm = ddm._river_detector
    new_ph = ph._river_detector

    assert new_adwin is not old_adwin
    assert new_ddm is not old_ddm
    assert new_ph is not old_ph

    assert new_adwin.calls == 1
    assert new_ddm.calls == 1
    assert new_ph.calls == 1

    assert all(
        result.metadata[
            "unresolved_timeout"
        ]
        is True
        for result in results
    )


def test_detector_reset_is_not_done_by_close_window():
    runner, adwin, ddm, ph, _ = _make_runner(
        drifting=False,
    )

    runner.process_observation(
        _obs(
            sample_index=0,
            window_id="WIN_000001",
        )
    )

    runner.close_window()

    assert adwin._river_detector.calls == 1
    assert ddm._river_detector.calls == 1
    assert ph._river_detector.calls == 1

    assert adwin.reset is not None
    assert ddm.reset is not None
    assert ph.reset is not None


# ============================================================
# New-window processing after timeout
# ============================================================

def test_new_window_can_be_processed_after_close():
    runner, *_ = _make_runner(
        drifting=False,
    )

    runner.process_observation(
        _obs(
            sample_index=0,
            window_id="WIN_000001",
        )
    )

    runner.close_window()

    results = runner.process_observation(
        _obs(
            sample_index=10,
            window_id="WIN_000002",
        )
    )

    assert len(results) == 3
    assert runner.current_window_id == "WIN_000002"
    assert runner.last_sample_index == 10


def test_window_level_vote_is_independent_between_windows():
    runner, adwin, ddm, ph, _ = _make_runner(
        drifting=True,
    )

    runner.process_observation(
        _obs(
            sample_index=0,
            window_id="WIN_000001",
            error=1,
        )
    )

    first_vote = runner.close_window()

    assert first_vote.error_vote_count == 3
    assert first_vote.error_drift is True

    # Replace current detector sources with non-drifting fakes
    # for the second window's new detector instances.
    adwin._build_river_detector = NeverDriftingFakeRiver
    ddm._build_river_detector = NeverDriftingFakeRiver
    ph._build_river_detector = NeverDriftingFakeRiver

    runner.process_observation(
        _obs(
            sample_index=10,
            window_id="WIN_000002",
            error=0,
        )
    )

    second_vote = runner.close_window()

    assert second_vote.error_vote_count == 0
    assert second_vote.error_drift is False


# ============================================================
# Global sample ordering survives window boundaries
# ============================================================

def test_sample_index_remains_monotonic_across_windows():
    runner, *_ = _make_runner(
        drifting=False,
    )

    runner.process_observation(
        _obs(
            sample_index=9,
            window_id="WIN_000001",
        )
    )

    runner.close_window()

    runner.process_observation(
        _obs(
            sample_index=11,
            window_id="WIN_000002",
        )
    )

    assert runner.last_sample_index == 11

    # sample_index=10 IS a valid index for WIN_000002 per the
    # WindowManager (window 2 spans indices 10-19), so the
    # WindowManager cross-check will NOT reject this pairing.
    # This isolates the adapter-level strictly-increasing-index
    # guard as the actual cause of rejection below -- not a
    # WindowManager window/index mismatch, which is what the
    # original version of this test accidentally exercised
    # instead of true monotonicity.
    with pytest.raises(ValueError):
        runner.process_observation(
            _obs(
                sample_index=10,
                window_id="WIN_000002",
            )
        )
