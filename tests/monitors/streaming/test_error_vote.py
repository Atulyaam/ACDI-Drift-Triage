import pytest

from src.monitors.streaming.error_contracts import (
    DetectorState,
    DetectorUpdateResult,
)

from src.monitors.streaming.error_vote import (
    ErrorDriftVoteResult,
    compute_error_drift_vote,
)


RUN_ID = "RUN_001"
WINDOW_ID = "WIN_001"
SAMPLE_INDEX = 100


def _result(detector_name, detection):
    return DetectorUpdateResult(
        detector_name=detector_name,
        run_id=RUN_ID,
        sample_index=SAMPLE_INDEX,
        reported_window_id=WINDOW_ID,
        detection=detection,
        state=(
            DetectorState.DRIFT_DETECTED
            if detection
            else DetectorState.ACTIVE
        ),
    )


def test_valid_zero_vote():
    result = compute_error_drift_vote(
        _result("ADWIN", False),
        _result("DDM", False),
        _result("PAGE_HINKLEY", False),
    )

    assert result.adwin_drift is False
    assert result.ddm_drift is False
    assert result.page_hinkley_drift is False
    assert result.error_vote_count == 0
    assert result.error_drift is False


def test_one_vote_is_not_drift():
    result = compute_error_drift_vote(
        _result("ADWIN", True),
        _result("DDM", False),
        _result("PAGE_HINKLEY", False),
    )

    assert result.error_vote_count == 1
    assert result.error_drift is False


def test_two_votes_are_drift():
    result = compute_error_drift_vote(
        _result("ADWIN", True),
        _result("DDM", True),
        _result("PAGE_HINKLEY", False),
    )

    assert result.adwin_drift is True
    assert result.ddm_drift is True
    assert result.page_hinkley_drift is False
    assert result.error_vote_count == 2
    assert result.error_drift is True


def test_three_votes_are_drift():
    result = compute_error_drift_vote(
        _result("ADWIN", True),
        _result("DDM", True),
        _result("PAGE_HINKLEY", True),
    )

    assert result.error_vote_count == 3
    assert result.error_drift is True


def test_result_preserves_traceability():
    result = compute_error_drift_vote(
        _result("ADWIN", True),
        _result("DDM", False),
        _result("PAGE_HINKLEY", True),
    )

    assert result.run_id == RUN_ID
    assert result.sample_index == SAMPLE_INDEX
    assert result.reported_window_id == WINDOW_ID


def test_result_is_frozen():
    result = ErrorDriftVoteResult(
        run_id=RUN_ID,
        sample_index=SAMPLE_INDEX,
        reported_window_id=WINDOW_ID,
        adwin_drift=True,
        ddm_drift=False,
        page_hinkley_drift=False,
    )

    with pytest.raises(Exception):
        result.adwin_drift = False


def test_bool_fields_reject_integer():
    with pytest.raises(TypeError):
        ErrorDriftVoteResult(
            run_id=RUN_ID,
            sample_index=SAMPLE_INDEX,
            reported_window_id=WINDOW_ID,
            adwin_drift=1,
            ddm_drift=False,
            page_hinkley_drift=False,
        )


def test_bool_fields_reject_zero_integer():
    with pytest.raises(TypeError):
        ErrorDriftVoteResult(
            run_id=RUN_ID,
            sample_index=SAMPLE_INDEX,
            reported_window_id=WINDOW_ID,
            adwin_drift=False,
            ddm_drift=0,
            page_hinkley_drift=False,
        )


def test_bool_fields_reject_integer_for_page_hinkley():
    with pytest.raises(TypeError):
        ErrorDriftVoteResult(
            run_id=RUN_ID,
            sample_index=SAMPLE_INDEX,
            reported_window_id=WINDOW_ID,
            adwin_drift=False,
            ddm_drift=False,
            page_hinkley_drift=1,
        )


def test_error_vote_count_is_not_constructor_supplied():
    result = ErrorDriftVoteResult(
        run_id=RUN_ID,
        sample_index=SAMPLE_INDEX,
        reported_window_id=WINDOW_ID,
        adwin_drift=True,
        ddm_drift=False,
        page_hinkley_drift=False,
    )

    assert result.error_vote_count == 1

    with pytest.raises(TypeError):
        ErrorDriftVoteResult(
            run_id=RUN_ID,
            sample_index=SAMPLE_INDEX,
            reported_window_id=WINDOW_ID,
            adwin_drift=True,
            ddm_drift=False,
            page_hinkley_drift=False,
            error_vote_count=3,
        )


def test_error_drift_is_not_constructor_supplied():
    with pytest.raises(TypeError):
        ErrorDriftVoteResult(
            run_id=RUN_ID,
            sample_index=SAMPLE_INDEX,
            reported_window_id=WINDOW_ID,
            adwin_drift=True,
            ddm_drift=False,
            page_hinkley_drift=False,
            error_drift=True,
        )


def test_duplicate_adwin_identity_is_rejected():
    with pytest.raises(ValueError):
        compute_error_drift_vote(
            _result("ADWIN", True),
            _result("ADWIN", False),
            _result("PAGE_HINKLEY", True),
        )


def test_duplicate_ddm_identity_is_rejected():
    with pytest.raises(ValueError):
        compute_error_drift_vote(
            _result("ADWIN", True),
            _result("DDM", False),
            _result("DDM", True),
        )


def test_duplicate_page_hinkley_identity_is_rejected():
    with pytest.raises(ValueError):
        compute_error_drift_vote(
            _result("ADWIN", True),
            _result("PAGE_HINKLEY", False),
            _result("PAGE_HINKLEY", True),
        )


def test_positional_swap_is_rejected():
    with pytest.raises(ValueError):
        compute_error_drift_vote(
            _result("DDM", True),
            _result("ADWIN", False),
            _result("PAGE_HINKLEY", True),
        )


def test_invalid_adwin_detector_identity_is_rejected():
    with pytest.raises(ValueError):
        compute_error_drift_vote(
            _result("SOME_OTHER", True),
            _result("DDM", False),
            _result("PAGE_HINKLEY", True),
        )


def test_non_detector_update_result_is_rejected():
    with pytest.raises(TypeError):
        compute_error_drift_vote(
            object(),
            _result("DDM", False),
            _result("PAGE_HINKLEY", True),
        )


def test_run_id_mismatch_is_rejected():
    ddm_result = DetectorUpdateResult(
        detector_name="DDM",
        run_id="RUN_002",
        sample_index=SAMPLE_INDEX,
        reported_window_id=WINDOW_ID,
        detection=False,
        state=DetectorState.ACTIVE,
    )

    with pytest.raises(ValueError):
        compute_error_drift_vote(
            _result("ADWIN", True),
            ddm_result,
            _result("PAGE_HINKLEY", True),
        )


def test_sample_index_mismatch_is_rejected():
    ph_result = DetectorUpdateResult(
        detector_name="PAGE_HINKLEY",
        run_id=RUN_ID,
        sample_index=SAMPLE_INDEX + 1,
        reported_window_id=WINDOW_ID,
        detection=True,
        state=DetectorState.DRIFT_DETECTED,
    )

    with pytest.raises(ValueError):
        compute_error_drift_vote(
            _result("ADWIN", True),
            _result("DDM", True),
            ph_result,
        )


def test_window_mismatch_is_rejected():
    ph_result = DetectorUpdateResult(
        detector_name="PAGE_HINKLEY",
        run_id=RUN_ID,
        sample_index=SAMPLE_INDEX,
        reported_window_id="WIN_002",
        detection=True,
        state=DetectorState.DRIFT_DETECTED,
    )

    with pytest.raises(ValueError):
        compute_error_drift_vote(
            _result("ADWIN", True),
            _result("DDM", True),
            ph_result,
        )


def test_exact_two_of_three_boundary():
    cases = [
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ]

    for adwin, ddm, ph in cases:
        result = compute_error_drift_vote(
            _result("ADWIN", adwin),
            _result("DDM", ddm),
            _result("PAGE_HINKLEY", ph),
        )

        assert result.error_vote_count == 2
        assert result.error_drift is True


def test_exact_one_of_three_boundary():
    cases = [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ]

    for adwin, ddm, ph in cases:
        result = compute_error_drift_vote(
            _result("ADWIN", adwin),
            _result("DDM", ddm),
            _result("PAGE_HINKLEY", ph),
        )

        assert result.error_vote_count == 1
        assert result.error_drift is False


def test_derived_fields_always_match_flags():
    combinations = [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ]

    for adwin, ddm, ph in combinations:
        result = compute_error_drift_vote(
            _result("ADWIN", adwin),
            _result("DDM", ddm),
            _result("PAGE_HINKLEY", ph),
        )

        expected_count = int(adwin) + int(ddm) + int(ph)

        assert result.error_vote_count == expected_count
        assert result.error_drift == (expected_count >= 2)
