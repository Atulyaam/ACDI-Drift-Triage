
import pytest

from src.monitors.batch.feature_summary import (
    FeatureDriftSummary,
)

from src.monitors.batch.confidence import (
    ConfidenceDriftResult,
)

from src.monitors.streaming.error_vote import (
    ErrorDriftVoteResult,
)

from src.triage.signal_report import (
    SignalReport,
    build_signal_report,
)


RUN_ID = "RUN_001"
REFERENCE = "WIN_000001"
CURRENT = "WIN_000002"


def _feature_summary(
    significant=0,
    reference=REFERENCE,
    current=CURRENT,
):
    names = tuple(
        f"feature_{index}"
        for index in range(significant)
    )

    return FeatureDriftSummary(
        reference_window_id=reference,
        current_window_id=current,
        q=0.05,
        n_features_total=5,
        n_features_significant=significant,
        drifted_feature_names=names,
    )


def _confidence_result(
    significant=False,
    reference=REFERENCE,
    current=CURRENT,
):
    return ConfidenceDriftResult(
        reference_window_id=reference,
        current_window_id=current,
        d_statistic=0.2,
        p_value=0.01 if significant else 0.5,
        n_ref=20,
        n_cur=20,
        significant=significant,
        alpha=0.05,
        entropy_constant_reference=False,
        entropy_constant_current=False,
        metadata={
            "signal": "predictive_entropy"
        },
    )


def _error_vote(
    adwin=False,
    ddm=False,
    ph=False,
    window=CURRENT,
):
    return ErrorDriftVoteResult(
        run_id=RUN_ID,
        sample_index=999,
        reported_window_id=window,
        adwin_drift=adwin,
        ddm_drift=ddm,
        page_hinkley_drift=ph,
    )


def test_no_signal_produces_clean_report():
    report = build_signal_report(
        _feature_summary(0),
        _confidence_result(False),
        _error_vote(),
    )

    assert report.run_id == RUN_ID
    assert report.reference_window_id == REFERENCE
    assert report.current_window_id == CURRENT

    assert report.feature_drift is False
    assert report.confidence_drift is False
    assert report.error_drift is False
    assert report.error_vote_count == 0


def test_feature_drift_is_derived_from_significant_feature_count():
    report = build_signal_report(
        _feature_summary(2),
        _confidence_result(False),
        _error_vote(),
    )

    assert report.feature_drift is True


def test_confidence_drift_copies_significance():
    report = build_signal_report(
        _feature_summary(0),
        _confidence_result(True),
        _error_vote(),
    )

    assert report.confidence_drift is True


def test_error_drift_and_vote_count_are_copied():
    report = build_signal_report(
        _feature_summary(0),
        _confidence_result(False),
        _error_vote(
            adwin=True,
            ddm=True,
        ),
    )

    assert report.error_drift is True
    assert report.error_vote_count == 2


def test_three_of_three_is_preserved():
    report = build_signal_report(
        _feature_summary(1),
        _confidence_result(True),
        _error_vote(
            adwin=True,
            ddm=True,
            ph=True,
        ),
    )

    assert report.feature_drift is True
    assert report.confidence_drift is True
    assert report.error_drift is True
    assert report.error_vote_count == 3


def test_reference_window_mismatch_is_rejected():
    with pytest.raises(ValueError):
        build_signal_report(
            _feature_summary(
                reference="WIN_ABC"
            ),
            _confidence_result(),
            _error_vote(),
        )


def test_confidence_current_window_mismatch_is_rejected():
    with pytest.raises(ValueError):
        build_signal_report(
            _feature_summary(),
            _confidence_result(
                current="WIN_000003"
            ),
            _error_vote(),
        )


def test_error_vote_current_window_mismatch_is_rejected():
    with pytest.raises(ValueError):
        build_signal_report(
            _feature_summary(),
            _confidence_result(),
            _error_vote(
                window="WIN_000003"
            ),
        )


def test_feature_summary_type_is_required():
    with pytest.raises(TypeError):
        build_signal_report(
            object(),
            _confidence_result(),
            _error_vote(),
        )


def test_confidence_result_type_is_required():
    with pytest.raises(TypeError):
        build_signal_report(
            _feature_summary(),
            object(),
            _error_vote(),
        )


def test_error_vote_type_is_required():
    with pytest.raises(TypeError):
        build_signal_report(
            _feature_summary(),
            _confidence_result(),
            object(),
        )


def test_malformed_error_vote_object_is_rejected_before_defensive_check():
    class MalformedErrorVote:
        run_id = RUN_ID
        sample_index = 999
        reported_window_id = CURRENT
        error_vote_count = 4
        error_drift = True

    with pytest.raises(TypeError):
        build_signal_report(
            _feature_summary(),
            _confidence_result(),
            MalformedErrorVote(),
        )


def test_defensive_vote_count_validation_rejects_corrupted_high_value():
    vote = _error_vote(
        adwin=True,
        ddm=True,
    )

    object.__setattr__(
        vote,
        "error_vote_count",
        99,
    )

    with pytest.raises(ValueError):
        build_signal_report(
            _feature_summary(),
            _confidence_result(),
            vote,
        )


def test_defensive_vote_count_validation_rejects_corrupted_negative_value():
    vote = _error_vote()

    object.__setattr__(
        vote,
        "error_vote_count",
        -1,
    )

    with pytest.raises(ValueError):
        build_signal_report(
            _feature_summary(),
            _confidence_result(),
            vote,
        )


def test_signal_report_requires_strict_boolean_fields():
    with pytest.raises(TypeError):
        SignalReport(
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            feature_drift=1,
            confidence_drift=False,
            error_drift=False,
            error_vote_count=0,
        )


def test_signal_report_rejects_bool_vote_count():
    with pytest.raises(TypeError):
        SignalReport(
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            feature_drift=False,
            confidence_drift=False,
            error_drift=False,
            error_vote_count=True,
        )


def test_signal_report_rejects_vote_count_above_three():
    with pytest.raises(ValueError):
        SignalReport(
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            feature_drift=False,
            confidence_drift=False,
            error_drift=False,
            error_vote_count=4,
        )


def test_signal_report_rejects_negative_vote_count():
    with pytest.raises(ValueError):
        SignalReport(
            run_id=RUN_ID,
            reference_window_id=REFERENCE,
            current_window_id=CURRENT,
            feature_drift=False,
            confidence_drift=False,
            error_drift=False,
            error_vote_count=-1,
        )


def test_signal_report_rejects_same_reference_and_current_window():
    with pytest.raises(ValueError):
        SignalReport(
            run_id=RUN_ID,
            reference_window_id=CURRENT,
            current_window_id=CURRENT,
            feature_drift=False,
            confidence_drift=False,
            error_drift=False,
            error_vote_count=0,
        )


def test_source_signal_names_are_recorded():
    report = build_signal_report(
        _feature_summary(),
        _confidence_result(),
        _error_vote(),
    )

    assert (
        report.metadata["source_signal_names"]
        == [
            "feature_drift_summary",
            "confidence_drift_result",
            "error_drift_vote_result",
        ]
    )


def test_metadata_uses_dict_default():
    report = SignalReport(
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        feature_drift=False,
        confidence_drift=False,
        error_drift=False,
        error_vote_count=0,
    )

    assert isinstance(
        report.metadata,
        dict,
    )


def test_signal_report_is_frozen():
    report = SignalReport(
        run_id=RUN_ID,
        reference_window_id=REFERENCE,
        current_window_id=CURRENT,
        feature_drift=False,
        confidence_drift=False,
        error_drift=False,
        error_vote_count=0,
    )

    with pytest.raises(Exception):
        report.error_drift = True


def test_signal_report_contains_no_action_field():
    report = build_signal_report(
        _feature_summary(),
        _confidence_result(),
        _error_vote(),
    )

    assert not hasattr(
        report,
        "action",
    )

    assert not hasattr(
        report,
        "triage_action",
    )
