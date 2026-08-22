import pytest

from src.monitors.batch.fdr import FDRResult
from src.monitors.batch.feature_summary import (
    FeatureDriftSummary,
    build_feature_drift_summary,
)


def _make_result(
    feature_name,
    significant,
    q=0.05,
    ref="REF_001",
    cur="WIN_001",
):
    return FDRResult(
        feature_name=feature_name,
        reference_window_id=ref,
        current_window_id=cur,
        raw_p_value=0.01 if significant else 0.50,
        adjusted_p_value=0.02 if significant else 0.50,
        significant=significant,
        q=q,
    )


def test_build_summary_counts_significant_features():
    results = (
        _make_result("f1", True),
        _make_result("f2", False),
        _make_result("f3", True),
    )

    summary = build_feature_drift_summary(
        results
    )

    assert summary.n_features_total == 3
    assert summary.n_features_significant == 2
    assert summary.proportion_drifted == pytest.approx(
        2 / 3
    )


def test_build_summary_preserves_input_order():
    results = (
        _make_result("f3", True),
        _make_result("f1", False),
        _make_result("f2", True),
    )

    summary = build_feature_drift_summary(
        results
    )

    assert summary.drifted_feature_names == (
        "f3",
        "f2",
    )


def test_summary_rejects_empty_results():
    with pytest.raises(ValueError):
        build_feature_drift_summary(())


def test_summary_rejects_duplicate_features():
    results = (
        _make_result("f1", True),
        _make_result("f1", False),
    )

    with pytest.raises(ValueError):
        build_feature_drift_summary(
            results
        )


def test_summary_rejects_mixed_reference_windows():
    results = (
        _make_result(
            "f1",
            True,
            ref="REF_001",
        ),
        _make_result(
            "f2",
            False,
            ref="REF_002",
        ),
    )

    with pytest.raises(ValueError):
        build_feature_drift_summary(
            results
        )


def test_summary_rejects_mixed_current_windows():
    results = (
        _make_result(
            "f1",
            True,
            cur="WIN_001",
        ),
        _make_result(
            "f2",
            False,
            cur="WIN_002",
        ),
    )

    with pytest.raises(ValueError):
        build_feature_drift_summary(
            results
        )


def test_summary_rejects_mixed_q():
    results = (
        _make_result(
            "f1",
            True,
            q=0.05,
        ),
        _make_result(
            "f2",
            False,
            q=0.10,
        ),
    )

    with pytest.raises(ValueError):
        build_feature_drift_summary(
            results
        )


def test_summary_metadata_is_hash_safe():
    summary = FeatureDriftSummary(
        reference_window_id="REF_001",
        current_window_id="WIN_001",
        q=0.05,
        n_features_total=2,
        n_features_significant=1,
        drifted_feature_names=("f1",),
        metadata={"source": "fdr"},
    )

    hash(summary)


def test_summary_rejects_invalid_counts():
    with pytest.raises(ValueError):
        FeatureDriftSummary(
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            q=0.05,
            n_features_total=2,
            n_features_significant=3,
            drifted_feature_names=(
                "f1",
                "f2",
                "f3",
            ),
        )


def test_summary_rejects_count_name_mismatch():
    with pytest.raises(ValueError):
        FeatureDriftSummary(
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            q=0.05,
            n_features_total=3,
            n_features_significant=2,
            drifted_feature_names=("f1",),
        )


def test_summary_rejects_same_window_pair():
    with pytest.raises(ValueError):
        FeatureDriftSummary(
            reference_window_id="WIN_001",
            current_window_id="WIN_001",
            q=0.05,
            n_features_total=1,
            n_features_significant=1,
            drifted_feature_names=("f1",),
        )


def test_summary_rejects_duplicate_drifted_feature_names():
    with pytest.raises(ValueError):
        FeatureDriftSummary(
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            q=0.05,
            n_features_total=2,
            n_features_significant=2,
            drifted_feature_names=(
                "f1",
                "f1",
            ),
        )
