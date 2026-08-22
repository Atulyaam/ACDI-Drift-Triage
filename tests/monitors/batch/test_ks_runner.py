
import numpy as np
import pytest

from src.monitors.batch.ks import KSResult
from src.monitors.batch.ks_runner import run_ks_for_features


REF_WIN = "REF_001"
CUR_WIN = "WIN_001"


def _uniform_data(
    features,
    size=50,
    offset=0.0,
):
    rng = np.random.default_rng(1)

    return {
        name: rng.normal(
            loc=offset,
            scale=1.0,
            size=size,
        )
        for name in features
    }


def test_runner_returns_results_in_requested_feature_order():
    features = [
        "feature_c",
        "feature_a",
        "feature_b",
    ]

    reference = _uniform_data(features)
    current = _uniform_data(
        features,
        offset=0.0,
    )

    results = run_ks_for_features(
        reference_data=reference,
        current_data=current,
        feature_names=features,
        reference_window_id=REF_WIN,
        current_window_id=CUR_WIN,
    )

    assert isinstance(results, tuple)
    assert len(results) == 3
    assert all(
        isinstance(result, KSResult)
        for result in results
    )

    assert [
        result.feature_name
        for result in results
    ] == features


def test_runner_rejects_duplicate_feature_names():
    features = [
        "feature_a",
        "feature_a",
    ]

    reference = _uniform_data(
        ["feature_a"]
    )

    current = _uniform_data(
        ["feature_a"]
    )

    with pytest.raises(
        ValueError,
        match="feature_a",
    ):
        run_ks_for_features(
            reference_data=reference,
            current_data=current,
            feature_names=features,
            reference_window_id=REF_WIN,
            current_window_id=CUR_WIN,
        )


def test_runner_rejects_feature_missing_from_reference():
    features = [
        "feature_a",
        "feature_b",
    ]

    reference = _uniform_data(
        ["feature_a"]
    )

    current = _uniform_data(
        features
    )

    with pytest.raises(
        ValueError,
        match="feature_b",
    ):
        run_ks_for_features(
            reference_data=reference,
            current_data=current,
            feature_names=features,
            reference_window_id=REF_WIN,
            current_window_id=CUR_WIN,
        )


def test_runner_rejects_feature_missing_from_current():
    features = [
        "feature_a",
        "feature_b",
    ]

    reference = _uniform_data(
        features
    )

    current = _uniform_data(
        ["feature_a"]
    )

    with pytest.raises(
        ValueError,
        match="feature_b",
    ):
        run_ks_for_features(
            reference_data=reference,
            current_data=current,
            feature_names=features,
            reference_window_id=REF_WIN,
            current_window_id=CUR_WIN,
        )


def test_runner_rejects_empty_feature_names():
    reference = _uniform_data(
        ["feature_a"]
    )

    current = _uniform_data(
        ["feature_a"]
    )

    with pytest.raises(ValueError):
        run_ks_for_features(
            reference_data=reference,
            current_data=current,
            feature_names=[],
            reference_window_id=REF_WIN,
            current_window_id=CUR_WIN,
        )


def test_runner_rejects_same_window_ids():
    features = ["feature_a"]

    reference = _uniform_data(features)
    current = _uniform_data(features)

    with pytest.raises(ValueError):
        run_ks_for_features(
            reference_data=reference,
            current_data=current,
            feature_names=features,
            reference_window_id="WIN_SAME",
            current_window_id="WIN_SAME",
        )


def test_runner_propagates_feature_name_for_non_numeric_failure():
    features = [
        "feature_good",
        "feature_bad",
    ]

    reference = {
        "feature_good": [
            1.0,
            2.0,
            3.0,
        ],
        "feature_bad": [
            "x",
            "y",
            "z",
        ],
    }

    current = {
        "feature_good": [
            1.5,
            2.5,
            3.5,
        ],
        "feature_bad": [
            "x",
            "y",
            "z",
        ],
    }

    with pytest.raises(
        TypeError,
        match="feature_bad",
    ):
        run_ks_for_features(
            reference_data=reference,
            current_data=current,
            feature_names=features,
            reference_window_id=REF_WIN,
            current_window_id=CUR_WIN,
        )


def test_runner_does_not_return_partial_results_on_failure():
    features = [
        "feature_good",
        "feature_nan",
    ]

    reference = {
        "feature_good": [
            1.0,
            2.0,
            3.0,
        ],
        "feature_nan": [
            1.0,
            np.nan,
            3.0,
        ],
    }

    current = {
        "feature_good": [
            1.5,
            2.5,
            3.5,
        ],
        "feature_nan": [
            1.0,
            2.0,
            3.0,
        ],
    }

    with pytest.raises(
        ValueError,
        match="feature_nan",
    ):
        run_ks_for_features(
            reference_data=reference,
            current_data=current,
            feature_names=features,
            reference_window_id=REF_WIN,
            current_window_id=CUR_WIN,
        )


def test_runner_passes_window_ids_to_every_result():
    features = [
        "feature_a",
        "feature_b",
    ]

    reference = _uniform_data(features)
    current = _uniform_data(features)

    results = run_ks_for_features(
        reference_data=reference,
        current_data=current,
        feature_names=features,
        reference_window_id=REF_WIN,
        current_window_id=CUR_WIN,
    )

    for result in results:
        assert result.reference_window_id == REF_WIN
        assert result.current_window_id == CUR_WIN


def test_runner_returns_raw_ksresults_without_aggregation_fields():
    features = ["feature_a"]

    reference = _uniform_data(features)
    current = _uniform_data(features)

    results = run_ks_for_features(
        reference_data=reference,
        current_data=current,
        feature_names=features,
        reference_window_id=REF_WIN,
        current_window_id=CUR_WIN,
    )

    result = results[0]

    assert isinstance(
        result,
        KSResult,
    )

    assert not hasattr(
        result,
        "drift_detected",
    )

    assert not hasattr(
        result,
        "adjusted_p_value",
    )
