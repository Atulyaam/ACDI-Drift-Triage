
import pytest

from src.monitors.batch.fdr import (
    FDRConfig,
    FDRResult,
    apply_bh_fdr,
)
from src.monitors.batch.ks import KSResult


# ================================================================
# ASSUMPTION: KSResult accepts these fields directly as kwargs.
# If your real KSResult signature differs, this helper is the
# ONLY place that needs to change -- adjust field names here.
# ================================================================

def _make_ks_result(
    feature_name,
    p_value,
    reference_window_id="REF_001",
    current_window_id="WIN_001",
    d_statistic=0.5,
    n_ref=50,
    n_cur=50,
    is_constant_reference=False,
    is_constant_current=False,
):
    return KSResult(
        feature_name=feature_name,
        reference_window_id=reference_window_id,
        current_window_id=current_window_id,
        d_statistic=d_statistic,
        p_value=p_value,
        n_ref=n_ref,
        n_cur=n_cur,
        is_constant_reference=is_constant_reference,
        is_constant_current=is_constant_current,
        metadata={"method": "asymp", "alternative": "two-sided"},
    )


# ================================================================
# EXISTING CONTRACT TESTS (18 -- unchanged from STEP 10.3.5.1,
# minus the obsolete NotImplementedError test, since apply_bh_fdr
# now has a real implementation)
# ================================================================

def test_fdr_config_default_q():
    config = FDRConfig()
    assert config.q == 0.05


def test_fdr_config_accepts_valid_q():
    config = FDRConfig(q=0.10)
    assert config.q == 0.10


def test_fdr_config_rejects_zero():
    with pytest.raises(ValueError):
        FDRConfig(q=0.0)


def test_fdr_config_rejects_one():
    with pytest.raises(ValueError):
        FDRConfig(q=1.0)


def test_fdr_config_rejects_negative():
    with pytest.raises(ValueError):
        FDRConfig(q=-0.1)


def test_fdr_config_rejects_above_one():
    with pytest.raises(ValueError):
        FDRConfig(q=1.1)


def test_fdr_config_rejects_bool():
    with pytest.raises(TypeError):
        FDRConfig(q=True)


def test_fdr_config_rejects_nan():
    with pytest.raises(ValueError):
        FDRConfig(q=float("nan"))


def test_fdr_config_rejects_infinity():
    with pytest.raises(ValueError):
        FDRConfig(q=float("inf"))


def test_valid_fdr_result():
    result = FDRResult(
        feature_name="flow_duration",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
        raw_p_value=0.01,
        adjusted_p_value=0.02,
        significant=True,
        q=0.05,
    )
    assert result.feature_name == "flow_duration"
    assert result.raw_p_value == 0.01
    assert result.adjusted_p_value == 0.02
    assert result.significant is True


def test_fdr_result_rejects_same_window_pair():
    with pytest.raises(ValueError):
        FDRResult(
            feature_name="flow_duration",
            reference_window_id="WIN_001",
            current_window_id="WIN_001",
            raw_p_value=0.01,
            adjusted_p_value=0.02,
            significant=True,
            q=0.05,
        )


def test_fdr_result_rejects_invalid_raw_p():
    with pytest.raises(ValueError):
        FDRResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            raw_p_value=1.5,
            adjusted_p_value=0.20,
            significant=False,
            q=0.05,
        )


def test_fdr_result_rejects_invalid_adjusted_p():
    with pytest.raises(ValueError):
        FDRResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            raw_p_value=0.10,
            adjusted_p_value=-0.01,
            significant=False,
            q=0.05,
        )


def test_fdr_result_rejects_bool_raw_p():
    with pytest.raises(TypeError):
        FDRResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            raw_p_value=True,
            adjusted_p_value=0.20,
            significant=False,
            q=0.05,
        )


def test_fdr_result_rejects_bool_adjusted_p():
    with pytest.raises(TypeError):
        FDRResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            raw_p_value=0.10,
            adjusted_p_value=False,
            significant=False,
            q=0.05,
        )


def test_fdr_result_rejects_bool_significant():
    with pytest.raises(TypeError):
        FDRResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            raw_p_value=0.10,
            adjusted_p_value=0.20,
            significant=1,
            q=0.05,
        )


def test_fdr_result_rejects_invalid_q():
    with pytest.raises(ValueError):
        FDRResult(
            feature_name="flow_duration",
            reference_window_id="REF_001",
            current_window_id="WIN_001",
            raw_p_value=0.10,
            adjusted_p_value=0.20,
            significant=False,
            q=1.0,
        )


def test_fdr_result_metadata_is_hash_safe():
    result = FDRResult(
        feature_name="flow_duration",
        reference_window_id="REF_001",
        current_window_id="WIN_001",
        raw_p_value=0.01,
        adjusted_p_value=0.02,
        significant=True,
        q=0.05,
        metadata={"method": "benjamini_hochberg"},
    )
    hash(result)


# ================================================================
# NEW TESTS -- STEP 10.3.5.2 statistical / behavioral validation
# ================================================================

def test_bh_known_answer_matches_hand_computed_values():
    """
    p = [0.01, 0.02, 0.03, 0.50], m = 4
    Hand-computed BH adjusted p-values: [0.04, 0.04, 0.04, 0.50]
    This is the critical test that validates the rank formula
    itself, not just bounds.
    """
    ks_results = (
        _make_ks_result("f1", 0.01),
        _make_ks_result("f2", 0.02),
        _make_ks_result("f3", 0.03),
        _make_ks_result("f4", 0.50),
    )
    expected_features = ["f1", "f2", "f3", "f4"]

    results = apply_bh_fdr(
        ks_results=ks_results,
        expected_feature_names=expected_features,
        config=FDRConfig(q=0.05),
    )

    adjusted = [r.adjusted_p_value for r in results]

    assert adjusted == pytest.approx([0.04, 0.04, 0.04, 0.50], abs=1e-9)


def test_bh_tied_p_values_produce_equal_adjusted_p():
    """
    Three features all at p = 0.03, m = 3.
    Regardless of tie-break order in the internal sort, the
    reverse cumulative minimum must converge all three to the
    SAME adjusted p-value.
    """
    ks_results = (
        _make_ks_result("f1", 0.03),
        _make_ks_result("f2", 0.03),
        _make_ks_result("f3", 0.03),
    )
    expected_features = ["f1", "f2", "f3"]

    results = apply_bh_fdr(
        ks_results=ks_results,
        expected_feature_names=expected_features,
        config=FDRConfig(q=0.05),
    )

    adjusted = [r.adjusted_p_value for r in results]

    assert adjusted[0] == pytest.approx(adjusted[1], abs=1e-12)
    assert adjusted[1] == pytest.approx(adjusted[2], abs=1e-12)
    assert adjusted[0] == pytest.approx(0.03, abs=1e-9)


def test_bh_output_order_matches_input_ks_results_order():
    """
    Input order is deliberately NOT sorted by p-value.
    Output order must match ks_results input order exactly,
    not p-value-sorted order and not expected_feature_names order.
    """
    ks_results = (
        _make_ks_result("feature_c", 0.50),
        _make_ks_result("feature_a", 0.01),
        _make_ks_result("feature_b", 0.03),
    )
    expected_features = ["feature_a", "feature_b", "feature_c"]

    results = apply_bh_fdr(
        ks_results=ks_results,
        expected_feature_names=expected_features,
        config=FDRConfig(q=0.05),
    )

    assert [r.feature_name for r in results] == [
        "feature_c",
        "feature_a",
        "feature_b",
    ]


def test_bh_rejects_missing_expected_feature():
    ks_results = (
        _make_ks_result("f1", 0.01),
        _make_ks_result("f2", 0.02),
    )
    expected_features = ["f1", "f2", "f3"]  # f3 never produced

    with pytest.raises(ValueError):
        apply_bh_fdr(
            ks_results=ks_results,
            expected_feature_names=expected_features,
            config=FDRConfig(),
        )


def test_bh_rejects_extra_ks_result_feature():
    ks_results = (
        _make_ks_result("f1", 0.01),
        _make_ks_result("f2", 0.02),
        _make_ks_result("f3", 0.03),  # not expected
    )
    expected_features = ["f1", "f2"]

    with pytest.raises(ValueError):
        apply_bh_fdr(
            ks_results=ks_results,
            expected_feature_names=expected_features,
            config=FDRConfig(),
        )


def test_bh_rejects_same_count_but_mismatched_feature_set():
    """
    Count matches (3 == 3) but the actual feature names differ
    from expected -- must still be caught by set-equality check,
    not slip through because len() matched.
    """
    ks_results = (
        _make_ks_result("f1", 0.01),
        _make_ks_result("f2", 0.02),
        _make_ks_result("f_wrong", 0.03),
    )
    expected_features = ["f1", "f2", "f3"]

    with pytest.raises(ValueError):
        apply_bh_fdr(
            ks_results=ks_results,
            expected_feature_names=expected_features,
            config=FDRConfig(),
        )


def test_bh_rejects_duplicate_ks_result_feature_name():
    """
    Two KSResults with the same feature_name -- count may match
    len(expected_feature_names), but this must be rejected
    explicitly by the duplicate check, not silently pass via
    set-equality alone.
    """
    ks_results = (
        _make_ks_result("f1", 0.01),
        _make_ks_result("f1", 0.02),  # duplicate feature_name
    )
    expected_features = ["f1", "f2"]

    with pytest.raises(ValueError):
        apply_bh_fdr(
            ks_results=ks_results,
            expected_feature_names=expected_features,
            config=FDRConfig(),
        )


def test_bh_rejects_mixed_reference_windows():
    ks_results = (
        _make_ks_result("f1", 0.01, reference_window_id="REF_001"),
        _make_ks_result("f2", 0.02, reference_window_id="REF_002"),
    )
    expected_features = ["f1", "f2"]

    with pytest.raises(ValueError):
        apply_bh_fdr(
            ks_results=ks_results,
            expected_feature_names=expected_features,
            config=FDRConfig(),
        )


def test_bh_rejects_mixed_current_windows():
    ks_results = (
        _make_ks_result("f1", 0.01, current_window_id="WIN_001"),
        _make_ks_result("f2", 0.02, current_window_id="WIN_002"),
    )
    expected_features = ["f1", "f2"]

    with pytest.raises(ValueError):
        apply_bh_fdr(
            ks_results=ks_results,
            expected_feature_names=expected_features,
            config=FDRConfig(),
        )


def test_bh_rejects_invalid_p_value_defensively():
    """
    KSResult's own __post_init__ already forbids p outside [0,1],
    so we bypass it with object.__setattr__ AFTER valid
    construction (frozen dataclasses still allow this) to prove
    apply_bh_fdr's independent revalidation actually fires,
    rather than silently trusting the upstream object.
    """
    bad_result = _make_ks_result("f1", 0.02)
    object.__setattr__(bad_result, "p_value", 1.5)

    ks_results = (
        bad_result,
        _make_ks_result("f2", 0.02),
    )
    expected_features = ["f1", "f2"]

    with pytest.raises(ValueError):
        apply_bh_fdr(
            ks_results=ks_results,
            expected_feature_names=expected_features,
            config=FDRConfig(),
        )


def test_bh_rejects_same_reference_and_current_window_in_ks_results():
    """
    KSResult's own contract already forbids reference_window_id
    == current_window_id at construction, so this bypasses that
    guard the same way as the invalid-p-value test, to prove
    apply_bh_fdr's own defensive check (step 6) is not dead code.
    """
    result = _make_ks_result("f1", 0.02)
    object.__setattr__(result, "current_window_id", result.reference_window_id)

    ks_results = (result,)
    expected_features = ["f1"]

    with pytest.raises(ValueError):
        apply_bh_fdr(
            ks_results=ks_results,
            expected_feature_names=expected_features,
            config=FDRConfig(),
        )


def test_bh_rejects_empty_expected_feature_names():
    ks_results = (_make_ks_result("f1", 0.01),)

    with pytest.raises(ValueError):
        apply_bh_fdr(
            ks_results=ks_results,
            expected_feature_names=[],
            config=FDRConfig(),
        )


def test_bh_no_partial_output_on_failure():
    """
    One bad feature must abort the entire call -- there is no
    return value to inspect on failure, which is itself the proof
    that no partial tuple is ever produced.
    """
    ks_results = (
        _make_ks_result("f_good", 0.01),
        _make_ks_result("f_dup", 0.02),
        _make_ks_result("f_dup", 0.03),  # duplicate triggers failure
    )
    expected_features = ["f_good", "f_dup"]

    with pytest.raises(ValueError):
        apply_bh_fdr(
            ks_results=ks_results,
            expected_feature_names=expected_features,
            config=FDRConfig(),
        )


def test_bh_metadata_records_method_m_and_rank():
    ks_results = (
        _make_ks_result("f1", 0.01),
        _make_ks_result("f2", 0.02),
        _make_ks_result("f3", 0.03),
    )
    expected_features = ["f1", "f2", "f3"]

    results = apply_bh_fdr(
        ks_results=ks_results,
        expected_feature_names=expected_features,
        config=FDRConfig(q=0.05),
    )

    for result in results:
        assert result.metadata["method"] == "benjamini_hochberg"
        assert result.metadata["m"] == 3
        assert result.metadata["rank"] in (1, 2, 3)

    ranks = sorted(r.metadata["rank"] for r in results)
    assert ranks == [1, 2, 3]


def test_bh_significance_flag_matches_q_threshold():
    """
    Reuses the known-answer case: adjusted = [0.04, 0.04, 0.04, 0.50]
    with q = 0.05.
    0.04 <= 0.05  -> significant True
    0.50 >  0.05  -> significant False
    """
    ks_results = (
        _make_ks_result("f1", 0.01),
        _make_ks_result("f2", 0.02),
        _make_ks_result("f3", 0.03),
        _make_ks_result("f4", 0.50),
    )
    expected_features = ["f1", "f2", "f3", "f4"]

    results = apply_bh_fdr(
        ks_results=ks_results,
        expected_feature_names=expected_features,
        config=FDRConfig(q=0.05),
    )

    by_name = {r.feature_name: r for r in results}

    assert by_name["f1"].significant is True
    assert by_name["f2"].significant is True
    assert by_name["f3"].significant is True
    assert by_name["f4"].significant is False


def test_bh_rejects_non_ksresult_objects():
    """
    apply_bh_fdr must not silently accept duck-typed objects that
    merely look like a KSResult -- only real KSResult instances.
    """
    class FakeResult:
        feature_name = "f1"
        p_value = 0.02
        reference_window_id = "REF_001"
        current_window_id = "WIN_001"

    ks_results = (FakeResult(),)
    expected_features = ["f1"]

    with pytest.raises(TypeError):
        apply_bh_fdr(
            ks_results=ks_results,
            expected_feature_names=expected_features,
            config=FDRConfig(),
        )
