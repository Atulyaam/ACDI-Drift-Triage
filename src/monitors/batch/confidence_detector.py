from __future__ import annotations

from src.monitors.batch.confidence import (
    _DEFAULT_ALPHA,
    _validate_alpha,
    ConfidenceDriftResult,
)

from src.monitors.batch.entropy import (
    EntropyComputationInput,
)

from src.monitors.batch.ks import (
    KSComputationInput,
    compute_ks,
)


_ENTROPY_FEATURE_NAME = "prediction_entropy"


def compute_confidence_drift(
    computation_input: EntropyComputationInput,
    reference_window_id: str,
    current_window_id: str,
    alpha: float = _DEFAULT_ALPHA,
) -> ConfidenceDriftResult:
    """
    Compute predictive-confidence drift from binary predictive
    entropy using the existing KS engine.

    Pipeline:

        probability streams
                v
        EntropyComputationInput
                v
        per-sample entropy
                v
        KSComputationInput
                v
        existing compute_ks()
                v
        ConfidenceDriftResult

    The wrapper introduces no new statistical test.

    Scope limitation:
    Binary entropy captures uncertainty, not directional
    probability flips, because H(p) == H(1-p).
    """

    if not isinstance(computation_input, EntropyComputationInput):
        raise TypeError(
            "computation_input must be an EntropyComputationInput."
        )

    # --------------------------------------------------------
    # Validate alpha FIRST, using the real validator, before
    # any coercion happens. float(alpha) alone would silently
    # accept bool (float(True) == 1.0) and defer the failure
    # to a later, wrong-typed error inside ConfidenceDriftResult.
    # Using _validate_alpha here preserves fail-fast propagation
    # of the correct exception type (TypeError for bool, etc).
    # --------------------------------------------------------

    validated_alpha = _validate_alpha(alpha)

    # --------------------------------------------------------
    # Load validated entropy arrays.
    # This is the signal transformation layer.
    # --------------------------------------------------------

    reference_entropy = computation_input.reference_entropy
    current_entropy = computation_input.current_entropy

    # --------------------------------------------------------
    # CONNECTIVE STEP:
    # Convert entropy data into the EXISTING KS input contract.
    #
    # No new KS logic is implemented here.
    # --------------------------------------------------------

    ks_input = KSComputationInput(
        reference_values=reference_entropy,
        current_values=current_entropy,
        min_samples=computation_input.min_samples,
    )

    # --------------------------------------------------------
    # Reuse the existing KS statistical engine.
    # --------------------------------------------------------

    ks_result = compute_ks(
        computation_input=ks_input,
        feature_name=_ENTROPY_FEATURE_NAME,
        reference_window_id=reference_window_id,
        current_window_id=current_window_id,
    )

    # --------------------------------------------------------
    # Significance belongs to this semantic layer.
    # KS only provides p_value.
    # --------------------------------------------------------

    significant = ks_result.p_value <= validated_alpha

    # --------------------------------------------------------
    # Explicit field mapping:
    #
    # KSResult.is_constant_reference
    #     ->
    # ConfidenceDriftResult.entropy_constant_reference
    #
    # KSResult.is_constant_current
    #     ->
    # ConfidenceDriftResult.entropy_constant_current
    # --------------------------------------------------------

    return ConfidenceDriftResult(
        reference_window_id=ks_result.reference_window_id,
        current_window_id=ks_result.current_window_id,
        d_statistic=ks_result.d_statistic,
        p_value=ks_result.p_value,
        n_ref=ks_result.n_ref,
        n_cur=ks_result.n_cur,
        significant=significant,
        alpha=validated_alpha,
        entropy_constant_reference=ks_result.is_constant_reference,
        entropy_constant_current=ks_result.is_constant_current,
        metadata={
            "signal": "predictive_entropy",
        },
    )
