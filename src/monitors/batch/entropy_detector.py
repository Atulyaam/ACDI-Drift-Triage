from __future__ import annotations

from src.monitors.batch.entropy import (
    EntropyComputationInput,
)

from src.monitors.batch.ks import (
    KSComputationInput,
    KSResult,
    compute_ks,
)


def _build_entropy_ks_input(
    entropy_input: EntropyComputationInput,
) -> KSComputationInput:
    """
    Convert validated entropy input into the existing
    KS computation contract.

    No statistical test is implemented here.
    """

    if not isinstance(
        entropy_input,
        EntropyComputationInput,
    ):
        raise TypeError(
            "entropy_input must be an "
            "EntropyComputationInput."
        )

    return KSComputationInput(
        reference_values=(
            entropy_input.reference_entropy
        ),
        current_values=(
            entropy_input.current_entropy
        ),
        min_samples=entropy_input.min_samples,
    )


def compute_entropy_drift(
    computation_input: EntropyComputationInput,
    feature_name: str,
    reference_window_id: str,
    current_window_id: str,
) -> KSResult:
    """
    Compare reference and current predictive-entropy
    distributions using the EXISTING compute_ks() engine.

    Pipeline:

        probability streams
              ↓
        binary predictive entropy
              ↓
        KSComputationInput
              ↓
        existing compute_ks()
              ↓
        KSResult

    No duplicate KS implementation exists here.

    Scope limitation:
    Binary entropy measures predictive uncertainty only.
    H(p) == H(1-p), so pure confidence-direction flips
    are intentionally outside this signal's scope.
    """

    ks_input = _build_entropy_ks_input(
        computation_input
    )

    return compute_ks(
        computation_input=ks_input,
        feature_name=feature_name,
        reference_window_id=reference_window_id,
        current_window_id=current_window_id,
    )
