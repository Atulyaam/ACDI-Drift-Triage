from __future__ import annotations

from dataclasses import dataclass, field

from src.monitors.streaming.error_contracts import (
    DetectorUpdateResult,
)


_ADWIN_NAME = "ADWIN"
_DDM_NAME = "DDM"
_PAGE_HINKLEY_NAME = "PAGE_HINKLEY"


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class ErrorDriftVoteResult:
    """
    Canonical prediction-error 2-of-3 consensus result.

    Canonical architecture fields:

        run_id
        sample_index
        reported_window_id
        adwin_drift
        ddm_drift
        page_hinkley_drift
        error_vote_count
        error_drift

    error_vote_count and error_drift are derived fields.
    Callers cannot supply inconsistent values for them.
    """

    run_id: str
    sample_index: int
    reported_window_id: str

    adwin_drift: bool
    ddm_drift: bool
    page_hinkley_drift: bool

    error_vote_count: int = field(init=False)
    error_drift: bool = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string.")

        if not _is_strict_int(self.sample_index):
            raise TypeError(
                "sample_index must be an integer (not bool)."
            )

        if self.sample_index < 0:
            raise ValueError("sample_index cannot be negative.")

        if (
            not isinstance(self.reported_window_id, str)
            or not self.reported_window_id.strip()
        ):
            raise ValueError(
                "reported_window_id must be a non-empty string."
            )

        for field_name, value in (
            ("adwin_drift", self.adwin_drift),
            ("ddm_drift", self.ddm_drift),
            ("page_hinkley_drift", self.page_hinkley_drift),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a boolean.")

        vote_count = (
            int(self.adwin_drift)
            + int(self.ddm_drift)
            + int(self.page_hinkley_drift)
        )

        object.__setattr__(self, "error_vote_count", vote_count)
        object.__setattr__(self, "error_drift", vote_count >= 2)


def _validate_detector_result(
    result: DetectorUpdateResult,
    expected_detector_name: str,
    parameter_name: str,
) -> None:
    if not isinstance(result, DetectorUpdateResult):
        raise TypeError(
            f"{parameter_name} must be a DetectorUpdateResult."
        )

    # IMPORTANT:
    # Positional identity is enforced explicitly.
    #
    # This catches both:
    #   ADWIN + ADWIN + PH
    # and:
    #   DDM passed into adwin_result (a positional swap)
    #
    # A set-membership check alone would not catch swaps, since
    # {"ADWIN","DDM","PAGE_HINKLEY"} is still satisfied even if
    # two arguments are exchanged.
    if result.detector_name != expected_detector_name:
        raise ValueError(
            f"{parameter_name} must come from "
            f"{expected_detector_name!r}; received "
            f"{result.detector_name!r}."
        )


def compute_error_drift_vote(
    adwin_result: DetectorUpdateResult,
    ddm_result: DetectorUpdateResult,
    page_hinkley_result: DetectorUpdateResult,
) -> ErrorDriftVoteResult:
    """
    Compute the canonical 2-of-3 error-drift consensus.

    Positional detector identity is mandatory:

        adwin_result        -> ADWIN
        ddm_result          -> DDM
        page_hinkley_result -> PAGE_HINKLEY

    All three results must describe the same:

        run_id
        sample_index
        reported_window_id

    This function only aggregates detector results. It does not
    mutate detector state or perform resets.
    """

    _validate_detector_result(adwin_result, _ADWIN_NAME, "adwin_result")
    _validate_detector_result(ddm_result, _DDM_NAME, "ddm_result")
    _validate_detector_result(
        page_hinkley_result, _PAGE_HINKLEY_NAME, "page_hinkley_result"
    )

    if (
        adwin_result.run_id != ddm_result.run_id
        or adwin_result.run_id != page_hinkley_result.run_id
    ):
        raise ValueError(
            "All detector results must have the same run_id."
        )

    if (
        adwin_result.sample_index != ddm_result.sample_index
        or adwin_result.sample_index != page_hinkley_result.sample_index
    ):
        raise ValueError(
            "All detector results must have the same sample_index."
        )

    if (
        adwin_result.reported_window_id != ddm_result.reported_window_id
        or adwin_result.reported_window_id
        != page_hinkley_result.reported_window_id
    ):
        raise ValueError(
            "All detector results must have the same "
            "reported_window_id."
        )

    return ErrorDriftVoteResult(
        run_id=adwin_result.run_id,
        sample_index=adwin_result.sample_index,
        reported_window_id=adwin_result.reported_window_id,
        adwin_drift=bool(adwin_result.detection),
        ddm_drift=bool(ddm_result.detection),
        page_hinkley_drift=bool(page_hinkley_result.detection),
    )
