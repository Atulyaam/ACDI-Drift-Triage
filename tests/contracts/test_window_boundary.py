import pytest

from src.contracts.window_boundary import (
    WindowBoundary,
)


def valid_boundary() -> WindowBoundary:

    return WindowBoundary(
        run_id="RUN_001",
        window_id="WIN_000001",
        start_index=0,
        end_index=999,
        sample_count=1000,
        internal_key="RUN_001::WIN_000001",
        is_partial=False,
    )


def test_valid_boundary():

    boundary = valid_boundary()

    assert boundary.sample_count == 1000
    assert boundary.is_partial is False


def test_end_before_start_rejected():

    with pytest.raises(ValueError):

        WindowBoundary(
            run_id="RUN_001",
            window_id="WIN_000001",
            start_index=100,
            end_index=50,
            sample_count=51,
            internal_key="RUN_001::WIN_000001",
            is_partial=False,
        )


def test_sample_count_mismatch_rejected():

    with pytest.raises(ValueError):

        WindowBoundary(
            run_id="RUN_001",
            window_id="WIN_000001",
            start_index=0,
            end_index=999,
            sample_count=500,
            internal_key="RUN_001::WIN_000001",
            is_partial=False,
        )


def test_internal_key_mismatch_rejected():

    with pytest.raises(ValueError):

        WindowBoundary(
            run_id="RUN_001",
            window_id="WIN_000001",
            start_index=0,
            end_index=999,
            sample_count=1000,
            internal_key="RUN_002::WIN_000001",
            is_partial=False,
        )


def test_boundary_is_hashable():

    boundary = valid_boundary()

    hash(boundary)


def test_boundary_usable_in_set():

    b1 = valid_boundary()
    b2 = valid_boundary()

    assert {b1, b2} == {b1}
