from datetime import datetime, timezone

import pytest

from src.contracts.windows import WindowContext


def valid_window() -> WindowContext:
    return WindowContext(
        run_id="TEST_RUN_001",
        window_id="WIN_001",
        start_index=100,
        end_index=199,
        start_timestamp=datetime(
            2026, 8, 22, 10, 0, tzinfo=timezone.utc
        ),
        end_timestamp=datetime(
            2026, 8, 22, 10, 5, tzinfo=timezone.utc
        ),
        sample_count=100,
    )


def test_valid_window_context():
    window = valid_window()

    assert window.window_id == "WIN_001"
    assert window.sample_count == 100


def test_negative_start_index_rejected():
    with pytest.raises(ValueError):
        WindowContext(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            start_index=-1,
            end_index=99,
            start_timestamp=datetime.now(timezone.utc),
            end_timestamp=datetime.now(timezone.utc),
            sample_count=101,
        )


def test_end_before_start_rejected():
    with pytest.raises(ValueError):
        WindowContext(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            start_index=100,
            end_index=99,
            start_timestamp=datetime.now(timezone.utc),
            end_timestamp=datetime.now(timezone.utc),
            sample_count=0,
        )


def test_sample_count_mismatch_rejected():
    with pytest.raises(ValueError):
        WindowContext(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            start_index=100,
            end_index=199,
            start_timestamp=datetime.now(timezone.utc),
            end_timestamp=datetime.now(timezone.utc),
            sample_count=50,
        )


def test_naive_start_timestamp_rejected():
    with pytest.raises(ValueError):
        WindowContext(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            start_index=100,
            end_index=199,
            start_timestamp=datetime.now(),
            end_timestamp=datetime.now(timezone.utc),
            sample_count=100,
        )


def test_naive_end_timestamp_rejected():
    with pytest.raises(ValueError):
        WindowContext(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            start_index=100,
            end_index=199,
            start_timestamp=datetime.now(timezone.utc),
            end_timestamp=datetime.now(),
            sample_count=100,
        )


def test_end_timestamp_before_start_rejected():
    with pytest.raises(ValueError):
        WindowContext(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            start_index=100,
            end_index=199,
            start_timestamp=datetime(
                2026, 8, 22, 10, 5, tzinfo=timezone.utc
            ),
            end_timestamp=datetime(
                2026, 8, 22, 10, 0, tzinfo=timezone.utc
            ),
            sample_count=100,
        )


def test_metadata_must_be_dict():
    with pytest.raises(TypeError):
        WindowContext(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            start_index=100,
            end_index=199,
            start_timestamp=datetime.now(timezone.utc),
            end_timestamp=datetime.now(timezone.utc),
            sample_count=100,
            metadata="invalid",
        )


def test_bool_start_index_rejected():
    # bool is a subclass of int; without an explicit check,
    # True/False would silently pass as valid indices.
    with pytest.raises(TypeError):
        WindowContext(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            start_index=True,
            end_index=199,
            start_timestamp=datetime.now(timezone.utc),
            end_timestamp=datetime.now(timezone.utc),
            sample_count=100,
        )


def test_bool_sample_count_rejected():
    with pytest.raises(TypeError):
        WindowContext(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            start_index=0,
            end_index=0,
            start_timestamp=datetime.now(timezone.utc),
            end_timestamp=datetime.now(timezone.utc),
            sample_count=True,
        )


def test_window_context_is_hashable():
    window = valid_window()
    # Should not raise TypeError even though metadata is a dict
    hash(window)


def test_window_metadata_non_string_keys_rejected():
    with pytest.raises(TypeError):
        WindowContext(
            run_id="TEST_RUN_001",
            window_id="WIN_001",
            start_index=100,
            end_index=199,
            start_timestamp=datetime.now(timezone.utc),
            end_timestamp=datetime.now(timezone.utc),
            sample_count=100,
            metadata={17: "invalid_key"},
        )
