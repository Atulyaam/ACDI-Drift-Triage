import pytest

from src.contracts.window_config import (
    WindowManagerConfig,
)

from src.windows.manager import (
    WindowManager,
)


def make_config(**overrides):

    defaults = dict(
        run_id="RUN_001",
        window_size=1000,
        total_samples=2500,
        start_index=0,
        window_id_prefix="WIN_",
        drop_last_partial=False,
    )

    defaults.update(overrides)

    return WindowManagerConfig(
        **defaults
    )


def test_creates_expected_number_of_windows():

    manager = WindowManager(
        make_config()
    )

    assert len(
        manager.windows
    ) == 3


def test_last_window_is_partial_when_not_dropped():

    manager = WindowManager(
        make_config()
    )

    last = manager.get_window(
        "WIN_000003"
    )

    assert last.sample_count == 500
    assert last.is_partial is True


def test_drop_last_partial_removes_final_window():

    manager = WindowManager(
        make_config(
            drop_last_partial=True
        )
    )

    assert len(
        manager.windows
    ) == 2

    assert not manager.has_window(
        "WIN_000003"
    )


def test_boundary_indices():

    manager = WindowManager(
        make_config()
    )

    assert (
        manager.window_for_index(0).window_id
        == "WIN_000001"
    )

    assert (
        manager.window_for_index(999).window_id
        == "WIN_000001"
    )

    assert (
        manager.window_for_index(1000).window_id
        == "WIN_000002"
    )

    assert (
        manager.window_for_index(1999).window_id
        == "WIN_000002"
    )

    assert (
        manager.window_for_index(2000).window_id
        == "WIN_000003"
    )

    assert (
        manager.window_for_index(2499).window_id
        == "WIN_000003"
    )


def test_index_before_start_rejected():

    manager = WindowManager(
        make_config()
    )

    with pytest.raises(IndexError):

        manager.window_for_index(-1)


def test_index_after_end_rejected():

    manager = WindowManager(
        make_config()
    )

    with pytest.raises(IndexError):

        manager.window_for_index(2500)


def test_index_rejects_bool():

    manager = WindowManager(
        make_config()
    )

    with pytest.raises(TypeError):

        manager.window_for_index(True)


def test_index_in_dropped_partial_window_raises():

    manager = WindowManager(
        make_config(
            drop_last_partial=True
        )
    )

    with pytest.raises(IndexError):

        manager.window_for_index(2000)


def test_unknown_window_id_raises():

    manager = WindowManager(
        make_config()
    )

    with pytest.raises(KeyError):

        manager.get_window(
            "WIN_999999"
        )


def test_run_isolation_via_internal_key():

    manager_a = WindowManager(
        make_config(
            run_id="RUN_001"
        )
    )

    manager_b = WindowManager(
        make_config(
            run_id="RUN_002"
        )
    )

    win_a = manager_a.get_window(
        "WIN_000001"
    )

    win_b = manager_b.get_window(
        "WIN_000001"
    )

    assert (
        win_a.internal_key
        != win_b.internal_key
    )

    assert (
        win_a.internal_key
        == "RUN_001::WIN_000001"
    )

    assert (
        win_b.internal_key
        == "RUN_002::WIN_000001"
    )


def test_zero_windows_config_raises():

    with pytest.raises(ValueError):

        WindowManager(
            make_config(
                window_size=1000,
                total_samples=500,
                drop_last_partial=True,
            )
        )
