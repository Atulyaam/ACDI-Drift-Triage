import pytest

from src.contracts.window_config import (
    WindowManagerConfig,
)


def valid_config() -> WindowManagerConfig:
    return WindowManagerConfig(
        run_id="RUN_001",
        window_size=1000,
        total_samples=2500,
        start_index=0,
        window_id_prefix="WIN_",
    )


def test_valid_config():
    config = valid_config()

    assert config.run_id == "RUN_001"
    assert config.window_size == 1000
    assert config.total_samples == 2500
    assert config.start_index == 0
    assert config.window_id_prefix == "WIN_"
    assert config.drop_last_partial is False


def test_drop_last_partial_can_be_enabled():
    config = WindowManagerConfig(
        run_id="RUN_001",
        window_size=1000,
        total_samples=2500,
        start_index=0,
        window_id_prefix="WIN_",
        drop_last_partial=True,
    )

    assert config.drop_last_partial is True


def test_window_size_must_be_positive():
    with pytest.raises(ValueError):
        WindowManagerConfig(
            run_id="RUN_001",
            window_size=0,
            total_samples=2500,
            start_index=0,
            window_id_prefix="WIN_",
        )


def test_window_size_rejects_bool():
    with pytest.raises(TypeError):
        WindowManagerConfig(
            run_id="RUN_001",
            window_size=True,
            total_samples=2500,
            start_index=0,
            window_id_prefix="WIN_",
        )


def test_total_samples_must_be_positive():
    with pytest.raises(ValueError):
        WindowManagerConfig(
            run_id="RUN_001",
            window_size=1000,
            total_samples=0,
            start_index=0,
            window_id_prefix="WIN_",
        )


def test_total_samples_rejects_bool():
    with pytest.raises(TypeError):
        WindowManagerConfig(
            run_id="RUN_001",
            window_size=1000,
            total_samples=True,
            start_index=0,
            window_id_prefix="WIN_",
        )


def test_start_index_cannot_be_negative():
    with pytest.raises(ValueError):
        WindowManagerConfig(
            run_id="RUN_001",
            window_size=1000,
            total_samples=2500,
            start_index=-1,
            window_id_prefix="WIN_",
        )


def test_start_index_rejects_bool():
    with pytest.raises(TypeError):
        WindowManagerConfig(
            run_id="RUN_001",
            window_size=1000,
            total_samples=2500,
            start_index=False,
            window_id_prefix="WIN_",
        )


def test_start_index_must_be_less_than_total_samples():
    with pytest.raises(ValueError):
        WindowManagerConfig(
            run_id="RUN_001",
            window_size=1000,
            total_samples=2500,
            start_index=2500,
            window_id_prefix="WIN_",
        )


def test_valid_last_start_index():
    config = WindowManagerConfig(
        run_id="RUN_001",
        window_size=1000,
        total_samples=2500,
        start_index=2499,
        window_id_prefix="WIN_",
    )

    assert config.start_index == 2499


def test_invalid_prefix_rejected():
    with pytest.raises(ValueError):
        WindowManagerConfig(
            run_id="RUN_001",
            window_size=1000,
            total_samples=2500,
            start_index=0,
            window_id_prefix="WIN-",
        )


def test_prefix_with_spaces_rejected():
    with pytest.raises(ValueError):
        WindowManagerConfig(
            run_id="RUN_001",
            window_size=1000,
            total_samples=2500,
            start_index=0,
            window_id_prefix="WIN PREFIX",
        )


def test_empty_prefix_rejected():
    with pytest.raises(ValueError):
        WindowManagerConfig(
            run_id="RUN_001",
            window_size=1000,
            total_samples=2500,
            start_index=0,
            window_id_prefix="",
        )


def test_whitespace_only_run_id_rejected():
    with pytest.raises(ValueError):
        WindowManagerConfig(
            run_id="   ",
            window_size=1000,
            total_samples=2500,
            start_index=0,
            window_id_prefix="WIN_",
        )


def test_drop_last_partial_rejects_non_bool():
    with pytest.raises(TypeError):
        WindowManagerConfig(
            run_id="RUN_001",
            window_size=1000,
            total_samples=2500,
            start_index=0,
            window_id_prefix="WIN_",
            drop_last_partial=1,
        )


def test_config_is_immutable():
    config = valid_config()

    with pytest.raises(Exception):
        config.window_size = 500


def test_config_is_deterministic():
    config_a = valid_config()
    config_b = valid_config()

    assert config_a == config_b
