from __future__ import annotations

from typing import Dict, Tuple

from src.contracts.window_config import (
    WindowManagerConfig,
)

from src.contracts.window_boundary import (
    WindowBoundary,
)


class WindowManager:
    """
    Deterministic external experiment-window manager.

    Responsibilities:
    - create deterministic external windows
    - map sample indices to windows
    - retrieve windows by display ID
    - preserve run isolation
    - apply partial-window policy

    Timestamp information is intentionally outside this class.
    """

    def __init__(
        self,
        config: WindowManagerConfig,
    ) -> None:

        self._config = config

        self._windows: list[WindowBoundary] = []

        self._by_window_id: Dict[
            str,
            WindowBoundary,
        ] = {}

        self._build_windows()

        if not self._windows:
            raise ValueError(
                "WindowManager produced zero windows. "
                "This can happen when total_samples < "
                "window_size and drop_last_partial=True."
            )

    @property
    def config(self) -> WindowManagerConfig:
        return self._config

    @property
    def windows(self) -> Tuple[WindowBoundary, ...]:
        return tuple(self._windows)

    def _build_windows(self) -> None:

        start = self._config.start_index

        total_end = (
            self._config.start_index
            + self._config.total_samples
        )

        window_number = 1

        while start < total_end:

            end = min(
                start + self._config.window_size,
                total_end,
            )

            sample_count = end - start

            is_partial = (
                sample_count
                < self._config.window_size
            )

            if (
                is_partial
                and self._config.drop_last_partial
            ):
                break

            window_id = (
                f"{self._config.window_id_prefix}"
                f"{window_number:06d}"
            )

            internal_key = (
                f"{self._config.run_id}::{window_id}"
            )

            boundary = WindowBoundary(
                run_id=self._config.run_id,
                window_id=window_id,
                start_index=start,
                end_index=end - 1,
                sample_count=sample_count,
                internal_key=internal_key,
                is_partial=is_partial,
            )

            self._windows.append(
                boundary
            )

            self._by_window_id[
                window_id
            ] = boundary

            start = end
            window_number += 1

    def get_window(
        self,
        window_id: str,
    ) -> WindowBoundary:

        try:
            return self._by_window_id[
                window_id
            ]

        except KeyError as exc:
            raise KeyError(
                f"Unknown window_id: {window_id!r}. "
                "It may not exist, or it may have "
                "been dropped because "
                "drop_last_partial=True."
            ) from exc

    def window_for_index(
        self,
        sample_index: int,
    ) -> WindowBoundary:

        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
        ):
            raise TypeError(
                "sample_index must be an integer."
            )

        if (
            sample_index
            < self._config.start_index
        ):
            raise IndexError(
                "sample_index is before the "
                "configured start_index."
            )

        final_index = (
            self._config.start_index
            + self._config.total_samples
            - 1
        )

        if sample_index > final_index:
            raise IndexError(
                "sample_index is outside the "
                "configured sample range."
            )

        offset = (
            sample_index
            - self._config.start_index
        )

        window_number = (
            offset // self._config.window_size
        ) + 1

        window_id = (
            f"{self._config.window_id_prefix}"
            f"{window_number:06d}"
        )

        if window_id not in self._by_window_id:
            raise IndexError(
                f"sample_index {sample_index} belongs "
                f"to window {window_id!r}, which was "
                "dropped because "
                "drop_last_partial=True."
            )

        return self._by_window_id[
            window_id
        ]

    def has_window(
        self,
        window_id: str,
    ) -> bool:
        return (
            window_id
            in self._by_window_id
        )
