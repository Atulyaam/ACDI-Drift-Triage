from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class FeatureDistributionObservation:
    """
    Contract describing one feature-level distribution comparison.

    The observation contains references to the relevant windows.
    Raw feature arrays are intentionally NOT stored here.
    """

    run_id: str
    feature_name: str
    reference_window_id: str
    current_window_id: str
    timestamp: datetime
    # metadata excluded from eq/hash: dicts are unhashable, and frozen
    # dataclasses generate __hash__ from compared fields by default.
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if (
            not isinstance(self.feature_name, str)
            or not self.feature_name.strip()
        ):
            raise ValueError(
                "feature_name must be a non-empty string."
            )

        if (
            not isinstance(self.reference_window_id, str)
            or not self.reference_window_id.strip()
        ):
            raise ValueError(
                "reference_window_id must be a non-empty string."
            )

        if (
            not isinstance(self.current_window_id, str)
            or not self.current_window_id.strip()
        ):
            raise ValueError(
                "current_window_id must be a non-empty string."
            )

        if self.reference_window_id == self.current_window_id:
            raise ValueError(
                "reference_window_id and current_window_id must not be "
                "the same window (self-comparison is not meaningful)."
            )

        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                "timestamp must be a datetime instance."
            )

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        if not all(isinstance(k, str) for k in self.metadata.keys()):
            raise TypeError(
                "metadata keys must all be strings."
            )


from typing import Literal


ConfidenceDefinition = Literal[
    "max_class_prob",
    "full_vector",
]


@dataclass(frozen=True)
class ConfidenceDistributionObservation:
    """
    Contract describing a confidence/uncertainty comparison.

    Raw probability values are intentionally not stored here.
    They are retrieved from the WindowStore during computation.
    """

    run_id: str
    reference_window_id: str
    current_window_id: str
    confidence_definition: ConfidenceDefinition
    timestamp: datetime
    # metadata excluded from eq/hash: dicts are unhashable, and frozen
    # dataclasses generate __hash__ from compared fields by default.
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if (
            not isinstance(self.reference_window_id, str)
            or not self.reference_window_id.strip()
        ):
            raise ValueError(
                "reference_window_id must be a non-empty string."
            )

        if (
            not isinstance(self.current_window_id, str)
            or not self.current_window_id.strip()
        ):
            raise ValueError(
                "current_window_id must be a non-empty string."
            )

        if self.reference_window_id == self.current_window_id:
            raise ValueError(
                "reference_window_id and current_window_id must not be "
                "the same window (self-comparison is not meaningful)."
            )

        if self.confidence_definition not in (
            "max_class_prob",
            "full_vector",
        ):
            raise ValueError(
                "confidence_definition must be "
                "'max_class_prob' or 'full_vector'."
            )

        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                "timestamp must be a datetime instance."
            )

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        if not all(isinstance(k, str) for k in self.metadata.keys()):
            raise TypeError(
                "metadata keys must all be strings."
            )


@dataclass(frozen=True)
class PredictionErrorObservation:
    """
    Contract for the shared prediction-error stream.

    error_stream:
        0 = correct prediction
        1 = incorrect prediction

    Raw stream values are kept only for the active observation.
    Persistent detector results must not duplicate the stream.
    """

    run_id: str
    window_id: str
    error_stream: tuple[int, ...]
    sample_count: int
    start_index: int
    end_index: int
    timestamp: datetime
    # metadata excluded from eq/hash: dicts are unhashable, and frozen
    # dataclasses generate __hash__ from compared fields by default.
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if not isinstance(self.window_id, str) or not self.window_id.strip():
            raise ValueError(
                "window_id must be a non-empty string."
            )

        if not isinstance(self.error_stream, tuple):
            raise TypeError(
                "error_stream must be a tuple of integers."
            )

        for value in self.error_stream:
            # bool is a subclass of int in Python, and 1.0 == 1 for
            # floats, so "value not in (0, 1)" alone would silently
            # accept True/False/1.0/0.0. We reject anything that is
            # not EXACTLY a plain int of value 0 or 1.
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    "error_stream elements must be plain int "
                    "(0 or 1), not bool/float/other types. "
                    f"Got: {value!r} ({type(value).__name__})"
                )
            if value not in (0, 1):
                raise ValueError(
                    "error_stream may contain only 0 or 1."
                )

        if not isinstance(self.sample_count, int) or isinstance(self.sample_count, bool):
            raise TypeError(
                "sample_count must be an integer."
            )

        if self.sample_count < 0:
            raise ValueError(
                "sample_count cannot be negative."
            )

        if self.sample_count != len(self.error_stream):
            raise ValueError(
                "sample_count must equal len(error_stream)."
            )

        if not isinstance(self.start_index, int) or isinstance(self.start_index, bool):
            raise TypeError(
                "start_index must be an integer."
            )

        if not isinstance(self.end_index, int) or isinstance(self.end_index, bool):
            raise TypeError(
                "end_index must be an integer."
            )

        if self.start_index < 0 or self.end_index < 0:
            raise ValueError(
                "start_index and end_index cannot be negative."
            )

        if self.sample_count > 0:
            expected_end = (
                self.start_index
                + self.sample_count
                - 1
            )

            if self.end_index != expected_end:
                raise ValueError(
                    "end_index must equal "
                    "start_index + sample_count - 1."
                )

        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                "timestamp must be a datetime instance."
            )

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        if not all(isinstance(k, str) for k in self.metadata.keys()):
            raise TypeError(
                "metadata keys must all be strings."
            )
