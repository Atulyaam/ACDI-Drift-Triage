
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.defense.actions import (
    DefenseActionRequest,
)


class ExecutionStatus(str, Enum):
    """
    Outcome of a defense-executor invocation.
    """

    EXECUTED = "executed"
    SKIPPED = "skipped"
    REJECTED = "rejected"
    ALREADY_EXECUTED = "already_executed"


@dataclass(frozen=True)
class DefenseExecutionResult:
    """
    Immutable result of one defense execution attempt.

    This is an execution report, not an execution engine.

    request_id is the idempotency key and must identify the
    request that produced this result.
    """

    request_id: str
    status: ExecutionStatus

    action: str
    run_id: str

    reference_window_id: str
    current_window_id: str

    message: str

    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.request_id,
                str,
            )
            or not self.request_id.strip()
        ):
            raise ValueError(
                "request_id must be a non-empty string."
            )

        if not isinstance(
            self.status,
            ExecutionStatus,
        ):
            raise TypeError(
                "status must be an ExecutionStatus."
            )

        if (
            not isinstance(
                self.action,
                str,
            )
            or not self.action.strip()
        ):
            raise ValueError(
                "action must be a non-empty string."
            )

        if (
            not isinstance(
                self.run_id,
                str,
            )
            or not self.run_id.strip()
        ):
            raise ValueError(
                "run_id must be a non-empty string."
            )

        if (
            not isinstance(
                self.reference_window_id,
                str,
            )
            or not self.reference_window_id.strip()
        ):
            raise ValueError(
                "reference_window_id must be a "
                "non-empty string."
            )

        if (
            not isinstance(
                self.current_window_id,
                str,
            )
            or not self.current_window_id.strip()
        ):
            raise ValueError(
                "current_window_id must be a "
                "non-empty string."
            )

        if (
            self.reference_window_id
            == self.current_window_id
        ):
            raise ValueError(
                "reference_window_id and current_window_id "
                "must be different."
            )

        if (
            not isinstance(
                self.message,
                str,
            )
            or not self.message.strip()
        ):
            raise ValueError(
                "message must be a non-empty string."
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary."
            )

        if not all(
            isinstance(key, str)
            for key in self.metadata
        ):
            raise TypeError(
                "metadata keys must all be strings."
            )


class DefenseExecutor(ABC):
    """
    Abstract defense execution boundary.

    Contract:

        DefenseActionRequest
                ↓
        DefenseExecutor.execute()
                ↓
        DefenseExecutionResult

    Implementations may later perform actual system/model
    mutations. This base interface itself performs none.

    request_id is the idempotency key.

    An executor must not silently execute the same request
    twice. Concrete implementations are responsible for
    enforcing idempotency.
    """

    @abstractmethod
    def execute(
        self,
        request: DefenseActionRequest,
    ) -> DefenseExecutionResult:
        """
        Execute one defense request or return a non-executed
        result.

        Concrete implementations must validate that the
        request is compatible with their action.
        """
        raise NotImplementedError
