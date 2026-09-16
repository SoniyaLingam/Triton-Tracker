"""Pipeline execution reporting abstractions and structured log implementation."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Protocol


class PipelineReporter(Protocol):
    """Describe the lifecycle notifications required by pipeline orchestration."""

    def pipeline_started(self, data: Any, step_count: int) -> None:
        """Report that a pipeline run has started."""

    def step_started(self, step_name: str, data: Any) -> None:
        """Report that one pipeline step has started."""

    def step_completed(self, step_name: str, input_data: Any, output_data: Any) -> None:
        """Report that one pipeline step has completed."""

    def step_failed(self, step_name: str) -> None:
        """Report a step failure from inside an exception handler."""

    def pipeline_completed(self, data: Any) -> None:
        """Report that a pipeline run has completed."""


class StructuredPipelineReporter:
    """Emit JSON-ready pipeline lifecycle fields through Tracker's logger."""

    def __init__(self, logger_name: str = "src.pipeline"):
        self._logger = logging.getLogger(logger_name)
        self._pipeline_started_at: Optional[float] = None
        self._step_started_at: dict[str, float] = {}

    def pipeline_started(self, data: Any, step_count: int) -> None:
        """Log the start of a pipeline run."""
        self._pipeline_started_at = time.perf_counter()
        self._logger.info(
            "Pipeline started.",
            extra={
                "event": "pipeline_started",
                "records": record_count(data),
                "step_count": step_count,
            },
        )

    def step_started(self, step_name: str, data: Any) -> None:
        """Log a pipeline step start."""
        self._step_started_at[step_name] = time.perf_counter()
        self._logger.info(
            "Pipeline step started.",
            extra={"event": "step_started", "step": step_name, "input_records": record_count(data)},
        )

    def step_completed(self, step_name: str, input_data: Any, output_data: Any) -> None:
        """Log a pipeline step completion and record-count change."""
        self._logger.info(
            "Pipeline step completed.",
            extra={
                "event": "step_completed",
                "step": step_name,
                "duration_seconds": elapsed_since(self._step_started_at.pop(step_name, None)),
                "input_records": record_count(input_data),
                "output_records": record_count(output_data),
            },
        )

    def step_failed(self, step_name: str) -> None:
        """Log the active exception with its full traceback."""
        self._logger.exception(
            "Pipeline step failed.", extra={"event": "step_failed", "step": step_name}
        )

    def pipeline_completed(self, data: Any) -> None:
        """Log a successful pipeline completion."""
        self._logger.info(
            "Pipeline completed.",
            extra={
                "event": "pipeline_completed",
                "duration_seconds": elapsed_since(self._pipeline_started_at),
                "records": record_count(data),
            },
        )


def record_count(data: Any) -> Optional[int]:
    """Return a size without consuming iterators or other lazy inputs."""
    if isinstance(data, (list, tuple, set, dict)):
        return len(data)
    return None


def elapsed_since(started_at: Optional[float]) -> Optional[float]:
    """Return an elapsed duration when a matching lifecycle start exists."""
    if started_at is None:
        return None
    return round(time.perf_counter() - started_at, 6)


__all__ = ["PipelineReporter", "StructuredPipelineReporter", "record_count"]
