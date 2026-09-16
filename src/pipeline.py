"""Pipeline and step abstractions for CSV data processing.

This module demonstrates composition over inheritance in a beginner-friendly
way. The ``Pipeline`` depends only on the ``Step`` interface, not on any
concrete step implementation.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Union

from src.exceptions import PipelineError, TrackerError
from src.utils import timeit

logger = logging.getLogger(__name__)


def _record_count(data: Any) -> Optional[int]:
    """Return a record count for common collection inputs without consuming iterators."""
    if isinstance(data, (list, tuple, set, dict)):
        return len(data)
    return None


class Step(ABC):
    """Abstract contract for every processing step in a pipeline."""

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Apply one processing operation to the provided data."""


class CSVDataCleaner(Step):
    """Trim whitespace and normalize basic CSV field values."""

    def process(self, data: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Clean each row in a list of dictionaries."""
        cleaned_rows: List[dict[str, Any]] = []

        for row in data:
            cleaned_row: dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, str):
                    cleaned_row[key] = value.strip()
                else:
                    cleaned_row[key] = value

            if "score" in cleaned_row and cleaned_row["score"] is not None:
                try:
                    cleaned_row["score"] = int(str(cleaned_row["score"]).strip())
                except (TypeError, ValueError):
                    pass

            cleaned_rows.append(cleaned_row)

        return cleaned_rows


class DataFilter(Step):
    """Filter rows where a numeric field meets a minimum threshold."""

    def __init__(self, field_name: str, minimum: Union[int, float]):
        self.field_name = field_name
        self.minimum = minimum

    def process(self, data: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Return rows whose field value is greater than or equal to minimum."""
        filtered_rows: List[dict[str, Any]] = []

        for row in data:
            value = row.get(self.field_name)
            if value is None:
                continue
            try:
                if float(value) >= self.minimum:
                    filtered_rows.append(row)
            except (TypeError, ValueError):
                continue

        return filtered_rows


class HighScoreFilter(Step):
    """A second filter implementation that demonstrates runtime swapping."""

    def __init__(self, field_name: str, minimum: Union[int, float]):
        self.field_name = field_name
        self.minimum = minimum

    def process(self, data: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Keep only records with very high scores."""
        filtered_rows: List[dict[str, Any]] = []

        for row in data:
            value = row.get(self.field_name)
            if value is None:
                continue
            try:
                if float(value) >= self.minimum:
                    filtered_rows.append(row)
            except (TypeError, ValueError):
                continue

        return filtered_rows


class DataTransformer(Step):
    """Normalize values and add a standard status label after filtering."""

    def process(self, data: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Convert names to uppercase and annotate each row with a status."""
        transformed_rows: List[dict[str, Any]] = []

        for row in data:
            transformed = dict(row)
            transformed["name"] = str(transformed.get("name", "")).upper()
            transformed["status"] = "eligible"
            transformed_rows.append(transformed)

        return transformed_rows


class AddProcessedFlagStep(Step):
    """Add a new processing flag without changing the Pipeline implementation."""

    def process(self, data: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Attach a processed flag to each row."""
        return [{**row, "processed": True} for row in data]


class CSVStatisticsStep(Step):
    """Example extension step that summarizes record counts for a batch."""

    def process(self, data: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Add a summary field describing the dataset size."""
        summary: dict[str, Any] = {"record_count": len(data), "processed": True}
        if data:
            summary["first_id"] = data[0].get("id")
        return [{**row, "summary": summary.copy()} for row in data]


class Pipeline:
    """Execute a sequence of interchangeable steps in order."""

    def __init__(self, steps: Iterable[Step]):
        """Store the steps that make up the processing workflow."""
        validated_steps: List[Step] = []
        for step in steps:
            if not isinstance(step, Step):
                raise PipelineError("Every pipeline step must implement the Step interface.")
            validated_steps.append(step)
        self.steps = validated_steps

    @timeit
    def run(self, data: Any) -> Any:
        """Run each step sequentially and return the final transformed data."""
        current = data
        started_at = time.perf_counter()
        logger.info(
            "Pipeline started.",
            extra={
                "event": "pipeline_started",
                "records": _record_count(current),
                "step_count": len(self.steps),
            },
        )
        for step in self.steps:
            step_name = type(step).__name__
            input_records = _record_count(current)
            step_started_at = time.perf_counter()
            logger.info(
                "Pipeline step started.",
                extra={"event": "step_started", "step": step_name, "input_records": input_records},
            )
            try:
                current = step.process(current)
            except TrackerError:
                logger.exception(
                    "Pipeline step failed.", extra={"event": "step_failed", "step": step_name}
                )
                logger.exception(
                    "Pipeline failed.", extra={"event": "pipeline_failed", "step": step_name}
                )
                raise
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                logger.exception(
                    "Pipeline step failed.", extra={"event": "step_failed", "step": step_name}
                )
                raise PipelineError(
                    f"Pipeline step {type(step).__name__} could not process the supplied data."
                ) from exc
            logger.info(
                "Pipeline step completed.",
                extra={
                    "event": "step_completed",
                    "step": step_name,
                    "duration_seconds": round(time.perf_counter() - step_started_at, 6),
                    "input_records": input_records,
                    "output_records": _record_count(current),
                },
            )
        logger.info(
            "Pipeline completed.",
            extra={
                "event": "pipeline_completed",
                "duration_seconds": round(time.perf_counter() - started_at, 6),
                "records": _record_count(current),
            },
        )
        return current


__all__ = [
    "AddProcessedFlagStep",
    "CSVDataCleaner",
    "CSVStatisticsStep",
    "DataFilter",
    "DataTransformer",
    "HighScoreFilter",
    "Pipeline",
    "Step",
]
