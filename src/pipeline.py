"""Pipeline and step abstractions for CSV data processing.

This module demonstrates composition over inheritance in a beginner-friendly
way. The ``Pipeline`` depends only on the ``Step`` interface, not on any
concrete step implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List


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

    def __init__(self, field_name: str, minimum: int | float):
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

    def __init__(self, field_name: str, minimum: int | float):
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
        summary = {"record_count": len(data), "processed": True}
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
                raise TypeError("Every pipeline step must implement the Step interface.")
            validated_steps.append(step)
        self.steps = validated_steps

    def run(self, data: Any) -> Any:
        """Run each step sequentially and return the final transformed data."""
        current = data
        for step in self.steps:
            current = step.process(current)
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
