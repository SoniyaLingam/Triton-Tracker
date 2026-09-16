"""Validated configuration for the Tracker CSV processing pipeline.

The configuration model intentionally represents realistic settings for the
CSV/data-processing pipeline created in earlier internship days. Each value is
validated at runtime before it can be used by the pipeline.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class PipelineMode(str, Enum):
    """Supported pipeline execution modes."""

    TRAIN = "train"
    EVALUATE = "evaluate"
    INFERENCE = "inference"


class DeviceType(str, Enum):
    """Supported processing devices."""

    CPU = "cpu"
    GPU = "gpu"


class PipelineConfig(BaseModel):
    """Validated configuration for CSV pipeline execution."""

    model_config = ConfigDict(extra="forbid")

    data_path: Path
    batch_size: int = Field(..., gt=0)
    mode: PipelineMode
    device: DeviceType = DeviceType.CPU
    threshold: float = Field(..., ge=0, le=100)
    feature_columns: list[str]

    @field_validator("data_path")
    @classmethod
    def validate_data_path(cls, value: Path) -> Path:
        """The configured path must exist and point to a directory."""
        if not value.exists():
            raise ValueError("Path does not exist.")
        if not value.is_dir():
            raise ValueError("Path must be a directory.")
        return value

    @field_validator("feature_columns")
    @classmethod
    def validate_feature_columns(cls, value: list[str]) -> list[str]:
        """Feature columns must be non-empty and unique."""
        if not value:
            raise ValueError("Feature columns must not be empty.")

        cleaned_columns: list[str] = []
        seen: set[str] = set()

        for column in value:
            cleaned = column.strip()
            if not cleaned:
                raise ValueError("Feature columns cannot contain empty names.")
            if cleaned in seen:
                raise ValueError("Feature columns must be unique.")
            seen.add(cleaned)
            cleaned_columns.append(cleaned)

        return cleaned_columns

    def to_pipeline_kwargs(self) -> dict[str, Any]:
        """Return a dictionary suitable for pipeline setup code."""
        return {
            "batch_size": self.batch_size,
            "mode": self.mode,
            "device": self.device,
            "threshold": self.threshold,
            "feature_columns": self.feature_columns,
        }


def load_pipeline_config(values: Mapping[str, Any]) -> PipelineConfig:
    """Create a configuration and translate validation details for callers.

    ``PipelineConfig`` remains directly usable when a caller needs Pydantic's
    field-level ``ValidationError``. Application entry points can use this
    helper to receive Tracker's stable configuration error category instead.
    """
    try:
        return PipelineConfig(**values)
    except ValidationError as exc:
        logger.exception(
            "Pipeline configuration is invalid.", extra={"event": "configuration_failed"}
        )
        raise ConfigurationError("Invalid Tracker pipeline configuration.") from exc


__all__ = [
    "DeviceType",
    "PipelineConfig",
    "PipelineMode",
    "ValidationError",
    "load_pipeline_config",
]
