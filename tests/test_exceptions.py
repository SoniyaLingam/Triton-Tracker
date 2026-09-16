"""Tests for Tracker's Day 6 exception hierarchy and integration points."""

from pathlib import Path

import pytest

from src.config import PipelineMode, load_pipeline_config
from src.exceptions import (
    ConfigurationError,
    DataProcessingError,
    PipelineError,
    ResourceError,
    TrackerError,
)
from src.pipeline import Pipeline, Step
from src.utils import read_csv_rows_with_retry, retry


def test_tracker_error_can_be_raised_and_caught():
    """The base exception provides one catch point for application code."""
    with pytest.raises(TrackerError, match="Tracker failure"):
        raise TrackerError("Tracker failure")


@pytest.mark.parametrize(
    "error_type",
    [ConfigurationError, DataProcessingError, ResourceError, PipelineError],
)
def test_specific_errors_are_tracker_errors(error_type: type[TrackerError]):
    """Every specific project exception inherits from TrackerError."""
    with pytest.raises(TrackerError):
        raise error_type("specific failure")


def test_invalid_configuration_is_translated_to_configuration_error(tmp_path: Path):
    """Application configuration loading keeps Pydantic details as the cause."""
    with pytest.raises(ConfigurationError) as raised:
        load_pipeline_config(
            {
                "data_path": tmp_path,
                "batch_size": 0,
                "mode": PipelineMode.TRAIN,
                "threshold": 80,
                "feature_columns": ["score"],
            }
        )

    assert raised.value.__cause__ is not None


def test_invalid_csv_header_raises_data_processing_error(tmp_path: Path):
    """A blank CSV header is invalid data rather than a missing resource."""
    csv_path = tmp_path / "bad-header.csv"
    csv_path.write_text(",name\n1,Alice\n", encoding="utf-8")

    with pytest.raises(DataProcessingError, match="invalid header"):
        read_csv_rows_with_retry(csv_path)


def test_missing_csv_resource_preserves_os_error_cause(tmp_path: Path):
    """File-open failures are translated without losing the original error."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(ResourceError, match="Unable to read CSV resource") as raised:
        read_csv_rows_with_retry(missing_path)

    assert isinstance(raised.value.__cause__, FileNotFoundError)


class BrokenStep(Step):
    """A realistic custom operation that receives unsuitable batch data."""

    def process(self, data: object) -> object:
        """Attempt a mapping operation that fails for a non-mapping value."""
        return data["required"]  # type: ignore[index]


def test_pipeline_step_failure_is_translated_to_pipeline_error():
    """The orchestration layer adds the failing step's useful context."""
    with pytest.raises(PipelineError, match="BrokenStep") as raised:
        Pipeline([BrokenStep()]).run([])

    assert isinstance(raised.value.__cause__, TypeError)


def test_invalid_pipeline_operation_is_a_pipeline_error():
    """A non-Step object is rejected with the project exception category."""
    with pytest.raises(PipelineError, match="must implement"):
        Pipeline(["not-a-step"])  # type: ignore[list-item]


def test_retry_exhaustion_keeps_the_final_tracker_error():
    """Retry logs failed attempts but does not swallow the final exception."""
    attempts = {"count": 0}

    @retry(max_attempts=2)
    def unavailable_resource() -> None:
        attempts["count"] += 1
        raise ResourceError("resource is still unavailable")

    with pytest.raises(ResourceError, match="still unavailable"):
        unavailable_resource()

    assert attempts["count"] == 2


def test_valid_csv_read_still_succeeds(tmp_path: Path):
    """A valid CSV remains readable through the D5 retry workflow."""
    csv_path = tmp_path / "valid.csv"
    csv_path.write_text("id,name\n1,Alice\n", encoding="utf-8")

    assert read_csv_rows_with_retry(csv_path) == [{"id": "1", "name": "Alice"}]
