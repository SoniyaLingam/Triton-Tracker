"""Tests for Tracker's centralized structured logging."""

import json
import logging
from pathlib import Path

import pytest

from src.exceptions import PipelineError
from src.logging_config import configure_logging
from src.pipeline import CSVDataCleaner, DataFilter, Pipeline, Step
from src.utils import CSVResourceManager, timeit


def read_log_records(log_file: Path) -> list[dict[str, object]]:
    """Read newline-delimited JSON records emitted by Tracker."""
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]


def test_configuration_writes_structured_json_to_a_file(tmp_path: Path):
    """The central logger writes valid JSON with consistent core fields."""
    log_file = tmp_path / "tracker.jsonl"
    logger = configure_logging(log_file, level="INFO", console=False)

    logger.info("Test event.", extra={"event": "test_event", "records": 2})

    record = read_log_records(log_file)[0]
    assert record["event"] == "test_event"
    assert record["records"] == 2
    assert {"timestamp", "level", "logger", "message"} <= record.keys()


def test_reconfiguration_does_not_duplicate_handlers(tmp_path: Path):
    """Repeated central configuration replaces prior Tracker handlers."""
    log_file = tmp_path / "tracker.jsonl"
    logger = configure_logging(log_file, console=False)
    logger = configure_logging(log_file, console=False)

    assert len(logger.handlers) == 1


def test_pipeline_logs_lifecycle_and_step_metrics(tmp_path: Path):
    """A successful pipeline emits start, step, completion, and timing events."""
    log_file = tmp_path / "pipeline.jsonl"
    configure_logging(log_file, level="INFO", console=False)

    result = Pipeline([CSVDataCleaner(), DataFilter("score", 80)]).run(
        [{"name": " Alice ", "score": "90"}]
    )

    events = [record["event"] for record in read_log_records(log_file)]
    assert result[0]["name"] == "Alice"
    assert "pipeline_started" in events
    assert events.count("step_started") == 2
    assert events.count("step_completed") == 2
    assert "pipeline_completed" in events
    assert "function_timed" in events


def test_timeit_logs_execution_duration(tmp_path: Path):
    """The D5 timing decorator emits a structured duration field."""
    log_file = tmp_path / "timing.jsonl"
    configure_logging(log_file, level="INFO", console=False)

    @timeit
    def add(first: int, second: int) -> int:
        return first + second

    assert add(2, 3) == 5
    record = read_log_records(log_file)[0]
    assert record["event"] == "function_timed"
    assert isinstance(record["duration_seconds"], float)


def test_resource_open_and_close_are_logged(tmp_path: Path):
    """CSV resource lifecycle events are emitted without logging row data."""
    csv_path = tmp_path / "data.csv"
    log_file = tmp_path / "resource.jsonl"
    csv_path.write_text("id\n1\n", encoding="utf-8")
    configure_logging(log_file, level="INFO", console=False)

    with CSVResourceManager(csv_path) as handle:
        assert handle.read()

    events = [record["event"] for record in read_log_records(log_file)]
    assert {"resource_opening", "resource_opened", "resource_closed"} <= set(events)


class FailingLoggingStep(Step):
    """Trigger an expected pipeline processing error for log verification."""

    def process(self, data: object) -> object:
        """Use invalid list indexing to simulate unsuitable operation input."""
        return data["required"]  # type: ignore[index]


def test_pipeline_failure_logs_a_traceback(tmp_path: Path):
    """Pipeline failures include error event metadata and the full traceback."""
    log_file = tmp_path / "failure.jsonl"
    configure_logging(log_file, level=logging.INFO, console=False)

    with pytest.raises(PipelineError):
        Pipeline([FailingLoggingStep()]).run([])

    records = read_log_records(log_file)
    failed_records = [record for record in records if record["event"] == "step_failed"]
    assert failed_records
    assert "Traceback" in str(failed_records[0]["exception"])
    assert failed_records[0]["exception_type"] == "TypeError"
