"""Tests for Day 5 reusable decorators, context manager, and caching helpers."""

from pathlib import Path

import pytest

from src.pipeline import CSVDataCleaner, DataFilter, Pipeline
from src.utils import (
    CSVResourceManager,
    normalize_feature_name,
    read_csv_rows_with_retry,
    retry,
    timeit,
)


@pytest.fixture
def sample_records():
    """Return a small dataset similar to CSV rows."""
    return [
        {"id": "1", "name": " Alice ", "score": "85", "subject": "Math"},
        {"id": "2", "name": "Bob", "score": "72", "subject": "Science"},
        {"id": "3", "name": "Charlie", "score": "93", "subject": "Math"},
        {"id": "4", "name": " Diana ", "score": "88", "subject": "Science"},
    ]


def test_timeit_preserves_metadata_and_result():
    """The timing decorator should preserve function identity and result."""

    @timeit
    def add_values(first: int, second: int) -> int:
        """Add two integers together."""
        return first + second

    assert add_values(2, 3) == 5
    assert add_values.__name__ == "add_values"
    assert add_values.__doc__ == "Add two integers together."


def test_retry_succeeds_after_failures():
    """A flaky function should eventually recover before the last allowed try."""
    attempts = {"count": 0}

    @retry(max_attempts=3)
    def flaky_operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("temporary failure")
        return "success"

    assert flaky_operation() == "success"
    assert attempts["count"] == 3


def test_retry_raises_final_exception_after_all_attempts():
    """The final failure should be re-raised without swallowing the exception."""
    attempts = {"count": 0}

    @retry(max_attempts=2)
    def always_fails() -> str:
        attempts["count"] += 1
        raise RuntimeError("still failing")

    with pytest.raises(RuntimeError, match="still failing"):
        always_fails()

    assert attempts["count"] == 2


def test_retry_rejects_invalid_max_attempts():
    """Retry should reject invalid attempt counts immediately."""
    with pytest.raises(ValueError, match="positive integer"):
        retry(max_attempts=0)

    with pytest.raises(ValueError, match="positive integer"):
        retry(max_attempts=-2)


def test_resource_manager_releases_after_success(tmp_path: Path):
    """The resource manager should close a file safely after successful use."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("id,name\n1,Alice\n", encoding="utf-8")

    resource = CSVResourceManager(csv_path)
    with resource as handle:
        assert handle is not None
        assert resource.is_open is True
        assert handle.read().startswith("id,name")

    assert resource.is_open is False


def test_resource_manager_releases_after_exception(tmp_path: Path):
    """The resource manager should still close even when an exception escapes."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("id,name\n1,Alice\n", encoding="utf-8")

    resource = CSVResourceManager(csv_path)
    with pytest.raises(RuntimeError, match="boom"):
        with resource as handle:
            assert handle is not None
            raise RuntimeError("boom")

    assert resource.is_open is False


def test_pipeline_run_is_timed(sample_records):
    """The actual pipeline runner should use the timing decorator."""
    pipeline = Pipeline(
        [
            CSVDataCleaner(),
            DataFilter(field_name="score", minimum=80),
        ]
    )

    result = pipeline.run(sample_records)

    assert len(result) == 3
    assert Pipeline.run.__wrapped__ is not None


def test_retry_csv_reader_uses_context_manager(tmp_path: Path):
    """CSV ingestion should be retried and safely managed through the resource context."""
    csv_path = tmp_path / "tracker.csv"
    csv_path.write_text("id,name,score\n1,Alice,95\n2,Bob,70\n", encoding="utf-8")

    rows = read_csv_rows_with_retry(csv_path)

    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"


def test_lru_cache_normalizes_feature_names():
    """The safe cached function should return normalized feature names consistently."""
    assert normalize_feature_name("  Score ") == "score"
    assert normalize_feature_name("score") == "score"
    assert normalize_feature_name(" Subject ") == "subject"
