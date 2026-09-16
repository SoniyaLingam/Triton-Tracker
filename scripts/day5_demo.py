"""Demonstrate the Day 5 decorators, context manager, and caching usage."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from src.pipeline import CSVDataCleaner, DataFilter
    from src.utils import (
        CSVResourceManager,
        normalize_feature_name,
        read_csv_rows_with_retry,
        retry,
        timeit,
    )
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.pipeline import CSVDataCleaner, DataFilter
    from src.utils import (
        CSVResourceManager,
        normalize_feature_name,
        read_csv_rows_with_retry,
        retry,
        timeit,
    )


@timeit
def process_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Apply the same cleaning/filtering flow used by Tracker."""
    cleaned = CSVDataCleaner().process(rows)
    filtered = DataFilter(field_name="score", minimum=80).process(cleaned)
    return filtered


@retry(max_attempts=3)
def safe_read(file_path: str | Path) -> list[dict[str, str]]:
    """Read CSV rows using the reusable retry helper and resource manager."""
    return read_csv_rows_with_retry(file_path)


if __name__ == "__main__":
    data = [
        {"id": "1", "name": " Alice ", "score": "95", "subject": "Math"},
        {"id": "2", "name": "Bob", "score": "75", "subject": "Science"},
    ]

    print("Timing decorator demo:")
    processed = process_rows(data)
    print(processed)

    print("\nRetry + context manager demo:")
    csv_path = Path("data/sample/sample.csv")
    rows = safe_read(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    print("\nResource manager demo:")
    with CSVResourceManager(csv_path) as handle:
        print(f"Open handle: {handle is not None}")

    print("\nCached feature normalization:")
    print(normalize_feature_name(" Score "))
    print(normalize_feature_name("score"))
