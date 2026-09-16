"""Reusable decorators and resource management utilities for Tracker."""

from __future__ import annotations

import csv
import logging
import time
from contextlib import AbstractContextManager
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timeit(func: F) -> F:
    """Measure how long a function takes to execute and log the duration."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.info("%s executed in %.6f seconds", func.__qualname__, elapsed)
        return result

    return wrapper  # type: ignore[return-value]


def retry(max_attempts: int) -> Callable[[F], F]:
    """Retry a function until it succeeds or the configured attempt limit is reached."""
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - behavior is validated by tests
                    if attempt == max_attempts:
                        logger.exception(
                            "Function %s failed on final attempt (%s/%s): %s",
                            func.__qualname__,
                            attempt,
                            max_attempts,
                            exc,
                        )
                        raise
                    logger.warning(
                        "Retrying %s after failure on attempt %s/%s: %s",
                        func.__qualname__,
                        attempt,
                        max_attempts,
                        exc,
                    )
            raise RuntimeError(f"Retry loop exhausted for {func.__qualname__}")

        return wrapper  # type: ignore[return-value]

    return decorator


class CSVResourceManager(AbstractContextManager[Optional[Any]]):
    """Safely open and close a CSV file while guaranteeing cleanup on exceptions."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.handle: Optional[Any] = None
        self.is_open = False

    def __enter__(self) -> Any:
        self.handle = open(self.file_path, "r", newline="", encoding="utf-8")
        self.is_open = True
        return self.handle

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if self.handle is not None:
                self.handle.close()
        finally:
            self.is_open = False
            self.handle = None

        if exc_value is not None:
            logger.exception(
                "Resource cleanup completed for %s after an exception: %s",
                self.file_path,
                exc_value,
            )


@retry(max_attempts=3)
def read_csv_rows_with_retry(file_path: Union[str, Path]) -> list[dict[str, str]]:
    """Read CSV rows and retry a few times if the file resource is temporarily unavailable."""
    rows: list[dict[str, str]] = []
    with CSVResourceManager(file_path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: value for key, value in row.items()})
    return rows


@lru_cache(maxsize=128)
def normalize_feature_name(name: str) -> str:
    """Normalize a feature name into a predictable, cacheable key."""
    cleaned = name.strip().lower().replace(" ", "_")
    return cleaned


__all__ = [
    "CSVResourceManager",
    "normalize_feature_name",
    "read_csv_rows_with_retry",
    "retry",
    "timeit",
]
