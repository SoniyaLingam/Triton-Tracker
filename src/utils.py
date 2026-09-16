"""Reusable decorators and resource management utilities for Tracker."""

from __future__ import annotations

import csv
import logging
import time
from contextlib import AbstractContextManager
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

from src.exceptions import DataProcessingError, ResourceError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timeit(func: F) -> F:
    """Measure how long a function takes to execute and log the duration."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Function completed.",
            extra={
                "event": "function_timed",
                "function": func.__qualname__,
                "duration_seconds": round(elapsed, 6),
            },
        )
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
                            "Function failed after all retry attempts.",
                            extra={
                                "event": "retry_exhausted",
                                "function": func.__qualname__,
                                "attempt": attempt,
                                "max_attempts": max_attempts,
                            },
                        )
                        raise
                    logger.warning(
                        "Retrying function after a failure.",
                        extra={
                            "event": "retry_scheduled",
                            "function": func.__qualname__,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "error": str(exc),
                        },
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
        logger.info(
            "Opening CSV resource.", extra={"event": "resource_opening", "path": self.file_path}
        )
        try:
            self.handle = open(self.file_path, "r", newline="", encoding="utf-8")
        except OSError:
            logger.exception(
                "Unable to open CSV resource.",
                extra={"event": "resource_open_failed", "path": self.file_path},
            )
            raise
        self.is_open = True
        logger.info(
            "CSV resource opened.", extra={"event": "resource_opened", "path": self.file_path}
        )
        return self.handle

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if self.handle is not None:
                self.handle.close()
        finally:
            self.is_open = False
            self.handle = None

        logger.info(
            "CSV resource released.", extra={"event": "resource_closed", "path": self.file_path}
        )

        if exc_value is not None:
            logger.exception(
                "CSV resource closed after an exception.",
                extra={"event": "resource_failed", "path": self.file_path},
            )


@retry(max_attempts=3)
def read_csv_rows_with_retry(file_path: Union[str, Path]) -> list[dict[str, str]]:
    """Read CSV rows and retry a few times if the file resource is temporarily unavailable."""
    rows: list[dict[str, str]] = []
    path = Path(file_path)
    try:
        with CSVResourceManager(path) as handle:
            reader = csv.DictReader(handle)
            invalid_header = not reader.fieldnames or any(
                not field or not field.strip() for field in reader.fieldnames
            )
            if invalid_header:
                raise DataProcessingError(f"CSV file has an invalid header: {path}")
            for row in reader:
                rows.append({key: value for key, value in row.items()})
    except (OSError, UnicodeError) as exc:
        raise ResourceError(f"Unable to read CSV resource: {path}") from exc
    else:
        logger.info(
            "CSV rows loaded.",
            extra={"event": "csv_rows_loaded", "path": path, "records": len(rows)},
        )
        return rows
    finally:
        # The context manager closes its handle here even when parsing fails.
        logger.debug(
            "CSV read attempt finished.", extra={"event": "csv_read_finished", "path": path}
        )


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
