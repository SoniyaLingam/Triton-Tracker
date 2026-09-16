"""Demonstrate Tracker's Day 6 custom exceptions and their causes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

try:
    from src.config import PipelineMode, load_pipeline_config
    from src.exceptions import TrackerError
    from src.pipeline import Pipeline, Step
    from src.utils import read_csv_rows_with_retry
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.config import PipelineMode, load_pipeline_config
    from src.exceptions import TrackerError
    from src.pipeline import Pipeline, Step
    from src.utils import read_csv_rows_with_retry


class FailingStep(Step):
    """A step that demonstrates a real pipeline-operation failure."""

    def process(self, data: object) -> object:
        """Require a missing field from unsuitable pipeline input."""
        return data["required"]  # type: ignore[index]


def show_failure(label: str, operation: Callable[[], object]) -> None:
    """Run an operation and display its project-level error and root cause."""
    try:
        operation()
    except TrackerError as exc:
        print(f"{label}: {type(exc).__name__}: {exc}")
        if exc.__cause__ is not None:
            print(f"  Caused by: {type(exc.__cause__).__name__}: {exc.__cause__}")
    else:
        print(f"{label}: completed successfully")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    show_failure(
        "Configuration failure",
        lambda: load_pipeline_config(
            {
                "data_path": project_root / "data" / "sample",
                "batch_size": 0,
                "mode": PipelineMode.TRAIN,
                "threshold": 80,
                "feature_columns": ["score"],
            }
        ),
    )
    show_failure(
        "Resource failure",
        lambda: read_csv_rows_with_retry(project_root / "data" / "sample" / "missing.csv"),
    )
    show_failure("Pipeline failure", lambda: Pipeline([FailingStep()]).run([]))
