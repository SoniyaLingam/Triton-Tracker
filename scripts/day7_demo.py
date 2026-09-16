"""Create structured logs for successful and controlled failed Tracker runs."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from src.exceptions import PipelineError
    from src.logging_config import configure_logging
    from src.pipeline import CSVDataCleaner, DataFilter, Pipeline, Step
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.exceptions import PipelineError
    from src.logging_config import configure_logging
    from src.pipeline import CSVDataCleaner, DataFilter, Pipeline, Step


class RequiredFieldStep(Step):
    """Represent a pipeline operation that requires a mapping-like batch."""

    def process(self, data: object) -> object:
        """Access a required field and fail clearly for unsuitable input."""
        return data["required"]  # type: ignore[index]


def main() -> None:
    """Run and log both a successful pipeline and an expected failure."""
    project_root = Path(__file__).resolve().parents[1]
    logs_directory = project_root / "logs"
    success_log = logs_directory / "day7_success.jsonl"
    failure_log = logs_directory / "day7_failure.jsonl"

    configure_logging(success_log, level="INFO", console=False)
    records = [
        {"id": "1", "name": " Alice ", "score": "95"},
        {"id": "2", "name": "Bob", "score": "72"},
    ]
    result = Pipeline([CSVDataCleaner(), DataFilter("score", 80)]).run(records)
    print(f"Successful pipeline processed {len(result)} record(s): {success_log}")

    configure_logging(failure_log, level="INFO", console=False)
    try:
        Pipeline([RequiredFieldStep()]).run([])
    except PipelineError as exc:
        print(f"Controlled pipeline failure logged as {type(exc).__name__}: {failure_log}")


if __name__ == "__main__":
    main()
