"""Tests that protect the Day 8 SOLID refactoring boundaries."""

from typing import Any

from src.pipeline import Pipeline, Step


class AddSourceStep(Step):
    """A new step added without changing Pipeline orchestration."""

    def process(self, data: list[dict[str, object]]) -> list[dict[str, object]]:
        """Add source metadata to every record."""
        return [{**row, "source": "day8"} for row in data]


class RecordingReporter:
    """Small reporter implementation that demonstrates dependency inversion."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def pipeline_started(self, data: Any, step_count: int) -> None:
        self.events.append("pipeline_started")

    def step_started(self, step_name: str, data: Any) -> None:
        self.events.append(f"step_started:{step_name}")

    def step_completed(self, step_name: str, input_data: Any, output_data: Any) -> None:
        self.events.append(f"step_completed:{step_name}")

    def step_failed(self, step_name: str) -> None:
        self.events.append(f"step_failed:{step_name}")

    def pipeline_completed(self, data: Any) -> None:
        self.events.append("pipeline_completed")


def test_new_step_and_reporter_work_without_changing_pipeline():
    """Pipeline relies on Step and reporter contracts, not concrete implementations."""
    reporter = RecordingReporter()

    result = Pipeline([AddSourceStep()], reporter=reporter).run([{"id": "1"}])

    assert result == [{"id": "1", "source": "day8"}]
    assert reporter.events == [
        "pipeline_started",
        "step_started:AddSourceStep",
        "step_completed:AddSourceStep",
        "pipeline_completed",
    ]
