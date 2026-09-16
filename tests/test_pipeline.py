"""Tests for the composition-based pipeline architecture."""

import pytest

from src.pipeline import (
    AddProcessedFlagStep,
    CSVDataCleaner,
    DataFilter,
    DataTransformer,
    HighScoreFilter,
    Pipeline,
    Step,
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


def test_abstract_step_cannot_be_instantiated_directly():
    """The abstract Step contract cannot be instantiated."""
    with pytest.raises(TypeError):
        Step()


def test_concrete_steps_implement_step_interface():
    """Concrete steps all implement the shared Step interface."""
    steps = [
        CSVDataCleaner(),
        DataFilter(field_name="score", minimum=80),
        DataTransformer(),
        AddProcessedFlagStep(),
    ]

    for step in steps:
        assert isinstance(step, Step)


def test_pipeline_runs_steps_in_correct_order(sample_records):
    """Pipeline should run every step sequentially."""
    pipeline = Pipeline(
        [
            CSVDataCleaner(),
            DataFilter(field_name="score", minimum=80),
            DataTransformer(),
        ]
    )

    result = pipeline.run(sample_records)

    assert len(result) == 3
    assert result[0]["name"] == "ALICE"
    assert result[0]["status"] == "eligible"


def test_pipeline_passes_output_from_one_step_to_the_next(sample_records):
    """The output of one step becomes the input for the next step."""
    pipeline = Pipeline(
        [
            CSVDataCleaner(),
            DataFilter(field_name="score", minimum=80),
            AddProcessedFlagStep(),
        ]
    )

    result = pipeline.run(sample_records)

    assert all(row["processed"] is True for row in result)
    assert result[0]["name"] == "Alice"


def test_pipeline_supports_different_combinations_of_steps(sample_records):
    """Different step combinations should work without changing Pipeline."""
    pipeline = Pipeline(
        [
            CSVDataCleaner(),
            HighScoreFilter(field_name="score", minimum=90),
        ]
    )

    result = pipeline.run(sample_records)

    assert len(result) == 1
    assert {row["id"] for row in result} == {"3"}


def test_new_step_can_be_added_without_changing_pipeline(sample_records):
    """A brand-new step should plug into the existing pipeline unchanged."""
    pipeline = Pipeline(
        [
            CSVDataCleaner(),
            AddProcessedFlagStep(),
        ]
    )

    result = pipeline.run(sample_records)

    assert all("processed" in row for row in result)
    assert all(row["processed"] is True for row in result)


def test_runtime_step_swapping_works(sample_records):
    """Replacing one step with another should only require new step objects."""
    pipeline_a = Pipeline(
        [
            CSVDataCleaner(),
            DataFilter(field_name="score", minimum=80),
            DataTransformer(),
        ]
    )
    pipeline_b = Pipeline(
        [
            CSVDataCleaner(),
            HighScoreFilter(field_name="score", minimum=90),
            DataTransformer(),
        ]
    )

    result_a = pipeline_a.run(sample_records)
    result_b = pipeline_b.run(sample_records)

    assert len(result_a) == 3
    assert len(result_b) == 1
    assert {row["id"] for row in result_a} == {"1", "3", "4"}
    assert {row["id"] for row in result_b} == {"3"}


def test_empty_pipeline_returns_input_data(sample_records):
    """An empty pipeline is a pass-through pipeline."""
    pipeline = Pipeline([])

    assert pipeline.run(sample_records) == sample_records


def test_invalid_step_objects_are_rejected():
    """Only Step implementations are accepted by the pipeline."""
    with pytest.raises(TypeError):
        Pipeline(["not-a-step"])  # type: ignore[list-item]
