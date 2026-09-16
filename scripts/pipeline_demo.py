"""Demonstrate composition-based pipeline step swapping.

The key idea is that the ``Pipeline`` class never knows about concrete steps.
It only depends on the shared ``Step`` interface.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (  # noqa: E402
    AddProcessedFlagStep,
    CSVDataCleaner,
    CSVStatisticsStep,
    DataFilter,
    DataTransformer,
    HighScoreFilter,
    Pipeline,
)


def sample_records():
    """Return a small CSV-like dataset for demonstration."""
    return [
        {"id": "1", "name": " Alice ", "score": "85", "subject": "Math"},
        {"id": "2", "name": "Bob", "score": "72", "subject": "Science"},
        {"id": "3", "name": "Charlie", "score": "93", "subject": "Math"},
        {"id": "4", "name": " Diana ", "score": "88", "subject": "Science"},
    ]


if __name__ == "__main__":
    data = sample_records()

    pipeline_one = Pipeline(
        [
            CSVDataCleaner(),
            DataFilter(field_name="score", minimum=80),
            DataTransformer(),
        ]
    )
    result_one = pipeline_one.run(data)

    pipeline_two = Pipeline(
        [
            CSVDataCleaner(),
            HighScoreFilter(field_name="score", minimum=90),
            DataTransformer(),
        ]
    )
    result_two = pipeline_two.run(data)

    pipeline_three = Pipeline(
        [
            CSVDataCleaner(),
            AddProcessedFlagStep(),
            CSVStatisticsStep(),
        ]
    )
    result_three = pipeline_three.run(data)

    print("Pipeline 1: Cleaner -> Filter -> Transformer")
    print(result_one)
    print()

    print("Pipeline 2: Cleaner -> AlternativeFilter -> Transformer")
    print(result_two)
    print()

    print("Pipeline 3: New Step added without changing Pipeline")
    print(result_three)
