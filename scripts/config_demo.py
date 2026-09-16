"""Demonstrate runtime validation for the Tracker pipeline configuration."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import ValidationError  # noqa: E402

from src.config import DeviceType, PipelineConfig, PipelineMode  # noqa: E402


def print_invalid_case(label: str, payload: dict) -> None:
    """Print the validation error for a configuration payload."""
    print(f"\nInvalid configuration {label}:")
    try:
        PipelineConfig(**payload)
    except ValidationError as exc:
        for error in exc.errors():
            field_name = ".".join(str(item) for item in error["loc"])
            message = error["msg"]
            print(f"{field_name}: {message}")
    else:
        print("Unexpected success: configuration passed validation.")


if __name__ == "__main__":
    valid_data_path = Path("data/sample")

    print("Valid configuration:")
    try:
        config = PipelineConfig(
            data_path=valid_data_path,
            batch_size=32,
            mode=PipelineMode.TRAIN,
            device=DeviceType.CPU,
            threshold=85.0,
            feature_columns=["id", "score", "subject"],
        )
    except ValidationError as exc:
        print(f"Configuration failed unexpectedly: {exc}")
    else:
        print("Configuration loaded successfully.")
        print(config)

    invalid_cases = [
        (
            "1 (wrong type)",
            {
                "data_path": valid_data_path,
                "batch_size": "large",
                "mode": PipelineMode.EVALUATE,
                "threshold": 80,
                "feature_columns": ["score"],
            },
        ),
        (
            "2 (out of range)",
            {
                "data_path": valid_data_path,
                "batch_size": 0,
                "mode": PipelineMode.EVALUATE,
                "threshold": 80,
                "feature_columns": ["score"],
            },
        ),
        (
            "3 (missing required field)",
            {
                "batch_size": 16,
                "mode": PipelineMode.TRAIN,
                "threshold": 75,
                "feature_columns": ["score"],
            },
        ),
        (
            "4 (non-existent path)",
            {
                "data_path": Path("data/does_not_exist"),
                "batch_size": 8,
                "mode": PipelineMode.INFERENCE,
                "threshold": 50,
                "feature_columns": ["score"],
            },
        ),
    ]

    for label, payload in invalid_cases:
        print_invalid_case(label, payload)
