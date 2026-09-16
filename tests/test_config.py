"""Tests for the validated pipeline configuration model."""

import pytest
from pydantic import ValidationError

from src.config import DeviceType, PipelineConfig, PipelineMode


@pytest.fixture
def valid_data_dir(tmp_path):
    """Create a temporary directory that represents valid CSV data input."""
    data_dir = tmp_path / "sample_data"
    data_dir.mkdir()
    return data_dir


def test_valid_configuration(valid_data_dir):
    """A valid configuration should load successfully."""
    config = PipelineConfig(
        data_path=valid_data_dir,
        batch_size=32,
        mode=PipelineMode.TRAIN,
        device=DeviceType.CPU,
        threshold=85.0,
        feature_columns=["id", "score", "subject"],
    )

    assert config.data_path == valid_data_dir
    assert config.batch_size == 32
    assert config.mode is PipelineMode.TRAIN
    assert config.device is DeviceType.CPU
    assert config.threshold == 85.0
    assert config.feature_columns == ["id", "score", "subject"]


def test_batch_size_wrong_type(valid_data_dir):
    """batch_size must be an integer, not a string."""
    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        PipelineConfig(
            data_path=valid_data_dir,
            batch_size="large",
            mode=PipelineMode.EVALUATE,
            threshold=80,
            feature_columns=["score", "subject"],
        )


def test_batch_size_out_of_range(valid_data_dir):
    """batch_size must be greater than zero."""
    with pytest.raises(ValidationError, match="greater than 0"):
        PipelineConfig(
            data_path=valid_data_dir,
            batch_size=0,
            mode=PipelineMode.EVALUATE,
            threshold=80,
            feature_columns=["score", "subject"],
        )


def test_missing_data_path(valid_data_dir):
    """data_path is required and must be present."""
    with pytest.raises(ValidationError, match="Field required"):
        PipelineConfig(
            batch_size=16,
            mode=PipelineMode.TRAIN,
            threshold=75,
            feature_columns=["score"],
        )


def test_nonexistent_data_path():
    """A missing directory must be rejected."""
    with pytest.raises(ValidationError, match="Path does not exist"):
        PipelineConfig(
            data_path="data/does_not_exist",
            batch_size=8,
            mode=PipelineMode.INFERENCE,
            threshold=50,
            feature_columns=["score"],
        )


def test_invalid_enum_value(valid_data_dir):
    """Enum values must be from the allowed choices."""
    with pytest.raises(ValidationError, match="Input should be"):
        PipelineConfig(
            data_path=valid_data_dir,
            batch_size=16,
            mode="unknown-mode",
            threshold=70,
            feature_columns=["score"],
        )


def test_valid_enum_value(valid_data_dir):
    """A valid enum value should be accepted."""
    config = PipelineConfig(
        data_path=valid_data_dir,
        batch_size=12,
        mode=PipelineMode.INFERENCE,
        device=DeviceType.GPU,
        threshold=90,
        feature_columns=["score", "name"],
    )

    assert config.mode is PipelineMode.INFERENCE
    assert config.device is DeviceType.GPU


def test_threshold_validation(valid_data_dir):
    """Threshold must remain within a realistic score range."""
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        PipelineConfig(
            data_path=valid_data_dir,
            batch_size=10,
            mode=PipelineMode.EVALUATE,
            threshold=101,
            feature_columns=["score"],
        )


def test_feature_columns_validation(valid_data_dir):
    """Feature columns must be non-empty and unique."""
    with pytest.raises(ValidationError, match="Feature columns must not be empty"):
        PipelineConfig(
            data_path=valid_data_dir,
            batch_size=10,
            mode=PipelineMode.TRAIN,
            threshold=70,
            feature_columns=[],
        )


def test_data_path_must_be_directory(valid_data_dir, tmp_path):
    """A file path should not be accepted when a directory is required."""
    file_path = tmp_path / "not_a_directory.txt"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValidationError, match="Path must be a directory"):
        PipelineConfig(
            data_path=file_path,
            batch_size=10,
            mode=PipelineMode.TRAIN,
            threshold=60,
            feature_columns=["score"],
        )
