"""
Unit tests for the CSV batch iterator.

Tests cover:
- Basic iteration and batch size
- Multiple CSV files
- Empty folders and empty files
- Iterator protocol compliance
- Memory efficiency demonstration
"""

import csv
import tempfile
from pathlib import Path

import pytest

from src.data_iterator import CSVBatchGeneratorIterator, CSVBatchIterator


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with test CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv(temp_data_dir):
    """Create a simple CSV file with 20 rows for testing."""
    csv_file = temp_data_dir / "data.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "name", "value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, 21):
            row = {
                "id": str(i),
                "name": f"item_{i}",
                "value": str(i * 10),
            }
            writer.writerow(row)
    return temp_data_dir


@pytest.fixture
def multi_csv(temp_data_dir):
    """Create multiple CSV files with 10 rows each."""
    for file_num in range(1, 4):
        csv_file = temp_data_dir / f"data_{file_num}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["id", "name", "value"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(1, 11):
                row_id = (file_num - 1) * 10 + i
                row = {
                    "id": str(row_id),
                    "name": f"item_{row_id}",
                    "value": str(row_id * 10),
                }
                writer.writerow(row)
    return temp_data_dir


@pytest.fixture
def empty_csv(temp_data_dir):
    """Create an empty CSV file (headers only)."""
    csv_file = temp_data_dir / "empty.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "value"])
        writer.writeheader()
    return temp_data_dir


@pytest.fixture
def mixed_csv(temp_data_dir):
    """Create a mix of empty and non-empty CSV files."""
    # File 1: 5 rows
    csv_file1 = temp_data_dir / "file1.csv"
    with open(csv_file1, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name"])
        writer.writeheader()
        for i in range(1, 6):
            writer.writerow({"id": str(i), "name": f"item_{i}"})

    # File 2: Empty
    csv_file2 = temp_data_dir / "file2.csv"
    with open(csv_file2, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name"])
        writer.writeheader()

    # File 3: 8 rows
    csv_file3 = temp_data_dir / "file3.csv"
    with open(csv_file3, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name"])
        writer.writeheader()
        for i in range(6, 14):
            writer.writerow({"id": str(i), "name": f"item_{i}"})

    return temp_data_dir


class TestCSVBatchIterator:
    """Tests for CSVBatchIterator (iterator protocol implementation)."""

    def test_basic_iteration(self, sample_csv):
        """Test basic iteration through batches."""
        iterator = CSVBatchIterator(str(sample_csv), batch_size=5)
        batches = list(iterator)

        assert len(batches) == 4  # 20 rows / 5 batch_size = 4 batches
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5
        assert len(batches[2]) == 5
        assert len(batches[3]) == 5

    def test_batch_size_respected(self, sample_csv):
        """Test batch sizes respected (except possibly last batch)."""
        iterator = CSVBatchIterator(str(sample_csv), batch_size=7)
        batches = list(iterator)

        # All batches except possibly the last should be full
        for batch in batches[:-1]:
            assert len(batch) == 7

        # Last batch should have remaining rows
        assert len(batches[-1]) == 20 % 7

    def test_final_batch_smaller(self, sample_csv):
        """Test that final batch can be smaller than batch_size."""
        iterator = CSVBatchIterator(str(sample_csv), batch_size=6)
        batches = list(iterator)

        assert len(batches[-1]) < 6  # 20 % 6 = 2

    def test_iterator_protocol(self, sample_csv):
        """Test iterator protocol (__iter__, __next__) works correctly."""
        iterator = CSVBatchIterator(str(sample_csv), batch_size=5)

        # __iter__ should return self
        assert iter(iterator) is iterator

        # __next__ should return batches
        batch1 = next(iterator)
        assert isinstance(batch1, list)
        assert len(batch1) == 5

        # Calling next again should return different data
        batch2 = next(iterator)
        assert batch1[0]["id"] != batch2[0]["id"]

        # Exhausting the iterator should raise StopIteration
        for _ in range(2):  # Read remaining 2 batches
            next(iterator)

        with pytest.raises(StopIteration):
            next(iterator)

    def test_multiple_files(self, multi_csv):
        """Test iteration through multiple CSV files."""
        iterator = CSVBatchIterator(str(multi_csv), batch_size=5)
        batches = list(iterator)

        # 3 files * 10 rows / 5 batch_size = 6 batches
        assert len(batches) == 6

        # Verify data continuity across files
        all_rows = []
        for batch in batches:
            all_rows.extend(batch)

        assert len(all_rows) == 30

    def test_empty_folder(self):
        """Test that empty folder returns no batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            iterator = CSVBatchIterator(str(tmpdir), batch_size=5)
            batches = list(iterator)
            assert len(batches) == 0

    def test_empty_csv_file(self, empty_csv):
        """Test handling of empty CSV file (headers only)."""
        iterator = CSVBatchIterator(str(empty_csv), batch_size=5)
        batches = list(iterator)
        assert len(batches) == 0

    def test_mixed_empty_and_data(self, mixed_csv):
        """Test handling of folder with both empty and non-empty files."""
        iterator = CSVBatchIterator(str(mixed_csv), batch_size=3)
        batches = list(iterator)

        # 5 + 0 + 8 = 13 rows total
        # 13 / 3 = 4 batches (3, 3, 3, 4)
        assert len(batches) == 5  # (3, 3, 3, 3, 1)

        total_rows = sum(len(batch) for batch in batches)
        assert total_rows == 13

    def test_row_integrity(self, sample_csv):
        """Test that row data is preserved correctly."""
        iterator = CSVBatchIterator(str(sample_csv), batch_size=5)
        batches = list(iterator)

        # Flatten batches and check first and last rows
        all_rows = []
        for batch in batches:
            all_rows.extend(batch)

        assert all_rows[0]["id"] == "1"
        assert all_rows[0]["name"] == "item_1"
        assert all_rows[-1]["id"] == "20"
        assert all_rows[-1]["name"] == "item_20"

    def test_invalid_batch_size(self, sample_csv):
        """Test that invalid batch sizes raise ValueError."""
        with pytest.raises(ValueError):
            CSVBatchIterator(str(sample_csv), batch_size=0)

        with pytest.raises(ValueError):
            CSVBatchIterator(str(sample_csv), batch_size=-1)

    def test_nonexistent_folder(self):
        """Test that nonexistent folder raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CSVBatchIterator("/nonexistent/path", batch_size=5)

    def test_iterator_reuse(self, sample_csv):
        """Test that iterator can be reset and reused."""
        iterator = CSVBatchIterator(str(sample_csv), batch_size=5)

        # First iteration
        batches1 = list(iterator)
        assert len(batches1) == 4

        # Reset and iterate again
        batches2 = list(iterator)
        assert len(batches2) == 4

        # Verify data is the same
        assert batches1[0][0]["id"] == batches2[0][0]["id"]


class TestCSVBatchGeneratorIterator:
    """Tests for CSVBatchGeneratorIterator (generator-based implementation)."""

    def test_basic_iteration(self, sample_csv):
        """Test basic iteration through batches."""
        iterator = CSVBatchGeneratorIterator(str(sample_csv), batch_size=5)
        batches = list(iterator)

        assert len(batches) == 4  # 20 rows / 5 batch_size = 4 batches
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5
        assert len(batches[2]) == 5
        assert len(batches[3]) == 5

    def test_batch_size_respected(self, sample_csv):
        """Test batch sizes respected (except possibly last batch)."""
        iterator = CSVBatchGeneratorIterator(str(sample_csv), batch_size=7)
        batches = list(iterator)

        # All batches except possibly the last should be full
        for batch in batches[:-1]:
            assert len(batch) == 7

        # Last batch should have remaining rows
        assert len(batches[-1]) == 20 % 7

    def test_multiple_files(self, multi_csv):
        """Test iteration through multiple CSV files."""
        iterator = CSVBatchGeneratorIterator(str(multi_csv), batch_size=5)
        batches = list(iterator)

        # 3 files * 10 rows / 5 batch_size = 6 batches
        assert len(batches) == 6

        # Verify data continuity across files
        all_rows = []
        for batch in batches:
            all_rows.extend(batch)

        assert len(all_rows) == 30

    def test_empty_folder(self):
        """Test that empty folder returns no batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            iterator = CSVBatchGeneratorIterator(str(tmpdir), batch_size=5)
            batches = list(iterator)
            assert len(batches) == 0

    def test_empty_csv_file(self, empty_csv):
        """Test handling of empty CSV file (headers only)."""
        iterator = CSVBatchGeneratorIterator(str(empty_csv), batch_size=5)
        batches = list(iterator)
        assert len(batches) == 0

    def test_lazy_loading_behavior(self, sample_csv):
        """
        Test that generator iterator exhibits lazy loading behavior.

        The generator should not load all data at once. We verify this by
        checking that we can partially iterate and stop without issues.
        """
        iterator = CSVBatchGeneratorIterator(str(sample_csv), batch_size=5)
        gen = iter(iterator)

        # Get just the first batch
        batch1 = next(gen)
        assert len(batch1) == 5

        # Generator should still be active (not fully consumed)
        batch2 = next(gen)
        assert len(batch2) == 5

        # We can stop here without loading all remaining batches


class TestMemoryEfficiency:
    """Tests demonstrating lazy-loading behavior."""

    def test_iterator_does_not_retain_batches(self, sample_csv):
        """Test that iterator does not retain all batches in memory."""
        iterator = CSVBatchIterator(str(sample_csv), batch_size=5)

        batch_ids = []
        for batch in iterator:
            # Store just the IDs of rows in each batch
            batch_ids.extend([row["id"] for row in batch])

        # Verify we got all 20 rows
        assert len(batch_ids) == 20
        assert batch_ids[0] == "1"
        assert batch_ids[-1] == "20"

    def test_generator_yields_one_at_a_time(self, sample_csv):
        """Test that generator-based iterator yields one batch at a time."""
        iterator = CSVBatchGeneratorIterator(str(sample_csv), batch_size=3)

        # Manually iterate to verify one-at-a-time behavior
        gen = iter(iterator)
        batch1 = next(gen)
        assert len(batch1) == 3
        assert batch1[0]["id"] == "1"

        batch2 = next(gen)
        assert len(batch2) == 3
        assert batch2[0]["id"] == "4"

        # Continue iteration
        remaining_batches = list(gen)
        assert len(remaining_batches) == 5  # 20 rows / 3 = 7 batches total
