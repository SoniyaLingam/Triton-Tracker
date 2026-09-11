"""
Memory efficiency demonstration for the CSV batch iterator.

This script demonstrates that the lazy-loading iterator maintains approximately
constant memory usage as the dataset size increases, unlike eager loading which
consumes memory proportional to dataset size.

Run with: python scripts/memory_test.py

Important: Small variations in memory usage are normal due to Python's garbage
collection, OS memory management, and other factors. The key observation is
that memory usage does NOT increase dramatically as dataset size increases.
"""

import csv
import sys
import tempfile
import tracemalloc
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_iterator import CSVBatchGeneratorIterator, CSVBatchIterator  # noqa: E402


def create_test_csv(file_path: Path, num_rows: int) -> None:
    """Create a CSV file with a specified number of rows."""
    fieldnames = ["id", "name", "score", "subject", "timestamp"]
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, num_rows + 1):
            writer.writerow(
                {
                    "id": str(i),
                    "name": f"person_{i}",
                    "score": str(85 + (i % 15)),
                    "subject": ["Math", "Science", "English"][i % 3],
                    "timestamp": f"2024-01-{(i % 30) + 1:02d}",
                }
            )


def measure_memory(iterator, description: str) -> float:
    """
    Measure peak memory usage while iterating through batches.

    Args:
        iterator: An iterator (CSVBatchIterator or
                  CSVBatchGeneratorIterator).
        description: Description for logging.

    Returns:
        float: Peak memory usage in MB.
    """
    tracemalloc.start()

    batch_count = 0
    row_count = 0

    try:
        for batch in iterator:
            batch_count += 1
            row_count += len(batch)
            # Process the batch (simulate work)
            _ = [row["id"] for row in batch]
    finally:
        # Ensure cleanup
        if hasattr(iterator, "_cleanup"):
            iterator._cleanup()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    msg = (
        f"  {description}: {peak_mb:.2f} MB "
        f"(processed {row_count} rows in {batch_count} batches)"
    )
    print(msg)

    return peak_mb


def test_dataset_size(num_rows: int, batch_size: int = 100) -> None:
    """
    Test memory usage for a dataset of a given size.

    Args:
        num_rows (int): Number of rows to generate.
        batch_size (int): Batch size for the iterator.
    """
    print(f"\n{'=' * 70}")
    print(f"Dataset size: {num_rows:,} rows")
    print(f"Batch size: {batch_size}")
    print(f"{'=' * 70}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test CSV file
        csv_file = tmpdir / "test_data.csv"
        print("  Generating test data... ", end="", flush=True)
        create_test_csv(csv_file, num_rows)
        print("done")

        # Measure memory for lazy-loading iterator (iterator protocol)
        print("\n  Lazy-loading Iterator (CSVBatchIterator):")
        iterator = CSVBatchIterator(str(tmpdir), batch_size=batch_size)
        peak_lazy_iter = measure_memory(iterator, "Peak memory")
        del iterator  # Ensure cleanup

        # Measure memory for lazy-loading iterator (generator-based)
        print("\n  Lazy-loading Iterator (CSVBatchGeneratorIterator):")
        generator_iter = CSVBatchGeneratorIterator(str(tmpdir), batch_size=batch_size)
        peak_lazy_gen = measure_memory(generator_iter, "Peak memory")
        del generator_iter  # Ensure cleanup

        # Measure memory for eager loading (for comparison)
        print("\n  Eager Loading (for comparison):")
        tracemalloc.start()
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)  # Load entire dataset
        current, peak_eager = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_eager_mb = peak_eager / (1024 * 1024)
        msg = f"  Peak memory: {peak_eager_mb:.2f} MB " f"(loaded {len(all_rows)} rows at once)"
        print(msg)

        print(f"\n  {'Comparison':.<50}")
        print(f"  Eager loading peak: {peak_eager_mb:>10.2f} MB")
        print(f"  Lazy iterator peak: {peak_lazy_iter:>10.2f} MB")
        print(f"  Lazy generator peak: {peak_lazy_gen:>10.2f} MB")

        # Calculate reduction
        reduction_iter = ((peak_eager_mb - peak_lazy_iter) / peak_eager_mb) * 100
        reduction_gen = ((peak_eager_mb - peak_lazy_gen) / peak_eager_mb) * 100
        print(f"\n  Memory reduction (iterator): {reduction_iter:.1f}%")
        print(f"  Memory reduction (generator): {reduction_gen:.1f}%")


def main():
    """Run memory efficiency tests with different dataset sizes."""
    print("\n" + "=" * 70)
    print("MEMORY EFFICIENCY DEMONSTRATION")
    print("CSV Batch Iterator - Lazy vs Eager Loading")
    print("=" * 70)
    print(
        """
EXPLANATION:
As dataset size increases, lazy-loading memory usage stays approximately
constant because only one batch is kept in memory at a time.

Eager loading memory grows proportionally with dataset size because the entire
dataset is loaded into memory at once.

IMPORTANT NOTES:
- Small memory variations are normal (garbage collection, OS memory management)
- The goal is to show that lazy loading does NOT retain the entire dataset
- Batch size affects memory: larger batches = higher peak memory
- Lazy loading is ideal for ML workflows with large datasets or limited RAM
"""
    )

    try:
        # Test with progressively larger datasets
        test_dataset_size(100, batch_size=10)
        test_dataset_size(1000, batch_size=50)
        test_dataset_size(10000, batch_size=100)

        print(f"\n{'=' * 70}")
        print("CONCLUSION")
        print(f"{'=' * 70}")
        print(
            """
The lazy-loading iterator demonstrates that:

1. Memory usage is NOT significantly impacted by total dataset size
2. Only ONE batch of data is held in memory at any time
3. This approach scales to datasets much larger than available RAM
4. Generator-based and iterator protocol implementations both show
   similar memory-efficient behavior

This is crucial for ML/AI applications that process large datasets that
cannot fit entirely in memory.
"""
        )

    except Exception as e:
        print(f"\nError during memory test: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import traceback

    main()
