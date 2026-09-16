"""
Lazy-loading CSV batch iterator for memory-efficient data processing.

This module demonstrates practical understanding of Python's iteration
protocol, generators, and yield for memory-efficient data handling in
ML workflows.
"""

import csv
from pathlib import Path
from typing import Any, Iterator, List, Optional, TextIO


class CSVBatchIterator:
    """
    A lazy-loading iterator that reads CSV files from a folder in batches.

    This iterator reads CSV files progressively without loading the entire
    dataset into memory. It yields one batch at a time, moving to the next
    CSV file when the current one is exhausted.

    Attributes:
        folder_path (Path): Path to the folder containing CSV files.
        batch_size (int): Number of rows to include in each batch.

    Example:
        >>> iterator = CSVBatchIterator("data/sample", batch_size=10)
        >>> for batch in iterator:
        ...     print(len(batch))  # Number of rows in the batch
    """

    def __init__(self, folder_path: str, batch_size: int = 32):
        """
        Initialize the CSV batch iterator.

        Args:
            folder_path (str): Path to the folder containing CSV files.
            batch_size (int): Number of rows per batch. Defaults to 32.

        Raises:
            ValueError: If batch_size is not positive.
            FileNotFoundError: If the folder does not exist.
        """
        # Initialize attributes first (before validation) to avoid
        # AttributeError in __del__
        self._current_file_index = 0
        self._current_reader: Optional[csv.DictReader] = None
        self._current_file_handle: Optional[TextIO] = None

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

        self.batch_size = batch_size
        # Get all CSV files in sorted order
        self._csv_files = sorted(self.folder_path.glob("*.csv"))

    def __iter__(self) -> "CSVBatchIterator":
        """
        Return the iterator object itself.

        Returns:
            CSVBatchIterator: This iterator instance.
        """
        # Clean up any existing file handle before resetting state
        self._cleanup()
        # Reset state when iteration starts
        self._current_file_index = 0
        self._current_reader = None
        self._current_file_handle = None
        return self

    def __next__(self) -> List[dict]:
        """
        Return the next batch of rows.

        This method implements the iterator protocol. It reads rows
        progressively from CSV files and yields batches without loading
        the entire dataset.

        Returns:
            List[dict]: A batch of rows (dicts with column headers as keys).

        Raises:
            StopIteration: When all files have been processed.
        """
        batch: List[dict[str, Any]] = []

        # Keep reading rows until we have a full batch or reach end of
        # all files
        while len(batch) < self.batch_size:
            # Load the next CSV file if needed
            if self._current_reader is None:
                if self._current_file_index >= len(self._csv_files):
                    # All files exhausted
                    if batch:
                        return batch
                    raise StopIteration

                # Open the next CSV file
                if self._current_file_handle is not None:
                    self._current_file_handle.close()

                csv_file = self._csv_files[self._current_file_index]
                self._current_file_handle = open(csv_file, "r", newline="", encoding="utf-8")
                if self._current_file_handle is not None:
                    self._current_reader = csv.DictReader(self._current_file_handle)
                self._current_file_index += 1

            # Try to read a row from the current file
            try:
                row = next(self._current_reader)
                batch.append(row)
            except StopIteration:
                # Current file exhausted, move to next
                self._current_reader = None
                continue

        return batch

    def _cleanup(self) -> None:
        """Clean up open file handles."""
        if self._current_file_handle is not None:
            try:
                self._current_file_handle.close()
            except (OSError, ValueError):
                # File already closed or invalid handle
                pass
            self._current_file_handle = None
        self._current_reader = None

    def __del__(self):
        """Clean up open file handles when iterator is destroyed."""
        self._cleanup()


class CSVBatchGeneratorIterator:
    """
    Alternative implementation using a generator for batch iteration.

    This class demonstrates using generators (yield) with the iterator
    protocol for cleaner, more Pythonic lazy-loading behavior.

    Attributes:
        folder_path (Path): Path to the folder containing CSV files.
        batch_size (int): Number of rows to include in each batch.

    Example:
        >>> iterator = CSVBatchGeneratorIterator("data/sample",
        ...                                      batch_size=10)
        >>> for batch in iterator:
        ...     print(len(batch))
    """

    def __init__(self, folder_path: str, batch_size: int = 32):
        """
        Initialize the generator-based CSV batch iterator.

        Args:
            folder_path (str): Path to the folder containing CSV files.
            batch_size (int): Number of rows per batch. Defaults to 32.

        Raises:
            ValueError: If batch_size is not positive.
            FileNotFoundError: If the folder does not exist.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[dict]]:
        """
        Return a generator that yields batches of rows.

        This is the key difference from CSVBatchIterator: it uses yield
        to create a generator, which is more memory-efficient and
        Pythonic.

        Returns:
            Iterator[List[dict]]: An iterator yielding batches.
        """
        return self._batch_generator()

    def _batch_generator(self) -> Iterator[List[dict]]:
        """
        Generator function that yields batches of rows from CSV files.

        This generator reads rows progressively and yields batches
        without loading the entire dataset into memory.

        Yields:
            List[dict]: One batch of rows at a time.
        """
        csv_files = sorted(self.folder_path.glob("*.csv"))

        for csv_file in csv_files:
            batch: List[dict] = []
            try:
                with open(csv_file, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        batch.append(row)
                        if len(batch) >= self.batch_size:
                            yield batch
                            batch = []

                # Yield remaining rows if any
                if batch:
                    yield batch
            except (IOError, OSError) as e:
                # Log the error and continue with next file
                print(f"Warning: Could not read file {csv_file}: {e}")
                continue
