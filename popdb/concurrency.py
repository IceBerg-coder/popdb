"""
Concurrency and Reader-Writer Locking for PopDB.

Provides thread-safe concurrent reads with exclusive write protection,
allowing multiple concurrent search workers while ensuring safe WAL flushes,
appends, and storage compaction.
"""

import threading
from contextlib import contextmanager
from typing import Generator


class RWLock:
    """
    Reader-Writer Lock with writer priority to prevent writer starvation.
    Allows multiple concurrent readers, but only one exclusive writer.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._readers = 0
        self._writers = 0
        self._writers_waiting = 0

    def acquire_read(self) -> None:
        """Acquire a shared read lock."""
        with self._cond:
            while self._writers > 0 or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        """Release the shared read lock."""
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        """Acquire an exclusive write lock."""
        with self._cond:
            self._writers_waiting += 1
            while self._readers > 0 or self._writers > 0:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writers = 1

    def release_write(self) -> None:
        """Release the exclusive write lock."""
        with self._cond:
            self._writers = 0
            self._cond.notify_all()

    @contextmanager
    def read_lock(self) -> Generator[None, None, None]:
        """Context manager for shared read operations."""
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_lock(self) -> Generator[None, None, None]:
        """Context manager for exclusive write operations."""
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

