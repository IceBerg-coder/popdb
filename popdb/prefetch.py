"""
Native OS Asynchronous Memory Prefetching for PopDB.

Asynchronously pre-faults disk-backed memory-mapped pages into physical RAM
using OS kernel calls:
- Windows: kernel32.PrefetchVirtualMemory
- Linux/macOS: libc.madvise(MADV_WILLNEED)

Eliminates NVMe/SSD page fault stalls during Tier 2 and Tier 3 candidate reads.
"""

import sys
import ctypes
from typing import Sequence, Union
import numpy as np


class OSMemoryPrefetcher:
    """
    Cross-platform kernel-level memory prefetcher for np.memmap arrays.
    """

    def __init__(self):
        self._is_windows = sys.platform == "win32"
        self._prefetch_fn = None
        self._get_process_fn = None
        self._entry_cls = None
        self._madvise_fn = None
        self._available = False

        self._init_os_hooks()

    def _init_os_hooks(self) -> None:
        if self._is_windows:
            try:
                from ctypes import wintypes
                class WIN32_MEMORY_RANGE_ENTRY(ctypes.Structure):
                    _fields_ = [
                        ("VirtualAddress", ctypes.c_void_p),
                        ("NumberOfBytes", ctypes.c_size_t),
                    ]

                k32 = ctypes.windll.kernel32
                if hasattr(k32, "PrefetchVirtualMemory"):
                    self._prefetch_fn = k32.PrefetchVirtualMemory
                    self._prefetch_fn.argtypes = [
                        wintypes.HANDLE,
                        ctypes.c_size_t,
                        ctypes.POINTER(WIN32_MEMORY_RANGE_ENTRY),
                        wintypes.DWORD,
                    ]
                    self._prefetch_fn.restype = wintypes.BOOL

                    self._get_process_fn = k32.GetCurrentProcess
                    self._get_process_fn.restype = wintypes.HANDLE
                    self._entry_cls = WIN32_MEMORY_RANGE_ENTRY
                    self._available = True
            except Exception:
                self._available = False
        else:
            # POSIX / Linux / Darwin
            try:
                libc = ctypes.CDLL(None)
                if hasattr(libc, "madvise"):
                    self._madvise_fn = libc.madvise
                    self._madvise_fn.argtypes = [
                        ctypes.c_void_p,
                        ctypes.c_size_t,
                        ctypes.c_int,
                    ]
                    self._madvise_fn.restype = ctypes.c_int
                    self._available = True
            except Exception:
                self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def prefetch(
        self,
        memmap_arr: np.ndarray,
        offsets: Union[Sequence[int], np.ndarray],
        bytes_per_row: int = 768 * 4,
    ) -> bool:
        """
        Advises the OS kernel to prefetch physical pages for candidate offsets.

        Args:
            memmap_arr: The memory-mapped numpy array.
            offsets: Sequence of physical row indices to prefetch.
            bytes_per_row: Byte size of each row (default 768 * 4 = 3072 bytes).

        Returns:
            bool: True if prefetch was successfully dispatched, False otherwise.
        """
        if not self._available or len(offsets) == 0:
            return False

        base_addr = memmap_arr.ctypes.data
        if base_addr == 0:
            return False

        try:
            if self._is_windows and self._prefetch_fn and self._entry_cls:
                n_entries = len(offsets)
                entries = (self._entry_cls * n_entries)()
                for i, off in enumerate(offsets):
                    entries[i].VirtualAddress = base_addr + int(off) * bytes_per_row
                    entries[i].NumberOfBytes = bytes_per_row

                h_proc = self._get_process_fn()
                res = self._prefetch_fn(h_proc, n_entries, entries, 0)
                return bool(res)

            elif self._madvise_fn:
                # Linux: MADV_WILLNEED = 3
                MADV_WILLNEED = 3
                for off in offsets:
                    addr = base_addr + int(off) * bytes_per_row
                    self._madvise_fn(addr, bytes_per_row, MADV_WILLNEED)
                return True

        except Exception:
            return False

        return False

