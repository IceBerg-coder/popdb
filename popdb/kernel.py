"""Fused SIMD POPCOUNT Kernel for high-throughput Tier 1 Hamming distance filtering."""

from typing import Tuple
import numpy as np


class FusedHammingKernel:
    """Zero-allocation cache-friendly hardware POPCOUNT kernel.
    
    Optimizations:
    - Reinterprets 96-byte (768-bit) uint8 bitmasks as 12 x uint64 (8x fewer CPU ops).
    - Processes in L3-cache-friendly blocks (65,536 vectors) to prevent cache thrashing.
    - Avoids large heap allocations for multi-million vector datasets.
    """

    CHUNK_SIZE: int = 65536

    @classmethod
    def compute_hamming_distances(
        cls,
        doc_bits_u8: np.ndarray,
        query_bits_u8: np.ndarray,
    ) -> np.ndarray:
        """Compute Hamming distances across all vectors using uint64 SIMD.
        
        Args:
            doc_bits_u8: Array of shape (N, 96) uint8.
            query_bits_u8: Array of shape (96,) uint8.
            
        Returns:
            np.ndarray of shape (N,) int32 containing exact Hamming distances.
        """
        n = doc_bits_u8.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int32)

        # Reinterpret 96 uint8 as 12 uint64
        doc_bits_u64 = np.ascontiguousarray(doc_bits_u8).view(np.uint64)
        query_bits_u64 = np.ascontiguousarray(query_bits_u8).view(np.uint64)

        if n <= cls.CHUNK_SIZE:
            xor = np.bitwise_xor(doc_bits_u64, query_bits_u64)
            return np.bitwise_count(xor).sum(axis=-1, dtype=np.int32)

        # Chunked computation to remain in CPU L3 cache
        dists = np.empty(n, dtype=np.int32)
        for start in range(0, n, cls.CHUNK_SIZE):
            end = min(start + cls.CHUNK_SIZE, n)
            chunk = doc_bits_u64[start:end]
            xor = np.bitwise_xor(chunk, query_bits_u64)
            dists[start:end] = np.bitwise_count(xor).sum(axis=-1, dtype=np.int32)

        return dists

    @classmethod
    def top_k_hamming(
        cls,
        doc_bits_u8: np.ndarray,
        query_bits_u8: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Hamming distances and select top-k lowest distance candidates.
        
        Args:
            doc_bits_u8: Array of shape (N, 96) uint8.
            query_bits_u8: Array of shape (96,) uint8.
            k: Number of survivors to return.
            
        Returns:
            Tuple of (top_k_indices: np.ndarray, top_k_distances: np.ndarray).
        """
        n = doc_bits_u8.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)

        target_k = min(k, n)
        dists = cls.compute_hamming_distances(doc_bits_u8, query_bits_u8)

        if n <= target_k:
            order = np.argsort(dists)
        else:
            partition = np.argpartition(dists, target_k)[:target_k]
            order = partition[np.argsort(dists[partition])]

        return order, dists[order]

