"""2-Bit Scalar Quantization (SQ2) for In-Memory Tier 1.5 Asymmetric Dot Product."""

from typing import Tuple
import numpy as np


class ScalarQuantizer2Bit:
    """2-Bit Scalar Quantizer for 256d Matryoshka sub-vectors.
    
    Compresses 256 float32 dimensions (1,024 bytes) into 64 bytes per vector.
    Enables in-memory Tier 1.5 asymmetric re-ranking to prune candidates
    before triggering disk memmap page reads.
    
    Encoding:
    - 4 bins per dimension (2 bits):
        0 (00): v < -threshold
        1 (01): -threshold <= v < 0
        2 (10): 0 <= v < threshold
        3 (11): v >= threshold
    - 4 values packed per uint8 byte -> 64 bytes for 256 dimensions.
    """

    DIM = 256
    PACKED_BYTES = 64  # 256 * 2 bits / 8 = 64 bytes
    # For unit 256d vectors, standard deviation is ~1/sqrt(256) = 0.0625
    DEFAULT_THRESHOLD = 0.042
    # Reconstruction centroids
    CENTROIDS = np.array([-0.085, -0.021, 0.021, 0.085], dtype=np.float32)

    @classmethod
    def quantize(cls, vectors_256: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
        """Quantize and pack 256d float32 vectors into 64 bytes uint8.
        
        Args:
            vectors_256: Array of shape (N, 256) float32.
            threshold: Boundary threshold separating small and large magnitude bins.
            
        Returns:
            np.ndarray of shape (N, 64) uint8.
        """
        n = vectors_256.shape[0]
        if n == 0:
            return np.empty((0, cls.PACKED_BYTES), dtype=np.uint8)

        v = vectors_256
        # Compute bin indices: 0, 1, 2, 3
        # 0: v < -tau
        # 1: -tau <= v < 0
        # 2: 0 <= v < tau
        # 3: v >= tau
        b0 = (v >= -threshold).astype(np.uint8)
        b1 = (v >= 0).astype(np.uint8)
        b2 = (v >= threshold).astype(np.uint8)
        bins = b0 + b1 + b2  # values 0, 1, 2, 3

        # Pack 4 consecutive 2-bit values into each uint8 byte
        # Shape: (N, 64, 4)
        reshaped = bins.reshape(n, cls.PACKED_BYTES, 4)
        packed = (
            (reshaped[:, :, 0] << 6)
            | (reshaped[:, :, 1] << 4)
            | (reshaped[:, :, 2] << 2)
            | reshaped[:, :, 3]
        ).astype(np.uint8)

        return packed

    @classmethod
    def unpack(cls, packed_codes: np.ndarray) -> np.ndarray:
        """Unpack (N, 64) uint8 into (N, 256) bin codes (0..3)."""
        n = packed_codes.shape[0]
        if n == 0:
            return np.empty((0, cls.DIM), dtype=np.uint8)

        unpacked = np.empty((n, cls.PACKED_BYTES, 4), dtype=np.uint8)
        unpacked[:, :, 0] = (packed_codes >> 6) & 0x03
        unpacked[:, :, 1] = (packed_codes >> 4) & 0x03
        unpacked[:, :, 2] = (packed_codes >> 2) & 0x03
        unpacked[:, :, 3] = packed_codes & 0x03

        return unpacked.reshape(n, cls.DIM)

    @classmethod
    def asymmetric_dot_product(
        cls,
        packed_codes: np.ndarray,
        query_256: np.ndarray,
    ) -> np.ndarray:
        """Compute fast asymmetric dot product between unquantized query and 2-bit codes.
        
        Args:
            packed_codes: Array of shape (N, 64) uint8.
            query_256: Array of shape (256,) float32.
            
        Returns:
            np.ndarray of shape (N,) float32 approximate dot products.
        """
        n = packed_codes.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float32)

        # Unpack codes
        codes = cls.unpack(packed_codes)  # (N, 256) uint8

        # Map codes to approximate float values via centroids
        reconstructed = cls.CENTROIDS[codes]  # (N, 256) float32

        # Compute dot product
        return np.dot(reconstructed, query_256)
