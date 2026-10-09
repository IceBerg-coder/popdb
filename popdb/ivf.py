"""
Inverted File Binary Quantization (IVF-BQ) Index for PopDB.

Partitions vector space into K Voronoi clusters using spherical k-means.
Allows sub-linear pruning from 10M+ vectors down to small candidate sets
by probing only the top-n_probe nearest cluster inverted lists.
"""

from typing import Dict, List, Optional, Tuple, Set
import numpy as np


class IVFBQIndex:
    """
    IVF-BQ index partitioning high-dimensional space into Voronoi cells.
    
    Attributes:
        k (int): Number of cluster centroids.
        n_probe (int): Number of nearest centroids to probe during search.
        centroids (np.ndarray): Shape (K, 768), float32 unit-normalized cluster centers.
        centroid_bitmasks (np.ndarray): Shape (K, 96), 1-bit packed bitmasks of centroids.
        inverted_lists (Dict[int, List[int]]): Mapping cluster_id -> list of physical offsets.
        doc_clusters (Dict[int, int]): Mapping physical offset -> cluster_id.
        is_trained (bool): Whether the IVF index has been trained.
    """

    def __init__(self, k: int = 16, n_probe: int = 4):
        self.k = k
        self.n_probe = min(n_probe, k)
        self.centroids: Optional[np.ndarray] = None  # (K, 768)
        self.centroid_bitmasks: Optional[np.ndarray] = None  # (K, 96)
        self.inverted_lists: Dict[int, List[int]] = {i: [] for i in range(k)}
        self.doc_clusters: Dict[int, int] = {}
        self.is_trained: bool = False

    def train(
        self,
        vectors: np.ndarray,
        offsets: Optional[List[int]] = None,
        max_iters: int = 20,
        tol: float = 1e-4,
    ) -> None:
        """
        Train IVF centroids using spherical k-means on normalized vectors.
        
        Args:
            vectors: (N, 768) float32 vectors.
            offsets: Optional list of physical offsets corresponding to vectors rows.
                     Defaults to range(N).
            max_iters: Maximum Lloyd iterations.
            tol: Convergence tolerance for centroid movement.
        """
        n_docs, dim = vectors.shape
        if n_docs < self.k:
            # Not enough documents to populate k clusters; scale down k
            effective_k = max(1, n_docs)
            self.k = effective_k
            self.n_probe = min(self.n_probe, effective_k)
        else:
            effective_k = self.k

        if offsets is None:
            offsets = list(range(n_docs))

        # 1. Initialize centroids with k-means++ style random sampling
        rng = np.random.RandomState(42)
        initial_indices = rng.choice(n_docs, size=effective_k, replace=False)
        centroids = vectors[initial_indices].copy()

        # Ensure unit L2 normalization
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        centroids /= norms

        # 2. Spherical k-means Lloyd iterations
        for _ in range(max_iters):
            # Compute cosine similarities: (N, K)
            sims = np.dot(vectors, centroids.T)
            assignments = np.argmax(sims, axis=1)

            new_centroids = np.zeros_like(centroids)
            for c in range(effective_k):
                cluster_members = vectors[assignments == c]
                if len(cluster_members) > 0:
                    mean_vec = np.sum(cluster_members, axis=0)
                    m_norm = np.linalg.norm(mean_vec)
                    if m_norm > 0:
                        new_centroids[c] = mean_vec / m_norm
                    else:
                        new_centroids[c] = centroids[c]
                else:
                    # Re-seed empty cluster with random vector
                    rand_idx = rng.randint(0, n_docs)
                    new_centroids[c] = vectors[rand_idx]

            diff = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids
            if diff < tol:
                break

        self.centroids = centroids.astype(np.float32)

        # 3. Compute 1-bit packed bitmasks for centroids (K, 96)
        binary_bool = (self.centroids > 0)
        self.centroid_bitmasks = np.packbits(binary_bool, axis=1)

        # 4. Populate inverted posting lists
        self.inverted_lists = {i: [] for i in range(effective_k)}
        self.doc_clusters = {}

        sims = np.dot(vectors, self.centroids.T)
        final_assignments = np.argmax(sims, axis=1)

        for off, cluster_id in zip(offsets, final_assignments):
            c_int = int(cluster_id)
            self.inverted_lists[c_int].append(off)
            self.doc_clusters[off] = c_int

        self.is_trained = True

    def assign(self, vector: np.ndarray) -> int:
        """Assign a single 768d vector to the nearest centroid."""
        if not self.is_trained or self.centroids is None:
            return 0
        sims = np.dot(self.centroids, vector)
        return int(np.argmax(sims))

    def add(self, offset: int, vector: np.ndarray) -> None:
        """Add a newly inserted vector offset into the appropriate inverted list."""
        if not self.is_trained or self.centroids is None:
            return
        cluster_id = self.assign(vector)
        self.inverted_lists[cluster_id].append(offset)
        self.doc_clusters[offset] = cluster_id

    def remove(self, offset: int) -> None:
        """Remove a tombstoned vector offset from its inverted list."""
        if offset in self.doc_clusters:
            cluster_id = self.doc_clusters.pop(offset)
            if cluster_id in self.inverted_lists:
                try:
                    self.inverted_lists[cluster_id].remove(offset)
                except ValueError:
                    pass

    def probe(self, query_vector: np.ndarray, n_probe: Optional[int] = None) -> np.ndarray:
        """
        Probe nearest centroids and return candidate physical offsets.
        
        Args:
            query_vector: (768,) float32 normalized query vector.
            n_probe: Number of clusters to probe (defaults to self.n_probe).
            
        Returns:
            np.ndarray of candidate physical offsets.
        """
        if not self.is_trained or self.centroids is None:
            return np.array([], dtype=np.int64)

        probe_k = min(n_probe or self.n_probe, len(self.centroids))

        # Dot product against K centroids
        centroid_scores = np.dot(self.centroids, query_vector)
        if probe_k >= len(self.centroids):
            top_clusters = np.arange(len(self.centroids))
        else:
            top_clusters = np.argpartition(-centroid_scores, probe_k)[:probe_k]

        candidates: List[int] = []
        for c in top_clusters:
            candidates.extend(self.inverted_lists.get(int(c), []))

        if not candidates:
            return np.array([], dtype=np.int64)

        return np.unique(np.array(candidates, dtype=np.int64))

