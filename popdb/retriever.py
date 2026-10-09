"""Hierarchical 3-tier cascade retriever and Reciprocal Rank Fusion (RRF) engine."""

import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from popdb.bm25 import BM25Index
from popdb.client import EmbeddingGemmaClient
from popdb.filter import MetadataFilter
from popdb.kernel import FusedHammingKernel
from popdb.quantization import ScalarQuantizer2Bit
from popdb.storage import StorageEngine
from popdb.types import HybridResult, SearchResult, TierStats


class HierarchicalRetriever:
    """3-Tier coarse-to-fine cascading vector retriever with RRF hybrid fusion.
    
    Pipeline:
    - Tier 1: 1-bit binary quantization with hardware POPCOUNT Hamming filtering (N -> 500)
    - Tier 2: Matryoshka 256d normalized dot product via memmap slice (500 -> 50)
    - Tier 3: Full 768d exact dot product via memmap slice (50 -> top_k)
    - Hybrid: Reciprocal Rank Fusion (RRF) blending dense and BM25 sparse rankings
    """

    def __init__(self, storage: StorageEngine, bm25: BM25Index):
        self.storage = storage
        self.bm25 = bm25

    def search_dense(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_predicate: Optional[Dict[str, Any]] = None,
        candidate_k: int = 500,
        tier2_k: int = 50,
        return_stats: bool = False,
        use_ivf: bool = True,
        n_probe: Optional[int] = None,
    ) -> Tuple[List[SearchResult], Optional[TierStats]]:
        """Run 3-tier cascading search.
        
        Args:
            query_vector: Unit-normalized 768d float32 query embedding.
            top_k: Final number of nearest neighbors to return.
            filter_predicate: Structured metadata filter dictionary.
            candidate_k: Number of candidates retained from Tier 1 (default: 500).
            tier2_k: Number of candidates retained from Tier 2 (default: 50).
            return_stats: If True, return timing and candidate telemetry.
            use_ivf: If True and IVF index trained, probe nearest clusters.
            n_probe: Number of nearest IVF clusters to probe.
            
        Returns:
            Tuple of (results_list, optional_tier_stats).
        """
        n_physical = len(self.storage.offset_to_id)
        if n_physical == 0:
            stats = TierStats(0, 0, 0, 0.0, 0, 0.0, 0, 0.0, 0.0) if return_stats else None
            return [], stats

        t_start = time.perf_counter()

        # =====================================================================
        # TIER 1: Predicate Pushdown + 1-Bit Binary Quantization + POPCOUNT
        # =====================================================================
        t1_start = time.perf_counter()

        # 1. Apply metadata pre-filtering directly via Columnar Inverted Bitsets
        if filter_predicate:
            col_mask = self.storage.columnar.evaluate_mask(filter_predicate)
            mask = col_mask if col_mask is not None else np.ones(n_physical, dtype=bool)
        else:
            mask = np.ones(n_physical, dtype=bool)

        # Apply tombstone soft-delete mask
        if self.storage.segment_mgr.tombstones:
            mask &= self.storage.segment_mgr.get_active_mask(self.storage.offset_to_id)

        # Apply IVF Voronoi cluster probing mask if trained
        if use_ivf and self.storage.ivf.is_trained:
            ivf_offsets = self.storage.ivf.probe(query_vector, n_probe=n_probe)
            if len(ivf_offsets) > 0:
                ivf_mask = np.zeros(n_physical, dtype=bool)
                ivf_mask[ivf_offsets] = True
                combined = mask & ivf_mask
                if np.any(combined):
                    mask = combined

        eligible_indices = np.flatnonzero(mask)

        n_filtered = len(eligible_indices)
        if n_filtered == 0:
            t_total = (time.perf_counter() - t_start) * 1000.0
            stats = (
                TierStats(n_physical, 0, 0, 0.0, 0, 0.0, 0, 0.0, t_total)
                if return_stats
                else None
            )
            return [], stats

        # 2. Quantize query vector to 1-bit packed uint8 (96 bytes)
        query_bits = EmbeddingGemmaClient.quantize_1bit(query_vector)

        # 3. Bitwise XOR and hardware POPCOUNT via Fused SIMD uint64 Kernel
        if n_filtered == n_physical:
            cand_bits = self.storage.bitmask_index
            target_indices = np.arange(n_physical)
        else:
            cand_bits = self.storage.bitmask_index[eligible_indices]
            target_indices = np.array(eligible_indices, dtype=np.int64)

        order1, t1_hamming_dists = FusedHammingKernel.top_k_hamming(
            doc_bits_u8=cand_bits,
            query_bits_u8=query_bits,
            k=candidate_k,
        )

        t1_survivor_indices = target_indices[order1].tolist()
        t1_survivor_hamming = t1_hamming_dists.tolist()
        t1_time = (time.perf_counter() - t1_start) * 1000.0

        # =====================================================================
        # TIER 1.5: In-Memory 2-Bit Scalar Quantization Asymmetric Filter
        # =====================================================================
        # Prunes from candidate_k down to 2 * tier2_k entirely in RAM before disk I/O
        norm_query_256 = EmbeddingGemmaClient.truncate_mrl(query_vector, dim=EmbeddingGemmaClient.MRL_DIM)
        if (
            len(t1_survivor_indices) > tier2_k * 2
            and self.storage.sq2_index.shape[0] == n_physical
        ):
            cand_sq2 = self.storage.sq2_index[t1_survivor_indices]
            sq2_scores = ScalarQuantizer2Bit.asymmetric_dot_product(cand_sq2, norm_query_256)
            k1_5 = min(tier2_k * 2, len(t1_survivor_indices))
            if len(sq2_scores) <= k1_5:
                order1_5 = np.argsort(-sq2_scores)
            else:
                part1_5 = np.argpartition(-sq2_scores, k1_5)[:k1_5]
                order1_5 = part1_5[np.argsort(-sq2_scores[part1_5])]
            t1_survivor_indices = [t1_survivor_indices[i] for i in order1_5]
            t1_survivor_hamming = [t1_survivor_hamming[i] for i in order1_5]

        # Prefetch surviving candidates into OS page cache before disk reads
        self.storage.prefetch_candidates(t1_survivor_indices)

        # =====================================================================
        # TIER 2: Matryoshka 256d Dot Product via Zero-Copy Memmap Slice
        # =====================================================================
        t2_start = time.perf_counter()
        k2 = min(tier2_k, len(t1_survivor_indices))

        # 1. Zero-copy read first 256 dimensions from disk memmap
        raw_vecs_256 = self.storage.get_mrl_slice(t1_survivor_indices, dim=EmbeddingGemmaClient.MRL_DIM)

        # 2. Re-normalize 256d candidate vectors
        norm_vecs_256 = EmbeddingGemmaClient.truncate_mrl(raw_vecs_256, dim=EmbeddingGemmaClient.MRL_DIM)

        # 3. Truncate and re-normalize 256d query vector
        norm_query_256 = EmbeddingGemmaClient.truncate_mrl(query_vector, dim=EmbeddingGemmaClient.MRL_DIM)

        # 4. Compute 256d dot products
        scores_256 = np.dot(norm_vecs_256, norm_query_256)

        # 5. Prune candidates down to tier2_k
        if len(scores_256) <= k2:
            order2 = np.argsort(-scores_256)
        else:
            partition = np.argpartition(-scores_256, k2)[:k2]
            order2 = partition[np.argsort(-scores_256[partition])]

        t2_survivor_indices = [t1_survivor_indices[i] for i in order2]
        t2_survivor_scores = [float(scores_256[i]) for i in order2]
        t2_survivor_hamming = [t1_survivor_hamming[i] for i in order2]
        t2_time = (time.perf_counter() - t2_start) * 1000.0

        # =====================================================================
        # TIER 3: Full 768d Exact Re-ranking via Zero-Copy Memmap Slice
        # =====================================================================
        t3_start = time.perf_counter()
        k3 = min(top_k, len(t2_survivor_indices))

        # 1. Zero-copy read full 768d vectors from disk memmap
        raw_vecs_768 = self.storage.get_full_vectors(t2_survivor_indices)

        # 2. Compute exact dot products
        scores_768 = np.dot(raw_vecs_768, query_vector)

        # 3. Final top_k sorting
        order3 = np.argsort(-scores_768)[:k3]

        final_results: List[SearchResult] = []
        for i in order3:
            orig_idx = t2_survivor_indices[i]
            doc_id = self.storage.offset_to_id[orig_idx]
            text = self.storage.offset_to_text[orig_idx]
            metadata = self.storage.metadata_cache[orig_idx]
            final_results.append(
                SearchResult(
                    id=doc_id,
                    score=float(scores_768[i]),
                    text=text,
                    metadata=metadata,
                    hamming_distance=t2_survivor_hamming[i],
                    tier2_score=t2_survivor_scores[i],
                    exact_score=float(scores_768[i]),
                )
            )

        t3_time = (time.perf_counter() - t3_start) * 1000.0
        total_time = (time.perf_counter() - t_start) * 1000.0

        stats = None
        if return_stats:
            stats = TierStats(
                total_records=n_physical,
                filtered_records=n_filtered,
                tier1_candidates=len(t1_survivor_indices),
                tier1_time_ms=t1_time,
                tier2_candidates=len(t2_survivor_indices),
                tier2_time_ms=t2_time,
                tier3_candidates=len(final_results),
                tier3_time_ms=t3_time,
                total_time_ms=total_time,
            )

        return final_results, stats

    def search_hybrid(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_predicate: Optional[Dict[str, Any]] = None,
        candidate_k: int = 500,
        tier2_k: int = 50,
        bm25_top_k: int = 50,
        rrf_constant: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        use_ivf: bool = True,
        n_probe: Optional[int] = None,
    ) -> List[HybridResult]:
        """Perform Hybrid Search fusing Dense Retrieval and BM25 via Reciprocal Rank Fusion.
        
        Formula:
            RRF_score(d) = (dense_weight / (rrf_constant + rank_dense(d))) +
                           (sparse_weight / (rrf_constant + rank_sparse(d)))
                           
        Args:
            query_text: Text string for BM25 keyword matching.
            query_vector: 768d float32 vector for hierarchical dense retrieval.
            top_k: Number of hybrid results to return.
            filter_predicate: Metadata pre-filtering dictionary.
            candidate_k: Tier 1 candidate pool size.
            tier2_k: Tier 2 candidate pool size.
            bm25_top_k: Candidate pool size for BM25.
            rrf_constant: Smoothing factor k (default: 60).
            dense_weight: Multiplier for dense rank contribution.
            sparse_weight: Multiplier for sparse rank contribution.
            use_ivf: If True and IVF trained, probe nearest clusters.
            n_probe: Number of nearest IVF clusters to probe.
            
        Returns:
            List of HybridResult items ranked by fused RRF score.
        """
        # 1. Run 3-tier dense retrieval
        dense_results, _ = self.search_dense(
            query_vector=query_vector,
            top_k=max(top_k * 2, tier2_k),
            filter_predicate=filter_predicate,
            candidate_k=candidate_k,
            tier2_k=tier2_k,
            use_ivf=use_ivf,
            n_probe=n_probe,
        )

        # 2. Run BM25 search with metadata pushdown and tombstones
        candidate_set: Optional[Set[int]] = None
        if filter_predicate or self.storage.segment_mgr.tombstones:
            n_tot = len(self.storage.offset_to_id)
            mask = np.ones(n_tot, dtype=bool)
            if filter_predicate:
                col_mask = self.storage.columnar.evaluate_mask(filter_predicate)
                if col_mask is not None:
                    mask &= col_mask
            if self.storage.segment_mgr.tombstones:
                mask &= self.storage.segment_mgr.get_active_mask(self.storage.offset_to_id)
            candidate_set = set(np.flatnonzero(mask).tolist())

        bm25_matches = self.bm25.search(
            query=query_text,
            candidate_indices=candidate_set,
            top_k=bm25_top_k,
        )

        # 3. Reciprocal Rank Fusion aggregation
        # doc_id -> {dense_rank, sparse_rank, dense_score, sparse_score}
        fusion_map: Dict[str, Dict[str, Any]] = {}

        # Record dense ranks (1-indexed)
        for rank, res in enumerate(dense_results, start=1):
            fusion_map[res.id] = {
                "dense_rank": rank,
                "dense_score": res.score,
                "sparse_rank": None,
                "sparse_score": None,
                "text": res.text,
                "metadata": res.metadata,
            }

        # Record sparse ranks (1-indexed)
        for rank, (offset, b_score) in enumerate(bm25_matches, start=1):
            doc_id = self.storage.offset_to_id[offset]
            if doc_id not in fusion_map:
                fusion_map[doc_id] = {
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": rank,
                    "sparse_score": b_score,
                    "text": self.storage.offset_to_text[offset],
                    "metadata": self.storage.metadata_cache[offset],
                }
            else:
                fusion_map[doc_id]["sparse_rank"] = rank
                fusion_map[doc_id]["sparse_score"] = b_score

        # 4. Compute RRF scores
        hybrid_items: List[HybridResult] = []
        for doc_id, data in fusion_map.items():
            rrf_score = 0.0
            d_rank = data["dense_rank"]
            s_rank = data["sparse_rank"]

            if d_rank is not None:
                rrf_score += dense_weight / (rrf_constant + d_rank)
            if s_rank is not None:
                rrf_score += sparse_weight / (rrf_constant + s_rank)

            hybrid_items.append(
                HybridResult(
                    id=doc_id,
                    rrf_score=rrf_score,
                    dense_rank=d_rank,
                    sparse_rank=s_rank,
                    dense_score=data["dense_score"],
                    sparse_score=data["sparse_score"],
                    text=data["text"],
                    metadata=data["metadata"],
                )
            )

        # 5. Sort descending by RRF score
        hybrid_items.sort(key=lambda item: item.rrf_score, reverse=True)
        return hybrid_items[:top_k]

