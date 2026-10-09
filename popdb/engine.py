"""High-level database engine facade unifying storage, WAL, BM25, and EmbeddingGemma."""

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from popdb.bm25 import BM25Index
from popdb.client import EmbeddingGemmaClient
from popdb.retriever import HierarchicalRetriever
from popdb.storage import StorageEngine
from popdb.types import Document, HybridResult, SearchResult, TierStats


class Engine:
    """PopDB Vector Database Engine.
    
    A high-performance embedded vector database engine optimized specifically
    for Google's EmbeddingGemma 2 model.
    
    Key Features:
    - Zero-copy reads via np.memmap for raw 768d float32 vectors.
    - Append-only vectors.bin and metadata.jsonl storage.
    - Write-Ahead Log (WAL) with CRC32 framing and atomic crash recovery.
    - In-memory 1-bit packed bitmask index (96 bytes/vector).
    - 3-tier coarse-to-fine hierarchical search:
        Tier 1: Hardware POPCOUNT Hamming filtering (prunes N -> 500).
        Tier 2: Matryoshka 256d normalized dot product (prunes 500 -> 50).
        Tier 3: Full 768d exact dot product (50 -> top_k).
    - Native Python BM25Okapi inverted index.
    - Hybrid Search with Reciprocal Rank Fusion (RRF).
    - Structured metadata predicate pushdown.
    - Instruction prefix routing:
        prompt_name="Document" for ingestion.
        prompt_name="SearchQuery" for search.
    """

    def __init__(
        self,
        db_path: str,
        model_name_or_path: str = "google/embeddinggemma-2",
        device: Optional[str] = None,
        use_mock_client: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Initialize Engine instance.
        
        Args:
            db_path: Directory path for database files.
            model_name_or_path: Embedding model name or directory.
            device: 'cuda', 'cpu', or None.
            use_mock_client: If True, use fast mock embeddings generator.
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
        """
        self.db_path = os.path.abspath(db_path)
        self.client = EmbeddingGemmaClient(
            model_name_or_path=model_name_or_path,
            device=device,
            lazy_load=True,
            use_mock=use_mock_client,
        )

        self.storage = StorageEngine(self.db_path)
        self.bm25 = BM25Index(k1=k1, b=b)

        # Hydrate BM25 inverted index from storage if recovering existing database
        if self.storage.count > 0:
            doc_indices = list(range(self.storage.count))
            self.bm25.add_documents(doc_indices, self.storage.offset_to_text)

        self.retriever = HierarchicalRetriever(self.storage, self.bm25)

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def count(self) -> int:
        """Return total number of documents in database."""
        return self.storage.count

    def insert(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
    ) -> str:
        """Insert a single document into the database.
        
        Args:
            text: Text body to index and embed.
            metadata: Structured attributes dict.
            doc_id: Optional unique identifier (generated if omitted).
            vector: Optional pre-computed 768d float32 vector.
            
        Returns:
            The document ID.
        """
        return self.insert_batch(
            texts=[text],
            metadatas=[metadata] if metadata is not None else None,
            doc_ids=[doc_id] if doc_id is not None else None,
            vectors=vector[np.newaxis, :] if vector is not None else None,
        )[0]

    def insert_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        doc_ids: Optional[List[str]] = None,
        vectors: Optional[np.ndarray] = None,
        batch_size: int = 32,
    ) -> List[str]:
        """Atomically insert a batch of documents with WAL durability.
        
        Args:
            texts: List of text bodies.
            metadatas: List of metadata dicts.
            doc_ids: List of unique document IDs.
            vectors: Optional pre-computed (N, 768) float32 embeddings.
            batch_size: Batch size for model embedding generation.
            
        Returns:
            List of document IDs.
        """
        n = len(texts)
        if n == 0:
            return []

        # Generate IDs if not provided
        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in range(n)]
        elif len(doc_ids) != n:
            raise ValueError(f"doc_ids length ({len(doc_ids)}) != texts length ({n})")

        # Normalize metadata
        if metadatas is None:
            metadatas = [{} for _ in range(n)]
        elif len(metadatas) != n:
            raise ValueError(f"metadatas length ({len(metadatas)}) != texts length ({n})")

        # Compute embeddings with prompt_name="Document" if vectors not passed
        if vectors is None:
            vectors = self.client.encode_documents(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        else:
            if vectors.shape != (n, EmbeddingGemmaClient.FULL_DIM):
                raise ValueError(
                    f"Expected vectors shape ({n}, {EmbeddingGemmaClient.FULL_DIM}), "
                    f"got {vectors.shape}"
                )
            # Ensure float32 contiguous array
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        # 1. Commit batch to storage and WAL
        offsets = self.storage.append_batch(doc_ids, texts, metadatas, vectors)

        # 2. Add to BM25 inverted index
        self.bm25.add_documents(offsets, texts)

        return doc_ids

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        candidate_k: int = 500,
        tier2_k: int = 50,
        query_vector: Optional[np.ndarray] = None,
        return_stats: bool = False,
    ) -> Union[List[SearchResult], Tuple[List[SearchResult], TierStats]]:
        """Run 3-tier coarse-to-fine cascading dense search.
        
        Tier 1: 1-bit binary Hamming filter + metadata pushdown (N -> candidate_k)
        Tier 2: Matryoshka 256d dot product (candidate_k -> tier2_k)
        Tier 3: Full 768d exact re-ranking (tier2_k -> top_k)
        
        Args:
            query: Query text string (encoded with prompt_name="SearchQuery").
            top_k: Number of final results.
            filter: Structured metadata filter dictionary.
            candidate_k: Number of Tier 1 survivors (default: 500).
            tier2_k: Number of Tier 2 survivors (default: 50).
            query_vector: Optional pre-computed 768d query vector.
            return_stats: If True, return (results, TierStats).
            
        Returns:
            List of SearchResult items (or tuple with TierStats).
        """
        if query_vector is None:
            query_vector = self.client.encode_query(query, normalize_embeddings=True)

        results, stats = self.retriever.search_dense(
            query_vector=query_vector,
            top_k=top_k,
            filter_predicate=filter,
            candidate_k=candidate_k,
            tier2_k=tier2_k,
            return_stats=return_stats,
        )
        if return_stats:
            return results, stats
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        candidate_k: int = 500,
        tier2_k: int = 50,
        bm25_top_k: int = 50,
        rrf_constant: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        query_vector: Optional[np.ndarray] = None,
    ) -> List[HybridResult]:
        """Run Hybrid Search combining Dense Cascade and BM25 via Reciprocal Rank Fusion.
        
        Args:
            query: Query string.
            top_k: Number of fused results to return.
            filter: Structured metadata filter.
            candidate_k: Number of Tier 1 dense candidates.
            tier2_k: Number of Tier 2 dense candidates.
            bm25_top_k: Number of BM25 candidates.
            rrf_constant: RRF constant factor k (default: 60).
            dense_weight: Weight for dense ranking.
            sparse_weight: Weight for BM25 sparse ranking.
            query_vector: Optional pre-computed query embedding.
            
        Returns:
            List of HybridResult items sorted by RRF score.
        """
        if query_vector is None:
            query_vector = self.client.encode_query(query, normalize_embeddings=True)

        return self.retriever.search_hybrid(
            query_text=query,
            query_vector=query_vector,
            top_k=top_k,
            filter_predicate=filter,
            candidate_k=candidate_k,
            tier2_k=tier2_k,
            bm25_top_k=bm25_top_k,
            rrf_constant=rrf_constant,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform pure BM25 keyword search with metadata pre-filtering."""
        from popdb.filter import MetadataFilter
        candidate_set = None
        if filter:
            eligible_list = MetadataFilter.filter_indices(
                self.storage.metadata_cache,
                filter,
            )
            candidate_set = set(eligible_list)

        matches = self.bm25.search(query, candidate_indices=candidate_set, top_k=top_k)
        results = []
        for offset, score in matches:
            doc_id = self.storage.offset_to_id[offset]
            results.append(
                SearchResult(
                    id=doc_id,
                    score=score,
                    text=self.storage.offset_to_text[offset],
                    metadata=self.storage.metadata_cache[offset],
                )
            )
        return results

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Fetch document text and metadata by ID."""
        return self.storage.get_document(doc_id)

    def delete(self, doc_id: str) -> bool:
        """Atomically soft-delete a document with WAL durability."""
        return self.storage.delete(doc_id)

    def compact(self) -> int:
        """Perform physical storage compaction: defragment vectors and purge deleted records.
        
        Returns:
            Number of physically purged records.
        """
        purged = self.storage.compact()
        # Re-hydrate BM25 index with cleaned corpus
        self.bm25 = BM25Index(k1=self.bm25.k1, b=self.bm25.b)
        if self.storage.count > 0:
            self.bm25.add_documents(list(range(self.storage.count)), self.storage.offset_to_text)
        self.retriever.bm25 = self.bm25
        return purged

    def checkpoint(self) -> None:
        """Checkpoint the Write-Ahead Log to optimize disk usage."""
        self.storage.checkpoint()

    def close(self) -> None:
        """Safely close storage handles and memory mappings."""
        self.storage.close()
