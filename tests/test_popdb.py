"""Comprehensive unit and integration tests for PopDB."""

import os
import shutil
import tempfile
import numpy as np
import pytest

from popdb import (
    BM25Index,
    EmbeddingGemmaClient,
    Engine,
    HierarchicalRetriever,
    MetadataFilter,
    StorageEngine,
    WALManager,
)
from popdb.wal import WALRecordType


class TestEmbeddingGemmaClient:
    """Test client logic: MRL, 1-bit packing, and embedding generation."""

    def test_mrl_truncation(self):
        rng = np.random.RandomState(42)
        vecs = rng.randn(10, 768).astype(np.float32)
        # Normalize
        vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)

        mrl_256 = EmbeddingGemmaClient.truncate_mrl(vecs, dim=256)
        assert mrl_256.shape == (10, 256)
        # Verify L2 unit norm
        norms = np.linalg.norm(mrl_256, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_quantize_1bit(self):
        rng = np.random.RandomState(42)
        vecs = rng.randn(5, 768).astype(np.float32)
        packed = EmbeddingGemmaClient.quantize_1bit(vecs)
        assert packed.shape == (5, 96)
        assert packed.dtype == np.uint8

        # Verify bits
        unpacked = np.unpackbits(packed, axis=-1)
        expected = (vecs > 0).astype(np.uint8)
        np.testing.assert_array_equal(unpacked, expected)

    def test_mock_client_prefixes(self):
        client = EmbeddingGemmaClient(use_mock=True)
        docs = ["Hello world", "Vector search database"]
        doc_embs = client.encode_documents(docs)
        assert doc_embs.shape == (2, 768)

        query_emb = client.encode_query("search engine")
        assert query_emb.shape == (768,)

        # Verify unit norm
        np.testing.assert_allclose(np.linalg.norm(doc_embs, axis=-1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(query_emb), 1.0, atol=1e-5)

    def test_real_embeddinggemma_model(self):
        """Test actual google/embeddinggemma-2 model inference with Document & SearchQuery prompts."""
        client = EmbeddingGemmaClient(
            model_name_or_path="google/embeddinggemma-2",
            device="cpu",
            use_mock=False,
        )
        texts = ["Vector database indexing document", "Reciprocal rank fusion"]
        doc_embs = client.encode_documents(texts)
        assert doc_embs.shape == (2, 768)
        assert doc_embs.dtype == np.float32

        query_emb = client.encode_query("vector search engine")
        assert query_emb.shape == (768,)
        assert query_emb.dtype == np.float32

        # Verify unit norm
        np.testing.assert_allclose(np.linalg.norm(doc_embs, axis=-1), 1.0, atol=1e-3)
        np.testing.assert_allclose(np.linalg.norm(query_emb), 1.0, atol=1e-3)


class TestMetadataFilter:
    """Test metadata pushdown filter engine."""

    def test_filter_operators(self):
        doc = {
            "title": "EmbeddingGemma Paper",
            "category": "AI",
            "year": 2024,
            "rating": 4.8,
            "tags": ["nlp", "embeddings", "mrl"],
        }

        # Exact match
        assert MetadataFilter.matches(doc, {"category": "AI"})
        assert not MetadataFilter.matches(doc, {"category": "Database"})

        # Comparison operators
        assert MetadataFilter.matches(doc, {"year": {"$gte": 2024}})
        assert MetadataFilter.matches(doc, {"year": {"$gt": 2023, "$lte": 2025}})
        assert not MetadataFilter.matches(doc, {"year": {"$lt": 2024}})

        # $in and $nin
        assert MetadataFilter.matches(doc, {"tags": {"$in": ["nlp", "vision"]}})
        assert not MetadataFilter.matches(doc, {"tags": {"$in": ["vision", "audio"]}})
        assert MetadataFilter.matches(doc, {"category": {"$nin": ["Web", "DevOps"]}})

        # Compound $and, $or, $not
        assert MetadataFilter.matches(doc, {
            "$and": [
                {"category": "AI"},
                {"year": {"$gte": 2024}},
            ]
        })

        assert MetadataFilter.matches(doc, {
            "$or": [
                {"category": "Physics"},
                {"rating": {"$gt": 4.5}},
            ]
        })

        assert MetadataFilter.matches(doc, {
            "$not": {"category": "Web"}
        })


class TestWALDurability:
    """Test Write-Ahead Log crash recovery and atomic guarantees."""

    def test_wal_recovery_committed_vs_uncommitted(self):
        temp_dir = tempfile.mkdtemp()
        try:
            wal_path = os.path.join(temp_dir, "wal.log")
            wal = WALManager(wal_path)

            v1 = np.ones(768, dtype=np.float32)
            v2 = np.ones(768, dtype=np.float32) * 2

            # Tx 1: Committed
            tx1 = wal.next_tx_id()
            wal.begin_transaction(tx1, 1)
            wal.log_insert(tx1, "doc1", "text1", {"val": 1}, v1)
            wal.commit_transaction(tx1)

            # Tx 2: Uncommitted (simulating crash before commit)
            tx2 = wal.next_tx_id()
            wal.begin_transaction(tx2, 1)
            wal.log_insert(tx2, "doc2", "text2", {"val": 2}, v2)
            # No commit!

            wal.close()

            # Recover with new WALManager instance
            wal_recovery = WALManager(wal_path)
            recovered = wal_recovery.recover()
            wal_recovery.close()

            # Only tx1 should be recovered!
            assert len(recovered) == 1
            doc_id, text, meta, vec = recovered[0]
            assert doc_id == "doc1"
            assert text == "text1"
            assert meta == {"val": 1}
            np.testing.assert_array_equal(vec, v1)
        finally:
            shutil.rmtree(temp_dir)


class TestBM25Index:
    """Test native BM25 inverted index."""

    def test_bm25_search_and_pushdown(self):
        bm25 = BM25Index()
        texts = [
            "Information retrieval using BM25 and inverted index",
            "Dense vector embeddings with neural models",
            "Hybrid search combining BM25 and vector embeddings",
        ]
        bm25.add_documents([0, 1, 2], texts)

        # Keyword match
        results = bm25.search("retrieval inverted")
        assert len(results) > 0
        assert results[0][0] == 0  # First doc is top match

        # Hybrid keyword match
        results_hybrid = bm25.search("hybrid embeddings")
        assert len(results_hybrid) > 0
        assert results_hybrid[0][0] == 2  # Third doc is top match

        # Metadata pushdown: exclude doc 2
        results_filtered = bm25.search("hybrid embeddings", candidate_indices={0, 1})
        top_indices = [idx for idx, _ in results_filtered]
        assert 2 not in top_indices


class TestHierarchicalEngine:
    """Integration test for full Engine: cascade, RRF, persistence, and zero-copy memmap."""

    def test_engine_lifecycle_and_persistence(self):
        temp_dir = tempfile.mkdtemp()
        try:
            # 1. Initialize Engine and ingest sample data
            with Engine(temp_dir, use_mock_client=True) as db:
                docs = [
                    "Google announced EmbeddingGemma 2 with native MRL support",
                    "PostgreSQL uses WAL for write ahead logging and ACID consistency",
                    "Binary quantization accelerates vector retrieval using hardware popcount",
                    "Matryoshka Representation Learning allows truncating embeddings to 256 dimensions",
                    "Reciprocal Rank Fusion fuses dense semantic search and BM25 sparse search",
                ]
                metadatas = [
                    {"category": "AI", "year": 2024, "tags": ["gemma", "google"]},
                    {"category": "Database", "year": 2023, "tags": ["postgres", "wal"]},
                    {"category": "AI", "year": 2024, "tags": ["quantization", "popcount"]},
                    {"category": "AI", "year": 2024, "tags": ["mrl", "embeddings"]},
                    {"category": "Search", "year": 2024, "tags": ["rrf", "hybrid"]},
                ]
                ids = [f"doc_{i}" for i in range(len(docs))]
                db.insert_batch(docs, metadatas=metadatas, doc_ids=ids)

                assert db.count() == 5

                # Search with predicate pushdown: category="AI"
                results, stats = db.search(
                    query="EmbeddingGemma MRL models",
                    top_k=3,
                    filter={"category": "AI"},
                    return_stats=True,
                )
                assert len(results) <= 3
                for r in results:
                    assert r.metadata["category"] == "AI"
                assert stats.filtered_records == 3

                # Hybrid search
                hybrid_results = db.hybrid_search(
                    query="popcount binary quantization",
                    top_k=2,
                )
                assert len(hybrid_results) == 2
                assert hybrid_results[0].id == "doc_2"
                assert hybrid_results[0].sparse_rank is not None

                # Test document retrieval
                doc0 = db.get("doc_0")
                assert doc0 is not None
                assert "EmbeddingGemma 2" in doc0["text"]

            # 2. Re-open engine from same directory to verify disk persistence & WAL recovery
            with Engine(temp_dir, use_mock_client=True) as db_reloaded:
                assert db_reloaded.count() == 5

                # Search on reloaded engine
                reloaded_results = db_reloaded.search(
                    query="hardware popcount binary",
                    top_k=1,
                )
                assert len(reloaded_results) == 1
                assert reloaded_results[0].id in [f"doc_{i}" for i in range(5)]
                assert reloaded_results[0].exact_score is not None

                # Hybrid search on reloaded engine (keyword BM25 + dense)
                reloaded_hybrid = db_reloaded.hybrid_search(
                    query="popcount binary quantization",
                    top_k=1,
                )
                assert len(reloaded_hybrid) == 1
                assert reloaded_hybrid[0].id == "doc_2"

                # Insert one more document
                new_id = db_reloaded.insert(
                    text="New document added after reload",
                    metadata={"category": "AI", "year": 2025},
                )
                assert db_reloaded.count() == 6
                assert db_reloaded.get(new_id)["metadata"]["year"] == 2025

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
