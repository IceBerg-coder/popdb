"""PopDB: End-to-End Demonstration and Architecture Benchmark.

Showcases:
1. EmbeddingGemma 2 Model Integration (MRL 768d -> 256d, Document/SearchQuery prompt names).
2. Zero-Copy np.memmap Storage Engine & Append-Only vectors.bin / metadata.jsonl.
3. Write-Ahead Log (WAL) with CRC32 framing and atomic crash recovery.
4. 3-Tier Hierarchical Coarse-to-Fine Search (Hardware POPCOUNT -> MRL 256d -> Full 768d).
5. Structured Metadata Predicate Pushdown ($gte, $in, $or, $and).
6. Hybrid Search with BM25 Okapi & Reciprocal Rank Fusion (RRF).
7. Cold Restart, Disk Persistence & WAL Replay Verification.
"""

import os
import shutil
import sys
import time
from typing import Dict, List
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure popdb is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from popdb.client import EmbeddingGemmaClient
from popdb.engine import Engine
from popdb.wal import WALManager, WALRecordType


SAMPLE_DOCUMENTS = [
    {
        "id": "doc_01",
        "text": "Google released EmbeddingGemma 2, an open-weight embedding model featuring native Matryoshka Representation Learning (MRL) from 768d down to 256d.",
        "metadata": {"category": "AI", "year": 2024, "rating": 4.9, "tags": ["gemma", "google", "mrl", "embeddings"], "author": "Google DeepMind"},
    },
    {
        "id": "doc_02",
        "text": "Matryoshka Representation Learning (MRL) allows slicing embeddings at 256 dimensions with negligible loss in downstream retrieval accuracy.",
        "metadata": {"category": "AI", "year": 2024, "rating": 4.8, "tags": ["mrl", "embeddings", "compression"], "author": "Research Group"},
    },
    {
        "id": "doc_03",
        "text": "Binary Quantization compresses 768-dimensional float32 vectors into 96-byte packed bitmasks using 1-bit sign thresholding v > 0.",
        "metadata": {"category": "AI", "year": 2024, "rating": 4.7, "tags": ["quantization", "popcount", "compression"], "author": "VectorDB Lab"},
    },
    {
        "id": "doc_04",
        "text": "Hardware POPCOUNT instructions execute bitwise XOR and count set bits at tens of gigabytes per second across modern x86-64 and ARM processors.",
        "metadata": {"category": "Hardware", "year": 2023, "rating": 4.6, "tags": ["popcount", "cpu", "simd", "performance"], "author": "Systems Architect"},
    },
    {
        "id": "doc_05",
        "text": "PostgreSQL Write-Ahead Logging (WAL) ensures atomic ACID durability by logging all data modifications to disk before altering data pages.",
        "metadata": {"category": "Database", "year": 2023, "rating": 4.9, "tags": ["wal", "acid", "postgres", "storage"], "author": "Database Team"},
    },
    {
        "id": "doc_06",
        "text": "Memory-mapped files via np.memmap allow zero-copy reading of multi-gigabyte vector binary arrays directly backed by OS virtual memory pages.",
        "metadata": {"category": "Database", "year": 2024, "rating": 4.8, "tags": ["memmap", "numpy", "zero-copy", "storage"], "author": "Systems Engineer"},
    },
    {
        "id": "doc_07",
        "text": "BM25 Okapi is a probabilistic ranking function used in information retrieval to score document relevance based on term frequency and inverted indices.",
        "metadata": {"category": "Search", "year": 2022, "rating": 4.5, "tags": ["bm25", "ir", "inverted-index", "sparse"], "author": "Information Retrieval Group"},
    },
    {
        "id": "doc_08",
        "text": "Reciprocal Rank Fusion (RRF) seamlessly combines dense semantic neural rankings and sparse BM25 lexical rankings without needing score normalization.",
        "metadata": {"category": "Search", "year": 2024, "rating": 4.8, "tags": ["rrf", "hybrid", "search", "dense-sparse"], "author": "Search Team"},
    },
    {
        "id": "doc_09",
        "text": "Task-specific instruction prefixes in EmbeddingGemma route Document indexing and SearchQuery retrieval to optimize asymmetric search alignment.",
        "metadata": {"category": "AI", "year": 2024, "rating": 4.7, "tags": ["gemma", "prompts", "retrieval"], "author": "Google DeepMind"},
    },
    {
        "id": "doc_10",
        "text": "Vector databases leverage hierarchical cascading pipelines to filter millions of candidates into hundreds using coarse hamming distance before exact reranking.",
        "metadata": {"category": "Database", "year": 2024, "rating": 4.9, "tags": ["vectordb", "cascade", "hierarchical", "reranking"], "author": "PopDB Engineer"},
    },
    {
        "id": "doc_11",
        "text": "Linux page cache transparently caches frequently accessed memmap blocks in RAM, reducing SSD read latency for candidate vector retrieval.",
        "metadata": {"category": "Systems", "year": 2023, "rating": 4.6, "tags": ["os", "kernel", "page-cache", "memmap"], "author": "Kernel Developer"},
    },
    {
        "id": "doc_12",
        "text": "Metadata predicate pushdown eliminates unqualified documents during coarse binary filtering, drastically reducing memory bandwidth and compute.",
        "metadata": {"category": "Database", "year": 2024, "rating": 4.8, "tags": ["predicate-pushdown", "filtering", "query-optimizer"], "author": "PopDB Engineer"},
    },
]


def print_banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def run_demo() -> None:
    demo_db_path = os.path.abspath("./popdb_demo_data")

    # Clean existing demo folder if present
    if os.path.exists(demo_db_path):
        shutil.rmtree(demo_db_path, ignore_errors=True)

    print_banner("PopDB: High-Performance Embedded Vector Database for EmbeddingGemma 2")
    print("Architecture Overview:")
    print(" • Model: google/embeddinggemma-2 (768d native, 256d Matryoshka MRL)")
    print(" • Prompts: prompt_name='Document' (index) | prompt_name='SearchQuery' (search)")
    print(" • Storage: append-only 'vectors.bin' (np.memmap zero-copy) & 'metadata.jsonl'")
    print(" • Durability: 'wal.log' (CRC32 checksummed Write-Ahead Log)")
    print(" • Tier 1 Index: 1-bit packed bitmasks (96 bytes/vector) with hardware POPCOUNT")
    print(" • Tier 2 Re-rank: Matryoshka 256d normalized dot product via memmap slice")
    print(" • Tier 3 Re-rank: Full 768d exact dot product via memmap slice")
    print(" • Hybrid Search: Native BM25Okapi + Reciprocal Rank Fusion (RRF)")

    # Detect if we should run with real model or fallback
    use_mock = "--mock" in sys.argv
    model_name = "google/embeddinggemma-2"
    device = "cpu"

    print(f"\nInitializing PopDB Engine at: {demo_db_path}")
    print(f"Embedding model mode: {'Synthetic MRL Mock' if use_mock else 'Real SentenceTransformers (google/embeddinggemma-2)'}")

    # =========================================================================
    # STEP 1: Ingestion & Write-Ahead Log Verification
    # =========================================================================
    print_banner("Step 1: Document Ingestion & Atomic WAL Durability")
    with Engine(
        db_path=demo_db_path,
        model_name_or_path=model_name,
        device=device,
        use_mock_client=use_mock,
    ) as db:
        print(f"Ingesting {len(SAMPLE_DOCUMENTS)} technical documents...")
        t0 = time.perf_counter()

        texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]
        metas = [doc["metadata"] for doc in SAMPLE_DOCUMENTS]
        doc_ids = [doc["id"] for doc in SAMPLE_DOCUMENTS]

        assigned_ids = db.insert_batch(texts=texts, metadatas=metas, doc_ids=doc_ids)
        t_ingest = (time.perf_counter() - t0) * 1000.0

        print(f"[OK] Ingested {len(assigned_ids)} documents in {t_ingest:.2f} ms")
        print(f"Total committed database count: {db.count()} documents")

        # Inspect disk footprint
        vec_size = os.path.getsize(db.storage.vectors_path)
        meta_size = os.path.getsize(db.storage.metadata_path)
        wal_size = os.path.getsize(db.storage.wal_path)
        ram_bits_size = db.storage.bitmask_index.nbytes

        print("\nPhysical Storage Footprint:")
        print(f" • vectors.bin   : {vec_size:,} bytes ({len(assigned_ids)} x 768 float32)")
        print(f" • metadata.jsonl: {meta_size:,} bytes")
        print(f" • wal.log       : {wal_size:,} bytes (CRC32 framed WAL entries)")
        print(f" • RAM bitmasks  : {ram_bits_size:,} bytes (96 bytes/vector in RAM)")

        # =========================================================================
        # STEP 2: 3-Tier Hierarchical Coarse-to-Fine Search
        # =========================================================================
        print_banner("Step 2: 3-Tier Hierarchical Cascading Search")
        query = "How does hardware popcount and binary quantization speed up vector similarity?"
        print(f"Query: '{query}'")
        print("Routing query via prompt_name='SearchQuery'...")

        results, stats = db.search(
            query=query,
            top_k=3,
            candidate_k=8,  # Tier 1 pool size
            tier2_k=4,      # Tier 2 pool size
            return_stats=True,
        )

        print_section("Cascade Pipeline Telemetry (TierStats)")
        print(f"Total Corpus Records    : {stats.total_records}")
        print(f"Tier 1 (Bitwise POPCOUNT): {stats.total_records} -> {stats.tier1_candidates} candidates in {stats.tier1_time_ms:.3f} ms")
        print(f"Tier 2 (MRL 256d Dot)    : {stats.tier1_candidates} -> {stats.tier2_candidates} candidates in {stats.tier2_time_ms:.3f} ms (zero-copy memmap slice)")
        print(f"Tier 3 (Full 768d Exact) : {stats.tier2_candidates} -> {stats.tier3_candidates} candidates in {stats.tier3_time_ms:.3f} ms (zero-copy memmap slice)")
        print(f"Total Retrieval Latency : {stats.total_time_ms:.3f} ms")

        print_section("Top-3 Retrieved Results (Dense Cascade)")
        for rank, res in enumerate(results, start=1):
            print(f"#{rank} [ID: {res.id}] Exact Score: {res.score:.4f} | MRL 256d: {res.tier2_score:.4f} | Hamming Dist: {res.hamming_distance}/768")
            print(f"   Text: {res.text}")
            print(f"   Meta: {res.metadata}\n")

        # =========================================================================
        # STEP 3: Structured Metadata Predicate Pushdown
        # =========================================================================
        print_banner("Step 3: Structured Metadata Predicate Pushdown")
        query3 = "embedding models and vector representation learning"
        
        # Predicate: Category == 'AI' and Year >= 2024
        predicate = {
            "category": "AI",
            "year": {"$gte": 2024},
            "rating": {"$gte": 4.8},
        }

        print(f"Query: '{query3}'")
        print(f"Pushdown Filter Predicate: {predicate}")
        print("Pushdown Engine: Filtering evaluated directly at Tier 1 before Hamming/memmap computation!")

        results3, stats3 = db.search(
            query=query3,
            top_k=3,
            filter=predicate,
            return_stats=True,
        )

        print(f"\nPushdown Pruning Stats: Total={stats3.total_records} -> Eligible={stats3.filtered_records} -> Returned={len(results3)}")
        for rank, res in enumerate(results3, start=1):
            print(f"#{rank} [ID: {res.id}] Score: {res.score:.4f} | Category: {res.metadata['category']} | Year: {res.metadata['year']} | Rating: {res.metadata['rating']}")
            print(f"   Text: {res.text}\n")

        # =========================================================================
        # STEP 4: Hybrid Search with BM25 & Reciprocal Rank Fusion (RRF)
        # =========================================================================
        print_banner("Step 4: Hybrid Search (BM25 Okapi + Dense Cascade + RRF)")
        hybrid_query = "PostgreSQL WAL atomic ACID durability"
        print(f"Hybrid Query: '{hybrid_query}'")
        print("Formula: RRF_score(d) = (1 / (60 + rank_dense(d))) + (1 / (60 + rank_sparse(d)))")

        hybrid_results = db.hybrid_search(
            query=hybrid_query,
            top_k=4,
            dense_weight=1.0,
            sparse_weight=1.0,
        )

        print_section("Rank Fusion Results")
        for rank, item in enumerate(hybrid_results, start=1):
            d_rank_str = f"Rank {item.dense_rank}" if item.dense_rank else "N/A"
            s_rank_str = f"Rank {item.sparse_rank} (score: {item.sparse_score:.2f})" if item.sparse_rank else "N/A"
            print(f"#{rank} [ID: {item.id}] RRF Score: {item.rrf_score:.5f}")
            print(f"   Dense: {d_rank_str} | Sparse (BM25): {s_rank_str}")
            print(f"   Text: {item.text}")
            print(f"   Category: {item.metadata.get('category')} | Tags: {item.metadata.get('tags')}\n")

    # =========================================================================
    # STEP 5: Persistence, Cold Restart & WAL Crash Recovery
    # =========================================================================
    print_banner("Step 5: Database Cold Restart & Durability Verification")
    print("1. Simulating power loss with an uncommitted transaction in WAL...")

    # Inject an uncommitted transaction into WAL to test crash recovery
    wal_path = os.path.join(demo_db_path, "wal.log")
    wal = WALManager(wal_path)
    crash_tx = wal.next_tx_id()
    wal.begin_transaction(crash_tx, 1)
    fake_vec = np.zeros(768, dtype=np.float32)
    wal.log_insert(crash_tx, "corrupt_tx_doc", "This transaction was interrupted mid-flight", {"status": "uncommitted"}, fake_vec)
    # Intentionally do NOT call commit_transaction!
    wal.close()
    print("   [Injected] Uncommitted transaction record written to WAL (simulating power failure).")

    print("\n2. Reopening PopDB Engine from cold disk storage...")
    with Engine(
        db_path=demo_db_path,
        model_name_or_path=model_name,
        device=device,
        use_mock_client=use_mock,
    ) as db_reloaded:
        print(f"   [Recovered] Active committed documents: {db_reloaded.count()}")
        assert "corrupt_tx_doc" not in db_reloaded.storage.id_to_offset
        print("   [Verified] Uncommitted transaction was discarded cleanly; database integrity intact.")

        # Re-run query on cold-started database
        res = db_reloaded.search("Matryoshka representation learning 256 dimensions", top_k=1)
        print(f"\n   Cold Retrieval Verification Match:")
        print(f"   [ID: {res[0].id}] Score: {res[0].score:.4f}")
        print(f"   Text: {res[0].text}")

        # Add new document to verify continued append operation
        new_doc_id = db_reloaded.insert(
            text="Antigravity high-speed AI engineering with vector databases and Matryoshka learning.",
            metadata={"category": "AI", "year": 2026, "tags": ["antigravity", "vectordb"]},
        )
        print(f"\n   [OK] Appended new document '{new_doc_id}' after recovery. New count: {db_reloaded.count()}")

    print_banner("Demonstration Completed Successfully!")


if __name__ == "__main__":
    run_demo()
