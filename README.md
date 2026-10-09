# PopDB: High-Performance Embedded Vector Database Engine for EmbeddingGemma 2

PopDB is a lightweight, embedded vector database engine written in Python, custom-engineered for Google's **EmbeddingGemma 2** (`google/embeddinggemma-2`) model. It integrates hardware POPCOUNT-accelerated binary quantization, Matryoshka Representation Learning (MRL), zero-copy memory mapping (`np.memmap`), ACID write-ahead logging (WAL), native BM25Okapi sparse search, and Reciprocal Rank Fusion (RRF).

---

## Key Architectural Highlights

```
                          ┌───────────────────────────────────────────────┐
                          │   Search Query: q (prompt_name="SearchQuery") │
                          └──────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Tier 1: In-Memory 1-Bit Binary Quantization & Hardware POPCOUNT Filtering                       │
│ • Predicate Pushdown: Metadata boolean filters prune disqualified records before distance eval  │
│ • Hamming Dist: bitwise_count(bitwise_xor(bits_doc, bits_query)).sum(axis=-1)                   │
│ • Hardware POPCNT instructions prune N candidates -> 500 (96 bytes RAM/vector)                  │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼ (Top 500 Candidates)
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Tier 2: Matryoshka (MRL 256d) Dot Product via Zero-Copy Disk Memmap                             │
│ • np.memmap reads ONLY the first 256 float32 dimensions directly from OS page cache             │
│ • L2-renormalize sub-vectors: v_256 / ||v_256||_2                                               │
│ • Dot product reduces 500 candidates -> 50 candidates                                           │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼ (Top 50 Candidates)
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Tier 3: Full 768d Exact Dot Product Re-ranking via Zero-Copy Disk Memmap                        │
│ • np.memmap reads full 768d float32 vectors for the 50 survivors                                │
│ • Exact cosine similarity dot product yields top_k final results                                │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         ▼                                               ▼
              ┌─────────────────────┐                         ┌─────────────────────┐
              │ Dense Ranked List   │                         │ BM25 Inverted Index │
              └──────────┬──────────┘                         └──────────┬──────────┘
                         │                                               │
                         └───────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
        ┌─────────────────────────────────────────────────────────────────────────────────┐
        │ Hybrid Search: Reciprocal Rank Fusion (RRF)                                     │
        │ RRF_score(d) = 1 / (60 + rank_dense(d)) + 1 / (60 + rank_sparse(d))             │
        └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components & Features

### 1. Model Integration (EmbeddingGemma 2)
- **Model**: `google/embeddinggemma-2` loaded natively via `sentence-transformers`.
- **Instruction Prefix Routing**:
  - `prompt_name="Document"` for document ingestion.
  - `prompt_name="SearchQuery"` for query execution.
- **Matryoshka Representation Learning (MRL)**:
  - Full dimension: $768$ float32 values.
  - Truncated dimension: $256$ float32 values with unit L2 re-normalization.

### 2. Zero-Copy Storage & Memory Engine
- **`vectors.bin`**: Contiguous append-only binary array of raw IEEE 754 float32 vectors (3,072 bytes per vector).
- **`np.memmap` Zero-Copy Reads**: Direct operating system page cache access without duplicating memory in Python RAM.
- **`metadata.jsonl`**: Append-only JSON Lines format storing doc ID, payload text, and structured attributes.
- **`wal.log`**: Write-Ahead Log featuring binary framing, monotonically increasing `tx_id`, operation codes (`BEGIN`, `INSERT`, `COMMIT`, `CHECKPOINT`), CRC32 checksums, and `os.fsync` durability.
- **In-Memory 1-Bit Index**: 1 byte per 8 dimensions = **96 bytes/vector** in RAM (a **32x RAM reduction** compared to raw 768d float32).

### 3. Hierarchical 3-Tier Coarse-to-Fine Cascade
1. **Tier 1 (Binary Quantization & Hardware POPCOUNT)**:
   - Query vector quantized to 1-bit ($\mathbf{q} > 0$).
   - Vectorized bitwise XOR with `np.bitwise_xor` and hardware `np.bitwise_count` (compiles to CPU `POPCNT` instruction).
   - Fast $O(N)$ candidate pruning via `np.argpartition` down to top 500.
2. **Tier 2 (Matryoshka 256d Dot Product)**:
   - Zero-copy memmap slice of first 256 dimensions.
   - Truncated vectors re-normalized to unit hypersphere.
   - Prunes from 500 down to 50 candidates.
3. **Tier 3 (Full 768d Exact Re-ranking)**:
   - Zero-copy memmap slice of full 768d vectors for top 50 candidates.
   - Exact cosine dot product produces final `top_k`.

### 4. Hybrid Search with Reciprocal Rank Fusion (RRF)
- Lightweight native Python **BM25Okapi** inverted index with term frequency saturation ($k_1=1.5$) and document length normalization ($b=0.75$).
- Blended with Dense retrieval rankings via RRF:
  $$\text{RRF\_score}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{sparse}}(d)}$$

### 5. Metadata Predicate Pushdown
- Evaluated directly during **Tier 1** before Hamming distance calculations or disk memmap reads.
- Supports MongoDB-style operators:
  `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$contains`, `$exists`, `$and`, `$or`, `$not`.

---

## Quickstart

```python
from popdb import Engine

# Initialize embedded database
with Engine("./my_vector_db") as db:
    # 1. Ingest documents with structured metadata
    db.insert(
        text="Hardware POPCOUNT accelerates 1-bit binary vector search.",
        metadata={"category": "Hardware", "year": 2024, "tags": ["simd", "popcount"]},
        doc_id="doc_1",
    )
    db.insert(
        text="Google EmbeddingGemma 2 supports native 256d Matryoshka truncation.",
        metadata={"category": "AI", "year": 2024, "tags": ["gemma", "mrl"]},
        doc_id="doc_2",
    )

    # 2. 3-Tier Cascading Dense Search with Metadata Pushdown
    results = db.search(
        query="fast binary quantized similarity search",
        top_k=5,
        filter={"category": "Hardware", "year": {"$gte": 2024}},
    )
    for res in results:
        print(f"[{res.id}] Score: {res.score:.4f} | Text: {res.text}")

    # 3. Hybrid Search (Dense + BM25 via RRF)
    hybrid_results = db.hybrid_search(
        query="EmbeddingGemma Matryoshka",
        top_k=5,
    )
    for res in hybrid_results:
        print(f"[{res.id}] RRF Score: {res.rrf_score:.5f} | Dense Rank: {res.dense_rank} | Sparse Rank: {res.sparse_rank}")
```

---

## Running the Demonstration & Tests

### Run the full demonstration:
```bash
python demo.py
```
Or with synthetic mock embeddings (offline mode):
```bash
python demo.py --mock
```

### Run the test suite:
```bash
python -m pytest tests/test_popdb.py -v
```

---

## Project Structure

```
popdb/
├── __init__.py           # Package exports
├── types.py              # Document, SearchResult, HybridResult, TierStats
├── client.py             # EmbeddingGemmaClient (SentenceTransformer, prompts, MRL, 1-bit)
├── wal.py                # WALManager (atomic framing, CRC32, fsync, crash replay)
├── storage.py            # StorageEngine (vectors.bin, metadata.jsonl, np.memmap)
├── bm25.py               # BM25Index native inverted index
├── filter.py             # MetadataFilter predicate pushdown engine
├── retriever.py          # HierarchicalRetriever (Tier 1 -> Tier 2 -> Tier 3 + RRF)
└── engine.py             # Engine facade
demo.py                   # Complete runnable demonstration
tests/
└── test_popdb.py         # Pytest test suite
```

