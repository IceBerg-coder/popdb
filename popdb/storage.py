"""Storage & Memory Engine managing vectors.bin, metadata.jsonl, memmap, and WAL."""

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from popdb.client import EmbeddingGemmaClient
from popdb.wal import WALManager


class StorageEngine:
    """Manages zero-copy vector storage, JSONL metadata, WAL, and in-memory bitmasks.
    
    Files:
    - vectors.bin: Contiguous binary float32 array file (append-only, shape (N, 768)).
    - metadata.jsonl: Append-only JSON lines file containing IDs, texts, and attributes.
    - wal.log: Write-Ahead Log for atomic transaction durability.
    
    In-memory indexes:
    - bitmask_index: (N, 96) uint8 array (1 byte per 8 dimensions = 96 bytes/vector).
    - metadata_cache: List of metadata dicts for predicate pushdown evaluation.
    - id_to_offset: Map of document string IDs to contiguous row indices.
    """

    VECTOR_DIM = 768
    BYTES_PER_VECTOR = VECTOR_DIM * 4  # float32 = 4 bytes -> 3072 bytes per vector
    PACKED_BYTES = 96  # 768 bits / 8 = 96 bytes

    def __init__(self, db_dir: str):
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)

        self.vectors_path = os.path.join(db_dir, "vectors.bin")
        self.metadata_path = os.path.join(db_dir, "metadata.jsonl")
        self.wal_path = os.path.join(db_dir, "wal.log")

        self.wal = WALManager(self.wal_path)
        self._memmap: Optional[np.memmap] = None
        self._vector_file = None
        self._metadata_file = None

        # In-memory structures
        self.bitmask_index: np.ndarray = np.empty((0, self.PACKED_BYTES), dtype=np.uint8)
        self.metadata_cache: List[Dict[str, Any]] = []
        self.id_to_offset: Dict[str, int] = {}
        self.offset_to_id: List[str] = []
        self.offset_to_text: List[str] = []

        self._recover_and_initialize()

    @property
    def count(self) -> int:
        """Total number of committed vectors in database."""
        return len(self.offset_to_id)

    def _recover_and_initialize(self) -> None:
        """Synchronize disk files, replay uncommitted/committed WAL, and build in-memory indexes."""
        # 1. Ensure files exist
        if not os.path.exists(self.vectors_path):
            open(self.vectors_path, "wb").close()
        if not os.path.exists(self.metadata_path):
            open(self.metadata_path, "wb").close()

        # 2. Check current physical count
        vec_bytes = os.path.getsize(self.vectors_path)
        num_vecs = vec_bytes // self.BYTES_PER_VECTOR

        # Read metadata entries
        metadata_records = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        metadata_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        break

        num_meta = len(metadata_records)

        # 3. Handle file desynchronization (truncate to minimum consistent length)
        consistent_count = min(num_vecs, num_meta)
        if num_vecs > consistent_count:
            with open(self.vectors_path, "r+b") as f:
                f.truncate(consistent_count * self.BYTES_PER_VECTOR)
        if num_meta > consistent_count:
            metadata_records = metadata_records[:consistent_count]
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                for rec in metadata_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 4. Check WAL for committed records that need replay
        wal_committed = self.wal.recover()
        if wal_committed:
            existing_ids = {r["id"] for r in metadata_records}
            to_append = [rec for rec in wal_committed if rec[0] not in existing_ids]
            if to_append:
                # Replay missing committed records
                with open(self.vectors_path, "ab") as vf, open(self.metadata_path, "a", encoding="utf-8") as mf:
                    curr_offset = consistent_count
                    for doc_id, text, meta, vec in to_append:
                        vf.write(vec.astype(np.float32).tobytes())
                        rec = {"id": doc_id, "text": text, "metadata": meta, "offset": curr_offset}
                        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        metadata_records.append(rec)
                        curr_offset += 1
                    vf.flush()
                    mf.flush()
                    os.fsync(vf.fileno())
                    os.fsync(mf.fileno())
                consistent_count = len(metadata_records)

        # 5. Build in-memory indexes and attach memmap
        self._remap_memmap(consistent_count)

        self.offset_to_id.clear()
        self.offset_to_text.clear()
        self.id_to_offset.clear()
        self.metadata_cache.clear()

        for idx, rec in enumerate(metadata_records):
            doc_id = rec["id"]
            self.id_to_offset[doc_id] = idx
            self.offset_to_id.append(doc_id)
            self.offset_to_text.append(rec.get("text", ""))
            self.metadata_cache.append(rec.get("metadata", {}))

        # Build in-memory 1-bit packed bitmasks from memmap
        if consistent_count > 0 and self._memmap is not None:
            # Vectorized bit-packing: 96 bytes per vector
            bitmask_list = []
            chunk_size = 10000
            for start in range(0, consistent_count, chunk_size):
                end = min(start + chunk_size, consistent_count)
                chunk_vecs = self._memmap[start:end]
                bitmask_chunk = EmbeddingGemmaClient.quantize_1bit(chunk_vecs)
                bitmask_list.append(bitmask_chunk)
            self.bitmask_index = np.vstack(bitmask_list)
        else:
            self.bitmask_index = np.empty((0, self.PACKED_BYTES), dtype=np.uint8)

    def _remap_memmap(self, count: int) -> None:
        """Create or update np.memmap pointer."""
        if self._memmap is not None:
            # Safely release underlying Windows OS mapping handle
            if hasattr(self._memmap, "_mmap") and self._memmap._mmap is not None:
                try:
                    self._memmap._mmap.close()
                except Exception:
                    pass
            self._memmap = None

        if count > 0:
            self._memmap = np.memmap(
                self.vectors_path,
                dtype=np.float32,
                mode="r",
                shape=(count, self.VECTOR_DIM),
            )

    def append_batch(
        self,
        doc_ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        vectors: np.ndarray,
    ) -> List[int]:
        """Atomically append batch of records via WAL and update disk and memory structures.
        
        Args:
            doc_ids: List of unique document IDs.
            texts: List of raw document texts.
            metadatas: List of structured metadata dicts.
            vectors: Array of shape (N, 768) float32.
            
        Returns:
            List of integer row offsets assigned to newly inserted documents.
        """
        n = len(doc_ids)
        if n == 0:
            return []

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        # 1. WAL Phase: write and fsync transaction log
        tx_id = self.wal.next_tx_id()
        self.wal.begin_transaction(tx_id, n)
        for i in range(n):
            self.wal.log_insert(
                tx_id,
                doc_ids[i],
                texts[i],
                metadatas[i],
                vectors[i],
            )
        self.wal.commit_transaction(tx_id)

        # 2. Main Storage Append Phase
        start_offset = self.count
        offsets = list(range(start_offset, start_offset + n))

        with open(self.vectors_path, "ab") as vf, open(self.metadata_path, "a", encoding="utf-8") as mf:
            vf.write(vectors.tobytes())
            for i, offset in enumerate(offsets):
                rec = {
                    "id": doc_ids[i],
                    "text": texts[i],
                    "metadata": metadatas[i],
                    "offset": offset,
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            vf.flush()
            mf.flush()
            os.fsync(vf.fileno())
            os.fsync(mf.fileno())

        # 3. Update In-Memory Indexes
        packed_batch = EmbeddingGemmaClient.quantize_1bit(vectors)
        if self.bitmask_index.shape[0] == 0:
            self.bitmask_index = packed_batch
        else:
            self.bitmask_index = np.vstack([self.bitmask_index, packed_batch])

        for i, offset in enumerate(offsets):
            d_id = doc_ids[i]
            self.id_to_offset[d_id] = offset
            self.offset_to_id.append(d_id)
            self.offset_to_text.append(texts[i])
            self.metadata_cache.append(metadatas[i])

        # 4. Refresh memmap view
        self._remap_memmap(self.count)

        return offsets

    def get_mrl_slice(self, indices: Sequence[int], dim: int = 256) -> np.ndarray:
        """Zero-copy read of Matryoshka sub-vectors (first `dim` floats) via memmap.
        
        Args:
            indices: List/array of document row indices.
            dim: Matryoshka dimension (default: 256).
            
        Returns:
            np.ndarray of shape (len(indices), dim) float32.
        """
        if len(indices) == 0:
            return np.empty((0, dim), dtype=np.float32)
        if self._memmap is None:
            raise RuntimeError("Database contains no vectors.")

        # NumPy advanced indexing on memmap reads only the requested byte ranges
        return np.asarray(self._memmap[indices, :dim])

    def get_full_vectors(self, indices: Sequence[int]) -> np.ndarray:
        """Zero-copy read of full 768d vectors via memmap.
        
        Args:
            indices: List/array of document row indices.
            
        Returns:
            np.ndarray of shape (len(indices), 768) float32.
        """
        if len(indices) == 0:
            return np.empty((0, self.VECTOR_DIM), dtype=np.float32)
        if self._memmap is None:
            raise RuntimeError("Database contains no vectors.")

        return np.asarray(self._memmap[indices, :])

    def get_vector_by_id(self, doc_id: str) -> Optional[np.ndarray]:
        """Fetch full 768d vector for a document ID."""
        if doc_id not in self.id_to_offset or self._memmap is None:
            return None
        idx = self.id_to_offset[doc_id]
        return np.asarray(self._memmap[idx])

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Fetch document text and metadata by ID."""
        if doc_id not in self.id_to_offset:
            return None
        idx = self.id_to_offset[doc_id]
        return {
            "id": doc_id,
            "text": self.offset_to_text[idx],
            "metadata": self.metadata_cache[idx],
            "offset": idx,
        }

    def checkpoint(self) -> None:
        """Checkpoint WAL log to reduce disk footprint."""
        self.wal.checkpoint()

    def close(self) -> None:
        """Safely flush and release all file descriptors and memory mappings."""
        if self._memmap is not None:
            if hasattr(self._memmap, "_mmap") and self._memmap._mmap is not None:
                try:
                    self._memmap._mmap.close()
                except Exception:
                    pass
            self._memmap = None

        self.wal.close()

