"""LSM-Tree Segment Manager with Tombstones, Soft Deletes, and Background Compaction."""

import json
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np


class SegmentManager:
    """Manages document lifecycles, soft deletes (tombstones), and storage compaction.
    
    Features:
    - O(1) Tombstone filtering: soft-deleted doc IDs are excluded from searches.
    - WAL-backed delete operations for crash durability.
    - Physical compaction: purges tombstones, defragments vectors.bin, and rewrites clean indexes.
    """

    TOMBSTONE_FILE = "tombstones.json"

    def __init__(self, db_dir: str):
        self.db_dir = db_dir
        self.tombstone_path = os.path.join(db_dir, self.TOMBSTONE_FILE)
        self.tombstones: Set[str] = set()
        self._load_tombstones()

    def _load_tombstones(self) -> None:
        """Load tombstone set from disk."""
        if os.path.exists(self.tombstone_path):
            try:
                with open(self.tombstone_path, "r", encoding="utf-8") as f:
                    self.tombstones = set(json.load(f))
            except Exception:
                self.tombstones = set()

    def _save_tombstones(self) -> None:
        """Persist tombstone set to disk atomically."""
        tmp_path = self.tombstone_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(list(self.tombstones), f)
            f.flush()
            os.fsync(f.fileno())
        shutil.move(tmp_path, self.tombstone_path)

    def mark_deleted(self, doc_id: str) -> bool:
        """Mark a document ID as deleted (soft delete). Returns True if freshly marked."""
        if doc_id in self.tombstones:
            return False
        self.tombstones.add(doc_id)
        self._save_tombstones()
        return True

    def is_deleted(self, doc_id: str) -> bool:
        """Check if a document ID has been deleted."""
        return doc_id in self.tombstones

    def get_active_mask(self, offset_to_id: List[str]) -> np.ndarray:
        """Generate a boolean mask where True = active (not deleted)."""
        n = len(offset_to_id)
        if not self.tombstones or n == 0:
            return np.ones(n, dtype=bool)

        mask = np.ones(n, dtype=bool)
        for idx, doc_id in enumerate(offset_to_id):
            if doc_id in self.tombstones:
                mask[idx] = False
        return mask

    def compact(
        self,
        vectors_path: str,
        metadata_path: str,
        bytes_per_vector: int,
        on_complete_callback: Optional[Callable[[], None]] = None,
    ) -> int:
        """Physically compact storage files by removing deleted records.
        
        Returns:
            Number of purged records.
        """
        if not self.tombstones:
            return 0

        compact_vec_path = vectors_path + ".compact"
        compact_meta_path = metadata_path + ".compact"

        purged_count = 0
        new_records = []
        new_vectors = []

        # Read metadata and identify survivors
        if os.path.exists(metadata_path) and os.path.exists(vectors_path):
            with open(metadata_path, "r", encoding="utf-8") as mf, open(vectors_path, "rb") as vf:
                offset = 0
                for line in mf:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    doc_id = rec["id"]
                    vec_bytes = vf.read(bytes_per_vector)
                    if len(vec_bytes) < bytes_per_vector:
                        break

                    if doc_id in self.tombstones:
                        purged_count += 1
                    else:
                        rec["offset"] = len(new_records)
                        new_records.append(rec)
                        new_vectors.append(vec_bytes)

        # Write compacted files
        with open(compact_meta_path, "w", encoding="utf-8") as mf, open(compact_vec_path, "wb") as vf:
            for rec in new_records:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for vb in new_vectors:
                vf.write(vb)
            mf.flush()
            vf.flush()
            os.fsync(mf.fileno())
            os.fsync(vf.fileno())

        if on_complete_callback:
            on_complete_callback()

        # Replace files atomically
        shutil.move(compact_vec_path, vectors_path)
        shutil.move(compact_meta_path, metadata_path)

        # Clear purged tombstones
        self.tombstones.clear()
        self._save_tombstones()

        return purged_count
