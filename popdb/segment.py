"""LSM-Tree Segment Manager with Tombstones, Soft Deletes, and Background Compaction."""

import json
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np


class SegmentManager:
    """Manages document lifecycles, soft deletes (tombstones), and storage compaction.
    
    Features:
    - O(1) Tombstone filtering: soft-deleted row offsets are excluded from searches.
    - WAL-backed delete operations for crash durability.
    - Physical compaction: purges tombstones, defragments vectors.bin, and rewrites clean indexes.

    Tombstones are keyed by physical row offset rather than doc ID, so a deleted ID can be
    re-inserted as a new live row while its old row stays hidden until compaction.
    """

    TOMBSTONE_FILE = "tombstones.json"

    def __init__(self, db_dir: str):
        self.db_dir = db_dir
        self.tombstone_path = os.path.join(db_dir, self.TOMBSTONE_FILE)
        self.tombstones: Set[int] = set()
        # Doc IDs from a pre-offset tombstones.json, resolved by StorageEngine once rows are loaded
        self.legacy_id_tombstones: Set[str] = set()
        self._load_tombstones()

    def _load_tombstones(self) -> None:
        """Load tombstone set from disk."""
        if os.path.exists(self.tombstone_path):
            try:
                with open(self.tombstone_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                self.tombstones = {e for e in entries if isinstance(e, int)}
                self.legacy_id_tombstones = {e for e in entries if isinstance(e, str)}
            except Exception:
                self.tombstones = set()

    def resolve_legacy_tombstones(self, offset_to_id: List[str]) -> None:
        """Convert ID-keyed tombstones from older databases into row offsets."""
        if not self.legacy_id_tombstones:
            return
        for offset, doc_id in enumerate(offset_to_id):
            if doc_id in self.legacy_id_tombstones:
                self.tombstones.add(offset)
        self.legacy_id_tombstones.clear()
        self._save_tombstones()

    def _save_tombstones(self) -> None:
        """Persist tombstone set to disk atomically."""
        tmp_path = self.tombstone_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self.tombstones), f)
            f.flush()
            os.fsync(f.fileno())
        shutil.move(tmp_path, self.tombstone_path)

    def mark_deleted(self, offset: int) -> bool:
        """Mark a row offset as deleted (soft delete). Returns True if freshly marked."""
        if offset in self.tombstones:
            return False
        self.tombstones.add(offset)
        self._save_tombstones()
        return True

    def is_deleted(self, offset: int) -> bool:
        """Check if a row offset has been deleted."""
        return offset in self.tombstones

    def get_active_mask(self, n: int) -> np.ndarray:
        """Generate a boolean mask over n rows where True = active (not deleted)."""
        mask = np.ones(n, dtype=bool)
        if self.tombstones and n > 0:
            dead = np.fromiter((t for t in self.tombstones if t < n), dtype=np.int64)
            mask[dead] = False
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
                    vec_bytes = vf.read(bytes_per_vector)
                    if len(vec_bytes) < bytes_per_vector:
                        break

                    if offset in self.tombstones:
                        purged_count += 1
                    else:
                        rec["offset"] = len(new_records)
                        new_records.append(rec)
                        new_vectors.append(vec_bytes)
                    offset += 1

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

