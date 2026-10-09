"""Columnar Inverted Bitset Index for microsecond metadata predicate pushdown."""

from typing import Any, Dict, List, Optional, Set, Union
import numpy as np


class ColumnarIndex:
    """High-performance columnar bitset index for structured metadata filtering.
    
    Stores:
    - Inverted boolean bitsets for categorical string/enum/tag fields.
    - Contiguous numeric arrays for fast range comparisons ($gt, $gte, $lt, $lte).
    - Presence bitmasks to distinguish missing/null values from 0/False.
    """

    def __init__(self):
        self.total_docs: int = 0
        # field -> value -> np.ndarray(bool, shape=(total_docs,))
        self.categorical_bitsets: Dict[str, Dict[Any, np.ndarray]] = {}
        # field -> np.ndarray(float64, shape=(total_docs,))
        self.numeric_columns: Dict[str, np.ndarray] = {}
        # field -> np.ndarray(bool, shape=(total_docs,))
        self.presence_masks: Dict[str, np.ndarray] = {}

    def add_documents(self, start_offset: int, metadatas: List[Dict[str, Any]]) -> None:
        """Incrementally index a batch of metadata dictionaries."""
        batch_size = len(metadatas)
        if batch_size == 0:
            return

        new_total = start_offset + batch_size
        old_total = self.total_docs

        # 1. Resize existing bitsets and numeric arrays if needed
        if new_total > old_total:
            for field, val_map in self.categorical_bitsets.items():
                for val, mask in val_map.items():
                    expanded = np.zeros(new_total, dtype=bool)
                    expanded[:old_total] = mask
                    val_map[val] = expanded

            for field, col in self.numeric_columns.items():
                expanded_col = np.zeros(new_total, dtype=np.float64)
                expanded_col[:old_total] = col
                self.numeric_columns[field] = expanded_col

            for field, pmask in self.presence_masks.items():
                expanded_pmask = np.zeros(new_total, dtype=bool)
                expanded_pmask[:old_total] = pmask
                self.presence_masks[field] = expanded_pmask

            self.total_docs = new_total

        # 2. Populate newly added rows
        for i, meta in enumerate(metadatas):
            doc_idx = start_offset + i
            for key, val in meta.items():
                # Track presence
                if key not in self.presence_masks:
                    self.presence_masks[key] = np.zeros(self.total_docs, dtype=bool)
                self.presence_masks[key][doc_idx] = True

                # Numerical columns
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if key not in self.numeric_columns:
                        self.numeric_columns[key] = np.zeros(self.total_docs, dtype=np.float64)
                    self.numeric_columns[key][doc_idx] = float(val)

                # Categorical & list-tag bitsets
                if isinstance(val, (str, bool, int)):
                    self._set_categorical_bit(key, val, doc_idx)
                elif isinstance(val, (list, tuple, set)):
                    for item in val:
                        if isinstance(item, (str, bool, int)):
                            self._set_categorical_bit(key, item, doc_idx)

    def _set_categorical_bit(self, field: str, val: Any, doc_idx: int) -> None:
        """Set a single bit in the categorical inverted index."""
        if field not in self.categorical_bitsets:
            self.categorical_bitsets[field] = {}

        if val not in self.categorical_bitsets[field]:
            mask = np.zeros(self.total_docs, dtype=bool)
            self.categorical_bitsets[field][val] = mask

        self.categorical_bitsets[field][val][doc_idx] = True

    def evaluate_mask(self, predicate: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Evaluate predicate query into a boolean array of shape (total_docs,).
        
        Returns:
            np.ndarray of bool where True = match, or None if no filter applied.
        """
        if not predicate or self.total_docs == 0:
            return None

        # Start with all True
        result_mask = np.ones(self.total_docs, dtype=bool)

        for key, cond in predicate.items():
            if key == "$and":
                if not isinstance(cond, list):
                    raise ValueError("$and must be a list of conditions")
                for sub in cond:
                    sub_mask = self.evaluate_mask(sub)
                    if sub_mask is not None:
                        result_mask &= sub_mask
                continue

            if key == "$or":
                if not isinstance(cond, list):
                    raise ValueError("$or must be a list of conditions")
                or_mask = np.zeros(self.total_docs, dtype=bool)
                for sub in cond:
                    sub_mask = self.evaluate_mask(sub)
                    if sub_mask is not None:
                        or_mask |= sub_mask
                result_mask &= or_mask
                continue

            if key == "$not":
                sub_mask = self.evaluate_mask(cond)
                if sub_mask is not None:
                    result_mask &= ~sub_mask
                continue

            # Field level condition
            field_mask = self._eval_field_predicate(key, cond)
            result_mask &= field_mask

        return result_mask

    def _eval_field_predicate(self, field: str, cond: Any) -> np.ndarray:
        """Evaluate condition on a single field using columnar data."""
        # If field has never been indexed, return all False
        if field not in self.presence_masks:
            # Check if condition checks for $exists: False
            if isinstance(cond, dict) and cond.get("$exists") is False:
                return np.ones(self.total_docs, dtype=bool)
            return np.zeros(self.total_docs, dtype=bool)

        presence = self.presence_masks[field]

        # Case 1: Exact equality, e.g. {"category": "AI"}
        if not isinstance(cond, dict):
            val_map = self.categorical_bitsets.get(field, {})
            if cond in val_map:
                return val_map[cond].copy()
            # If checking a number that was indexed in numeric_columns
            if isinstance(cond, (int, float)) and field in self.numeric_columns:
                return (self.numeric_columns[field] == float(cond)) & presence
            return np.zeros(self.total_docs, dtype=bool)

        # Case 2: Operator dictionary
        mask = np.ones(self.total_docs, dtype=bool)

        for op, target in cond.items():
            if op == "$eq":
                sub = self._eval_field_predicate(field, target)
                mask &= sub

            elif op == "$ne":
                sub = self._eval_field_predicate(field, target)
                mask &= (~sub) & presence

            elif op in ("$gt", "$gte", "$lt", "$lte"):
                if field not in self.numeric_columns:
                    mask &= False
                    continue
                col = self.numeric_columns[field]
                target_val = float(target)
                if op == "$gt":
                    cmp_mask = col > target_val
                elif op == "$gte":
                    cmp_mask = col >= target_val
                elif op == "$lt":
                    cmp_mask = col < target_val
                else:
                    cmp_mask = col <= target_val
                mask &= cmp_mask & presence

            elif op == "$in":
                if not isinstance(target, (list, tuple, set)):
                    raise ValueError("$in target must be iterable")
                in_mask = np.zeros(self.total_docs, dtype=bool)
                val_map = self.categorical_bitsets.get(field, {})
                for item in target:
                    if item in val_map:
                        in_mask |= val_map[item]
                    elif isinstance(item, (int, float)) and field in self.numeric_columns:
                        in_mask |= (self.numeric_columns[field] == float(item)) & presence
                mask &= in_mask

            elif op == "$nin":
                if not isinstance(target, (list, tuple, set)):
                    raise ValueError("$nin target must be iterable")
                in_mask = np.zeros(self.total_docs, dtype=bool)
                val_map = self.categorical_bitsets.get(field, {})
                for item in target:
                    if item in val_map:
                        in_mask |= val_map[item]
                    elif isinstance(item, (int, float)) and field in self.numeric_columns:
                        in_mask |= (self.numeric_columns[field] == float(item)) & presence
                mask &= (~in_mask) & presence

            elif op == "$exists":
                if bool(target):
                    mask &= presence
                else:
                    mask &= ~presence

            elif op == "$contains":
                val_map = self.categorical_bitsets.get(field, {})
                if target in val_map:
                    mask &= val_map[target]
                else:
                    mask &= False

            else:
                raise ValueError(f"Unsupported operator in columnar filter: {op}")

        return mask
