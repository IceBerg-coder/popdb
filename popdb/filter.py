"""Structured metadata predicate evaluation and pushdown filtering engine."""

from typing import Any, Dict, List, Optional, Set


class MetadataFilter:
    """Evaluates structured metadata filter predicates.
    
    Supports:
    - Direct equality: {"category": "AI"}
    - Comparisons: {"year": {"$gte": 2024, "$lte": 2026}}
    - Set membership: {"tag": {"$in": ["nlp", "search"]}} or {"status": {"$nin": ["archived"]}}
    - Containment: {"tags": {"$contains": "vector-db"}}
    - Existence: {"author": {"$exists": True}}
    - Logical combinations: {"$and": [...], "$or": [...], "$not": {...}}
    """

    @classmethod
    def matches(cls, doc_metadata: Dict[str, Any], predicate: Optional[Dict[str, Any]]) -> bool:
        """Check if a document metadata dictionary satisfies the predicate."""
        if not predicate:
            return True

        for key, condition in predicate.items():
            if key == "$and":
                if not isinstance(condition, list):
                    raise ValueError("$and condition must be a list of filter expressions.")
                if not all(cls.matches(doc_metadata, sub) for sub in condition):
                    return False
                continue

            if key == "$or":
                if not isinstance(condition, list):
                    raise ValueError("$or condition must be a list of filter expressions.")
                if not any(cls.matches(doc_metadata, sub) for sub in condition):
                    return False
                continue

            if key == "$not":
                if not isinstance(condition, dict):
                    raise ValueError("$not condition must be a filter dict.")
                if cls.matches(doc_metadata, condition):
                    return False
                continue

            # Field-level evaluation
            val = doc_metadata.get(key)

            if isinstance(condition, dict):
                # Operator dictionary: e.g. {"$gte": 2024, "$lt": 2026}
                if not cls._eval_field_operators(val, condition, key in doc_metadata):
                    return False
            else:
                # Direct equality match
                if val != condition:
                    return False

        return True

    @classmethod
    def _eval_field_operators(cls, val: Any, ops: Dict[str, Any], exists: bool) -> bool:
        """Evaluate operator dict for a single field."""
        for op, target in ops.items():
            if op == "$eq":
                if val != target:
                    return False
            elif op == "$ne":
                if val == target:
                    return False
            elif op == "$gt":
                if val is None or val <= target:
                    return False
            elif op == "$gte":
                if val is None or val < target:
                    return False
            elif op == "$lt":
                if val is None or val >= target:
                    return False
            elif op == "$lte":
                if val is None or val > target:
                    return False
            elif op == "$in":
                if not isinstance(target, (list, tuple, set)):
                    raise ValueError("$in target must be an iterable list/tuple/set.")
                # If doc value is list, check intersection; otherwise check membership
                if isinstance(val, (list, tuple, set)):
                    if not any(elem in target for elem in val):
                        return False
                else:
                    if val not in target:
                        return False
            elif op == "$nin":
                if not isinstance(target, (list, tuple, set)):
                    raise ValueError("$nin target must be an iterable.")
                if isinstance(val, (list, tuple, set)):
                    if any(elem in target for elem in val):
                        return False
                else:
                    if val in target:
                        return False
            elif op == "$contains":
                if val is None:
                    return False
                if isinstance(val, (list, tuple, set, str)):
                    if target not in val:
                        return False
                else:
                    return False
            elif op == "$exists":
                if bool(exists) != bool(target):
                    return False
            else:
                raise ValueError(f"Unsupported filter operator: {op}")

        return True

    @classmethod
    def filter_indices(
        cls,
        metadata_cache: List[Dict[str, Any]],
        predicate: Optional[Dict[str, Any]],
    ) -> List[int]:
        """Compute matching document row offsets for pushdown filtering.
        
        Args:
            metadata_cache: In-memory cache of all document metadata dicts.
            predicate: Structured filter query.
            
        Returns:
            List of matching row indices.
        """
        if not predicate:
            return list(range(len(metadata_cache)))

        matching = []
        for idx, meta in enumerate(metadata_cache):
            if cls.matches(meta, predicate):
                matching.append(idx)
        return matching

