"""Core data structures and types for PopDB."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class Document:
    """Document record stored in PopDB."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    """Result of dense vector retrieval."""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    hamming_distance: Optional[int] = None
    tier2_score: Optional[float] = None
    exact_score: Optional[float] = None


@dataclass
class HybridResult:
    """Result of hybrid search combining dense retrieval and BM25 via RRF."""
    id: str
    rrf_score: float
    dense_rank: Optional[int]
    sparse_rank: Optional[int]
    dense_score: Optional[float]
    sparse_score: Optional[float]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TierStats:
    """Telemetry for 3-tier cascade pruning performance."""
    total_records: int
    filtered_records: int
    tier1_candidates: int
    tier1_time_ms: float
    tier2_candidates: int
    tier2_time_ms: float
    tier3_candidates: int
    tier3_time_ms: float
    total_time_ms: float

