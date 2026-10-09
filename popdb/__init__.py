"""PopDB: Embedded Vector Database Engine optimized for EmbeddingGemma 2."""

from popdb.bm25 import BM25Index
from popdb.client import EmbeddingGemmaClient
from popdb.engine import Engine
from popdb.filter import MetadataFilter
from popdb.retriever import HierarchicalRetriever
from popdb.storage import StorageEngine
from popdb.types import Document, HybridResult, SearchResult, TierStats
from popdb.wal import WALManager

__all__ = [
    "Engine",
    "EmbeddingGemmaClient",
    "BM25Index",
    "HierarchicalRetriever",
    "StorageEngine",
    "MetadataFilter",
    "WALManager",
    "Document",
    "SearchResult",
    "HybridResult",
    "TierStats",
]

__version__ = "0.1.0"

