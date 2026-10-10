"""PopDB: Embedded Vector Database Engine optimized for EmbeddingGemma 2."""

from popdb.bm25 import BM25Index
from popdb.client import EmbeddingGemmaClient
from popdb.columnar import ColumnarIndex
from popdb.concurrency import RWLock
from popdb.engine import Engine
from popdb.filter import MetadataFilter
from popdb.ivf import IVFBQIndex
from popdb.kernel import FusedHammingKernel, MicroJITKernel
from popdb.prefetch import OSMemoryPrefetcher
from popdb.quantization import ScalarQuantizer2Bit
from popdb.retriever import HierarchicalRetriever
from popdb.storage import StorageEngine
from popdb.types import Document, HybridResult, SearchResult, TierStats
from popdb.wal import WALManager

__all__ = [
    "Engine",
    "EmbeddingGemmaClient",
    "BM25Index",
    "ColumnarIndex",
    "FusedHammingKernel",
    "MicroJITKernel",
    "ScalarQuantizer2Bit",
    "IVFBQIndex",
    "OSMemoryPrefetcher",
    "RWLock",
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

