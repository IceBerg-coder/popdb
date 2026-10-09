"""EmbeddingGemma 2 Client integration with SentenceTransformers and MRL."""

from typing import List, Optional, Sequence, Union
import numpy as np


class EmbeddingGemmaClient:
    """High-performance wrapper for Google's EmbeddingGemma 2 embedding model.
    
    Features:
    - Native support for `google/embeddinggemma-2` via `sentence-transformers`.
    - Task-specific prefix routing via prompt names:
        - prompt_name="Document" for indexing documents.
        - prompt_name="SearchQuery" for search queries.
    - Native Matryoshka Representation Learning (MRL) utilities (768d -> 256d).
    - 1-bit binary quantization (768 bits -> 96 bytes packed uint8).
    - High-fidelity mock mode for rapid testing or offline environments.
    """

    FULL_DIM: int = 768
    MRL_DIM: int = 256
    PACKED_BYTES: int = 96  # 768 bits / 8 bits per byte

    DOCUMENT_PROMPT: str = "Document"
    QUERY_PROMPT: str = "SearchQuery"

    def __init__(
        self,
        model_name_or_path: str = "google/embeddinggemma-2",
        device: Optional[str] = None,
        lazy_load: bool = True,
        use_mock: bool = False,
    ):
        """Initialize EmbeddingGemmaClient.
        
        Args:
            model_name_or_path: Hugging Face model repository or local path.
            device: 'cuda', 'cpu', or None for automatic device resolution.
            lazy_load: If True, defer heavy model weight loading until first inference.
            use_mock: If True, use a deterministic synthetic embedding generator.
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.use_mock = use_mock
        self._model = None

        if not lazy_load and not use_mock:
            self._load_model()

    def _load_model(self) -> None:
        """Instantiate the underlying SentenceTransformer model."""
        if self._model is not None or self.use_mock:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name_or_path,
                device=self.device,
            )
        except Exception as e:
            # If model loading fails, provide informative message
            raise RuntimeError(
                f"Failed to load sentence-transformers model '{self.model_name_or_path}': {e}"
            ) from e

    def encode_documents(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode documents using task-specific instruction prefix: prompt_name="Document".
        
        Args:
            texts: Single text or sequence of texts to encode.
            batch_size: Mini-batch size for inference.
            normalize_embeddings: If True, L2-normalize vectors to unit hypersphere.
            
        Returns:
            np.ndarray of shape (N, 768) with dtype float32.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self.FULL_DIM), dtype=np.float32)

        if self.use_mock:
            return self._mock_encode(texts, prefix="Document", normalize=normalize_embeddings)

        self._load_model()
        vectors = self._model.encode(
            list(texts),
            prompt_name=self.DOCUMENT_PROMPT,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors = vectors.astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vectors = vectors / norms
        return vectors

    def encode_queries(
        self,
        queries: Sequence[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode multiple search queries using task-specific instruction prefix: prompt_name="SearchQuery".
        
        Args:
            queries: Sequence of query text strings.
            batch_size: Mini-batch size for query inference.
            normalize_embeddings: If True, L2-normalize vectors to unit hypersphere.
            
        Returns:
            np.ndarray of shape (N, 768) with dtype float32.
        """
        if not queries:
            return np.empty((0, self.FULL_DIM), dtype=np.float32)

        if self.use_mock:
            return self._mock_encode(queries, prefix="SearchQuery", normalize=normalize_embeddings)

        self._load_model()
        vectors = self._model.encode(
            list(queries),
            prompt_name=self.QUERY_PROMPT,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors = vectors.astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vectors = vectors / norms
        return vectors

    def encode_query(
        self,
        query: str,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode single search query using prompt_name="SearchQuery"."""
        return self.encode_queries([query], batch_size=1, normalize_embeddings=normalize_embeddings)[0]

    @classmethod
    def truncate_mrl(cls, vectors: np.ndarray, dim: int = 256) -> np.ndarray:
        """Slice vectors along first `dim` dimensions and re-normalize for MRL.
        
        EmbeddingGemma 2 natively trains prefix sub-vectors such that the first 256d
        slice forms a compressed representation preserving semantic geometry.
        
        Args:
            vectors: Array of shape (..., full_dim).
            dim: Matryoshka target dimension (default: 256).
            
        Returns:
            L2-normalized array of shape (..., dim).
        """
        sub = vectors[..., :dim]
        norms = np.linalg.norm(sub, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (sub / norms).astype(np.float32)

    @classmethod
    def quantize_1bit(cls, vectors: np.ndarray) -> np.ndarray:
        """Perform 1-bit binary quantization and bit-packing along the last dimension.
        
        Quantization rule: bit_i = 1 if v_i > 0 else 0.
        Packs 768 bits into 96 bytes (uint8) using np.packbits.
        
        Args:
            vectors: Array of shape (..., 768) float32.
            
        Returns:
            np.ndarray of shape (..., 96) uint8.
        """
        bitmask = (vectors > 0).astype(np.uint8)
        packed = np.packbits(bitmask, axis=-1)
        return packed

    def _mock_encode(
        self,
        texts: Sequence[str],
        prefix: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """Deterministic pseudo-semantic vector generator for unit tests/offline usage."""
        vectors = np.zeros((len(texts), self.FULL_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            # Seed based on text hash to ensure reproducibility
            # Include prefix to emulate asymmetric query/doc mapping
            combined = f"{prefix}:{text.lower().strip()}"
            h = abs(hash(combined))
            rng = np.random.RandomState(h % (2**31 - 1))
            
            # Create structured components:
            # First 256d: higher energy signal for MRL
            v1 = rng.randn(self.MRL_DIM).astype(np.float32) * 1.5
            # Remaining 512d: fine-grained residual signal
            v2 = rng.randn(self.FULL_DIM - self.MRL_DIM).astype(np.float32) * 0.8
            v = np.concatenate([v1, v2])

            if normalize:
                norm = np.linalg.norm(v)
                v = v / (norm if norm != 0 else 1.0)
            vectors[i] = v
        return vectors
