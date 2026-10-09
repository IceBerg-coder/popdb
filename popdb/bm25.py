"""Native Python BM25Okapi inverted index for keyword search with predicate pushdown."""

import math
import re
from typing import Dict, List, Optional, Set, Tuple


class BM25Index:
    """Lightweight, native Python BM25Okapi inverted index.
    
    Supports:
    - Incremental document ingestion.
    - Fast inverted index traversal.
    - Metadata predicate pushdown filtering.
    - Configurable k1 and b parameters.
    """

    TOKEN_PATTERN = re.compile(r"\b\w+\b")

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.doc_lengths: List[int] = []
        self.total_docs: int = 0
        self.total_length: int = 0
        self.avg_doc_len: float = 0.0

        # Term -> list of (doc_index, term_freq)
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        # Term -> document frequency (number of documents containing term)
        self.df: Dict[str, int] = {}
        self._idf_cache: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric words."""
        return self.TOKEN_PATTERN.findall(text.lower())

    def add_documents(self, doc_indices: List[int], texts: List[str]) -> None:
        """Add batch of documents to the inverted index.
        
        Args:
            doc_indices: Contiguous integer row indices matching storage engine.
            texts: Raw text contents of the documents.
        """
        for idx, text in zip(doc_indices, texts):
            tokens = self._tokenize(text)
            doc_len = len(tokens)
            
            # Ensure doc_lengths array can accommodate idx
            if idx >= len(self.doc_lengths):
                self.doc_lengths.extend([0] * (idx - len(self.doc_lengths) + 1))
            self.doc_lengths[idx] = doc_len
            self.total_length += doc_len
            self.total_docs += 1

            # Count term frequencies for this document
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for term, tf in tf_map.items():
                if term not in self.postings:
                    self.postings[term] = []
                    self.df[term] = 0
                self.postings[term].append((idx, tf))
                self.df[term] += 1

        if self.total_docs > 0:
            self.avg_doc_len = self.total_length / self.total_docs
            self._idf_cache.clear()

    def _get_idf(self, term: str) -> float:
        """Compute BM25 Okapi IDF with smoothing."""
        if term in self._idf_cache:
            return self._idf_cache[term]

        n = self.df.get(term, 0)
        if n == 0 or self.total_docs == 0:
            idf = 0.0
        else:
            # BM25+ / Okapi formula with +1 smoothing inside log
            idf = math.log((self.total_docs - n + 0.5) / (n + 0.5) + 1.0)
            if idf < 0:
                idf = 0.0

        self._idf_cache[term] = idf
        return idf

    def search(
        self,
        query: str,
        candidate_indices: Optional[Set[int]] = None,
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        """Compute BM25 scores for matching documents.
        
        Args:
            query: Keyword query text.
            candidate_indices: Optional set of allowed doc indices for metadata pre-filtering.
            top_k: Maximum number of ranked results to return.
            
        Returns:
            List of (doc_index, bm25_score) tuples sorted in descending order of score.
        """
        query_terms = self._tokenize(query)
        if not query_terms or self.total_docs == 0:
            return []

        scores: Dict[int, float] = {}
        avgdl = self.avg_doc_len if self.avg_doc_len > 0 else 1.0
        k1 = self.k1
        b = self.b

        for term in query_terms:
            if term not in self.postings:
                continue

            idf = self._get_idf(term)
            if idf <= 0.0:
                continue

            for doc_idx, tf in self.postings[term]:
                # Apply metadata predicate pushdown
                if candidate_indices is not None and doc_idx not in candidate_indices:
                    continue

                doc_len = self.doc_lengths[doc_idx]
                denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
                term_score = idf * (tf * (k1 + 1.0)) / denom if denom > 0 else 0.0

                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_score

        if not scores:
            return []

        # Sort top-k
        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_results[:top_k]

