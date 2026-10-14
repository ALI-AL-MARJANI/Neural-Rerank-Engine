from typing import Any, Dict, List, Optional

from rerank_engine.retrievers.dense import DenseRetriever
from rerank_engine.retrievers.fusion import reciprocal_rank_fusion
from rerank_engine.retrievers.sparse import BM25Retriever


class HybridRetriever:
    """
    Orchestrates Sparse (BM25) and Dense (FAISS) retrieval.
    Merges results using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, sparse: Optional[BM25Retriever] = None, dense: Optional[DenseRetriever] = None):
        self.sparse = sparse or BM25Retriever()
        self.dense = dense or DenseRetriever()

        # Load indices immediately
        self.sparse.load_index()
        self.dense.load_index()

    def search(self, query: str, k: int = 50, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Performs hybrid search.

        Returns:
            List[Dict]: Top-k merged documents (each with an 'rrf_score').
        """
        # 1. Get results from both engines independently
        # We ask for 'k' results from each to have enough candidates
        sparse_results = self.sparse.search(query, k=k)
        dense_results = self.dense.search(query, k=k)

        # 2. Apply Reciprocal Rank Fusion (RRF) and keep the top-k
        return reciprocal_rank_fusion([sparse_results, dense_results], k=k)
