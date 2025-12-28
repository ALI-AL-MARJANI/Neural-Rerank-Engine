from typing import List, Dict, Any
from collections import defaultdict

from src.retrievers.sparse import BM25Retriever
from src.retrievers.dense import DenseRetriever


class HybridRetriever:
    """
    Orchestrates Sparse (BM25) and Dense (FAISS) retrieval.
    Merges results using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self):
        self.sparse = BM25Retriever()
        self.dense = DenseRetriever()
        
        # Load indices immediately
        self.sparse.load_index()
        self.dense.load_index()

    def search(self, query: str, k: int = 50, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Performs hybrid search.
                
        Returns:
            List[Dict]: Top-k merged documents.
        """
        # 1. Get results from both engines independently
        # We ask for 'k' results from each to have enough candidates
        sparse_results = self.sparse.search(query, k=k)
        dense_results = self.dense.search(query, k=k)

        # 2. Apply Reciprocal Rank Fusion (RRF)
        # RRF Score = 1 / (rank + k_constant)
    
        
        rrf_scores = defaultdict(float)
        doc_map = {} 
        
        k_const = 60 
        
        # Process Sparse
        for rank, doc in enumerate(sparse_results):
            doc_id = doc['id']
            rrf_scores[doc_id] += 1 / (rank + k_const)
            doc_map[doc_id] = doc

        # Process Dense
        for rank, doc in enumerate(dense_results):
            doc_id = doc['id']
            rrf_scores[doc_id] += 1 / (rank + k_const)
            # If doc was not in sparse, add it to map
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # 3. Sort by accumulated RRF score 
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        
        
        final_results = []
        for doc_id in sorted_ids[:k]:
            final_results.append(doc_map[doc_id])
            
        return final_results