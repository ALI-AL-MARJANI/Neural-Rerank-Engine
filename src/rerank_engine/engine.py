"""
The search pipeline used by the API: hybrid retrieval, then optional reranking.
"""
from typing import Any, Dict, List

from rerank_engine.config import Settings


class SearchEngine:
    """Stage 1: HybridRetriever (BM25 + FAISS, RRF). Stage 2: BertReranker (cross-encoder)."""

    def __init__(self, retriever, reranker, candidates_k: int = 50):
        self.retriever = retriever
        self.reranker = reranker
        self.candidates_k = candidates_k

    def search(self, query: str, top_k: int = 10, rerank: bool = True) -> List[Dict[str, Any]]:
        # Never ask for fewer candidates than the number of results requested
        candidates = self.retriever.search(query, k=max(self.candidates_k, top_k))

        if rerank:
            docs = self.reranker.rerank(query, candidates, top_n=top_k)
            score_key = 'rerank_score'
        else:
            docs = candidates[:top_k]
            score_key = 'rrf_score'

        return [{'doc_id': d['id'], 'text': d['text'], 'score': d[score_key]} for d in docs]


def load_engine(settings: Settings) -> SearchEngine:
    """Loads both models and both indices from disk. Slow: call it once, at startup."""
    # Heavy imports (torch, faiss) are done here so that importing the API stays fast
    from rerank_engine.rerankers.bert_reranker import BertReranker
    from rerank_engine.retrievers.dense import DenseRetriever
    from rerank_engine.retrievers.hybrid import HybridRetriever
    from rerank_engine.retrievers.sparse import BM25Retriever

    retriever = HybridRetriever(
        sparse=BM25Retriever(index_path=settings.bm25_path),
        dense=DenseRetriever(model_name=settings.dense_model, index_dir=settings.dense_dir),
    )
    reranker = BertReranker(model_name=settings.rerank_model)
    return SearchEngine(retriever, reranker, candidates_k=settings.top_k)
