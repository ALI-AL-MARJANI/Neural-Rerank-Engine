import json
import time
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrievers.sparse import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever
from src.rerankers.bert_reranker import BertReranker

def calculate_mrr(expected_id, results):
    """Calculates Reciprocal Rank (1/rank) and returns 0 if not found """
    for rank, doc in enumerate(results):
        if doc['id'] == expected_id:
            return 1.0 / (rank + 1)
    return 0.0

def main():
    
    with open("data/test_queries.json", "r") as f:
        test_queries = json.load(f)

    bm25 = BM25Retriever()
    faiss_retriever = DenseRetriever()
    hybrid = HybridRetriever()
    reranker = BertReranker()
    
    # Pre-load indices to avoid cold start time in loop
    bm25.load_index()
    faiss_retriever.load_index()

    
    metrics = {
        "BM25": {"mrr": [], "time": []},
        "FAISS": {"mrr": [], "time": []},
        "Hybrid+Rerank": {"mrr": [], "time": []}
    }

    
    for q in test_queries:
        query = q['query']
        target_id = q['expected_doc_id']
        
        # Test BM25 
        start = time.time()
        res_bm25 = bm25.search(query, k=10)
        metrics["BM25"]["time"].append(time.time() - start)
        metrics["BM25"]["mrr"].append(calculate_mrr(target_id, res_bm25))

        # Test FAISS
        start = time.time()
        res_faiss = faiss_retriever.search(query, k=10)
        metrics["FAISS"]["time"].append(time.time() - start)
        metrics["FAISS"]["mrr"].append(calculate_mrr(target_id, res_faiss))

        # Test Hybrid + Reranker
        start = time.time()
        # Step 1: Hybrid Retrieval (Get top 50 to maximize recall)
        candidates = hybrid.search(query, k=50)
        # Step 2: Reranking (Refine to top 10)
        final_results = reranker.rerank(query, candidates, top_n=10)
        
        metrics["Hybrid+Rerank"]["time"].append(time.time() - start)
        metrics["Hybrid+Rerank"]["mrr"].append(calculate_mrr(target_id, final_results))

    print("\n" + "="*60)
    print(f"{'METHOD':<20} | {'MRR@10':<10} | {'AVG LATENCY (s)':<15}")
    print("-" * 60)
    
    for method, data in metrics.items():
        avg_mrr = np.mean(data['mrr'])
        avg_time = np.mean(data['time'])
        print(f"{method:<20} | {avg_mrr:.4f}     | {avg_time:.4f} s")
    print("="*60)
    
    with open("data/benchmark_results.json", "w") as f:
        final_data = {k: {"mrr": np.mean(v["mrr"]), "latency": np.mean(v["time"])} for k, v in metrics.items()}
        json.dump(final_data, f, indent=4)
        print("\n Results saved to data/benchmark_results.json")

if __name__ == "__main__":
    main()