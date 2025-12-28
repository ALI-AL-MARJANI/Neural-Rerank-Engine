import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.retrievers.hybrid import HybridRetriever
from src.rerankers.bert_reranker import BertReranker
def main():
    
    print("Initializing Pipeline..")
    hybrid_retriever = HybridRetriever()
    reranker = BertReranker()
    
    
    # (Choosing a query that needs understanding, not just keywords..)
    query = "what is the function of mitochondria?"
    print(f"\n Query: {query}")
    print("-" * 50)

    # Stage 1: Retrieval (High Recall)
    start = time.time()
    candidates = hybrid_retriever.search(query, k=50)
    print(f"Stage 1 (Hybrid) retrieved {len(candidates)} docs in {time.time() - start:.4f}s")
    
    print("\n--- Top 3 Before Reranking (Hybrid) ")
    for i, doc in enumerate(candidates[:3]):
        print(f"{i+1}. {doc['text'][:100]}")

    # Stage 2: Reranking (High Precision)
    start = time.time()
    final_results = reranker.rerank(query, candidates, top_n=3)
    print(f"\n Stage 2 (Reranking) finished in {time.time() - start:.4f}s")

    print("\n---  Top 3 Final Results (After BERT) ")
    for i, doc in enumerate(final_results):
        score = doc.get('rerank_score', 0.0)
        print(f"{i+1}. [Score: {score:.4f}] {doc['text']}")
        print("-" * 20)

if __name__ == "__main__":
    main()