import pickle
import os
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
    Sparse Retriever based on BM25 algorithm (Keyword Matching)
    Uses the rank_bm25 library for tokenization and scoring
    """
    def __init__(self, index_path="data/indices/bm25.pkl"):
        self.bm25 = None
        self.corpus = [] 
        self.index_path = index_path

    def index(self, corpus: List[Dict[str, str]]):
        """
        Builds the BM25 index from the corpus

        """
        
        self.corpus = corpus
        tokenized_corpus = [doc['text'].lower().split() for doc in corpus]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save_index()
        print(f" BM25 Index built with {len(corpus)} documents")

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top-k documents relevant to the query using BM25 score
        
        """
        if not self.bm25:
            self.load_index()
            
        tokenized_query = query.lower().split()
        
        # rank_bm25's get_top_n returns the document objects directly
        top_docs = self.bm25.get_top_n(tokenized_query, self.corpus, n=k)
        
        return top_docs

    def save_index(self):
        """Saves the BM25 object and corpus to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump((self.bm25, self.corpus), f)

    def load_index(self):
        """Loads the BM25 object and corpus from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found at {self.index_path}")
        
        with open(self.index_path, "rb") as f:
            self.bm25, self.corpus = pickle.load(f)