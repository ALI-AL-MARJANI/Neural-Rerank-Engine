import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    """
    Dense Retriever based on Semantic Embeddings and FAISS.
    Uses 'all-MiniLM-L6-v2' for embeddings 
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", index_dir="data/indices/dense"):
        self.model = SentenceTransformer(model_name)
        self.faiss_index = None 
        self.corpus = []
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "metadata.pkl")

    def index(self, corpus: List[Dict[str, str]]):
        """
        Encodes the corpus and builds a FAISS index.
        """
        
        self.corpus = corpus
        texts = [doc['text'] for doc in corpus]
        
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # FAISS expects float32
        embeddings = embeddings.astype('float32')
        dimension = embeddings.shape[1]

        # 2. Build FAISS Index (FlatL2)
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        
        self.save_index()
        print(f" Dense Index built with {len(corpus)} documents.")

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Semantic search using vector similarity.
        """
        if not self.faiss_index:
            self.load_index()

       
        query_vec = self.model.encode([query], convert_to_numpy=True).astype('float32')
    
        distances, indices = self.faiss_index.search(query_vec, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.corpus):
                results.append(self.corpus[idx])
        
        return results

    def save_index(self):
        """Saves the FAISS index and metadata to disk."""
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.faiss_index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.corpus, f)

    def load_index(self):
        """Loads the FAISS index and metadata from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f" Index not found at {self.index_dir}. Run build_indices.py first.")
        
        self.faiss_index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.corpus = pickle.load(f)