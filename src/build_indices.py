import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrievers.sparse import BM25Retriever
from src.retrievers.dense import DenseRetriever

def main():
    
    data_path = "data/raw/corpus.json"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run data_loader.py first to download data")
        return

    print(f" Loading data from {data_path}..")
    with open(data_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    # Building Sparse Index (BM25)
    sparse_retriever = BM25Retriever()
    sparse_retriever.index(corpus)
    
    print("-" * 30)

    # Building Dense Index (FAISS)
    dense_retriever = DenseRetriever()
    dense_retriever.index(corpus)

    print("\n All indices built successfully")

if __name__ == "__main__":
    main()