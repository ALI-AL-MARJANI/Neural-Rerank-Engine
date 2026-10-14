"""
Runtime settings, read from environment variables (with defaults).

The same code runs locally, in Docker and in Kubernetes: only the environment
variables change (in Kubernetes they come from k8s/configmap.yaml).
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    index_dir: str          # folder with bm25.pkl and dense/
    dense_model: str        # bi-encoder used by FAISS retrieval
    rerank_model: str       # cross-encoder used for reranking
    top_k: int              # number of hybrid candidates sent to the reranker

    @property
    def bm25_path(self) -> str:
        return os.path.join(self.index_dir, "bm25.pkl")

    @property
    def dense_dir(self) -> str:
        return os.path.join(self.index_dir, "dense")


def get_settings() -> Settings:
    """Reads the environment each time it is called (easy to override in tests)."""
    return Settings(
        index_dir=os.getenv("INDEX_DIR", "data/indices"),
        dense_model=os.getenv("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        rerank_model=os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        top_k=int(os.getenv("TOP_K", "50")),
    )
