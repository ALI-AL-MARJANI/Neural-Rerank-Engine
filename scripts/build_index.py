"""
Downloads the public BEIR SciFact corpus (5,183 scientific abstracts) and builds
the BM25 and FAISS indices in INDEX_DIR (default: data/indices).

It also downloads the cross-encoder once, so that every model is in the local
Hugging Face cache and the API can start without internet.

Usage:
    python scripts/build_index.py                 # full corpus
    python scripts/build_index.py --max-docs 500  # quick test
"""
import argparse
import time

from datasets import load_dataset
from sentence_transformers import CrossEncoder

from rerank_engine.config import get_settings
from rerank_engine.retrievers.dense import DenseRetriever
from rerank_engine.retrievers.sparse import BM25Retriever


def load_scifact(max_docs=None):
    """Returns the corpus in the format the retrievers expect: [{'id', 'text'}]."""
    ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    if max_docs:
        ds = ds.select(range(min(max_docs, len(ds))))
    return [
        {"id": str(row["_id"]), "text": f"{row['title']} {row['text']}".strip()}
        for row in ds
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None, help="Index only the first N documents")
    args = parser.parse_args()

    settings = get_settings()
    start = time.perf_counter()

    corpus = load_scifact(args.max_docs)
    print(f"Loaded {len(corpus)} SciFact documents")

    BM25Retriever(index_path=settings.bm25_path).index(corpus)
    DenseRetriever(model_name=settings.dense_model, index_dir=settings.dense_dir).index(corpus)

    # Not used here: only downloaded so that it is cached for the API
    CrossEncoder(settings.rerank_model)
    print(f"Cross-encoder {settings.rerank_model} cached")

    print(f"Indices written to {settings.index_dir} in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
