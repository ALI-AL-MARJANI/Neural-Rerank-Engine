"""
Retrieval quality on the BEIR SciFact test set, for each stage of the pipeline.

Scores the 300 SciFact test queries against the official relevance labels
(BeIR/scifact-qrels) with nDCG@10, MRR@10 and Recall@50, for:
  bm25          BM25 alone
  dense         FAISS (all-MiniLM-L6-v2) alone
  hybrid        BM25 + FAISS merged with RRF (the API with rerank=false)
  hybrid+rerank the same 50 candidates re-ordered by the cross-encoder (the API default)

Needs the indices built by scripts/build_index.py (full corpus, not --max-docs).

Usage:
    python scripts/eval_quality.py                      # writes results/quality_scifact.json
    python scripts/eval_quality.py --max-queries 20     # quick test
"""
import argparse
import json
import math
import os
import platform
import random
import statistics
import time
from collections import defaultdict

from datasets import load_dataset

from rerank_engine.config import get_settings
from rerank_engine.engine import load_engine

N_BOOTSTRAP = 2000


def load_scifact_test():
    """Returns {query_id: text} and {query_id: set of relevant doc ids} for the test split."""
    qrels = defaultdict(set)
    for row in load_dataset("BeIR/scifact-qrels", split="test"):
        if row["score"] > 0:
            qrels[str(row["query-id"])].add(str(row["corpus-id"]))
    queries = {
        str(row["_id"]): row["text"]
        for row in load_dataset("BeIR/scifact", "queries", split="queries")
        if str(row["_id"]) in qrels
    }
    return queries, qrels


def ndcg_at_k(ranked_ids, relevant, k=10):
    """Binary-relevance nDCG@k (SciFact labels are all 1)."""
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if d in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def mrr_at_k(ranked_ids, relevant, k=10):
    for i, d in enumerate(ranked_ids[:k]):
        if d in relevant:
            return 1 / (i + 1)
    return 0.0


def recall_at_k(ranked_ids, relevant, k=50):
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def bootstrap_ci(values, seed=0):
    """95% interval of the mean, resampling queries."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(N_BOOTSTRAP))
    return means[int(0.025 * N_BOOTSTRAP)], means[int(0.975 * N_BOOTSTRAP) - 1]


def paired_delta(a, b, seed=0):
    """Mean of b - a over queries, with a 95% paired-bootstrap interval."""
    diffs = [y - x for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(diffs, seed)
    return {"delta": statistics.mean(diffs), "ci95": [lo, hi]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--out", default="results/quality_scifact.json")
    args = parser.parse_args()

    settings = get_settings()
    engine = load_engine(settings)
    retriever, reranker = engine.retriever, engine.reranker
    candidates_k = settings.top_k

    queries, qrels = load_scifact_test()
    query_ids = sorted(queries, key=int)[: args.max_queries]
    print(f"{len(query_ids)} SciFact test queries, {candidates_k} candidates per query")

    stages = {
        "bm25": lambda q: retriever.sparse.search(q, k=candidates_k),
        "dense": lambda q: retriever.dense.search(q, k=candidates_k),
        "hybrid": lambda q: retriever.search(q, k=candidates_k),
    }
    per_query = {name: {"ndcg@10": [], "mrr@10": [], "recall@50": [], "latency_ms": []}
                 for name in [*stages, "hybrid+rerank"]}

    for i, qid in enumerate(query_ids):
        query, relevant = queries[qid], qrels[qid]
        for name, search in stages.items():
            start = time.perf_counter()
            docs = search(query)
            elapsed = (time.perf_counter() - start) * 1000
            if name == "hybrid":
                hybrid_docs, hybrid_ms = docs, elapsed
            ids = [d["id"] for d in docs]
            per_query[name]["ndcg@10"].append(ndcg_at_k(ids, relevant))
            per_query[name]["mrr@10"].append(mrr_at_k(ids, relevant))
            per_query[name]["recall@50"].append(recall_at_k(ids, relevant))
            per_query[name]["latency_ms"].append(elapsed)

        # Reranking reuses the hybrid candidates, so its latency is hybrid + cross-encoder
        start = time.perf_counter()
        reranked = reranker.rerank(query, hybrid_docs, top_n=candidates_k)
        elapsed = hybrid_ms + (time.perf_counter() - start) * 1000
        ids = [d["id"] for d in reranked]
        per_query["hybrid+rerank"]["ndcg@10"].append(ndcg_at_k(ids, relevant))
        per_query["hybrid+rerank"]["mrr@10"].append(mrr_at_k(ids, relevant))
        per_query["hybrid+rerank"]["recall@50"].append(recall_at_k(ids, relevant))
        per_query["hybrid+rerank"]["latency_ms"].append(elapsed)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(query_ids)} queries")

    results = {}
    for name, m in per_query.items():
        lo, hi = bootstrap_ci(m["ndcg@10"])
        results[name] = {
            "ndcg@10": statistics.mean(m["ndcg@10"]),
            "ndcg@10_ci95": [lo, hi],
            "mrr@10": statistics.mean(m["mrr@10"]),
            "recall@50": statistics.mean(m["recall@50"]),
            "latency_ms_p50": statistics.median(m["latency_ms"]),
        }

    ndcg = {name: m["ndcg@10"] for name, m in per_query.items()}
    comparisons = {
        "dense_vs_bm25": paired_delta(ndcg["bm25"], ndcg["dense"]),
        "hybrid_vs_dense": paired_delta(ndcg["dense"], ndcg["hybrid"]),
        "rerank_vs_hybrid": paired_delta(ndcg["hybrid"], ndcg["hybrid+rerank"]),
        "rerank_vs_dense": paired_delta(ndcg["dense"], ndcg["hybrid+rerank"]),
    }

    print(f"\n{'stage':<15} {'nDCG@10':>8} {'95% CI':>17} {'MRR@10':>7} {'R@50':>6} {'p50 ms':>7}")
    for name, r in results.items():
        lo, hi = r["ndcg@10_ci95"]
        print(f"{name:<15} {r['ndcg@10']:>8.3f} [{lo:.3f}, {hi:.3f}] {r['mrr@10']:>7.3f} "
              f"{r['recall@50']:>6.3f} {r['latency_ms_p50']:>7.1f}")
    for name, c in comparisons.items():
        print(f"{name:<18} {c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "dataset": "BeIR/scifact test",
            "n_queries": len(query_ids),
            "candidates_k": candidates_k,
            "dense_model": settings.dense_model,
            "rerank_model": settings.rerank_model,
            "bootstrap_resamples": N_BOOTSTRAP,
            "stages": results,
            "ndcg@10_paired_deltas": comparisons,
            "env": {"python": platform.python_version(), "machine": platform.machine(),
                    "platform": platform.platform()},
        }, f, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
