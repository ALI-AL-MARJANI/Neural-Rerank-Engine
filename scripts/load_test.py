"""
Small load test: sends N POST /search requests and prints latency percentiles.

Usage:
    python scripts/load_test.py --url http://localhost:8000 -n 200 -c 4
    python scripts/load_test.py --no-rerank            # stage 1 only (BM25 + FAISS)

-n = total number of requests, -c = number of requests in flight at the same time.
Latency is measured on the client side (network + server), in milliseconds.

Only the standard library is used, so the script also runs inside the API image
(in Kubernetes, see RUNBOOK.md):
    kubectl run loadgen -i --rm --restart=Never --image=<api image> -- \
        python - --url http://rerank-api -n 500 -c 8 < scripts/load_test.py
"""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Example scientific claims, in the style of SciFact queries
QUERIES = [
    "vitamin D deficiency increases the risk of multiple sclerosis",
    "smoking is associated with lung cancer",
    "aspirin reduces the risk of cardiovascular events",
    "obesity is a risk factor for type 2 diabetes",
    "antibiotics are effective against viral infections",
    "statins lower LDL cholesterol",
    "sleep deprivation impairs memory consolidation",
    "BRCA1 mutations increase breast cancer risk",
]


def percentile(values, p):
    """p-th percentile (0-100) with linear interpolation."""
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("-n", "--requests", type=int, default=200)
    parser.add_argument("-c", "--concurrency", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    def one_request(i):
        body = {"query": QUERIES[i % len(QUERIES)], "top_k": args.top_k, "rerank": not args.no_rerank}
        req = urllib.request.Request(
            args.url.rstrip("/") + "/search",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                ok = resp.status == 200
        except OSError:  # connection errors, timeouts and HTTP errors (4xx/5xx)
            ok = False
        return ok, (time.perf_counter() - start) * 1000

    one_request(0)  # warm-up request, not counted

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one_request, range(args.requests)))
    total_s = time.perf_counter() - start

    latencies = [ms for ok, ms in results if ok]
    errors = sum(1 for ok, _ in results if not ok)
    print(f"requests={args.requests} concurrency={args.concurrency} rerank={not args.no_rerank} errors={errors}")
    if latencies:
        print(f"p50={percentile(latencies, 50):.0f} ms  p90={percentile(latencies, 90):.0f} ms  "
              f"p99={percentile(latencies, 99):.0f} ms  mean={statistics.mean(latencies):.0f} ms")
    print(f"throughput={args.requests / total_s:.1f} req/s  total={total_s:.1f}s")


if __name__ == "__main__":
    main()
