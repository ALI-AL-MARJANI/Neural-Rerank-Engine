from collections import defaultdict
from typing import Any, Dict, List


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]], k: int, k_const: int = 60
) -> List[Dict[str, Any]]:
    """
    Merges several ranked lists with Reciprocal Rank Fusion (RRF).
    RRF Score = sum over lists of 1 / (rank + k_constant)

    Returns the top-k documents as new dicts with an added 'rrf_score' key
    (the input documents are not modified).
    """
    rrf_scores = defaultdict(float)
    doc_map = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc['id']
            rrf_scores[doc_id] += 1 / (rank + k_const)
            # If doc was not seen in a previous list, add it to map
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    # Sort by accumulated RRF score
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    return [{**doc_map[doc_id], 'rrf_score': rrf_scores[doc_id]} for doc_id in sorted_ids[:k]]
