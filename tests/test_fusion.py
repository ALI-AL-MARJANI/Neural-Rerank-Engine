from rerank_engine.retrievers.fusion import reciprocal_rank_fusion


def test_rrf_toy_example():
    sparse = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}, {"id": "c", "text": "C"}]
    dense = [{"id": "b", "text": "B"}, {"id": "d", "text": "D"}, {"id": "a", "text": "A"}]

    fused = reciprocal_rank_fusion([sparse, dense], k=10)

    # b: 1/61 + 1/60, a: 1/60 + 1/62, c: 1/62, d: 1/61
    assert [d["id"] for d in fused] == ["b", "a", "d", "c"]
    assert abs(fused[0]["rrf_score"] - (1 / 61 + 1 / 60)) < 1e-12


def test_rrf_keeps_top_k_and_does_not_modify_inputs():
    sparse = [{"id": str(i), "text": ""} for i in range(5)]
    fused = reciprocal_rank_fusion([sparse, []], k=2)
    assert len(fused) == 2
    assert "rrf_score" not in sparse[0]
