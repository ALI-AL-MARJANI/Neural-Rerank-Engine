from fastapi.testclient import TestClient

from rerank_engine.api import create_app


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_when_engine_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


def test_not_ready_before_startup():
    # Without `with`, the lifespan does not run: the engine is not loaded yet
    c = TestClient(create_app(engine_loader=lambda settings: None))
    assert c.get("/health").status_code == 200
    assert c.get("/ready").status_code == 503
    assert c.post("/search", json={"query": "x"}).status_code == 503


def test_search_response_schema(client):
    r = client.post("/search", json={"query": "vitamin D", "top_k": 3, "rerank": True})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"results", "latency_ms"}
    assert len(body["results"]) == 3
    for item in body["results"]:
        assert set(item) == {"doc_id", "text", "score"}
    assert body["latency_ms"] >= 0


def test_search_defaults(client):
    r = client.post("/search", json={"query": "vitamin D"})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 10  # default top_k


def test_empty_query_is_rejected(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "   "}).status_code == 422


def test_missing_query_is_rejected(client):
    assert client.post("/search", json={}).status_code == 422


def test_invalid_top_k_is_rejected(client):
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 51}).status_code == 422


def test_metrics_endpoint(client):
    client.post("/search", json={"query": "vitamin D"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text
    assert "http_request_duration_seconds" in r.text
