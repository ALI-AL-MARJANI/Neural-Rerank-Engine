"""Shared test fixtures: a fake search engine, so tests never load real models."""
import pytest
from fastapi.testclient import TestClient

from rerank_engine.api import create_app


class FakeEngine:
    """Same interface as SearchEngine.search(), returns fixed results instantly."""

    def search(self, query, top_k=10, rerank=True):
        docs = [{"doc_id": str(i), "text": f"document {i} about {query}", "score": 1.0 / (i + 1)} for i in range(50)]
        return docs[:top_k]


@pytest.fixture
def client():
    app = create_app(engine_loader=lambda settings: FakeEngine())
    # The `with` block runs the lifespan (startup/shutdown), like a real server
    with TestClient(app) as c:
        yield c
