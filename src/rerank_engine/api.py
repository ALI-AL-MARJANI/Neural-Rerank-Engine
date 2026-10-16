"""
FastAPI service for the Neural Rerank Engine.

Run locally:  uvicorn rerank_engine.api:app --port 8000
Interactive docs: http://localhost:8000/docs
"""
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Callable, List

from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from rerank_engine.config import Settings, get_settings
from rerank_engine.engine import load_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rerank_engine")
# sentence-transformers shows a progress bar on every encode() when its logger is at INFO
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# ---------- Request / response schemas (validated by Pydantic) ----------

class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)  # "   " becomes "" -> rejected (422)

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    rerank: bool = Field(True, description="Rerank candidates with the cross-encoder")


class SearchResult(BaseModel):
    doc_id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    latency_ms: float


# ---------- Application factory ----------

def create_app(engine_loader: Callable[[Settings], object] = load_engine) -> FastAPI:
    """
    Builds the FastAPI app. `engine_loader` is a parameter so that tests can
    pass a fake engine instead of loading the real models.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs ONCE when the server starts: load models + indices into memory.
        settings = get_settings()
        logger.info("Loading engine (index_dir=%s, top_k=%d)...", settings.index_dir, settings.top_k)
        start = time.perf_counter()
        app.state.engine = engine_loader(settings)
        # At most N searches run at the same time; the others wait for a free slot.
        # Each rerank allocates a lot of temporary memory: without this bound, a burst
        # of parallel requests goes over the pod's memory limit (OOMKilled).
        app.state.search_slots = threading.Semaphore(settings.max_concurrent_searches)
        logger.info("Engine ready in %.1fs", time.perf_counter() - start)
        yield
        # Runs once when the server stops.
        app.state.engine = None

    app = FastAPI(title="Neural Rerank Engine", version="1.0.0", lifespan=lifespan)
    app.state.engine = None

    # Prometheus metrics: request count, latency histogram, status codes -> GET /metrics
    Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(app, include_in_schema=False)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        # One log line per request: method, path, status code, duration
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, duration_ms)
        return response

    # /health and /ready are `async def`: they run directly on the event loop, so
    # they still answer quickly when all worker threads are busy with searches.
    @app.get("/health")
    async def health():
        """Liveness: the process is running and answers HTTP."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        """Readiness: models and indices are loaded, the pod can receive traffic."""
        if app.state.engine is None:
            raise HTTPException(status_code=503, detail="engine not loaded")
        return {"status": "ready"}

    # A plain `def` (not `async def`): FastAPI runs it in a thread pool, so the
    # CPU-heavy model inference does not block the server's event loop.
    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        engine = app.state.engine
        if engine is None:
            raise HTTPException(status_code=503, detail="engine not loaded")
        start = time.perf_counter()  # includes the time spent waiting for a slot
        with app.state.search_slots:
            results = engine.search(req.query, top_k=req.top_k, rerank=req.rerank)
        latency_ms = (time.perf_counter() - start) * 1000
        return SearchResponse(results=results, latency_ms=round(latency_ms, 2))

    return app


app = create_app()
