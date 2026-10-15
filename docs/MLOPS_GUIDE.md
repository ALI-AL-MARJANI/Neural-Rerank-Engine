# MLOps guide for the Neural Rerank Engine

This guide explains **every file** that turns the search engine into a production-style service.
For each file: what it does, what each line does, and why this choice rather than another.
Config files are also commented line by line in the code itself: read this guide with the file open next to it.

Contents:
1. [Overview](#1-overview)
2. [Glossary](#2-glossary)
3. [Python code: config.py, engine.py, api.py](#3-python-code)
4. [Dockerfile and .dockerignore](#4-dockerfile)
5. [docker-compose.yml and monitoring/](#5-docker-compose)
6. [CI/CD: .github/workflows/ci.yml](#6-cicd-github-actions)
7. [Kubernetes: k8s/](#7-kubernetes)
8. [Makefile and load_test.py](#8-makefile-and-load_testpy)
9. [Journey of a request](#9-journey-of-a-request)
10. [What happens after a `git push`](#10-what-happens-after-a-git-push)
11. [25 interview questions](#11-25-interview-questions)
12. [Limitations](#limitations)

---

## 1. Overview

```
             Python code (src/rerank_engine)
                       │
       pytest + ruff ──┤  (CI: GitHub Actions)
                       ▼
        Dockerfile ──► Docker image ──► ghcr.io  (CD: GitHub Actions)
                                            │
                                            ▼  kubectl apply -f k8s/
                         minikube: Deployment (2-4 pods) + Service + HPA
                                            │
                                 Prometheus reads /metrics
```

| File | Role in one sentence |
| :--- | :--- |
| `src/rerank_engine/config.py` | reads environment variables (paths, models, TOP_K) |
| `src/rerank_engine/engine.py` | the pipeline: hybrid retrieval, then reranking |
| `src/rerank_engine/api.py` | the FastAPI HTTP API |
| `src/rerank_engine/retrievers/fusion.py` | RRF fusion, isolated so it can be tested |
| `scripts/build_index.py` | downloads SciFact and the models, builds the indices |
| `scripts/load_test.py` | measures latency (p50/p90/p99) |
| `tests/` | fast unit tests with a fake engine |
| `Dockerfile`, `.dockerignore` | the image recipe |
| `docker-compose.yml`, `monitoring/` | API + Prometheus + Grafana locally |
| `.github/workflows/ci.yml` | CI (lint + tests) and CD (build + push the image) |
| `k8s/*.yaml` | Kubernetes deployment |
| `Makefile` | shortcuts for every command |
| `RUNBOOK.md` | commands to run, with the expected output |

**What changed in the ML code** (no change to the logic):
- `src/` became a real installable package, `src/rerank_engine/`, with `pyproject.toml`. Before, the scripts used
  `sys.path.append(...)` and `from src...`.
- `HybridRetriever` now accepts already-configured retrievers (paths and models come from the environment).
  The RRF fusion was moved as-is into `fusion.py`, with the same `1/(rank+60)` formula.
- **One bug fixed**: the reranker wrote `doc['rerank_score']` *into* the in-memory corpus documents. In a server, that
  corpus is shared by all concurrent requests, so the score of one request could "leak" into another. It now works on copies.
- The old MS MARCO scripts (`benchmark.py`, `generate_ground_truth.py`, `data_loader.py`, `build_indices.py`,
  `test_pipeline.py`) were removed (see [Limitations](#limitations) for the benchmark).

---

## 2. Glossary

| Term | Simple definition |
| :--- | :--- |
| **Image** | A frozen, read-only package with everything needed to run the app: base system, Python, libraries, code, models. The "baked recipe". |
| **Container** | An image *that is running*: an isolated process (own files, own network, limited CPU/memory). Many containers can run from the same image. |
| **Layer** | Each Dockerfile instruction (`RUN`, `COPY`…) produces a layer. Docker caches layers: if an instruction and everything before it did not change, it is not run again. |
| **Registry** | A "GitHub for images": a server that stores images. Here ghcr.io (GitHub Container Registry). Docker Hub is another one. |
| **CI** (Continuous Integration) | On every push, automatically check that the code is correct (lint, tests). |
| **CD** (Continuous Delivery/Deployment) | After CI, automatically produce the deliverable (here the image) and publish it. *Delivery* = ready to deploy, *Deployment* = deployed automatically. This project does **delivery**: `kubectl apply` stays manual. |
| **Pod** | The smallest Kubernetes unit: one or more containers sharing the same IP. Here, 1 pod = 1 API container. |
| **Deployment** | Kubernetes object that keeps *N identical pods* running, recreates them if they die, and handles progressive updates. |
| **Replica** | A copy of a pod. `replicas: 2` = 2 identical pods. |
| **Service** | A stable address (DNS name + IP) in front of a group of pods, which spreads traffic between them. Pods get a new IP each time they restart; the Service does not. |
| **Liveness probe** | "Is the process alive?" If it fails several times, Kubernetes **restarts** the container. |
| **Readiness probe** | "Can this pod receive traffic?" If it fails, the pod is **removed from the Service** (no more requests) but not restarted. |
| **ConfigMap** | Kubernetes object that stores non-secret configuration (key/value), injected into pods as environment variables or files. For passwords, use a **Secret**. |
| **HPA** (HorizontalPodAutoscaler) | Adds or removes pods automatically based on a metric (here average CPU). "Horizontal" = more pods; "vertical" = bigger pods. |
| **metrics-server** | Small component that collects pod CPU/memory. The HPA and `kubectl top` need it. |
| **Prometheus** | Time-series database that **fetches** application metrics at a regular interval and lets you query them (PromQL language). |
| **Scraping** | Prometheus doing a `GET /metrics` on each target every X seconds ("pull" model: Prometheus pulls, the app does not push). |
| **Histogram** | Metric that counts how many requests fall into each duration "bucket" (< 0.1 s, < 0.5 s, < 1 s…). Percentiles are computed from it. |
| **p50 / p90 / p99** | Latency percentiles. p50 (median): 50% of requests are faster than this value. p90: 90% are faster, so 10% are slower. We look at p90/p99 because the average hides the slow requests users actually feel. |
| **Rolling update** | Progressive update: create one pod with the new version, wait until it is *ready*, remove one old pod, and so on. Zero downtime. |
| **Lifespan** | In FastAPI, code run once when the server starts and once when it stops. |

---

## 3. Python code

### `config.py`
- **Purpose**: one place for the settings read from the environment: `INDEX_DIR`, `DENSE_MODEL`, `RERANK_MODEL`, `TOP_K`.
- **Key lines**:
  - `@dataclass(frozen=True) class Settings`: a simple, immutable object holding the settings.
  - `bm25_path` / `dense_dir`: paths derived from `INDEX_DIR` (`bm25.pkl` and `dense/`).
  - `get_settings()`: reads `os.getenv("NAME", "default")` **on every call**, so a test can change the environment.
- **Why**: the "12-factor app" principle: the same image runs everywhere (local, Docker, k8s), only the config
  changes. No `pydantic-settings`, to avoid a dependency for 4 variables.
- **TOP_K** = number of stage-1 **candidates** sent to the reranker (default 50). Not to be confused with the request's
  `top_k`, which is the number of **results returned** (default 10).

### `engine.py`
- **Purpose**: connect the existing ML building blocks into one pipeline.
- `SearchEngine.search(query, top_k, rerank)`:
  1. `self.retriever.search(query, k=max(candidates_k, top_k))`: BM25 + FAISS + RRF.
  2. if `rerank=True`: the cross-encoder rescores the candidates and keeps `top_k`, score = `rerank_score`;
     otherwise keep the first `top_k`, score = `rrf_score`.
  3. returns `{doc_id, text, score}`.
- `load_engine(settings)`: loads the 2 models and the 2 indices. **The heavy imports (torch, faiss) are inside the
  function**, so importing the API stays fast and the tests never load torch.
- **Why separate engine and API**: the API only knows an object with a `search()` method. Tests replace it with a fake
  (`FakeEngine`). This is *dependency injection*.

### `api.py`
Line by line (in file order):
- `logging.basicConfig(...)`: log format `date level module message`, level INFO.
- `logging.getLogger("sentence_transformers").setLevel(logging.WARNING)`: otherwise the library prints a progress bar on every request.
- `class SearchRequest(BaseModel)`: schema of the JSON body, validated automatically by **Pydantic**:
  - `str_strip_whitespace=True`: removes leading/trailing spaces, so `"   "` becomes `""`, which is rejected.
  - `query: str = Field(..., min_length=1, max_length=1000)`: `...` = required; empty → **422**.
  - `top_k: int = Field(10, ge=1, le=50)`: between 1 and 50 (otherwise 422); the bound stops a client from asking for 10,000 results.
  - `rerank: bool = True`.
- `SearchResult`, `SearchResponse`: response schema (`response_model`). FastAPI checks and documents the output.
- `create_app(engine_loader=load_engine)`: an application **factory**. In production it loads the real engine;
  in tests it receives `lambda settings: FakeEngine()`.
- `lifespan` (with `@asynccontextmanager`): code **before `yield`** runs **once at startup** (loading the models,
  ~11 s measured on M4), code **after `yield`** once at shutdown. Uvicorn accepts no connection before startup is done.
  - Why not `@app.on_event("startup")`: that syntax is deprecated; `lifespan` is the current recommended way.
- `app.state.engine`: where the engine is stored, reachable from every route.
- `Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(app, include_in_schema=False)`:
  - `.instrument(app)` adds a middleware that measures each request: counter `http_requests_total{handler, method, status}`
    and histogram `http_request_duration_seconds`;
  - `.expose(app)` creates the `GET /metrics` route in Prometheus text format;
  - `/metrics` itself is excluded, so scraping does not count as traffic.
  - **Errors** show up through the `status="4xx"` / `"5xx"` label.
- `@app.middleware("http") log_requests`: times each request and writes **one log line**:
  `POST /search 200 489.3ms`. Uvicorn runs with `--no-access-log` to avoid two lines per request.
- `GET /health` (`async def`): always returns `{"status": "ok"}` if the process answers.
- `GET /ready` (`async def`): 200 if `app.state.engine` exists, **503** otherwise.
  - Why `async def` for these two: they run directly on the event loop, so they answer even when every thread is busy
    with searches. Otherwise the liveness probe could fail under load and Kubernetes would kill a pod that is busy, not broken.
- `POST /search` (`def`, **not** `async def`): torch inference is **blocking** CPU work. With a plain `def`,
  FastAPI runs it in a thread pool and the server keeps answering other requests. In an `async def`, it would block the whole loop.
- `app = create_app()`: the object that `uvicorn rerank_engine.api:app` starts.

### `fusion.py` (RRF)
`score(doc) = sum over each list of 1 / (rank + 60)`. A document ranked well by BM25 **and** FAISS gets two
contributions. RRF only uses **ranks**, not raw scores, which is convenient because BM25 scores (0 to 30+) and FAISS
distances are not comparable. Rank starts at 0 in this code (the original code was kept as-is); the original paper
starts at 1, the effect is minimal.

### `tests/`
- `conftest.py`: `FakeEngine` (returns fixed results instantly) + a `client` fixture = `TestClient` used inside a
  `with` block, which **runs the lifespan** like a real server.
- `test_api.py`: `/health` 200, `/ready` 200 (and **503** when the lifespan did not run), `/search` schema,
  default `top_k`, **422** for an empty query, a whitespace-only query, a missing query and an out-of-range `top_k`,
  and `/metrics` contains the expected metrics.
- `test_fusion.py`: RRF on a toy example computed by hand, and a check that the inputs are not modified.
- **Why fakes**: tests run in 0.04 s without downloading 200 MB of models or the corpus. CI stays fast and
  deterministic. The real ML part is checked by `build_index.py` during the Docker build and by `/ready`.

### `scripts/build_index.py`
Downloads the **BEIR SciFact** corpus (5,183 scientific abstracts, public, CC BY-SA 4.0 license) with
`datasets.load_dataset("BeIR/scifact", "corpus")`, puts `title + text` into `text`, builds the BM25 index
(`bm25.pkl`) and the FAISS index (`dense/index.faiss` + `metadata.pkl`), then loads the cross-encoder once so it is in
the Hugging Face cache. Measured: 39 s on M4 (downloads included).

---

## 4. Dockerfile

**Purpose**: the recipe to build the image containing Python, the libraries, the code, **the models and the indices**,
so the container starts without internet.

**The 3 stages (multi-stage)**:

| Stage | Base | Does what | Ends up in the image? |
| :--- | :--- | :--- | :--- |
| `builder` | python:3.11-slim | creates `/opt/venv`, installs CPU torch + `requirements.txt` + the package | only `/opt/venv` |
| `indexer` | python:3.11-slim (build platform) | also installs `datasets`, runs `build_index.py` | only the models and indices |
| `runtime` | python:3.11-slim | copies venv, models, indices; non-root user; starts uvicorn | **yes, this is the final image** |

Line by line (the important points):
- `# syntax=docker/dockerfile:1`: use the recent Dockerfile syntax (BuildKit).
- `ARG TORCH_VERSION=2.8.0`: build variable, declared once at the top and re-declared in each stage that uses it.
- `FROM python:3.11-slim AS builder`: official minimal Debian image with Python 3.11. `slim` is several times smaller
  than the full image (no compilers or extra tools). Why not `alpine`: Alpine uses musl instead of glibc, and many Python
  wheels (torch in particular) are not available for it or must be compiled.
- `ENV PIP_NO_CACHE_DIR=1`: pip does not keep downloaded files, so layers are smaller.
- `RUN python -m venv /opt/venv` + `ENV PATH="/opt/venv/bin:$PATH"`: everything is installed in a single folder,
  easy to copy as-is into the final stage.
- `COPY requirements.txt .` **before** `COPY src`: the **layer cache** trick. Dependencies change rarely, code changes
  often. If only the code changes, Docker reuses the `pip install` layer (minutes saved).
- `pip install torch==... --index-url https://download.pytorch.org/whl/cpu`: the **CPU** build of PyTorch. The default
  torch from PyPI on Linux x86_64 ships CUDA (several GB of NVIDIA libraries), useless without a GPU. torch is installed
  **alone** with `--index-url`, then the rest comes from normal PyPI.
- `pip install --no-deps .`: installs our `rerank_engine` package (dependencies are already there).
- `FROM --platform=$BUILDPLATFORM python:3.11-slim AS indexer`: this stage runs on the **architecture of the machine
  doing the build** (amd64 on GitHub). When CI builds the arm64 image (for the Mac), only `builder` and `runtime` go
  through QEMU emulation, which is slow. Computing embeddings for 5,183 documents under emulation would take much longer.
  The output (index files, model weights) is **data**, identical on every CPU type.
- `ENV HF_HOME=/opt/hf INDEX_DIR=/opt/indices` + `RUN python scripts/build_index.py`: download and indexing happen **at build time**.
- `FROM python:3.11-slim AS runtime`: start again from a clean image. Nothing from `indexer` (the `datasets` library,
  the corpus cache) and no pip cache gets in.
- `RUN useradd --create-home --uid 1000 appuser` + `USER appuser`: the server does **not run as root**. If someone
  exploits a bug in the app, they do not get root inside the container. Many clusters refuse root containers anyway.
- `COPY --from=builder /opt/venv /opt/venv`: take only the output of the previous stage.
- `COPY --from=indexer --chown=appuser:appuser ...`: models and indices, owned by appuser.
- `ENV HF_HUB_OFFLINE=1`: forbids Hugging Face from going online, it uses the cache. If a model is missing, the app
  crashes at startup instead of silently downloading in production.
- `ENV PYTHONUNBUFFERED=1`: logs appear immediately in `docker logs` / `kubectl logs`.
- `EXPOSE 8000`: **documentation**: the app listens on 8000. It opens nothing; `-p 8000:8000` publishes the port.
- `HEALTHCHECK ... CMD python -c "urllib..."`: Docker calls `/health` every 30 s and `docker ps` shows `(healthy)`.
  Python is used because `curl` is not installed in `slim`. `--start-period=60s` leaves time to load the models.
  **Note**: Kubernetes **ignores** the Docker HEALTHCHECK and uses its own probes.
- `CMD ["uvicorn", "rerank_engine.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]`:
  JSON form ("exec form"): uvicorn is the main process (PID 1) and receives the stop signal (SIGTERM) from
  Docker/Kubernetes directly, so it shuts down cleanly. `0.0.0.0` = accept connections from outside the container
  (with `127.0.0.1`, only the container itself could connect).

**Index in the image vs mounted volume** (a choice to be able to defend):

| | Index and models **in the image** (chosen here) | Index and models on a **mounted volume** |
| :--- | :--- | :--- |
| Startup | nothing to download, works offline | depends on the volume or storage (S3, PVC) |
| Reproducibility | image = code + model + data, versioned together by the tag | data must be versioned separately |
| Image size | bigger (~200 MB of models + 34 MB of indices) | smaller |
| Updating the data | rebuild the image | replace the files, no rebuild |
| Good for | small models and corpus (this case) | big models (several GB), corpus that changes often |

In production with a large corpus, you would use a volume: a PersistentVolumeClaim in Kubernetes, or an
initContainer that downloads from object storage (S3/GCS).

**`.dockerignore`**: lists what is **not sent** to Docker during the build: `venv/` (hundreds of MB, and built for
macOS while the image is Linux), `data/` (rebuilt inside the image), `.git/`, notebooks, tests, docs, k8s…
Faster build, smaller image, and no risk of copying a `.env` with secrets.

**Image size**: `<to be measured>` with `docker images neural-rerank-engine:local` (see RUNBOOK).

---

## 5. Docker Compose

**Purpose**: start **several containers together** with one command (`docker compose up`), on a shared network where
each service is reachable by its name (`api`, `prometheus`).

- `api`: `build: .` builds the image from the Dockerfile; `ports: "8000:8000"` = host port → container port;
  `restart: unless-stopped` restarts the container if it crashes.
- `prometheus`: official image with a **pinned version** (`v3.14.0`, never `latest`, so it behaves the same in 6 months);
  `volumes` mounts our `monitoring/prometheus.yml` read-only (`:ro`); UI on `:9090`.
- `grafana`: `profiles: ["grafana"]` = **optional**, only started with `--profile grafana`. The Prometheus data source is
  configured automatically ("provisioning") by `monitoring/grafana/datasource.yml`.
- `depends_on`: **start** order only (it does not wait for the API to be *ready*).

`monitoring/prometheus.yml`: `scrape_interval: 5s`, one target `api:8000` on `/metrics`. In Docker Compose, `api` is
resolved to an IP by the internal DNS of the compose network.

---

## 6. CI/CD (GitHub Actions)

**Purpose**: on every push, GitHub starts a clean Linux virtual machine that runs `ci.yml`.

- `on: push: branches: ["**"], tags: ["v*"]` + `pull_request`: triggers. `"**"` = every branch.
- `permissions: contents: read`: by default the workflow token can only *read* the code (least privilege).
- `env: IMAGE_NAME: ghcr.io/ali-al-marjani/neural-rerank-engine`: an image name must be **lowercase**, while the GitHub
  repo name has capitals, so it is written explicitly.

**`test` job (CI)**:
- `runs-on: ubuntu-latest`: Linux VM provided for free by GitHub.
- `actions/checkout@v6`: gets the code.
- `actions/setup-python@v6` with `python-version: "3.11"` (same as the Docker image) and `cache: pip` (downloaded
  packages are reused between runs).
- Install: CPU torch, then `requirements.txt` + `requirements-dev.txt`, then `pip install -e .`.
- `ruff check .`: linter; fails on an undefined name, an unused import or badly sorted imports (rules `E`, `F`, `I`).
- `pytest -v`: the 11 tests. If a step fails, the job is red and CD does not start.

**`docker` job (CD)**:
- `needs: test`: only if CI is green.
- `if: github.event_name == 'push' && (ref == main || tag v*)`: no image published for a pull request or a work branch.
- `permissions: packages: write`: allows `GITHUB_TOKEN` to push to ghcr.io. **No secret to create**: GitHub generates
  this token automatically for each run, and it expires at the end of the run.
- `docker/setup-qemu-action@v4`: emulator that lets an amd64 VM build an arm64 image.
- `docker/setup-buildx-action@v4`: enables Buildx (BuildKit), Docker's modern builder (multi-platform, cache).
- `docker/login-action@v4`: `docker login ghcr.io` with `github.actor` and `secrets.GITHUB_TOKEN`.
- `docker/metadata-action@v6`: computes the tags:
  - `type=raw,value=latest,enable={{is_default_branch}}`: `latest`, only on `main`;
  - `type=sha`: `sha-1a2b3c4`, one **immutable** tag per commit (you know exactly which code runs);
  - `type=semver,pattern={{version}}`: `1.0.0` when pushing the git tag `v1.0.0`.
- `docker/build-push-action@v7`: build + push.
  - `platforms: linux/amd64,linux/arm64`: a "multi-arch" image. amd64 for typical cloud servers, arm64 for the
    Apple Silicon Mac (minikube runs arm64 there). Docker picks the right one automatically on `pull`.
  - `cache-from/cache-to: type=gha`: layers are kept in the GitHub Actions cache, which speeds up later builds.

Why ghcr.io rather than Docker Hub: built into GitHub (same account, same token, no secret to create), free for public
images, and the package is linked to the repo.

---

## 7. Kubernetes

Apply with `kubectl apply -f k8s/`. Kubernetes is **declarative**: you describe *the desired state* ("2 pods of this
image") and Kubernetes keeps doing whatever is needed to reach it (recreate a dead pod, etc.).

Every YAML has the same structure: `apiVersion` (Kubernetes API version), `kind` (object type), `metadata` (name,
labels), `spec` (desired state).

### `configmap.yaml`
- `TOP_K`, `DENSE_MODEL`, `RERANK_MODEL`: app settings. Model names must match the models **inside the image** (offline mode).
- `OMP_NUM_THREADS: "2"`: number of torch CPU threads = the pod's CPU limit. Without it, torch sees every core of the
  node, starts too many threads, and Kubernetes throttles it as soon as it exceeds its limit.
- Why a ConfigMap: change a setting **without rebuilding the image**. After `kubectl apply`, the pods must be restarted
  (`kubectl rollout restart`), because environment variables are read at startup.

### `deployment.yaml`
- `replicas: 2`: 2 pods, so if one crashes the other keeps serving (minimal high availability).
- `selector.matchLabels: app: rerank-api` + `template.metadata.labels: app: rerank-api`: the Deployment recognises
  "its" pods by this label. The Service uses the same label.
- `strategy: RollingUpdate`, `maxUnavailable: 0`, `maxSurge: 1`: during an update, add 1 new pod, wait until it is
  **ready**, remove 1 old pod, and so on. There are never fewer than 2 ready pods.
- `annotations: prometheus.io/scrape|port|path`: free-form labels that **our Prometheus** (`k8s/prometheus.yaml`) reads
  to know which pods to scrape and on which port/path. Nothing magic: without a Prometheus configured to read them,
  they do nothing.
- `image: ghcr.io/ali-al-marjani/neural-rerank-engine:latest` + `imagePullPolicy: Always`: since `latest` changes on
  every push, Kubernetes checks the registry each time it creates a pod. In real production you would deploy the
  immutable `sha-xxxxxxx` tag.
- `envFrom: configMapRef: rerank-config`: every key of the ConfigMap becomes an environment variable.
- `resources`:
  - `requests` (`cpu: 500m` = half a core, `memory: 1Gi`): what is **reserved**. The scheduler only places the pod on
    a node with that much free capacity. The HPA computes its percentages **relative to the requests**.
  - `limits` (`cpu: 2`, `memory: 2Gi`): the maximum. CPU above it = slowed down (throttling); memory above it =
    container **killed** (`OOMKilled`).
  - The values: locally the process used ~350 to 460 MB of RAM (RSS measured on macOS, Python 3.9). 1 Gi leaves a
    margin. Tune them with `kubectl top pods` (`<to be measured>`).
- `readinessProbe` on `/ready`: every 5 s, after 10 s; 3 failures → the pod is removed from the Service.
- `livenessProbe` on `/health`: every 10 s, after **60 s** (so a pod loading its models is not killed); 3 failures → restart.
- Why `/health` does **not** check the model: if liveness depended on an external service or a slow state, a temporary
  problem would make every pod restart in a loop. Liveness should only say "the process is not stuck".

### `service.yaml`
- `type: ClusterIP`: an **internal** cluster IP, with a DNS name `rerank-api`. Other pods call `http://rerank-api`,
  and the Service spreads requests over the **ready** pods.
- `port: 80` → `targetPort: http` (the port named `http` = 8000 in the Deployment).
- Access from the Mac: `kubectl port-forward service/rerank-api 8000:80`. Note: `minikube service` needs a `NodePort`
  or `LoadBalancer` Service; with ClusterIP, use `port-forward`. **Careful**: `port-forward` sends everything to *one*
  pod (it is a debug tunnel, not a load balancer). To test load balancing and the HPA, the RUNBOOK sends the load
  **from a pod inside the cluster**.
- Service types: `ClusterIP` (internal), `NodePort` (a port opened on every node), `LoadBalancer` (public IP provided by
  the cloud), and `Ingress` (HTTP routing by domain/path, in front of Services).

### `hpa.yaml`
- `apiVersion: autoscaling/v2`: current HPA API version.
- `scaleTargetRef`: the `rerank-api` Deployment.
- `minReplicas: 2`, `maxReplicas: 4`.
- `averageUtilization: 70`: if the **average** CPU of the pods exceeds 70% of their *request* (= 350m), the HPA adds
  pods. Formula: `desired_pods = ceil(current_pods × current_CPU / target_CPU)`.
- Prerequisite: `minikube addons enable metrics-server`.
- After the load, scaling down waits 5 minutes (default stabilization window) to avoid flapping.
- Why CPU: this service is CPU-bound (the cross-encoder). For an LLM on GPU you would scale on queue length or
  in-flight requests instead (custom metrics).

### `prometheus.yaml` (Prometheus in the cluster)
Several objects separated by `---`:
- `ServiceAccount` `prometheus`: an identity for the Prometheus pod.
- `ClusterRole` + `ClusterRoleBinding` (RBAC): gives that identity the right to `get/list/watch` pods. Prometheus asks
  the Kubernetes API to **discover** pods ("service discovery").
- `ConfigMap prometheus-config`: the Prometheus config. The `relabel_configs` (taken from the official Prometheus example):
  1. `keep` only pods with `prometheus.io/scrape: "true"`;
  2. path = the `prometheus.io/path` annotation;
  3. address = `pod IP:` + the `prometheus.io/port` annotation;
  4. add a `pod` label to tell pods apart in graphs.
- `Deployment prometheus` (1 pod) that mounts the ConfigMap as a file in `/etc/prometheus`.
- `Service prometheus` to reach it with `kubectl port-forward service/prometheus 9090:9090`.
- Why not the "kube-prometheus-stack" (Helm, Prometheus Operator): it is the industry standard, but it installs dozens
  of objects. The goal here is to understand the mechanism, and no Helm.

---

## 8. Makefile and load_test.py

**Makefile**: aliases. `make run` = `uvicorn rerank_engine.api:app --port 8000 --no-access-log`. `make help` lists them.
In a Makefile, command lines must start with a **tab**. `-kubectl top pods`: the `-` tells make to continue even if the
command fails (metrics-server not ready yet). `make k8s-deploy` runs `minikube start --driver=docker --cpus=4 --memory=6g`:
minikube creates a Kubernetes "node" inside a Docker Desktop container.

**load_test.py**:
- sends `-n` `POST /search` requests, `-c` of them in parallel (`ThreadPoolExecutor`);
- 1 warm-up request, not counted (the first one is always slower);
- prints p50/p90/p99, mean, throughput (req/s) and the number of errors;
- **standard library only** (`urllib`), so it can be piped into a pod through `stdin`
  (`python - ... < scripts/load_test.py`) to load the Service from inside the cluster.

**Real measurements** (Apple M4, 16 GB, native Python 3.9, no Docker, `top_k=10`, 50 candidates reranked):

| Case | n | c | p50 | p90 | p99 | throughput |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid + rerank | 100 | 1 | 489 ms | 501 ms | 541 ms | 2.0 req/s |
| hybrid + rerank | 200 | 4 | 2248 ms | 2408 ms | 2568 ms | 1.8 req/s |
| hybrid only | 200 | 1 | 13 ms | 17 ms | 23 ms | 71.8 req/s |

Reading: reranking is ~97% of the time (~490 ms out of ~500). With 4 parallel requests, throughput does not increase:
one process already saturates the CPU and requests queue up (latency ×4). To get more throughput you need **more pods**
(HPA), **fewer candidates** (`TOP_K`), or a faster model (ONNX, quantization, GPU). Docker/k8s: `<to be measured>`.

---

## 9. Journey of a request

Example: `curl -X POST localhost:8000/search -d '{"query":"statins lower LDL"}'` on minikube.

1. **Mac → tunnel**: `kubectl port-forward` listens on `localhost:8000` and forwards to the Service `rerank-api:80`.
   From another pod you would call `http://rerank-api` directly (internal DNS).
2. **Service → pod**: the Service picks one of the **ready** pods (those whose `/ready` returns 200) and forwards to its port 8000.
3. **Uvicorn** (ASGI server) receives the HTTP connection and passes it to FastAPI.
4. **Middlewares**: the instrumentator starts its timer, and so does our logging middleware.
5. **Pydantic validation**: the JSON becomes a `SearchRequest`. Empty query or out-of-range `top_k` → 422 immediately,
   without touching the model.
6. **`search` route** (in a pool thread) → `engine.search()`:
   1. BM25: tokenization `lower().split()`, scoring of the 5,183 documents → top 50;
   2. FAISS: query encoded by MiniLM (384-dimension vector), exact L2 search → top 50;
   3. RRF: the two lists are merged → 50 candidates;
   4. cross-encoder: 50 (query, document) pairs → 50 scores → sort → top 10.
7. **Response**: `{"results":[...], "latency_ms": ...}`, validated by `SearchResponse`.
8. **Metrics and logs**: `http_requests_total{handler="/search",status="2xx"}` +1, the duration goes into the histogram,
   and a line `POST /search 200 495.1ms` is written to the logs (`kubectl logs`).
9. **Later**: Prometheus scrapes this pod's `/metrics` every 5 s. metrics-server measures its CPU, and the HPA decides
   whether more pods are needed.

---

## 10. What happens after a `git push`

1. GitHub receives the commit and triggers the `CI` workflow (`.github/workflows/ci.yml`).
2. **`test` job** on a fresh Ubuntu VM: checkout → Python 3.11 → install → `ruff check .` → `pytest`.
   The result ✅/❌ shows in the Actions tab, next to the commit and on the README badge.
3. If the branch is `main` (or a `v*` tag) **and** `test` is green → **`docker` job**:
   QEMU + Buildx → ghcr.io login with `GITHUB_TOKEN` → tag computation → amd64 and arm64 build (the Dockerfile downloads
   SciFact and the models and builds the index) → push of `latest` and `sha-xxxxxxx`.
4. The image appears in the *Packages* tab of the GitHub profile.
5. **Deployment stays manual**: `kubectl rollout restart deployment/rerank-api`; the new pods pull the new `latest`
   (`imagePullPolicy: Always`) through a rolling update. To automate it you would add a `kubectl set image ... :sha-xxx`
   step to the workflow (with cluster credentials), or a GitOps tool like Argo CD that syncs the cluster with the repo.

---

## 11. 25 interview questions

1. **Why a multi-stage Dockerfile?**
   So the final image only contains what is needed at runtime. Build steps (pip cache, `datasets`, corpus download)
   stay in intermediate stages that are thrown away. Smaller image, faster pulls, smaller attack surface.

2. **Liveness vs readiness?**
   Liveness = "is the process alive?" → a failure restarts the container. Readiness = "can it serve traffic?" → a
   failure removes the pod from the Service, without a restart. Here `/ready` checks that the models are loaded,
   `/health` only that the process answers.

3. **Why load the model at startup and not on each request?**
   Loading models and indices takes ~11 s (measured); a search ~0.5 s. Loading per request would multiply latency by
   20 and memory by the number of requests. The `lifespan` loads them once and every request shares them.

4. **How do you update without downtime?**
   Rolling update of the Deployment with `maxUnavailable: 0` and `maxSurge: 1`: Kubernetes creates a pod with the new
   version, waits until it is *ready* (readiness probe), then removes an old one. There are always at least 2 ready
   pods. Rollback: `kubectl rollout undo deployment/rerank-api`.

5. **Why CPU torch?**
   No GPU here. The default torch on Linux x86_64 ships CUDA and NVIDIA libraries (several GB) that would be useless.
   The CPU build is much lighter.

6. **Why a non-root user?**
   Defense in depth: if the app is compromised, the attacker is not root in the container. Many clusters (Pod Security
   Standards "restricted") refuse root containers anyway.

7. **Why `python:3.11-slim` and not `alpine`?**
   Alpine uses musl instead of glibc, and the torch/faiss wheels are built for glibc (manylinux). On Alpine you would
   often have to compile, which makes builds slow and fragile. `slim` is minimal Debian, and compatible.

8. **Why copy `requirements.txt` before the code?**
   Layer cache: Docker reuses the `pip install` layer as long as `requirements.txt` does not change. Editing a line
   of code does not reinstall torch.

9. **Index in the image or on a mounted volume?**
   Here, in the image: the corpus and models are small, and the image is self-contained, reproducible and works
   offline. With multi-GB models or an index that changes daily, I would put them on a volume (PVC) or download them
   from S3/GCS in an initContainer, to avoid rebuilding the image every time.

10. **CI vs CD in your project?**
    CI: ruff lint + pytest on every push and PR. CD: on `main` and `v*` tags, multi-arch image build and push to
    ghcr.io. This is *continuous delivery*: the artifact is ready, but applying it to the cluster stays manual.

11. **How does the workflow log in to ghcr.io without a secret?**
    With `GITHUB_TOKEN`, a token GitHub creates automatically for each run and that expires at the end of it. It only
    gets `packages: write` in the job that pushes the image (least privilege).

12. **Why tag with the SHA as well as `latest`?**
    `latest` moves on every push: you don't know which version is running, and you can't roll back cleanly.
    `sha-1a2b3c4` is immutable and ties the image to one commit. In production I would deploy the SHA tag.

13. **Why fakes in the tests?**
    To test the API's behavior (HTTP codes, validation, schema, metrics) in 0.04 s without downloading 200 MB of
    models. Tests are fast and deterministic, and CI is cheap. The real ML part is validated during the Docker build
    (index construction) and by the readiness probe.

14. **What do you monitor, and how?**
    `prometheus-fastapi-instrumentator` exposes `/metrics`: number of requests per route and per status (so the 5xx
    error rate) and a latency histogram. Prometheus scrapes it every 5 s. Useful queries:
    `rate(http_requests_total[1m])`, `histogram_quantile(0.9, ...)`. Plus one log line per request.

15. **Why a histogram rather than an average latency?**
    The average hides slow requests. With a histogram, Prometheus computes p50/p90/p99, including across several pods,
    which is impossible with percentiles computed inside each pod.

16. **How does your HPA work?**
    metrics-server measures pod CPU. The HPA compares the average with 70% of the *request* (350m) and adjusts:
    `ceil(pods × current_CPU / target)`, between 2 and 4 pods. Scaling down waits 5 minutes to avoid oscillation.

17. **Why does throughput not increase with more concurrency (c=4)?**
    The cross-encoder already saturates every CPU core for a single request, so requests queue. With c=4 I measured
    the same ~2 req/s and ~4× the latency. Throughput comes from adding pods (horizontal scaling), lowering `TOP_K`, or
    speeding up the model.

18. **How would you reduce reranking latency?**
    Rerank fewer candidates (TOP_K 50 → 20); truncate documents; export the cross-encoder to ONNX and quantize it to
    int8; batch several requests together; use a GPU; or cache frequent queries. Each change must be checked with a
    quality benchmark (nDCG) so relevance does not drop.

19. **Why `def` and not `async def` for `/search`?**
    Inference is blocking CPU work. Inside an `async def` it would block the event loop and the whole server. With
    `def`, FastAPI runs it in a thread pool. On the other hand, `/health` and `/ready` are `async` so they still answer
    when every thread is busy.

20. **Why requests/limits?**
    Requests let the scheduler place pods where there is room, and are the HPA's baseline. Limits stop one pod from
    eating the whole node. Above the memory limit the pod is killed (OOMKilled); above the CPU limit it is slowed down.

21. **Why a ConfigMap?**
    To separate config from code: I can change `TOP_K` without rebuilding the image. For sensitive data (an API key) I
    would use a Secret.

22. **How would you move to production in the cloud?**
    The same manifests work on managed Kubernetes (GKE, EKS, AKS), where the cloud runs the control plane. I would add:
    an Ingress + TLS certificate, the SHA tag instead of `latest`, Secrets, a cluster autoscaler to add nodes, and
    managed Prometheus/Grafana or the Operator. Simpler option: **Cloud Run**, which runs the Docker image directly,
    handles HTTPS and scales on requests (down to zero). Watch out for cold starts (~11 s to load the models): keep at
    least one minimum instance.

23. **How would you serve a bigger LLM?**
    Not with FastAPI + torch in the same process. I would use a dedicated inference server like **vLLM** (PagedAttention
    to manage KV-cache memory, *continuous batching* to group requests) on GPU nodes
    (`resources.limits: nvidia.com/gpu: 1`). The weights would sit on a volume rather than in the image. Autoscaling
    would use queue length or in-flight requests instead of CPU, and our FastAPI API would call that server over HTTP.
    This is not implemented here.

24. **How is the arm64 image built on an amd64 runner?**
    With Buildx + QEMU (emulation). Emulation is slow, so the stage that computes the embeddings runs natively
    (`FROM --platform=$BUILDPLATFORM`) and only its output files (indices, models) are copied into the arm64 image.
    That output is data, independent of the CPU.

25. **What happens if a pod crashes?**
    The Deployment sees 1 pod out of 2 and creates a new one. Meanwhile the Service only sends traffic to the *ready*
    pod. If the process hangs without crashing, the liveness probe fails 3 times and the kubelet restarts the container.

**Likely bonus questions**:
- *"Why BM25 + dense?"* BM25 is strong on exact words (gene names, acronyms), dense retrieval on meaning (synonyms).
  RRF combines both without normalizing scores.
- *"Did you evaluate quality?"* See the limitations below: no quality metric is published for now, and I can explain
  why the old benchmark was wrong.

---

<a id="limitations"></a>
## 12. Limitations

- **Local only**: minikube on a MacBook, a single node. No real high availability (if the Mac turns off, everything
  stops), no cloud, no node autoscaling.
- **No GPU**: CPU inference, ~0.5 s per search with reranking (measured). Throughput per pod is limited to ~2 req/s.
- **No authentication**, no rate limiting, no TLS: anyone who can reach the Service can call it.
- **Basic monitoring**: HTTP metrics + text logs. No alerts (Alertmanager), no versioned Grafana dashboard, no
  distributed tracing, no centralized logs, no ML metrics (query drift, score distribution).
- **Manual deployment**: CD publishes the image, but `kubectl apply` / `rollout restart` are run by hand. `latest` is used
  in the Deployment for simplicity; in production you would deploy the SHA tag.
- **In-memory, pickled indices**: BM25 (`rank_bm25`) rescores the whole corpus on every request, and FAISS does an exact
  search (`IndexFlatL2`). Fine for 5,000 documents, not for millions (you would need an approximate HNSW/IVF index or a
  vector database). Pickle files must only be loaded if they come from our own build (`pickle` can run code).
- **No quality metric**: the old MS MARCO table (MRR@10 0.5594 / 0.7977 / 0.8282) was removed because its protocol was
  wrong. The "correct answer" was the *first candidate passage* of each query, not the passage annotated as relevant.
  Also, the test queries came from the MS MARCO *train* split, on which the `ms-marco-MiniLM` cross-encoder was trained.
  A correct evaluation would use SciFact's **test** queries with their relevance judgments (qrels) and a metric such as
  nDCG@10. This is not done here.
- **A fairly large image**: torch + models + indices inside the image (size `<to be measured>`). This is the price of
  starting offline.
- **`notebooks/demo.ipynb`** still uses the old MS MARCO benchmark and the old `src.*` imports: it is out of date.
