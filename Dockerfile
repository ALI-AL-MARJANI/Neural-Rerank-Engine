# syntax=docker/dockerfile:1
# ^ Use the latest stable Dockerfile syntax (BuildKit). Must be the very first line.

# =============================================================================
# Multi-stage build: 3 stages, only the last one becomes the final image.
#   1. builder : installs Python dependencies in a virtual environment
#   2. indexer : downloads the corpus + models and builds the search indices
#   3. runtime : small final image = Python + venv + indices + models
# =============================================================================

# Version of the CPU-only PyTorch build (no CUDA = ~10x smaller download)
ARG TORCH_VERSION=2.8.0


# ---------------------------------------------------------------- 1. builder
# Start from the official slim Python 3.11 image (Debian without extras)
FROM python:3.11-slim AS builder
# Re-declare the ARG so it is visible inside this stage
ARG TORCH_VERSION
# Do not keep pip's download cache (smaller layers), no "new pip version" warning
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# Create a virtual environment: the whole folder is copied to the final stage later
RUN python -m venv /opt/venv
# Put the venv first in PATH, so "pip" and "python" now mean the venv's ones
ENV PATH="/opt/venv/bin:$PATH"
# Work in /app inside the image
WORKDIR /app
# Copy ONLY the requirements first: this layer stays cached while the code changes
COPY requirements.txt .
# Install CPU-only torch from the PyTorch index, then the other pinned dependencies
RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt
# Now copy the project code and install our package (dependencies are already there)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .


# ---------------------------------------------------------------- 2. indexer
# --platform=$BUILDPLATFORM: this stage runs on the machine that builds the image
# (native speed), even when CI builds an arm64 image on an amd64 runner.
# Its output (index files + model weights) is data, identical on every CPU type.
FROM --platform=$BUILDPLATFORM python:3.11-slim AS indexer
ARG TORCH_VERSION
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
# Same dependencies as the builder, plus `datasets` (only needed to download SciFact)
COPY requirements.txt requirements-index.txt ./
RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt -r requirements-index.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
COPY scripts/build_index.py scripts/
# Where the models (Hugging Face cache) and the indices are written
ENV HF_HOME=/opt/hf INDEX_DIR=/opt/indices
# Download SciFact + both models and build the BM25 and FAISS indices
RUN python scripts/build_index.py


# ---------------------------------------------------------------- 3. runtime
# Fresh slim image: no compilers, no pip cache, no `datasets`, no build files
FROM python:3.11-slim AS runtime
# Create an unprivileged user: the server must not run as root
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
# Copy the ready-to-use virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
# Copy the downloaded models and the built indices from the indexer stage
COPY --from=indexer --chown=appuser:appuser /opt/hf/hub /opt/hf/hub
COPY --from=indexer --chown=appuser:appuser /opt/indices /app/data/indices
# PATH          : use the venv
# HF_HOME       : where the models are cached
# HF_HUB_OFFLINE: never call the internet, use the cached models only
# INDEX_DIR     : where the app reads the indices
# PYTHONUNBUFFERED: print logs immediately (no buffering), so `docker logs` is live
ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    INDEX_DIR=/app/data/indices \
    PYTHONUNBUFFERED=1
# From here on, everything runs as appuser
USER appuser
# Documents that the app listens on port 8000 (the real mapping is done with -p)
EXPOSE 8000
# Docker checks /health every 30s (curl is not in slim images, so we use Python).
# start-period: loading the models takes a few seconds, failures are ignored meanwhile.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"
# Start the API. 0.0.0.0 = accept connections from outside the container.
# --no-access-log: our own middleware already logs one line per request.
CMD ["uvicorn", "rerank_engine.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
