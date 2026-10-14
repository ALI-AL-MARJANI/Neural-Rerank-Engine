# Shortcuts for the common commands. Run `make help` to list them.
# (Activate the venv first for the local targets: source venv/bin/activate)

IMAGE      ?= ghcr.io/ali-al-marjani/neural-rerank-engine
LOCAL_TAG  ?= neural-rerank-engine:local
PYTHON     ?= python

.PHONY: help install index run test lint load-test docker-build docker-run compose-up compose-down \
	    k8s-deploy k8s-status k8s-delete k8s-forward

help:            ## List the available targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

# ---------------------------------------------------------------- local
install:         ## Install CPU torch, the dependencies and the package (editable)
	$(PYTHON) -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements.txt -r requirements-index.txt -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

index:           ## Download SciFact + models, build the indices in data/indices
	$(PYTHON) scripts/build_index.py

run:             ## Start the API locally on http://localhost:8000 (docs: /docs)
	uvicorn rerank_engine.api:app --port 8000 --no-access-log

test:            ## Run the unit tests
	$(PYTHON) -m pytest -v

lint:            ## Lint the code with ruff
	ruff check .

load-test:       ## Send 200 requests (4 in parallel) to localhost:8000, print p50/p90
	$(PYTHON) scripts/load_test.py --url http://localhost:8000 -n 200 -c 4

# ---------------------------------------------------------------- docker
docker-build:    ## Build the Docker image (downloads models + builds the index inside)
	docker build -t $(LOCAL_TAG) .

docker-run:      ## Run the image on http://localhost:8000
	docker run --rm -p 8000:8000 --name rerank-api $(LOCAL_TAG)

compose-up:      ## API + Prometheus (http://localhost:9090) with docker compose
	docker compose up --build

compose-down:    ## Stop and remove the docker compose containers
	docker compose down

# ---------------------------------------------------------------- kubernetes (minikube)
k8s-deploy:      ## Start minikube, enable metrics-server, apply all manifests in k8s/
	minikube start --driver=docker --cpus=4 --memory=6g
	minikube addons enable metrics-server
	kubectl apply -f k8s/

k8s-status:      ## Show pods, service, HPA and pod CPU/memory
	kubectl get pods,svc,hpa -o wide
	-kubectl top pods

k8s-forward:     ## Make the Service reachable on http://localhost:8000
	kubectl port-forward service/rerank-api 8000:80

k8s-delete:      ## Delete everything created from k8s/ (minikube keeps running)
	kubectl delete -f k8s/
