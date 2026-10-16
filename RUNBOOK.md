# RUNBOOK: commands to run, in order

Each step shows **what you should see**. If it does not match, check the "Troubleshooting" section.
Values marked `<to be measured>` must be copied into the README once measured.

---

## 0. Prerequisites (once)

```bash
brew install minikube kubectl
docker --version
```

In **Docker Desktop → Settings → Resources**: at least **4 CPUs** and **8 GB** of memory
(minikube asks for 6 GB, and each API pod reserves 1 GB).

---

## 1. Local (Python)

```bash
cd ~/Desktop/projects/Neural-Rerank-Engine
source venv/bin/activate
make install
make index
```
Expected:
```
Loaded 5183 SciFact documents
 BM25 Index built with 5183 documents
 Dense Index built with 5183 documents.
Cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 cached
Indices written to data/indices in ...s
```

```bash
make test
make lint
make run
```
`make test` → `11 passed`. `make lint` → `All checks passed!`.
`make run` → `Engine ready in ...s`, then `Uvicorn running on http://127.0.0.1:8000`. Keep this terminal open.

In a **second terminal**:
```bash
curl localhost:8000/health
curl localhost:8000/ready
curl -X POST localhost:8000/search -H 'Content-Type: application/json' \
     -d '{"query": "vitamin D deficiency and multiple sclerosis", "top_k": 3}'
curl localhost:8000/metrics | grep http_requests_total
python scripts/load_test.py -n 100 -c 1
```
Expected: `{"status":"ok"}`, `{"status":"ready"}`, then
`{"results":[{"doc_id":"...","text":"...","score":...}, ...],"latency_ms":...}`.
Also open http://localhost:8000/docs (Swagger) and try `POST /search` from the browser.
Stop the server with Ctrl+C in the first terminal.

---

## 2. Docker

```bash
make docker-build
```
- The first build is slow: it downloads torch, the corpus and the models, and builds the index.
  Duration: `<to be measured>`. Rebuilding after a code-only change is faster thanks to the layer cache.
- At the end you should see `naming to docker.io/library/neural-rerank-engine:local`.

**Image size** (copy into the README):
```bash
docker images neural-rerank-engine:local --format '{{.Repository}}:{{.Tag}}  {{.Size}}'
```
Size: `<to be measured>`.

```bash
make docker-run
```
In a second terminal:
```bash
docker ps
curl localhost:8000/ready
python scripts/load_test.py -n 100 -c 1
docker logs rerank-api | tail
```
- `docker ps`: STATUS shows `Up ... (health: starting)`, then `(healthy)` after ~30 s.
- Latency in Docker: `<to be measured>`.
- `docker logs`: one line per request, e.g. `POST /search 200 ...ms`.

Optional "no internet" test: turn off Wi-Fi and run `make docker-run` again. `/ready` must still answer,
because everything is inside the image.

Stop with Ctrl+C.

---

## 3. Docker Compose (API + Prometheus + Grafana)

```bash
make compose-up
```
or, with Grafana:
```bash
docker compose --profile grafana up --build
```
1. Send some traffic: `python scripts/load_test.py -n 50 -c 2`
2. Open http://localhost:9090/targets: the `rerank-api` target (`api:8000`) must be **UP**.
3. Open http://localhost:9090/graph, type `sum(rate(http_requests_total{handler="/search"}[1m]))`, click Execute: you should see a curve.
4. Grafana: http://localhost:3000 (admin / admin) → Explore. The "Prometheus" data source is already configured.

```bash
make compose-down
```

---

## 4. GitHub: CI/CD

After `git push`:

1. **Actions**: https://github.com/ALI-AL-MARJANI/Neural-Rerank-Engine/actions
   - The `test` job must be green ✅ (lint + 11 tests).
   - The `docker` job then starts (on `main` only). It takes much longer because it also builds
     the arm64 image through emulation (QEMU). Duration: `<to be measured>`.
2. **Package**: https://github.com/ALI-AL-MARJANI?tab=packages
   The `neural-rerank-engine` image must appear with the tags `latest` and `sha-xxxxxxx`.
3. **Make the image public** (required for minikube to pull it without credentials):
   package → **Package settings → Danger Zone → Change visibility → Public**.
   New packages are private by default, and switching to public cannot be undone.
4. The CI badge at the top of the README must be green.

Check from the Mac:
```bash
docker pull ghcr.io/ali-al-marjani/neural-rerank-engine:latest
docker manifest inspect ghcr.io/ali-al-marjani/neural-rerank-engine:latest | grep architecture
```
Expected: `"architecture": "amd64"` and `"architecture": "arm64"`.

---

## 5. Kubernetes (minikube)

```bash
make k8s-deploy
```
Expected: `Done! kubectl is now configured to use "minikube"`, then `The 'metrics-server' addon is enabled`,
then one `... created` line per object (`configmap/rerank-config`, `deployment.apps/rerank-api`,
`horizontalpodautoscaler...`, `service/rerank-api`, `prometheus...`).

```bash
kubectl get pods -w
```
Expected, in order: `ContainerCreating` (pulling the image, several minutes the first time), then `Running`
with `READY 0/1`, then `READY 1/1` once `/ready` answers. Quit the watch mode with Ctrl+C.
```
NAME                          READY   STATUS    RESTARTS   AGE
prometheus-xxxxxxxxxx-xxxxx   1/1     Running   0          2m
rerank-api-xxxxxxxxxx-xxxxx   1/1     Running   0          2m
rerank-api-xxxxxxxxxx-yyyyy   1/1     Running   0          2m
```

```bash
make k8s-status
```
- `kubectl top pods` may print `metrics not available yet` for ~1 minute after startup. This is normal.
- Real memory of one API pod: `<to be measured>` (MEMORY column of `kubectl top pods`).

**Call the API**:
```bash
make k8s-forward
```
In a second terminal:
```bash
curl localhost:8000/ready
curl -X POST localhost:8000/search -H 'Content-Type: application/json' -d '{"query":"statins lower LDL cholesterol","top_k":3}'
```

**Prometheus in the cluster**:
```bash
kubectl port-forward service/prometheus 9090:9090
```
Open http://localhost:9090/targets: the `kubernetes-pods` job must list **the 2 rerank-api pods, UP**.

**See the HPA in action** (load sent from inside the cluster, so the Service spreads it over all pods).

Terminal 1, watch the HPA:
```bash
kubectl get hpa rerank-api -w
```
Terminal 2, generate load (the script is sent to the pod through stdin):
```bash
kubectl run loadgen -i --rm --restart=Never \
  --image=ghcr.io/ali-al-marjani/neural-rerank-engine:latest -- \
  python - --url http://rerank-api -n 600 -c 8 < scripts/load_test.py
```
Expected in terminal 1: `TARGETS` goes above `70%` (e.g. `cpu: 180%/70%`), then `REPLICAS` goes from 2 to 3, then 4.
Terminal 2 prints p50/p90 inside Kubernetes (`<to be measured>`).
After the load stops, the HPA waits **5 minutes** before scaling back down to 2 pods (default behavior, avoids flapping).

**Rolling update** (update without downtime):
```bash
kubectl rollout restart deployment/rerank-api
kubectl rollout status deployment/rerank-api
```
Expected: `deployment "rerank-api" successfully rolled out`.

**Cleanup**:
```bash
make k8s-delete
minikube stop
```
(`minikube delete` erases the cluster completely.)

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| :--- | :--- | :--- |
| `ImagePullBackOff` | ghcr.io package still private, or CD not finished | make the package public (step 4.3), check the Actions tab |
| `ErrImagePull ... no matching manifest for linux/arm64` | image built for amd64 only | check `platforms:` in ci.yml and re-run the workflow |
| Pod `OOMKilled` (`kubectl describe pod` → `Last State: Terminated, Reason: OOMKilled`) | too many searches at the same time in one pod (each rerank allocates a lot of temporary memory), or memory limit too low | keep `MAX_CONCURRENT_SEARCHES: "1"` in configmap.yaml; otherwise raise `limits.memory` in deployment.yaml |
| Pod `Pending` | not enough CPU/memory in minikube | `minikube stop && minikube start --cpus=4 --memory=6g`, or lower `requests` |
| `READY 0/1` for a long time | models still loading | `kubectl logs deploy/rerank-api` and wait for `Engine ready` |
| HPA shows `cpu: <unknown>/70%` | metrics-server not ready | `minikube addons enable metrics-server` and wait ~1 min |

**Without ghcr.io** (to test minikube before the CD has run):
```bash
make docker-build
minikube image load neural-rerank-engine:local
kubectl set image deployment/rerank-api api=neural-rerank-engine:local
kubectl patch deployment rerank-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","imagePullPolicy":"IfNotPresent"}]}}}}'
```
