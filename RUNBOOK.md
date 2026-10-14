# RUNBOOK : les commandes à lancer, dans l'ordre

Chaque étape indique **ce que tu dois voir**. Si ce n'est pas le cas, regarde la section « Dépannage ».
Les valeurs marquées `<à mesurer>` sont à reporter dans le README une fois mesurées.

---

## 0. Prérequis (une seule fois)

```bash
brew install minikube kubectl        # outils Kubernetes
docker --version                     # Docker Desktop doit être lancé
```

Dans **Docker Desktop → Settings → Resources** : au moins **4 CPUs** et **8 GB** de mémoire.
minikube va en demander 6 GB, et chaque pod de l'API en réserve 1 GB.

---

## 1. En local (Python)

```bash
cd ~/Desktop/projects/Neural-Rerank-Engine
source venv/bin/activate
make install          # déjà fait dans ce venv, à refaire sur une nouvelle machine
make index            # télécharge SciFact + les 2 modèles, construit data/indices
```
Tu dois voir :
```
Loaded 5183 SciFact documents
 BM25 Index built with 5183 documents
 Dense Index built with 5183 documents.
Cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 cached
Indices written to data/indices in ...s
```

```bash
make test             # 11 passed
make lint             # All checks passed!
make run              # laisse tourner ce terminal
```
Tu dois voir `Engine ready in ...s`, puis `Uvicorn running on http://127.0.0.1:8000`.

Dans un **2e terminal** :
```bash
curl localhost:8000/health        # {"status":"ok"}
curl localhost:8000/ready         # {"status":"ready"}
curl -X POST localhost:8000/search -H 'Content-Type: application/json' \
     -d '{"query": "vitamin D deficiency and multiple sclerosis", "top_k": 3}'
# -> {"results":[{"doc_id":"...","text":"...","score":...}, ...],"latency_ms":...}
curl localhost:8000/metrics | grep http_requests_total
python scripts/load_test.py -n 100 -c 1
```
Ouvre aussi http://localhost:8000/docs (Swagger) et essaie `POST /search` depuis le navigateur.
Ensuite, Ctrl+C dans le 1er terminal.

---

## 2. Docker

```bash
make docker-build
```
- Le premier build est long : téléchargement de torch, du corpus et des modèles, puis construction de l'index.
  Durée : `<à mesurer>`. Un rebuild après une simple modification du code est plus rapide grâce au cache des layers.
- À la fin, tu dois voir `naming to docker.io/library/neural-rerank-engine:local`.

**Taille de l'image** (à reporter dans le README) :
```bash
docker images neural-rerank-engine:local --format '{{.Repository}}:{{.Tag}}  {{.Size}}'
```
Taille : `<à mesurer>`.

```bash
make docker-run
```
Dans un 2e terminal :
```bash
docker ps                          # STATUS : "Up ... (health: starting)" puis "(healthy)" après ~30 s
curl localhost:8000/ready          # {"status":"ready"}
python scripts/load_test.py -n 100 -c 1     # latence dans Docker : <à mesurer>
docker logs rerank-api | tail      # une ligne par requête : "POST /search 200 ...ms"
```
Test « sans internet » (optionnel) : coupe le Wi-Fi, relance `make docker-run`. `/ready` doit répondre, parce que tout est dans l'image.

Ctrl+C pour arrêter.

---

## 3. Docker Compose (API + Prometheus + Grafana)

```bash
make compose-up                                # API + Prometheus
# ou, avec Grafana :
docker compose --profile grafana up --build
```
1. Envoie un peu de trafic : `python scripts/load_test.py -n 50 -c 2`
2. Ouvre http://localhost:9090/targets : la cible `rerank-api` (`api:8000`) doit être **UP**.
3. Ouvre http://localhost:9090/graph et tape :
   `sum(rate(http_requests_total{handler="/search"}[1m]))`, puis clique « Execute ». Tu dois voir une courbe.
4. Pour Grafana : http://localhost:3000 (admin / admin), puis Explore. La source « Prometheus » est déjà configurée.

```bash
make compose-down
```

---

## 4. GitHub : CI/CD

Après le `git push` (les commandes de commit sont dans le résumé donné par Claude) :

1. **Actions** : https://github.com/ALI-AL-MARJANI/Neural-Rerank-Engine/actions
   - Le job `test` doit devenir vert ✅ (lint + 11 tests).
   - Le job `docker` démarre ensuite, seulement sur `main`. Il est beaucoup plus long, car il construit
     aussi l'image arm64 par émulation (QEMU). Durée : `<à mesurer>`.
2. **Package** : https://github.com/ALI-AL-MARJANI?tab=packages
   L'image `neural-rerank-engine` doit apparaître, avec les tags `latest` et `sha-xxxxxxx`.
3. **Rendre l'image publique** (obligatoire pour que minikube puisse la télécharger sans mot de passe) :
   ouvre le package, puis **Package settings → Danger Zone → Change visibility → Public**.
   Une nouvelle image est privée par défaut, et le passage en public est définitif.
4. Le badge CI en haut du README doit être vert.

Vérification depuis le Mac :
```bash
docker pull ghcr.io/ali-al-marjani/neural-rerank-engine:latest
docker manifest inspect ghcr.io/ali-al-marjani/neural-rerank-engine:latest | grep architecture
# -> "architecture": "amd64"  et  "architecture": "arm64"
```

---

## 5. Kubernetes (minikube)

```bash
make k8s-deploy
```
Tu dois voir `Done! kubectl is now configured to use "minikube"`, puis
`The 'metrics-server' addon is enabled`, puis une ligne `... created` par objet
(`configmap/rerank-config`, `deployment.apps/rerank-api`, `horizontalpodautoscaler...`, `service/rerank-api`, `prometheus...`).

```bash
kubectl get pods -w       # Ctrl+C pour quitter le mode "watch"
```
Attendu, dans l'ordre : `ContainerCreating` (téléchargement de l'image, plusieurs minutes la 1re fois),
puis `Running` avec `READY 0/1`, puis `READY 1/1` une fois que `/ready` répond.
```
NAME                          READY   STATUS    RESTARTS   AGE
prometheus-xxxxxxxxxx-xxxxx   1/1     Running   0          2m
rerank-api-xxxxxxxxxx-xxxxx   1/1     Running   0          2m
rerank-api-xxxxxxxxxx-yyyyy   1/1     Running   0          2m
```

```bash
make k8s-status           # pods, service, HPA, et CPU/mémoire de chaque pod
```
- `kubectl top pods` peut afficher `metrics not available yet` pendant ~1 minute après le démarrage. C'est normal.
- Mémoire réelle d'un pod API : `<à mesurer>` (colonne MEMORY de `kubectl top pods`).

**Tester l'API** :
```bash
make k8s-forward          # laisse tourner
# 2e terminal :
curl localhost:8000/ready
curl -X POST localhost:8000/search -H 'Content-Type: application/json' -d '{"query":"statins lower LDL cholesterol","top_k":3}'
```

**Prometheus dans le cluster** :
```bash
kubectl port-forward service/prometheus 9090:9090
```
Ouvre http://localhost:9090/targets : le job `kubernetes-pods` doit lister **les 2 pods rerank-api, UP**.

**Voir l'HPA en action** (charge envoyée depuis l'intérieur du cluster, répartie par le Service sur tous les pods) :
```bash
# terminal 1 : surveiller l'HPA
kubectl get hpa rerank-api -w

# terminal 2 : générer de la charge (le script est envoyé au pod via stdin)
kubectl run loadgen -i --rm --restart=Never \
  --image=ghcr.io/ali-al-marjani/neural-rerank-engine:latest -- \
  python - --url http://rerank-api -n 600 -c 8 < scripts/load_test.py
```
Attendu dans le terminal 1 : la colonne `TARGETS` dépasse `70%` (par exemple `cpu: 180%/70%`), puis `REPLICAS` passe de 2 à 3, puis 4.
Le terminal 2 affiche p50/p90 dans Kubernetes (`<à mesurer>`).
Une fois la charge terminée, l'HPA attend **5 minutes** avant de redescendre à 2 pods. C'est le comportement par défaut, pour éviter les oscillations.

**Rolling update** (mise à jour sans coupure) :
```bash
kubectl rollout restart deployment/rerank-api
kubectl rollout status deployment/rerank-api     # "successfully rolled out"
```

**Nettoyage** :
```bash
make k8s-delete           # supprime les objets
minikube stop             # arrête la VM (minikube delete pour tout effacer)
```

---

## Dépannage

| Symptôme | Cause probable | Solution |
| :--- | :--- | :--- |
| `ImagePullBackOff` | package ghcr.io encore privé, ou CD pas encore terminé | rendre le package public (étape 4.3), vérifier l'onglet Actions |
| `ErrImagePull ... no matching manifest for linux/arm64` | image construite seulement pour amd64 | vérifier `platforms:` dans ci.yml et relancer le workflow |
| Pod `OOMKilled` | limite mémoire trop basse | augmenter `limits.memory` dans deployment.yaml |
| Pod `Pending` | pas assez de CPU/mémoire dans minikube | `minikube stop && minikube start --cpus=4 --memory=6g`, ou baisser `requests` |
| `READY 0/1` longtemps | modèles en cours de chargement | `kubectl logs deploy/rerank-api` et attendre `Engine ready` |
| HPA `cpu: <unknown>/70%` | metrics-server pas prêt | `minikube addons enable metrics-server` et attendre ~1 min |

**Alternative sans ghcr.io** (si tu veux tester minikube avant que le CD ait tourné) :
```bash
make docker-build
minikube image load neural-rerank-engine:local
kubectl set image deployment/rerank-api api=neural-rerank-engine:local
kubectl patch deployment rerank-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","imagePullPolicy":"IfNotPresent"}]}}}}'
```
