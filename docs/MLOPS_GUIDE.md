# Guide MLOps du Neural Rerank Engine (pour débutant)

Ce guide explique **chaque fichier** ajouté pour transformer le moteur de recherche en service
« comme en production ». Pour chaque fichier : à quoi il sert, ce que fait chaque ligne, et pourquoi
ce choix plutôt qu'un autre. Les fichiers de config sont aussi commentés ligne par ligne directement
dans le code. Lis le guide avec le fichier ouvert à côté.

Sommaire :
1. [Vue d'ensemble](#1-vue-densemble)
2. [Glossaire](#2-glossaire)
3. [Le code Python : config.py, engine.py, api.py](#3-le-code-python)
4. [Dockerfile et .dockerignore](#4-dockerfile)
5. [docker-compose.yml et monitoring/](#5-docker-compose)
6. [CI/CD : .github/workflows/ci.yml](#6-cicd-github-actions)
7. [Kubernetes : k8s/](#7-kubernetes)
8. [Makefile et load_test.py](#8-makefile-et-load_testpy)
9. [Le trajet d'une requête](#9-le-trajet-dune-requête)
10. [Ce qui se passe après un `git push`](#10-ce-qui-se-passe-après-un-git-push)
11. [25 questions d'entretien](#11-25-questions-dentretien)
12. [Limites honnêtes](#limites)

---

## 1. Vue d'ensemble

```
             code Python (src/rerank_engine)
                       │
       pytest + ruff ──┤  (CI : GitHub Actions)
                       ▼
        Dockerfile ──► image Docker ──► ghcr.io  (CD : GitHub Actions)
                                            │
                                            ▼  kubectl apply -f k8s/
                         minikube : Deployment (2-4 pods) + Service + HPA
                                            │
                                 Prometheus lit /metrics
```

| Fichier | Rôle en une phrase |
| :--- | :--- |
| `src/rerank_engine/config.py` | lit les variables d'environnement (chemins, modèles, TOP_K) |
| `src/rerank_engine/engine.py` | le pipeline : retrieval hybride puis reranking |
| `src/rerank_engine/api.py` | l'API HTTP FastAPI |
| `src/rerank_engine/retrievers/fusion.py` | la fusion RRF, isolée pour être testable |
| `scripts/build_index.py` | télécharge SciFact et les modèles, construit les index |
| `scripts/load_test.py` | mesure la latence (p50/p90/p99) |
| `tests/` | tests unitaires rapides, avec un faux moteur |
| `Dockerfile`, `.dockerignore` | recette de l'image |
| `docker-compose.yml`, `monitoring/` | API + Prometheus + Grafana en local |
| `.github/workflows/ci.yml` | CI (lint + tests) et CD (build + push de l'image) |
| `k8s/*.yaml` | déploiement sur Kubernetes |
| `Makefile` | raccourcis pour toutes les commandes |
| `RUNBOOK.md` | les commandes à lancer, avec le résultat attendu |

**Ce qui a changé dans le code ML** (rien dans la logique) :
- `src/` est devenu un vrai package installable, `src/rerank_engine/`, avec `pyproject.toml`. Avant,
  les scripts faisaient `sys.path.append(...)` et `from src...`.
- `HybridRetriever` accepte maintenant des retrievers déjà configurés (chemins et modèles venant de l'environnement).
  La fusion RRF a été déplacée telle quelle dans `fusion.py`, avec la même formule `1/(rank+60)`.
- **Un bug corrigé** : le reranker écrivait `doc['rerank_score']` *dans* les documents du corpus en mémoire.
  Dans un serveur, ce corpus est partagé par toutes les requêtes simultanées, donc le score d'une requête
  pouvait « fuiter » dans une autre. Il travaille maintenant sur des copies.
- Les anciens scripts MS MARCO (`benchmark.py`, `generate_ground_truth.py`, `data_loader.py`,
  `build_indices.py`, `test_pipeline.py`) ont été supprimés (voir la [section Limites](#limites) pour le benchmark).

---

## 2. Glossaire

| Terme | Définition simple |
| :--- | :--- |
| **Image** | Un paquet figé et en lecture seule, qui contient tout le nécessaire pour lancer l'app : système de base, Python, librairies, code, modèles. C'est la « recette cuite ». |
| **Conteneur** | Une image *en train de tourner*. Un processus isolé (fichiers, réseau, CPU/mémoire limités). On peut lancer 10 conteneurs à partir de la même image. |
| **Layer (couche)** | Chaque instruction du Dockerfile (`RUN`, `COPY`…) produit une couche. Docker met les couches en cache : si une instruction et tout ce qui la précède n'ont pas changé, elle n'est pas refaite. |
| **Registry** | Un « GitHub pour images » : un serveur qui stocke les images. Ici, ghcr.io (GitHub Container Registry). Docker Hub en est un autre. |
| **CI** (Continuous Integration) | À chaque push, on vérifie automatiquement que le code est correct (lint, tests). |
| **CD** (Continuous Delivery/Deployment) | Après la CI, on produit automatiquement l'artefact livrable (ici l'image) et on le publie. *Delivery* = prêt à déployer, *Deployment* = déployé automatiquement. Ici, c'est de la **delivery** : le `kubectl apply` reste manuel. |
| **Pod** | La plus petite unité de Kubernetes : un ou plusieurs conteneurs qui partagent la même IP. Ici, 1 pod = 1 conteneur de l'API. |
| **Deployment** | Objet Kubernetes qui garantit que *N pods identiques* tournent, les recrée s'ils meurent et gère les mises à jour progressives. |
| **Replica** | Une copie d'un pod. `replicas: 2` = 2 pods identiques. |
| **Service** | Une adresse stable (nom DNS + IP) devant un groupe de pods, qui répartit le trafic entre eux. Les pods changent d'IP à chaque redémarrage ; le Service, non. |
| **Probe liveness** | « Le processus est-il vivant ? » Si elle échoue plusieurs fois, Kubernetes **redémarre** le conteneur. |
| **Probe readiness** | « Le pod est-il prêt à recevoir du trafic ? » Si elle échoue, le pod est **retiré du Service** (il ne reçoit plus de requêtes) mais il n'est pas redémarré. |
| **ConfigMap** | Objet Kubernetes qui stocke de la configuration non secrète (clé/valeur), injectée dans les pods comme variables d'environnement ou fichiers. Pour les mots de passe, on utilise un **Secret**. |
| **HPA** (HorizontalPodAutoscaler) | Ajoute ou retire des pods automatiquement selon une métrique (ici le CPU moyen). « Horizontal » = plus de pods ; « vertical » = des pods plus gros. |
| **metrics-server** | Petit composant qui collecte le CPU/mémoire des pods. L'HPA et `kubectl top` en ont besoin. |
| **Prometheus** | Base de données de séries temporelles qui va **chercher** les métriques des applications à intervalle régulier et permet de les interroger (langage PromQL). |
| **Scraping** | L'action de Prometheus qui fait un `GET /metrics` sur chaque cible toutes les X secondes (modèle « pull » : c'est Prometheus qui tire, pas l'app qui pousse). |
| **Histogramme** | Métrique qui compte combien de requêtes tombent dans chaque « seau » de durée (< 0,1 s, < 0,5 s, < 1 s…). On en déduit des percentiles. |
| **p50 / p90 / p99** | Percentiles de latence. p50 (médiane) : 50 % des requêtes sont plus rapides que cette valeur. p90 : 90 % sont plus rapides, donc 10 % plus lentes. On regarde p90/p99 parce que la moyenne cache les requêtes lentes que les utilisateurs ressentent. |
| **Rolling update** | Mise à jour progressive : on crée un pod avec la nouvelle version, on attend qu'il soit *ready*, puis on supprime un ancien, et ainsi de suite. Zéro coupure. |
| **Lifespan** | Dans FastAPI, le code exécuté une seule fois au démarrage et à l'arrêt du serveur. |

---

## 3. Le code Python

### `config.py`
- **À quoi il sert** : centraliser les réglages lus dans l'environnement : `INDEX_DIR`, `DENSE_MODEL`, `RERANK_MODEL`, `TOP_K`.
- **Lignes clés** :
  - `@dataclass(frozen=True) class Settings` : un objet simple, non modifiable, qui contient les réglages.
  - `bm25_path` / `dense_dir` : chemins dérivés de `INDEX_DIR` (`bm25.pkl` et `dense/`).
  - `get_settings()` : lit `os.getenv("NOM", "défaut")` **à chaque appel**. Ça permet de changer l'environnement dans un test.
- **Pourquoi** : c'est le principe « 12-factor app » : la même image tourne partout (local, Docker, k8s), seule la
  config change. Je n'ai pas utilisé `pydantic-settings` pour ne pas ajouter de dépendance à 4 variables.
- **TOP_K** = nombre de **candidats** de l'étape 1 envoyés au reranker (50 par défaut). À ne pas confondre avec
  `top_k` de la requête, qui est le nombre de **résultats renvoyés** (10 par défaut).

### `engine.py`
- **À quoi il sert** : relier les briques ML existantes en un pipeline.
- `SearchEngine.search(query, top_k, rerank)` :
  1. `self.retriever.search(query, k=max(candidates_k, top_k))` : BM25 + FAISS + RRF.
  2. si `rerank=True` : le cross-encoder rescore les candidats et on garde `top_k`, avec score = `rerank_score` ;
     sinon on garde les `top_k` premiers avec score = `rrf_score`.
  3. renvoie `{doc_id, text, score}`.
- `load_engine(settings)` : charge les 2 modèles et les 2 index. **Les imports lourds (torch, faiss) sont dans la
  fonction** : importer l'API reste rapide, et les tests n'ont jamais à charger torch.
- **Pourquoi séparer engine et API** : l'API ne connaît qu'un objet avec une méthode `search()`. Dans les tests, on
  le remplace par un faux (`FakeEngine`). C'est de l'*injection de dépendance*.

### `api.py`
Ligne par ligne (dans l'ordre du fichier) :
- `logging.basicConfig(...)` : format des logs `date niveau module message`, niveau INFO.
- `logging.getLogger("sentence_transformers").setLevel(logging.WARNING)` : sinon la librairie affiche une barre de
  progression à chaque requête.
- `class SearchRequest(BaseModel)` : schéma du body JSON, validé automatiquement par **Pydantic** :
  - `str_strip_whitespace=True` : retire les espaces au début et à la fin, donc `"   "` devient `""`, ce qui est refusé.
  - `query: str = Field(..., min_length=1, max_length=1000)` : `...` = obligatoire ; vide → **422**.
  - `top_k: int = Field(10, ge=1, le=50)` : entre 1 et 50 (sinon 422) ; on borne pour éviter qu'un client demande 10 000 résultats.
  - `rerank: bool = True`.
- `SearchResult`, `SearchResponse` : schéma de la réponse (`response_model`). FastAPI vérifie et documente la sortie.
- `create_app(engine_loader=load_engine)` : une **fabrique** d'application. En prod, elle charge le vrai moteur ;
  dans les tests, on lui passe `lambda settings: FakeEngine()`.
- `lifespan` (avec `@asynccontextmanager`) : le code **avant `yield`** s'exécute **une fois au démarrage** (chargement
  des modèles, ~11 s mesurées sur M4), le code **après `yield`** une fois à l'arrêt. Uvicorn n'accepte aucune connexion
  avant la fin du démarrage.
  - Pourquoi pas `@app.on_event("startup")` : cette syntaxe est dépréciée, `lifespan` est la méthode actuelle recommandée.
- `app.state.engine` : là où on range le moteur, accessible depuis toutes les routes.
- `Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(app, include_in_schema=False)` :
  - `.instrument(app)` ajoute un middleware qui mesure chaque requête : compteur `http_requests_total{handler, method, status}`
    et histogramme `http_request_duration_seconds` ;
  - `.expose(app)` crée la route `GET /metrics` au format texte Prometheus ;
  - on exclut `/metrics` lui-même pour ne pas compter le scraping comme du trafic.
  - **Erreurs** : elles apparaissent via le label `status="4xx"` / `"5xx"`.
- `@app.middleware("http") log_requests` : chronomètre chaque requête et écrit **une ligne de log** :
  `POST /search 200 489.3ms`. On lance uvicorn avec `--no-access-log` pour ne pas avoir deux lignes par requête.
- `GET /health` (`async def`) : renvoie toujours `{"status": "ok"}` si le processus répond.
- `GET /ready` (`async def`) : 200 si `app.state.engine` existe, **503** sinon.
  - Pourquoi `async def` pour ces deux-là : elles tournent directement sur la boucle d'événements, donc elles répondent
    même quand tous les threads sont occupés par des recherches. Sinon la liveness probe pourrait échouer sous charge
    et Kubernetes tuerait un pod qui n'est pas malade, juste occupé.
- `POST /search` (`def`, **pas** `async def`) : l'inférence torch est du calcul CPU **bloquant**. Avec un `def`
  normal, FastAPI l'exécute dans un pool de threads et le serveur continue de répondre aux autres requêtes. Dans un
  `async def`, ce calcul bloquerait toute la boucle.
- `app = create_app()` : l'objet que lance `uvicorn rerank_engine.api:app`.

### `fusion.py` (RRF)
`score(doc) = Σ sur chaque liste de 1 / (rang + 60)`. Un document bien classé par BM25 **et** par FAISS cumule
deux contributions. RRF n'utilise que les **rangs**, pas les scores bruts : c'est pratique, parce que les scores
BM25 (0 à 30+) et les distances FAISS ne sont pas comparables. Le rang commence à 0 dans ce code (le code
d'origine a été gardé tel quel). L'article original utilise un rang qui commence à 1, mais l'effet est minime.

### `tests/`
- `conftest.py` : `FakeEngine` (renvoie des résultats fixes instantanément) + une fixture `client` = `TestClient`
  utilisé dans un `with`, ce qui **exécute le lifespan** comme un vrai serveur.
- `test_api.py` : `/health` 200, `/ready` 200 (et **503** quand le lifespan n'a pas tourné), schéma de `/search`,
  `top_k` par défaut, **422** pour une query vide, une query faite d'espaces, une query absente et un `top_k` hors bornes,
  et `/metrics` contient bien les métriques attendues.
- `test_fusion.py` : RRF sur un exemple jouet calculé à la main, plus la vérification que les entrées ne sont pas modifiées.
- **Pourquoi des fakes** : les tests tournent en 0,04 s, sans télécharger 200 Mo de modèles ni le corpus. La CI
  reste rapide et déterministe. La partie ML réelle est vérifiée par `build_index.py` au build Docker et par `/ready`.

### `scripts/build_index.py`
Télécharge le corpus **BEIR SciFact** (5 183 résumés scientifiques, public, licence CC BY-SA 4.0) via
`datasets.load_dataset("BeIR/scifact", "corpus")`, met `titre + texte` dans `text`, construit l'index BM25
(`bm25.pkl`) et l'index FAISS (`dense/index.faiss` + `metadata.pkl`), puis instancie le cross-encoder une fois
pour qu'il soit dans le cache Hugging Face. Mesuré : 39 s sur M4 (téléchargements compris).

---

## 4. Dockerfile

**À quoi il sert** : la recette pour construire l'image qui contient Python, les librairies, le code, **les modèles et
les index**. Le conteneur démarre donc sans internet.

**Les 3 étapes (multi-stage)** :

| Étape | Base | Fait quoi | Finit dans l'image ? |
| :--- | :--- | :--- | :--- |
| `builder` | python:3.11-slim | crée `/opt/venv`, installe torch CPU + `requirements.txt` + le package | seulement `/opt/venv` |
| `indexer` | python:3.11-slim (plateforme de build) | installe aussi `datasets`, lance `build_index.py` | seulement les modèles et les index |
| `runtime` | python:3.11-slim | copie le venv, les modèles et les index ; utilisateur non-root ; lance uvicorn | **oui, c'est l'image finale** |

Ligne par ligne (les points importants) :
- `# syntax=docker/dockerfile:1` : utilise la syntaxe Dockerfile récente (BuildKit).
- `ARG TORCH_VERSION=2.8.0` : variable de build, déclarée une fois en haut, redéclarée dans chaque étape qui l'utilise.
- `FROM python:3.11-slim AS builder` : image officielle Debian minimale avec Python 3.11. `slim` pèse plusieurs fois moins
  que l'image complète (pas de compilateurs ni d'outils inutiles). Pourquoi pas `alpine` : Alpine utilise musl au lieu
  de glibc, et beaucoup de wheels Python (torch notamment) ne sont pas disponibles ou doivent être compilés.
- `ENV PIP_NO_CACHE_DIR=1` : pip ne garde pas les fichiers téléchargés, ce qui donne des couches plus légères.
- `RUN python -m venv /opt/venv` + `ENV PATH="/opt/venv/bin:$PATH"` : tout s'installe dans un seul dossier, facile
  à copier tel quel dans l'étape finale.
- `COPY requirements.txt .` **avant** `COPY src` : c'est l'astuce du **cache de layers**. Les dépendances changent
  rarement, le code souvent. Si seul le code change, Docker réutilise la couche `pip install` (plusieurs minutes économisées).
- `pip install torch==... --index-url https://download.pytorch.org/whl/cpu` : la version **CPU** de PyTorch. Le torch
  par défaut de PyPI sur Linux x86_64 embarque CUDA (plusieurs Go de librairies NVIDIA), inutile sans GPU. On
  installe torch **seul** avec `--index-url`, puis le reste depuis PyPI normal.
- `pip install --no-deps .` : installe notre package `rerank_engine` (les dépendances sont déjà là).
- `FROM --platform=$BUILDPLATFORM python:3.11-slim AS indexer` : cette étape tourne sur **l'architecture de la machine
  qui construit** (amd64 sur GitHub). Quand la CI construit l'image arm64 (pour ton Mac), seules les étapes `builder`
  et `runtime` passent par l'émulation QEMU, qui est lente. Calculer les embeddings de 5 183 documents sous émulation
  prendrait beaucoup plus de temps. Le résultat (fichiers d'index, poids des modèles) est de la **donnée**, identique
  quel que soit le CPU.
- `ENV HF_HOME=/opt/hf INDEX_DIR=/opt/indices` + `RUN python scripts/build_index.py` : téléchargement et indexation **au build**.
- `FROM python:3.11-slim AS runtime` : on repart d'une image propre. Rien de l'étape `indexer` (la librairie
  `datasets`, le cache du corpus) ni du cache pip n'y entre.
- `RUN useradd --create-home --uid 1000 appuser` + `USER appuser` : le serveur ne tourne **pas en root**. Si
  quelqu'un exploite une faille de l'app, il n'a pas les droits root dans le conteneur. Beaucoup de clusters
  refusent d'ailleurs les conteneurs root.
- `COPY --from=builder /opt/venv /opt/venv` : on récupère seulement le résultat de l'étape précédente.
- `COPY --from=indexer --chown=appuser:appuser ...` : les modèles et les index, avec appuser comme propriétaire.
- `ENV HF_HUB_OFFLINE=1` : interdit à Hugging Face d'aller sur internet, il utilise le cache. Si un modèle manque,
  l'app plante au démarrage au lieu de télécharger en silence en production.
- `ENV PYTHONUNBUFFERED=1` : les logs s'affichent immédiatement dans `docker logs` / `kubectl logs`.
- `EXPOSE 8000` : **documentation** : l'app écoute sur 8000. Ça n'ouvre rien ; c'est `-p 8000:8000` qui publie le port.
- `HEALTHCHECK ... CMD python -c "urllib..."` : Docker appelle `/health` toutes les 30 s, et `docker ps` affiche
  `(healthy)`. On utilise Python parce que `curl` n'est pas installé dans `slim`. `--start-period=60s` laisse le
  temps de charger les modèles. **Note** : Kubernetes **ignore** le HEALTHCHECK Docker et utilise ses propres probes.
- `CMD ["uvicorn", "rerank_engine.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]` :
  forme JSON (« exec form ») : uvicorn est le processus principal (PID 1) et reçoit directement le signal d'arrêt
  (SIGTERM) de Docker/Kubernetes, ce qui permet un arrêt propre. `0.0.0.0` = accepter les connexions venant
  de l'extérieur du conteneur (avec `127.0.0.1`, seul le conteneur lui-même pourrait se parler).

**Index au build vs volume monté** (choix à savoir défendre) :

| | Index et modèles **dans l'image** (choix fait ici) | Index et modèles sur un **volume monté** |
| :--- | :--- | :--- |
| Démarrage | rien à télécharger, fonctionne hors ligne | dépend du volume ou du stockage (S3, PVC) |
| Reproductibilité | image = code + modèle + données, versionnés ensemble par le tag | il faut versionner les données à part |
| Taille de l'image | plus grosse (~200 Mo de modèles + 34 Mo d'index) | plus petite |
| Mise à jour des données | rebuild de l'image | on remplace les fichiers, sans rebuild |
| Adapté à | petits modèles et corpus (notre cas) | gros modèles (plusieurs Go), corpus qui change souvent |

En production avec un gros corpus, on utiliserait un volume : un PersistentVolumeClaim en Kubernetes, ou un
initContainer qui télécharge depuis un stockage objet (S3/GCS).

**`.dockerignore`** : liste ce qui n'est **pas envoyé** à Docker au build : `venv/` (des centaines de Mo, et pour macOS
alors que l'image est Linux), `data/` (reconstruit dans l'image), `.git/`, notebooks, tests, docs, k8s… Le build est
plus rapide, l'image plus petite, et on évite d'y copier un `.env` avec des secrets.

**Taille de l'image** : `<à mesurer>` avec `docker images neural-rerank-engine:local` (voir RUNBOOK).

---

## 5. Docker Compose

**À quoi il sert** : lancer **plusieurs conteneurs ensemble** avec une seule commande (`docker compose up`), sur un
réseau commun où chaque service est joignable par son nom (`api`, `prometheus`).

- `api` : `build: .` construit l'image depuis le Dockerfile ; `ports: "8000:8000"` = port de l'hôte → port du conteneur ;
  `restart: unless-stopped` redémarre le conteneur s'il plante.
- `prometheus` : image officielle à **version fixée** (`v3.14.0`, jamais `latest`, pour que ça marche pareil dans 6 mois) ;
  `volumes` monte notre `monitoring/prometheus.yml` en lecture seule (`:ro`) ; UI sur `:9090`.
- `grafana` : `profiles: ["grafana"]` = **optionnel**, lancé seulement avec `--profile grafana`. La source de données
  Prometheus est configurée automatiquement (« provisioning ») par `monitoring/grafana/datasource.yml`.
- `depends_on` : ordre de **démarrage** (pas d'attente que l'API soit *prête*).

`monitoring/prometheus.yml` : `scrape_interval: 5s`, une cible `api:8000` sur `/metrics`. Dans Docker Compose,
`api` est résolu en IP par le DNS interne du réseau compose.

---

## 6. CI/CD (GitHub Actions)

**À quoi il sert** : à chaque push, GitHub lance une machine virtuelle Linux propre qui exécute le fichier `ci.yml`.

- `on: push: branches: ["**"], tags: ["v*"]` + `pull_request` : déclencheurs. `"**"` = toutes les branches.
- `permissions: contents: read` : par défaut, le jeton du workflow peut seulement *lire* le code (principe du
  moindre privilège).
- `env: IMAGE_NAME: ghcr.io/ali-al-marjani/neural-rerank-engine` : un nom d'image doit être en **minuscules**,
  alors que le nom du repo GitHub contient des majuscules ; on l'écrit donc en dur.

**Job `test` (CI)** :
- `runs-on: ubuntu-latest` : VM Linux fournie gratuitement par GitHub.
- `actions/checkout@v6` : récupère le code.
- `actions/setup-python@v6` avec `python-version: "3.11"` (la même que l'image Docker) et `cache: pip` (les paquets
  téléchargés sont réutilisés d'un run à l'autre).
- Installation : torch CPU, puis `requirements.txt` + `requirements-dev.txt`, puis `pip install -e .`.
- `ruff check .` : linter ; il échoue sur une variable non définie, un import inutilisé ou des imports mal triés (règles `E`, `F`, `I`).
- `pytest -v` : les 11 tests. Si une étape échoue, le job est rouge et le CD ne démarre pas.

**Job `docker` (CD)** :
- `needs: test` : seulement si la CI est verte.
- `if: github.event_name == 'push' && (ref == main || tag v*)` : pas de publication d'image pour une pull request
  ou une branche de travail.
- `permissions: packages: write` : autorise le `GITHUB_TOKEN` à pousser sur ghcr.io. **Aucun secret à créer** :
  GitHub génère ce jeton automatiquement pour chaque run, et il expire à la fin du run.
- `docker/setup-qemu-action@v4` : émulateur qui permet à une VM amd64 de construire une image arm64.
- `docker/setup-buildx-action@v4` : active Buildx (BuildKit), le constructeur moderne de Docker
  (multi-plateformes, cache).
- `docker/login-action@v4` : `docker login ghcr.io` avec `github.actor` et `secrets.GITHUB_TOKEN`.
- `docker/metadata-action@v6` : calcule les tags :
  - `type=raw,value=latest,enable={{is_default_branch}}` : `latest`, uniquement sur `main` ;
  - `type=sha` : `sha-1a2b3c4`, un tag **immuable** par commit (on sait exactement quel code tourne) ;
  - `type=semver,pattern={{version}}` : `1.0.0` si on pousse le tag git `v1.0.0`.
- `docker/build-push-action@v7` : build + push.
  - `platforms: linux/amd64,linux/arm64` : une image « multi-arch ». amd64 pour les serveurs cloud classiques,
    arm64 pour ton Mac M4 (minikube y tourne en arm64). Docker choisit automatiquement la bonne au `pull`.
  - `cache-from/cache-to: type=gha` : les layers sont gardées dans le cache GitHub Actions, ce qui accélère les builds suivants.

Pourquoi ghcr.io plutôt que Docker Hub : c'est intégré à GitHub (même compte, même jeton, aucun secret à créer),
gratuit pour les images publiques, et le package est lié au repo.

---

## 7. Kubernetes

À appliquer avec `kubectl apply -f k8s/`. Kubernetes est **déclaratif** : on décrit *l'état voulu* (« 2 pods de
cette image »), et Kubernetes fait en permanence ce qu'il faut pour y arriver (recréer un pod mort, etc.).

Chaque YAML a la même structure : `apiVersion` (version de l'API Kubernetes), `kind` (type d'objet),
`metadata` (nom, labels), `spec` (l'état voulu).

### `configmap.yaml`
- `TOP_K`, `DENSE_MODEL`, `RERANK_MODEL` : les réglages de l'app. Les noms de modèles doivent correspondre aux
  modèles **présents dans l'image** (mode hors ligne).
- `OMP_NUM_THREADS: "2"` : nombre de threads CPU de torch = la limite CPU du pod. Sans ça, torch voit tous les
  cœurs du nœud, lance trop de threads, et Kubernetes le « throttle » (ralentit) dès qu'il dépasse sa limite.
- Pourquoi une ConfigMap : changer un réglage **sans reconstruire l'image**. Après un `kubectl apply`, il faut
  redémarrer les pods (`kubectl rollout restart`), car les variables d'environnement sont lues au démarrage.

### `deployment.yaml`
- `replicas: 2` : 2 pods, donc si un pod plante, l'autre continue de servir (haute disponibilité minimale).
- `selector.matchLabels: app: rerank-api` + `template.metadata.labels: app: rerank-api` : le Deployment
  reconnaît « ses » pods grâce à ce label. Le Service utilise le même label.
- `strategy: RollingUpdate`, `maxUnavailable: 0`, `maxSurge: 1` : pendant une mise à jour, on ajoute 1 nouveau pod,
  on attend qu'il soit **ready**, on supprime 1 ancien, et ainsi de suite. On n'a jamais moins de 2 pods prêts.
- `annotations: prometheus.io/scrape|port|path` : des étiquettes libres, que **notre Prometheus** (`k8s/prometheus.yaml`)
  lit pour savoir quels pods scraper et sur quel port/chemin. Ce n'est pas magique : sans un Prometheus configuré
  pour les lire, elles ne font rien.
- `image: ghcr.io/ali-al-marjani/neural-rerank-engine:latest` + `imagePullPolicy: Always` : comme `latest` change à
  chaque push, on demande à Kubernetes de vérifier le registry à chaque création de pod. En vraie production, on
  déploierait plutôt le tag immuable `sha-xxxxxxx`.
- `envFrom: configMapRef: rerank-config` : chaque clé de la ConfigMap devient une variable d'environnement.
- `resources` :
  - `requests` (`cpu: 500m` = un demi-cœur, `memory: 1Gi`) : ce qui est **réservé**. Le scheduler ne place le pod
    que sur un nœud qui a cette capacité libre. L'HPA calcule ses pourcentages **par rapport aux requests**.
  - `limits` (`cpu: 2`, `memory: 2Gi`) : le maximum. CPU au-delà = ralenti (throttling) ; mémoire au-delà =
    conteneur **tué** (`OOMKilled`).
  - Les valeurs : en local, le processus utilisait ~350 à 460 Mo de RAM (RSS mesuré sous macOS, Python 3.9).
    1 Gi laisse de la marge. Ces valeurs sont à ajuster avec `kubectl top pods` (`<à mesurer>`).
- `readinessProbe` sur `/ready` : toutes les 5 s, après 10 s ; 3 échecs → le pod est retiré du Service.
- `livenessProbe` sur `/health` : toutes les 10 s, après **60 s** (pour ne pas tuer un pod en train de charger
  ses modèles) ; 3 échecs → redémarrage.
- Pourquoi `/health` ne vérifie **pas** le modèle : si la liveness dépendait d'un service externe ou d'un état
  lent, un problème passager ferait redémarrer tous les pods en boucle. La liveness doit seulement dire « le
  processus n'est pas bloqué ».

### `service.yaml`
- `type: ClusterIP` : une IP **interne** au cluster, avec un nom DNS `rerank-api`. Les autres pods appellent
  `http://rerank-api`, et le Service répartit les requêtes entre les pods **ready**.
- `port: 80` → `targetPort: http` (le port nommé `http` = 8000 dans le Deployment).
- Accès depuis le Mac : `kubectl port-forward service/rerank-api 8000:80`. Note : `minikube service` a besoin d'un
  Service de type `NodePort` ou `LoadBalancer`. Avec ClusterIP, on utilise `port-forward`. **Attention** :
  `port-forward` envoie tout vers *un seul* pod (c'est un tunnel de debug, pas un load balancer). Pour tester la
  répartition et l'HPA, le RUNBOOK lance donc la charge **depuis un pod du cluster**.
- Types de Service : `ClusterIP` (interne), `NodePort` (un port ouvert sur chaque nœud), `LoadBalancer` (IP
  publique fournie par le cloud), et `Ingress` (routage HTTP par nom de domaine/chemin, devant des Services).

### `hpa.yaml`
- `apiVersion: autoscaling/v2` : version actuelle de l'API HPA.
- `scaleTargetRef` : le Deployment `rerank-api`.
- `minReplicas: 2`, `maxReplicas: 4`.
- `averageUtilization: 70` : si le CPU **moyen** des pods dépasse 70 % de leur *request* (soit 350m), l'HPA ajoute des
  pods. Formule : `pods_voulus = ceil(pods_actuels × CPU_actuel / CPU_cible)`.
- Prérequis : `minikube addons enable metrics-server`.
- Après la charge, la réduction attend 5 minutes (fenêtre de stabilisation par défaut) pour éviter le « yo-yo ».
- Pourquoi le CPU : notre service est limité par le CPU (le cross-encoder). Pour un LLM sur GPU, on scalerait
  plutôt sur la taille de la file d'attente ou le nombre de requêtes en cours (métriques personnalisées).

### `prometheus.yaml` (Prometheus dans le cluster)
Plusieurs objets séparés par `---` :
- `ServiceAccount` `prometheus` : une identité pour le pod Prometheus.
- `ClusterRole` + `ClusterRoleBinding` (RBAC) : donne à cette identité le droit de `get/list/watch` les pods.
  Prometheus interroge l'API Kubernetes pour **découvrir** les pods (« service discovery »).
- `ConfigMap prometheus-config` : la config Prometheus. Les `relabel_configs` (repris de l'exemple officiel de
  Prometheus) :
  1. `keep` seulement les pods avec `prometheus.io/scrape: "true"` ;
  2. chemin = annotation `prometheus.io/path` ;
  3. adresse = `IP du pod:` + annotation `prometheus.io/port` ;
  4. ajoute le label `pod` pour distinguer les pods dans les graphes.
- `Deployment prometheus` (1 pod) qui monte la ConfigMap comme fichier dans `/etc/prometheus`.
- `Service prometheus` pour y accéder avec `kubectl port-forward service/prometheus 9090:9090`.
- Pourquoi pas le « kube-prometheus-stack » (Helm, Prometheus Operator) : c'est le standard en entreprise, mais il
  installe des dizaines d'objets. Le but ici est de comprendre le mécanisme. Pas de Helm, comme demandé.

---

## 8. Makefile et load_test.py

**Makefile** : des alias. `make run` = `uvicorn rerank_engine.api:app --port 8000 --no-access-log`. `make help` les
liste. Dans un Makefile, les lignes de commande doivent commencer par une **tabulation**. `-kubectl top pods` :
le `-` dit à make de continuer même si la commande échoue (metrics-server pas encore prêt).
`make k8s-deploy` lance `minikube start --driver=docker --cpus=4 --memory=6g` : minikube crée un « nœud »
Kubernetes dans un conteneur Docker Desktop.

**load_test.py** :
- envoie `-n` requêtes `POST /search`, avec `-c` requêtes en parallèle (`ThreadPoolExecutor`) ;
- 1 requête d'échauffement non comptée (la première est toujours plus lente) ;
- affiche p50/p90/p99, la moyenne, le débit (req/s) et le nombre d'erreurs ;
- **seulement la librairie standard** (`urllib`), donc on peut l'envoyer dans un pod via `stdin`
  (`python - ... < scripts/load_test.py`) et charger le Service depuis l'intérieur du cluster.

**Mesures réelles** (Apple M4, 16 Go, Python 3.9 natif, hors Docker, `top_k=10`, 50 candidats rerankés) :

| Cas | n | c | p50 | p90 | p99 | débit |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybride + rerank | 100 | 1 | 489 ms | 501 ms | 541 ms | 2,0 req/s |
| hybride + rerank | 200 | 4 | 2248 ms | 2408 ms | 2568 ms | 1,8 req/s |
| hybride seul | 200 | 1 | 13 ms | 17 ms | 23 ms | 71,8 req/s |

Lecture : le reranking représente ~97 % du temps (~490 ms sur ~500). Avec 4 requêtes en parallèle, le débit ne monte
pas : un seul processus sature déjà le CPU, et les requêtes font la queue (latence ×4). Pour gagner en débit, il
faut **plus de pods** (HPA), **moins de candidats** (`TOP_K`), ou un modèle plus rapide (ONNX, quantization, GPU).
Docker/k8s : `<à mesurer>`.

---

## 9. Le trajet d'une requête

Exemple : `curl -X POST localhost:8000/search -d '{"query":"statins lower LDL"}'` sur minikube.

1. **Mac → tunnel** : `kubectl port-forward` écoute sur `localhost:8000` et transfère vers le Service `rerank-api:80`.
   Depuis un autre pod, on appellerait directement `http://rerank-api` (DNS interne).
2. **Service → pod** : le Service choisit un des pods **ready** (ceux dont `/ready` répond 200) et envoie vers son port 8000.
3. **Uvicorn** (serveur ASGI) reçoit la connexion HTTP et la passe à FastAPI.
4. **Middlewares** : l'instrumentator démarre son chronomètre, et notre middleware de log aussi.
5. **Validation Pydantic** : le JSON est converti en `SearchRequest`. Query vide ou `top_k` hors bornes → 422
   immédiatement, sans toucher au modèle.
6. **Route `search`** (dans un thread du pool) → `engine.search()` :
   1. BM25 : tokenisation `lower().split()`, score sur les 5 183 documents → top 50 ;
   2. FAISS : encodage de la query par MiniLM (vecteur de 384 dimensions), recherche exacte L2 → top 50 ;
   3. RRF : fusion des deux listes → 50 candidats ;
   4. cross-encoder : 50 paires (query, document) → 50 scores → tri → top 10.
7. **Réponse** : `{"results":[...], "latency_ms": ...}`, validée par `SearchResponse`.
8. **Métriques et logs** : `http_requests_total{handler="/search",status="2xx"}` +1, la durée va dans l'histogramme,
   et une ligne `POST /search 200 495.1ms` est écrite dans les logs (`kubectl logs`).
9. **Plus tard** : Prometheus scrape `/metrics` de ce pod toutes les 5 s. metrics-server mesure son CPU, et l'HPA
   décide s'il faut plus de pods.

---

## 10. Ce qui se passe après un `git push`

1. GitHub reçoit le commit et déclenche le workflow `CI` (`.github/workflows/ci.yml`).
2. **Job `test`** sur une VM Ubuntu neuve : checkout → Python 3.11 → install → `ruff check .` → `pytest`.
   Le résultat ✅/❌ apparaît dans l'onglet Actions, à côté du commit et sur le badge du README.
3. Si la branche est `main` (ou un tag `v*`) **et** que `test` est vert → **job `docker`** :
   QEMU + Buildx → login ghcr.io avec `GITHUB_TOKEN` → calcul des tags → build amd64 et arm64 (le Dockerfile
   télécharge SciFact et les modèles et construit l'index) → push de `latest` et `sha-xxxxxxx`.
4. L'image apparaît dans l'onglet *Packages* du profil GitHub.
5. **Le déploiement reste manuel** : `kubectl rollout restart deployment/rerank-api`, et les nouveaux pods tirent la
   nouvelle `latest` (`imagePullPolicy: Always`) via un rolling update. Pour l'automatiser, on ajouterait une étape
   `kubectl set image ... :sha-xxx` dans le workflow (avec les accès au cluster), ou un outil GitOps comme Argo CD
   qui synchronise le cluster avec le repo.

---

## 11. 25 questions d'entretien

1. **Pourquoi un Dockerfile multi-stage ?**
   Pour que l'image finale ne contienne que le nécessaire à l'exécution. Les étapes de build (cache pip,
   `datasets`, téléchargement du corpus) restent dans des étapes intermédiaires jetées à la fin. L'image est plus
   petite, plus rapide à télécharger, et sa surface d'attaque est réduite.

2. **Différence entre liveness et readiness ?**
   Liveness = « le processus est-il vivant ? ». Un échec entraîne le redémarrage du conteneur. Readiness = « peut-il
   servir du trafic ? ». Un échec retire le pod du Service, sans redémarrage. Ici `/ready` vérifie que les modèles
   sont chargés, `/health` seulement que le processus répond.

3. **Pourquoi charger le modèle au démarrage et pas à chaque requête ?**
   Charger les modèles et les index prend ~11 s (mesuré) ; une recherche, ~0,5 s. Les charger à chaque requête
   multiplierait la latence par 20 et la mémoire par le nombre de requêtes. Le `lifespan` les charge une fois, et
   toutes les requêtes les partagent.

4. **Comment tu mets à jour sans coupure ?**
   Rolling update du Deployment avec `maxUnavailable: 0` et `maxSurge: 1` : Kubernetes crée un pod de la nouvelle
   version, attend qu'il soit *ready* (readiness probe), puis supprime un ancien. On a toujours au moins 2 pods
   prêts. Retour arrière : `kubectl rollout undo deployment/rerank-api`.

5. **Pourquoi torch CPU ?**
   Pas de GPU ici. Le torch par défaut sur Linux x86_64 embarque CUDA et les librairies NVIDIA (plusieurs Go)
   inutiles. La version CPU est bien plus légère.

6. **Pourquoi un utilisateur non-root ?**
   Défense en profondeur : si l'app est compromise, l'attaquant n'est pas root dans le conteneur. Beaucoup de
   clusters (Pod Security Standards « restricted ») refusent d'ailleurs les conteneurs root.

7. **Pourquoi `python:3.11-slim` et pas `alpine` ?**
   Alpine utilise musl au lieu de glibc, et les wheels de torch/faiss sont faites pour glibc (manylinux). Sur
   Alpine, il faudrait souvent compiler, ce qui rend le build long et fragile. `slim` est Debian, minimal, compatible.

8. **Pourquoi copier `requirements.txt` avant le code ?**
   Pour le cache de layers : Docker réutilise la couche `pip install` tant que `requirements.txt` ne change pas.
   Modifier une ligne de code ne réinstalle pas torch.

9. **Index dans l'image ou volume monté ?**
   Ici, dans l'image : le corpus et les modèles sont petits, l'image est autonome, reproductible et fonctionne
   hors ligne. Avec des modèles de plusieurs Go ou un index qui change chaque jour, je les mettrais sur un volume
   (PVC) ou je les téléchargerais depuis S3/GCS dans un initContainer, pour ne pas reconstruire l'image à chaque fois.

10. **Différence CI / CD dans ton projet ?**
    CI : lint ruff + pytest à chaque push et chaque PR. CD : sur `main` et les tags `v*`, build de l'image
    multi-architecture et push sur ghcr.io. C'est du *continuous delivery* : l'artefact est prêt, mais
    l'application sur le cluster reste manuelle.

11. **Comment le workflow s'authentifie sur ghcr.io sans secret ?**
    Avec le `GITHUB_TOKEN`, un jeton créé automatiquement par GitHub pour chaque run et qui expire à la fin. On lui
    donne seulement `packages: write` dans le job qui pousse l'image (principe du moindre privilège).

12. **Pourquoi taguer avec le SHA en plus de `latest` ?**
    `latest` bouge à chaque push : on ne sait pas quelle version tourne, et on ne peut pas revenir en arrière
    proprement. `sha-1a2b3c4` est immuable et relie l'image à un commit précis. En production, je déploierais le tag SHA.

13. **Pourquoi des fakes dans les tests ?**
    Pour tester le comportement de l'API (codes HTTP, validation, schéma, métriques) en 0,04 s, sans télécharger
    200 Mo de modèles. Les tests sont rapides, déterministes, et la CI coûte peu. La partie ML réelle est validée au
    build Docker (construction de l'index) et par la readiness probe.

14. **Qu'est-ce que tu monitores, et comment ?**
    `prometheus-fastapi-instrumentator` expose `/metrics` : nombre de requêtes par route et par code (donc le taux
    d'erreurs 5xx) et un histogramme de latence. Prometheus le scrape toutes les 5 s. Requêtes utiles :
    `rate(http_requests_total[1m])`, `histogram_quantile(0.9, ...)`. Plus une ligne de log par requête.

15. **Pourquoi un histogramme plutôt qu'une moyenne de latence ?**
    La moyenne cache les requêtes lentes. Avec un histogramme, Prometheus calcule p50/p90/p99, y compris en
    agrégeant plusieurs pods, ce qui est impossible avec des percentiles calculés dans chaque pod.

16. **Comment fonctionne ton HPA ?**
    metrics-server mesure le CPU des pods. L'HPA compare la moyenne à 70 % de la *request* (350m) et ajuste :
    `ceil(pods × CPU_actuel / cible)`, entre 2 et 4 pods. La réduction attend 5 minutes pour éviter les oscillations.

17. **Pourquoi ton débit ne monte pas avec plus de concurrence (c=4) ?**
    Le cross-encoder sature déjà tous les cœurs CPU pour une seule requête : les requêtes font la queue. Avec c=4,
    j'ai mesuré la même latence ×4 et le même ~2 req/s. On gagne du débit en ajoutant des pods (scaling
    horizontal), en réduisant `TOP_K` ou en accélérant le modèle.

18. **Comment réduire la latence du reranking ?**
    Rerank moins de candidats (TOP_K 50 → 20) ; tronquer les documents ; exporter le cross-encoder en ONNX et le
    quantizer en int8 ; batcher plusieurs requêtes ; utiliser un GPU ; ou mettre en cache les requêtes fréquentes.
    À valider avec un benchmark de qualité (nDCG) pour ne pas perdre en pertinence.

19. **Pourquoi `def` et pas `async def` pour `/search` ?**
    L'inférence est du calcul CPU bloquant. Dans un `async def`, elle bloquerait la boucle d'événements et tout le
    serveur. Avec `def`, FastAPI l'exécute dans un pool de threads. À l'inverse, `/health` et `/ready` sont `async`
    pour répondre même quand les threads sont tous occupés.

20. **Pourquoi des requests/limits ?**
    Les requests permettent au scheduler de placer les pods là où il y a de la place, et servent de base à l'HPA.
    Les limits empêchent un pod de consommer tout le nœud. Si la mémoire dépasse la limite, le pod est tué
    (OOMKilled) ; si le CPU dépasse, il est ralenti.

21. **Pourquoi une ConfigMap ?**
    Pour séparer la config du code : je change `TOP_K` sans reconstruire l'image. Pour des données sensibles (clé
    API), j'utiliserais un Secret.

22. **Comment tu passerais en production sur le cloud ?**
    Les mêmes manifests marchent sur un Kubernetes managé (GKE, EKS, AKS), où le plan de contrôle est géré par le
    cloud. J'ajouterais : un Ingress + certificat TLS, le tag SHA au lieu de `latest`, des Secrets, un
    cluster-autoscaler pour ajouter des nœuds, et Prometheus/Grafana managés ou via l'Operator. Plus simple :
    **Cloud Run**, qui prend directement l'image Docker, gère HTTPS et scale selon les requêtes (jusqu'à zéro).
    Attention au démarrage à froid (~11 s de chargement des modèles) : il faudrait garder au moins une instance minimum.

23. **Comment servirais-tu un LLM plus gros ?**
    Pas avec FastAPI + torch dans le même processus : j'utiliserais un serveur d'inférence spécialisé comme **vLLM**
    (PagedAttention pour gérer la mémoire du cache KV, *continuous batching* pour regrouper les requêtes) sur des
    nœuds GPU (`resources.limits: nvidia.com/gpu: 1`). Les poids seraient sur un volume plutôt que dans l'image.
    L'autoscaling se ferait sur la file d'attente ou le nombre de requêtes en cours plutôt que sur le CPU, et notre
    API FastAPI appellerait ce serveur en HTTP. Je ne l'ai pas implémenté ici.

24. **Comment l'image arm64 est-elle construite sur un runner amd64 ?**
    Avec Buildx + QEMU (émulation). Comme l'émulation est lente, l'étape qui calcule les embeddings tourne en natif
    (`FROM --platform=$BUILDPLATFORM`), et seuls les fichiers résultants (index, modèles) sont copiés dans l'image
    arm64. Ce sont des données indépendantes du CPU.

25. **Que se passe-t-il si un pod plante ?**
    Le Deployment voit qu'il n'y a plus que 1 pod sur 2 et en recrée un. Pendant ce temps, le Service n'envoie le
    trafic qu'au pod *ready*. Si le processus est bloqué sans planter, la liveness probe échoue 3 fois et le
    kubelet redémarre le conteneur.

**Questions bonus probables** :
- *« Pourquoi BM25 + dense ? »* BM25 est fort sur les mots exacts (noms de gènes, sigles), le dense sur le sens
  (synonymes). RRF combine les deux sans normaliser les scores.
- *« Tu as évalué la qualité ? »* Voir les limites ci-dessous : pas de métrique de qualité publiée pour l'instant,
  et je sais expliquer pourquoi l'ancien benchmark était faux.

---

<a id="limites"></a>
## 12. Limites honnêtes

- **Local seulement** : minikube sur un MacBook, un seul nœud. Pas de vraie haute disponibilité (si le Mac
  s'éteint, tout s'arrête), pas de cloud, pas d'autoscaling de nœuds.
- **Pas de GPU** : inférence CPU, ~0,5 s par recherche avec reranking (mesuré). Le débit par pod est limité à ~2 req/s.
- **Pas d'authentification** ni de rate limiting, ni de TLS : n'importe qui ayant accès au Service peut l'appeler.
- **Monitoring basique** : métriques HTTP + logs texte. Pas d'alertes (Alertmanager), pas de dashboard Grafana
  versionné, pas de traces distribuées, pas de logs centralisés, pas de métriques ML (dérive des requêtes,
  distribution des scores).
- **Déploiement manuel** : le CD publie l'image, mais `kubectl apply` / `rollout restart` sont lancés à la main.
  `latest` est utilisé dans le Deployment pour la simplicité ; en production, on déploierait le tag SHA.
- **Index en mémoire et en pickle** : BM25 (`rank_bm25`) recalcule le score sur tout le corpus à chaque requête, et
  FAISS fait une recherche exacte (`IndexFlatL2`). C'est très bien pour 5 000 documents, pas pour des millions
  (il faudrait un index approximatif HNSW/IVF ou une base vectorielle). Les fichiers pickle ne doivent être chargés
  que s'ils viennent de notre propre build (`pickle` peut exécuter du code).
- **Pas de métrique de qualité** : l'ancien tableau MS MARCO (MRR@10 0,5594 / 0,7977 / 0,8282) a été retiré, parce
  que son protocole était faux. La « bonne réponse » était le *premier passage candidat* de chaque requête et non
  le passage annoté comme pertinent. De plus, les requêtes de test venaient du split *train* de MS MARCO, sur lequel
  le cross-encoder `ms-marco-MiniLM` a été entraîné. Une évaluation correcte utiliserait les requêtes de **test** de
  SciFact avec leurs jugements de pertinence (qrels) et une métrique comme nDCG@10. Ce n'est pas fait ici.
- **Une image assez grosse** : torch + modèles + index dans l'image (taille `<à mesurer>`). C'est le prix du
  démarrage hors ligne.
- **Le notebook `notebooks/demo.ipynb`** utilise encore l'ancien benchmark MS MARCO et les anciens imports `src.*` :
  il n'est plus à jour.
