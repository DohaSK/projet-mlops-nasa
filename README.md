# README.md

# Pipeline MLOps de Maintenance Predictive

Systeme hybride "Physics-Informed MLOps" sur le dataset NASA CMAPSS FD001.
Un modele de machine learning predit la duree de vie restante (RUL) des
moteurs. Quand la RUL est critique, un module RAG retrouve dans le papier
scientifique NASA les passages qui expliquent la degradation physique.

## Architecture

```
capteurs (JSON)
     |
     v
POST /predict --> modele ML (StandardScaler + RandomForest) --> RUL
                                   |
                     RUL < seuil critique (50 cycles) ?
                                   |
                  oui              v              non
        question fixe --> embeddings --> index FAISS      reponse sans
        --> top-k passages --> reponse extractive         explication
                                   |
                                   v
        JSON : rul_predite, statut, explication_physique_rag, references
```

Le ML et le RAG sont suivis dans le meme run MLflow : metriques RMSE et R2,
nombre de chunks de l'index, duree d'ingestion et temps de retrieval.

## Structure

```
params.yaml            hyperparametres, seuils du Quality Gate, parametres RAG
src/config.py          chemins et chargement de params.yaml
src/ml/                donnees, modele, entrainement MLflow
src/rag/               ingestion PDF, retrieval FAISS, generation
app.py                 API FastAPI unifiee
tests/                 tests ML, Quality Gate et tests RAG
docs/                  papier NASA (nasa_cmapss_paper.pdf)
dvc.yaml               pipeline DVC (ingestion RAG puis entrainement)
.github/workflows/     CI GitHub Actions
```

## Installation

```bash
python3.12 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Donnees et DVC

Placer dans `data/raw/` les fichiers `train_FD001.txt`, `test_FD001.txt` et
`RUL_FD001.txt` (NASA Prognostics Center of Excellence, Turbofan Engine
Degradation Simulation Data Set). Placer le papier "Damage Propagation
Modeling for Aircraft Engine Run-to-Failure Simulation" dans
`docs/nasa_cmapss_paper.pdf`. Les donnees sont versionnees avec DVC, pas Git :

```bash
dvc init
dvc add data/raw/train_FD001.txt data/raw/test_FD001.txt data/raw/RUL_FD001.txt
git add data/raw/*.dvc data/raw/.gitignore .dvc docs/nasa_cmapss_paper.pdf
git commit -m "Donnees FD001 via DVC et papier NASA"
```

Pour reproduire tout le pipeline : `dvc repro`.

## Lancer les tests

```bash
ruff check .
pytest -v
```

Le Quality Gate ML utilise des donnees synthetiques au format CMAPSS en CI.
Si les vrais fichiers FD001 sont presents, il est aussi execute sur eux. Les
tests RAG qui lisent le PDF sont ignores si `docs/nasa_cmapss_paper.pdf`
est absent. Les seuils (`min_r2`, `max_rmse`) sont lus dans `params.yaml`.

## Construire l'index RAG

```bash
python -m src.rag.ingestion
```

Cree `models/rag/faiss_index.bin`, `chunks.json` et `stats.json`. Le premier
lancement telecharge le modele d'embeddings all-MiniLM-L6-v2.

## Entrainer le modele

```bash
python -m src.ml.train
```

Les parametres, metriques ML et RAG et le modele sont enregistres dans
`./mlruns`. La commande retourne un code d'erreur si le Quality Gate n'est
pas respecte. Pour consulter les runs :

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Lancer l'API

```bash
uvicorn app:app --reload
```

- `GET /health` : statut de l'API, du modele et de l'index RAG
- `POST /predict` : corps JSON avec `setting1` a `setting3` et `s1` a `s21`
- Documentation interactive : http://127.0.0.1:8000/docs

Reponse type quand la RUL est critique :

```json
{
  "rul_predite": 42.0,
  "statut": "CRITIQUE",
  "explication_physique_rag": "D'apres la documentation NASA (modele C-MAPSS) : ...",
  "references": ["IV. DAMAGE PROPAGATION MODELING (p. 4)"]
}
```

Statuts : CRITIQUE (RUL inferieure a 50), SURVEILLANCE (inferieure a 100),
NORMAL sinon. Les seuils sont dans `params.yaml`, section `api`.

## Choix de generation

La reponse est construite par un template extractif : les phrases du papier
les plus proches de la question sont inserees dans une phrase structurelle.
Aucun LLM externe n'est utilise et aucune phrase n'est inventee. Une option
`generation.mode: flan_t5` active une reformulation locale par
google/flan-t5-small (necessite `pip install sentencepiece`).

## CI/CD

Chaque push sur `main` lance GitHub Actions : installation, lint `ruff`,
puis `pytest`. Le pipeline echoue si un test ou le Quality Gate echoue.