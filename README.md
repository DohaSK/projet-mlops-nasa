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
POST /predict --> modele ML champion (StandardScaler + HistGradientBoosting) --> RUL
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
nombre de chunks de l'index, duree d'ingestion et temps de retrieval. Un
assistant (`POST /assistant`, LLM Groq) repond aux questions des techniciens
a partir de la prediction, des capteurs inhabituels et du RAG.

## Structure

```
params.yaml            hyperparametres, seuils du Quality Gate, parametres RAG
src/config.py          chemins et chargement de params.yaml
src/ml/                donnees, modeles, comparaison et registre MLflow, analyse des erreurs
src/rag/               ingestion PDF, retrieval FAISS, generation
src/assistant/         assistant de maintenance (LLM Groq, garde-fous, evaluation)
src/journal.py         journal JSONL des requetes de l'API
app.py                 API FastAPI unifiee
monitor.py             indicateurs de monitoring a partir du journal
src/simulation.py      simulation de trafic de production sur l'API
src/reproductibilite.py fiche de reproductibilite (versions, hash DVC, runs MLflow)
reports/               comparaison, analyse des erreurs, monitoring, figures
REPRODUCIBILITE.md     procedure de reconstruction complete
data/eval/             jeu de questions d'evaluation de l'assistant
.env.example           modele de fichier .env (cle Groq, sans valeur)
tests/                 tests ML, Quality Gate, tests RAG et tests de l'assistant
docs/                  papier NASA (nasa_cmapss_paper.pdf)
dvc.yaml               pipeline DVC (ingestion RAG, comparaison, analyse, evaluation assistant)
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

## Entrainer et comparer les modeles

```bash
python -m src.ml.train
```

Trois modeles sont compares, chacun dans un run MLflow enfant du run
`comparaison-fd001` (parametres, metriques, modele avec signature) :

| Modele | RMSE validation | R2 validation | Alertes critiques manquees | RMSE test officiel |
|---|---|---|---|---|
| ridge (baseline) | 21.20 | 0.736 | 35.4 % | 20.78 |
| random_forest | 19.05 | 0.787 | 20.0 % | 17.39 |
| **hist_gradient_boosting** | **18.99** | **0.788** | **18.8 %** | **17.34** |

Le champion (plus faible RMSE de validation) est sauvegarde dans
`models/model.joblib` et enregistre dans le **MLflow Model Registry** sous le
nom `rul-fd001` avec l'alias `candidat`, puis `champion` s'il passe le Quality
Gate. La commande retourne un code d'erreur si le Quality Gate echoue.

```bash
mlflow ui --backend-store-uri ./mlruns     # onglets Experiments et Models
```

L'API peut charger le modele du registre :
`MLFLOW_MODEL_URI=models:/rul-fd001@champion` (sinon le fichier local).

### Quality Gate ML et justification des seuils

Le "taux d'alertes manquees" compte les cycles reellement critiques (RUL < 50)
que le modele classe non critiques : c'est l'erreur dangereuse en maintenance.

| Critere | Seuil | Origine |
|---|---|---|
| RMSE validation | < 25 cycles | moitie de la marge critique de 50 cycles ; une prediction constante fait 41.5 |
| R2 validation | > 0.7 | baseline Ridge : 0.74, le modele doit expliquer au moins 70 % de la variance |
| Alertes critiques manquees | <= 25 % | la baseline en manque 35 % ; un cycle manque est souvent rattrape quelques cycles plus tard |
| Mieux que la baseline | RMSE < RMSE Ridge | un modele complexe qui ne bat pas un modele lineaire est rejete |

## Analyse des erreurs

```bash
python -m src.ml.analyse_erreurs
```

Produit `reports/analyse_erreurs.json`, `reports/fig_predit_vs_reel.png` et
`reports/fig_rmse_par_tranche.png` (aussi dans le run MLflow du champion).
Constats sur FD001 :

- les erreurs sont faibles en fin de vie (RMSE 9 cycles sous 25 cycles) mais
  maximales entre 25 et 75 cycles (RMSE 26 a 29), justement autour du seuil
  critique, avec un biais optimiste (+12 a +15 cycles) ;
- au-dessus de 100 cycles, le modele sous-estime (biais -11) a cause du
  plafond de RUL a 125 : erreur prudente, sans danger ;
- sur le test officiel, 3 moteurs critiques sur 30 sont classes SURVEILLANCE ;
  le pire cas est le moteur 41 (RUL reelle 18, predite 70.7).

## Simulation de production et monitoring

```bash
python -m src.simulation --local                       # API en memoire
python -m src.simulation --url http://127.0.0.1:8000   # API lancee avec uvicorn
```

Le script rejoue 300 cycles reels de `test_FD001`, 15 requetes invalides
(capteur manquant ou valeur non numerique), 100 cycles avec une derive de
calibration simulee (T30 et T50 decales de +3 ecarts-types) et 6 questions a
l'assistant. Il appelle ensuite `monitor.py`, qui ecrit `reports/monitoring.json`
et `reports/fig_monitoring_derive.png`. La derive injectee est detectee sur s3
et s4 uniquement dans la phase de derive.

## Reproductibilite

Voir `REPRODUCIBILITE.md`. La commande `python -m src.reproductibilite`
ecrit `reports/reproductibilite.json` (versions, commit, hash DVC, runs MLflow,
version du registre).

## Lancer l'API

```bash
uvicorn app:app --reload
```

- `GET /health` : statut de l'API, du modele, de l'index RAG et mode de l'assistant
- `POST /predict` : corps JSON avec `setting1` a `setting3` et `s1` a `s21`
- `POST /assistant` : question libre + capteurs (voir section Assistant)
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

## Assistant de maintenance

L'endpoint `POST /assistant` repond en francais a une question libre d'un
technicien sur un moteur. Il combine trois sources :

1. la RUL et le statut predits par le modele ML (jamais recalcules par le LLM) ;
2. les 3 capteurs les plus eloignes de la moyenne d'entrainement (z-score
   calcule avec le StandardScaler du modele), nommes selon la table 2 du
   papier NASA (T30, P30, Nf...) ;
3. les passages du papier NASA retrouves par le RAG, cites sous la forme [n].

```
question + capteurs --> garde-fou injection --> LLM : perimetre + requete anglaise
   --> recherche FAISS --> LLM : reponse avec citations --> verification
   --> si probleme (citation invalide, RUL inventee, erreur LLM) : reponse extractive
```

Le LLM est `openai/gpt-oss-20b` via l'API Groq (les modeles Llama de Groq sont
reserves aux comptes Enterprise). Configuration : section `assistant` de
`params.yaml`. La cle se met dans un fichier `.env`, jamais dans Git :

```bash
cp .env.example .env        # Windows : copy .env.example .env
# puis ecrire la cle : GROQ_API_KEY=gsk_...
```

Sans cle, l'assistant fonctionne en mode extractif (filtre de perimetre par
mots-cles et phrases du papier). Exemple de requete :

```json
{
  "question": "Pourquoi ce moteur est critique et que dois-je faire ?",
  "capteurs": {"setting1": -0.0006, "setting2": 0.0004, "setting3": 100.0,
               "s1": 518.67, "s2": 643.95, "...": "...", "s21": 23.0}
}
```

La reponse contient `reponse`, `references`, `capteurs_anormaux`, `mode`
(`llm`, `extractif`, `extractif_secours` ou `refus`), `garde_fous`, les tokens,
le cout estime et la latence.

### Evaluation et Quality Gate de l'assistant

```bash
python -m src.assistant.evaluation
```

Les 14 questions de `data/eval/questions_assistant.yaml` (9 de maintenance,
5 hors sujet ou tentatives d'injection) sont posees sur 3 moteurs de test
(RUL predite minimale, mediane et maximale), soit 42 requetes. Metriques
enregistrees dans MLflow (run `assistant-eval-llm` ou `assistant-eval-extractif`)
et dans `assistant_metrics.json`. Le Quality Gate exige :

| Critere | Seuil | Justification |
|---|---|---|
| Refus hors perimetre | 100 % | systeme critique : aucune reponse hors sujet ou manipulee |
| Reponse dans le perimetre | >= 90 % | l'assistant doit rester utile, 1 echec sur 9 tolere |
| Citation d'une source | >= 80 % | chaque explication physique doit etre verifiable |
| Latence P95 | <= 8000 ms | deux appels LLM, usage interactif par un technicien |

Le compte gratuit Groq limite gpt-oss-20b a 8000 tokens par minute, et une
question consomme environ 2000 tokens. Sans precaution, l'evaluation depasse
cette limite apres 4 questions : Groq met les suivantes en attente et la latence
mesuree passe de 1 a 2 s a 13 a 15 s. L'evaluation espace donc ses appels
(`limite_tokens_par_minute` dans `params.yaml`) ; le temps d'attente est
enregistre a part (`attente_debit_s`) et n'entre pas dans la latence. En
production, plus de 3 a 4 questions par minute demanderaient un compte payant.

### Monitoring

Chaque appel a `/predict` et `/assistant` est ecrit dans `logs/requetes.jsonl`
(code HTTP, latence, statut, mode de l'assistant, tokens, cout, capteurs).

```bash
python monitor.py
```

affiche le taux d'erreur, le taux d'entrees invalides, la latence moyenne et
P95, la repartition des statuts, les refus et modes de l'assistant, les tokens
et le cout, et la derive moyenne de chaque capteur (en ecarts-types
d'entrainement, alerte au-dela de 0,5).

## Choix de generation

La reponse est construite par un template extractif : les phrases du papier
les plus proches de la question sont inserees dans une phrase structurelle.
Aucun LLM externe n'est utilise et aucune phrase n'est inventee. Une option
`generation.mode: flan_t5` active une reformulation locale par
google/flan-t5-small (necessite `pip install sentencepiece`).

## CI/CD

Chaque push sur `main` lance GitHub Actions : installation, lint `ruff`,
puis `pytest`. Le pipeline echoue si un test ou le Quality Gate echoue.