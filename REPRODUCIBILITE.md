# Dossier de reproductibilite

Ce document permet de reconstruire toutes les experiences du projet a
l'identique. La version machine de cette fiche (commit, versions installees,
identifiants des runs) est generee par `python -m src.reproductibilite` dans
`reports/reproductibilite.json`.

## 1. Environnement

| Element | Version |
|---|---|
| Python | 3.12 |
| Systemes testes | Windows 11 (PowerShell), Ubuntu (GitHub Actions) |
| Dependances | figees dans `requirements.txt` (scikit-learn 1.5.2, mlflow 2.17.2, dvc 3.56.0, sentence-transformers 3.3.1, faiss-cpu 1.9.0.post1, fastapi 0.115.4, groq 1.7.0...) |
| Modele d'embeddings | sentence-transformers/all-MiniLM-L6-v2 (telecharge au premier lancement) |
| LLM de l'assistant | openai/gpt-oss-20b via l'API Groq (optionnel) |

## 2. Donnees et documents

| Fichier | Source | Version (md5 DVC) |
|---|---|---|
| data/raw/train_FD001.txt | NASA Prognostics Center of Excellence, Turbofan Engine Degradation Simulation Data Set | 259f340bac32ce6fa8894815600fa757 |
| data/raw/test_FD001.txt | idem | 62316d1dd70f88a9a43a1e2690765437 |
| data/raw/RUL_FD001.txt | idem | 8f4a8b3d2692b2f7d8e72b344927bf4b |
| docs/nasa_cmapss_paper.pdf | Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation" (2008) | versionne dans Git |
| data/eval/questions_assistant.yaml | jeu d'evaluation de l'assistant (14 questions) | versionne dans Git |

Les donnees brutes sont suivies par DVC (fichiers `.dvc`), pas par Git.
`dvc status` verifie que les fichiers locaux correspondent aux empreintes.

## 3. Parametres

Tous les parametres sont dans `params.yaml` : preparation des donnees
(plafond de RUL 125, 20 % des moteurs en validation), hyperparametres des
trois modeles, seuils des Quality Gates ML et assistant, parametres RAG
(chunks de 150 mots, chevauchement 30, top_k 4), seuils de statut de l'API
(50 et 100 cycles), configuration de l'assistant. Les graines aleatoires sont
fixees (`random_state: 42`). L'empreinte SHA-256 de `params.yaml` est
enregistree dans la fiche generee et le fichier est logue dans chaque run.

## 4. Procedure de reconstruction

```bash
git clone <depot> && cd projet-mlops-nasa
python -m venv venv
venv\Scripts\activate            # Linux : source venv/bin/activate
pip install -r requirements.txt

# Donnees : dvc pull, ou copier les 3 fichiers FD001 dans data/raw/
dvc status

# Optionnel : cle Groq pour l'assistant en mode LLM
copy .env.example .env           # puis renseigner GROQ_API_KEY

dvc repro                        # 4 etapes, voir ci-dessous
pytest -v
python -m src.simulation --local
python -m src.reproductibilite
```

Etapes de `dvc repro` (definies dans `dvc.yaml`) :

1. `ingestion_rag` : extraction du PDF, 53 chunks, index FAISS (`models/rag`) ;
2. `train` : comparaison ridge / random_forest / hist_gradient_boosting,
   Quality Gate, enregistrement du champion dans le registre MLflow,
   `metrics.json` et `reports/comparaison_modeles.json` ;
3. `analyse_erreurs` : `reports/analyse_erreurs.json` et figures ;
4. `eval_assistant` : `assistant_metrics.json` et run MLflow `assistant-eval-*`.

Une etape dont les dependances et parametres n'ont pas change n'est pas
relancee. `dvc repro` echoue si un Quality Gate echoue.

## 5. Artefacts produits

| Artefact | Emplacement |
|---|---|
| Runs MLflow (parametres, metriques, modeles, figures) | `mlruns/` (`mlflow ui --backend-store-uri ./mlruns`) |
| Modele enregistre | registre MLflow `rul-fd001`, alias `candidat` et `champion` |
| Modele servi par l'API | `models/model.joblib` (ou `MLFLOW_MODEL_URI=models:/rul-fd001@champion`) |
| Index RAG | `models/rag/faiss_index.bin`, `chunks.json`, `stats.json` |
| Metriques | `metrics.json`, `assistant_metrics.json` |
| Rapports | `reports/*.json` et `reports/*.png` |
| Journal de l'API | `logs/requetes.jsonl` (non versionne) |

## 6. Resultats de reference

Obtenus avec la procedure ci-dessus (split par moteur, graine 42) :

| Modele | RMSE validation | R2 validation | Alertes manquees | RMSE test officiel |
|---|---|---|---|---|
| ridge | 21.20 | 0.736 | 35.4 % | 20.78 |
| random_forest | 19.05 | 0.787 | 20.0 % | 17.39 |
| hist_gradient_boosting (champion) | 18.99 | 0.788 | 18.8 % | 17.34 |

Les durees (ingestion, retrieval, latence de l'API et du LLM) dependent de la
machine et ne sont pas reproductibles a l'identique. Les reponses du LLM
peuvent varier legerement malgre une temperature de 0.

## 7. Ce qui n'est jamais dans Git

La cle `GROQ_API_KEY` (fichier `.env`, ignore), les donnees brutes (DVC),
`mlruns/`, les journaux `logs/` et l'environnement virtuel.
