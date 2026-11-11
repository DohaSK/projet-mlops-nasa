# app.py
"""API unifiee : prediction de RUL (ML) et explication physique (RAG)."""

import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import create_model

from src.config import charger_params, chemin_projet
from src.ml.data import COLONNES_FEATURES
from src.rag.generation import extraire_references, generer_reponse
from src.rag.retrieval import rechercher

# 24 champs obligatoires : setting1-3 et s1-s21
Capteurs = create_model("Capteurs", **{c: (float, ...) for c in COLONNES_FEATURES})


def charger_modele(params):
    """Charge le modele champion.

    Si MLFLOW_MODEL_URI est defini (ex : runs:/<run_id>/model), il est
    charge depuis MLflow. Sinon on lit le fichier local.
    """
    uri = os.getenv("MLFLOW_MODEL_URI")
    if uri:
        import mlflow.sklearn

        return mlflow.sklearn.load_model(uri)

    chemin = os.getenv("MODEL_PATH") or str(
        chemin_projet(params["training"]["model_path"])
    )
    if not os.path.exists(chemin):
        return None
    return joblib.load(chemin)


def calculer_statut(rul, params):
    seuils = params["api"]
    if rul < seuils["seuil_critique"]:
        return "CRITIQUE"
    if rul < seuils["seuil_attention"]:
        return "SURVEILLANCE"
    return "NORMAL"


def expliquer(params):
    """Explication physique via RAG. Ne fait jamais echouer la prediction."""
    question = params["rag"]["question_critique"]
    try:
        chunks = rechercher(question, params)
    except FileNotFoundError:
        return (
            "Index RAG indisponible. Lancez : python -m src.rag.ingestion",
            [],
        )
    return generer_reponse(question, chunks, params), extraire_references(chunks)


@asynccontextmanager
async def lifespan(app):
    app.state.params = charger_params()
    app.state.modele = charger_modele(app.state.params)
    yield


app = FastAPI(title="API RUL Maintenance Predictive + RAG", lifespan=lifespan)


@app.get("/health")
def health():
    params = app.state.params
    return {
        "status": "ok",
        "model_loaded": app.state.modele is not None,
        "rag_index_ready": chemin_projet(params["rag"]["index_path"]).exists(),
    }


@app.post("/predict")
def predict(capteurs: Capteurs):
    if app.state.modele is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    params = app.state.params

    X = pd.DataFrame([capteurs.model_dump()])[COLONNES_FEATURES]
    rul = max(float(app.state.modele.predict(X)[0]), 0.0)
    statut = calculer_statut(rul, params)

    explication, references = None, []
    if statut == "CRITIQUE":
        explication, references = expliquer(params)

    return {
        "rul_predite": round(rul, 1),
        "statut": statut,
        "explication_physique_rag": explication,
        "references": references,
    }