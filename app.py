# app.py
"""API unifiee : prediction de RUL (ML), explication physique (RAG) et
assistant de maintenance (LLM Groq avec garde-fous)."""

import os
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field, create_model

from src.assistant.assistant import repondre
from src.assistant.capteurs import capteurs_anormaux
from src.assistant.llm import creer_client
from src.config import charger_params, chemin_projet
from src.flotte import Flotte
from src.journal import journaliser
from src.ml.data import COLONNES_FEATURES
from src.rag.generation import extraire_references, generer_reponse
from src.rag.retrieval import rechercher

# 24 champs obligatoires : setting1-3 et s1-s21
Capteurs = create_model("Capteurs", **{c: (float, ...) for c in COLONNES_FEATURES})


class QuestionAssistant(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    capteurs: Capteurs


def charger_modele(params):
    """Charge le modele champion.

    Si MLFLOW_MODEL_URI est defini, il est charge depuis MLflow, par exemple
    depuis le registre : models:/rul-fd001@champion. Sinon on lit le fichier
    local models/model.joblib.
    """
    uri = os.getenv("MLFLOW_MODEL_URI")
    if uri:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(
            chemin_projet(params["training"]["mlruns_dir"]).as_uri()
        )
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
    load_dotenv()  # lit GROQ_API_KEY depuis .env s'il existe
    app.state.params = charger_params()
    app.state.modele = charger_modele(app.state.params)
    app.state.client_llm = creer_client(app.state.params)
    app.state.flotte = None  # chargee a la premiere visite de l'interface
    yield


app = FastAPI(title="API RUL Maintenance Predictive + RAG", lifespan=lifespan)


@app.middleware("http")
async def journal_requetes(request: Request, call_next):
    """Journalise chaque appel a /predict et /assistant (monitoring)."""
    if request.url.path not in ("/predict", "/assistant"):
        return await call_next(request)
    debut = time.perf_counter()
    request.state.details = {}
    code = 500
    try:
        reponse = await call_next(request)
        code = reponse.status_code
        return reponse
    finally:
        journaliser(
            {
                "endpoint": request.url.path,
                "code_http": code,
                "latence_ms": round((time.perf_counter() - debut) * 1000, 1),
                # Etiquette optionnelle posee par le script de simulation
                "phase": request.headers.get("x-phase", "production"),
                **getattr(request.state, "details", {}),
            },
            app.state.params,
        )


def predire(capteurs):
    """RUL (bornee a 0) et statut pour un jeu de capteurs."""
    X = pd.DataFrame([capteurs.model_dump()])[COLONNES_FEATURES]
    rul = round(max(float(app.state.modele.predict(X)[0]), 0.0), 1)
    return rul, calculer_statut(rul, app.state.params)


@app.get("/health")
def health():
    params = app.state.params
    return {
        "status": "ok",
        "model_loaded": app.state.modele is not None,
        "rag_index_ready": chemin_projet(params["rag"]["index_path"]).exists(),
        "assistant_mode": "llm" if app.state.client_llm else "extractif",
        "seuils": {
            "critique": params["api"]["seuil_critique"],
            "attention": params["api"]["seuil_attention"],
        },
    }


@app.post("/predict")
def predict(capteurs: Capteurs, request: Request):
    if app.state.modele is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    params = app.state.params

    rul, statut = predire(capteurs)
    request.state.details = {
        "rul_predite": rul,
        "statut": statut,
        "capteurs": capteurs.model_dump(),
    }

    explication, references = None, []
    if statut == "CRITIQUE":
        explication, references = expliquer(params)

    return {
        "rul_predite": rul,
        "statut": statut,
        "explication_physique_rag": explication,
        "references": references,
    }


@app.post("/assistant")
def assistant(corps: QuestionAssistant, request: Request):
    """Question libre d'un technicien sur un moteur (capteurs du cycle courant)."""
    if app.state.modele is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    params = app.state.params

    rul, statut = predire(corps.capteurs)
    anomalies = capteurs_anormaux(
        app.state.modele,
        corps.capteurs.model_dump(),
        params["assistant"]["nb_capteurs_anormaux"],
    )
    resultat = repondre(
        corps.question, rul, statut, anomalies, params, client=app.state.client_llm
    )
    request.state.details = {
        "rul_predite": rul,
        "statut": statut,
        "capteurs": corps.capteurs.model_dump(),
        "mode_assistant": resultat["mode"],
        "refus": resultat["refus"],
        "garde_fous": resultat["garde_fous"],
        "tokens_entree": resultat["tokens_entree"],
        "tokens_sortie": resultat["tokens_sortie"],
        "cout_estime_usd": resultat["cout_estime_usd"],
    }
    return resultat


# Interface web
PAGE_UI = chemin_projet("src/web/index.html")


def flotte():
    """Flotte de test, calculee une seule fois (necessite le modele et FD001)."""
    if app.state.modele is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    if app.state.flotte is None:
        try:
            app.state.flotte = Flotte(app.state.params, app.state.modele)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
    return app.state.flotte


@app.get("/", include_in_schema=False)
def accueil():
    return RedirectResponse("/ui")


@app.get("/ui", include_in_schema=False)
def interface():
    return FileResponse(PAGE_UI, media_type="text/html")


@app.get("/moteurs")
def moteurs():
    """Les 100 moteurs de test : RUL predite et reelle au dernier cycle."""
    return flotte().resume()


@app.get("/moteurs/{moteur}")
def moteur(moteur: int):
    """Trajectoire d'un moteur, capteurs du dernier cycle et capteurs inhabituels."""
    detail = flotte().detail(moteur)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Moteur {moteur} introuvable")
    params = app.state.params
    detail["statut"] = calculer_statut(detail["rul_predite"], params)
    detail["capteurs_anormaux"] = capteurs_anormaux(
        app.state.modele, detail["capteurs"], params["assistant"]["nb_capteurs_anormaux"]
    )
    return detail
