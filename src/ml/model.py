# src/ml/train.py
"""Entrainement du modele, suivi MLflow unifie (metriques ML et RAG)."""

import json
import time

import joblib
import mlflow
import mlflow.sklearn

from src.config import charger_params, chemin_projet
from src.ml.data import charger_test_officiel, preparer_donnees
from src.ml.model import construire_modele, evaluer, verifier_quality_gate
from src.rag.ingestion import construire_index
from src.rag.retrieval import rechercher


def _metriques_test_officiel(modele, params):
    """Evalue sur le dernier cycle de test_FD001 si les fichiers sont presents."""
    if not (
        chemin_projet(params["data"]["test_path"]).exists()
        and chemin_projet(params["data"]["rul_path"]).exists()
    ):
        return {}
    X, y = charger_test_officiel(params)
    m = evaluer(modele, X, y)
    return {"rmse_test_officiel": m["rmse"], "r2_test_officiel": m["r2"]}


def _metriques_rag(params):
    """Nombre de chunks, duree d'ingestion et temps de retrieval du RAG."""
    r = params["rag"]
    chemin_index = chemin_projet(r["index_path"])
    if not chemin_index.exists():
        if not chemin_projet(r["pdf_path"]).exists():
            return {}
        construire_index(params)

    chemin_stats = chemin_index.parent / "stats.json"
    stats = {}
    if chemin_stats.exists():
        stats = json.loads(chemin_stats.read_text(encoding="utf-8"))

    question = r["question_critique"]
    rechercher(question, params)  # premier appel : chargement du modele d'embeddings
    debut = time.perf_counter()
    rechercher(question, params)
    duree_ms = (time.perf_counter() - debut) * 1000
    return {
        "rag_nb_chunks": float(stats.get("nb_chunks", 0)),
        "rag_duree_ingestion_s": float(stats.get("duree_ingestion_s", 0.0)),
        "rag_retrieval_ms": round(duree_ms, 2),
    }


def entrainer(params=None):
    """Entraine le modele, loggue ML et RAG dans un meme run MLflow."""
    params = params or charger_params()
    X_train, X_test, y_train, y_test = preparer_donnees(params)
    modele = construire_modele(params)

    # Tracking local dans ./mlruns
    mlflow.set_tracking_uri(chemin_projet(params["training"]["mlruns_dir"]).as_uri())
    mlflow.set_experiment(params["training"]["experiment_name"])

    with mlflow.start_run(run_name="random-forest-fd001"):
        mlflow.log_params({f"model_{k}": v for k, v in params["model"].items()})
        mlflow.log_params({f"data_{k}": v for k, v in params["data"].items()})
        mlflow.log_params({f"rag_{k}": v for k, v in params["rag"].items()
                           if k in ("chunk_size", "chunk_overlap", "top_k", "embedding_model")})

        modele.fit(X_train, y_train)

        metriques_test = evaluer(modele, X_test, y_test)
        metriques_train = evaluer(modele, X_train, y_train)
        gate_ok = verifier_quality_gate(metriques_test, params)

        mlflow.log_metrics(
            {
                "rmse": metriques_test["rmse"],
                "r2": metriques_test["r2"],
                "rmse_train": metriques_train["rmse"],
                "r2_train": metriques_train["r2"],
            }
        )
        officiel = _metriques_test_officiel(modele, params)
        if officiel:
            mlflow.log_metrics(officiel)

        rag = _metriques_rag(params)
        if rag:
            mlflow.log_metrics(rag)
            mlflow.log_artifact(
                str(chemin_projet(params["rag"]["index_path"])), artifact_path="rag"
            )
        mlflow.set_tag("quality_gate", "passed" if gate_ok else "failed")

        # Sauvegarde locale du modele (utilisee par l'API et DVC)
        chemin_modele = chemin_projet(params["training"]["model_path"])
        chemin_modele.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(modele, chemin_modele)

        resultat = {
            **metriques_test,
            **officiel,
            **rag,
            "quality_gate_passed": gate_ok,
        }
        chemin_metriques = chemin_projet(params["training"]["metrics_path"])
        with open(chemin_metriques, "w", encoding="utf-8") as f:
            json.dump(resultat, f, indent=2)

        mlflow.log_artifact(str(chemin_modele), artifact_path="local_model")
        mlflow.log_artifact(str(chemin_projet("params.yaml")))
        mlflow.sklearn.log_model(modele, artifact_path="model")

    print(
        f"RMSE={metriques_test['rmse']:.2f}  R2={metriques_test['r2']:.3f}  "
        f"Quality Gate={'OK' if gate_ok else 'ECHEC'}"
    )
    return resultat


if __name__ == "__main__":
    sortie = entrainer()
    raise SystemExit(0 if sortie["quality_gate_passed"] else 1)