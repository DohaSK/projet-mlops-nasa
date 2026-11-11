# src/train.py
"""Entrainement du modele et suivi dans MLflow."""

import json

import joblib
import mlflow
import mlflow.sklearn

from src.data import charger_params, chemin_projet, preparer_donnees
from src.model import construire_modele, evaluer, verifier_quality_gate


def entrainer(params=None):
    """Entraine le modele, loggue dans MLflow, sauvegarde l'artefact."""
    params = params or charger_params()
    X_train, X_test, y_train, y_test = preparer_donnees(params)
    modele = construire_modele(params)

    # Tracking local dans ./mlruns
    mlflow.set_tracking_uri(chemin_projet(params["training"]["mlruns_dir"]).as_uri())
    mlflow.set_experiment(params["training"]["experiment_name"])

    with mlflow.start_run(run_name="random-forest-fd001"):
        mlflow.log_params({f"model_{k}": v for k, v in params["model"].items()})
        mlflow.log_params({f"data_{k}": v for k, v in params["data"].items()})

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
        mlflow.set_tag("quality_gate", "passed" if gate_ok else "failed")

        # Sauvegarde locale du modele (utilisee par l'API et DVC)
        chemin_modele = chemin_projet(params["training"]["model_path"])
        chemin_modele.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(modele, chemin_modele)

        chemin_metriques = chemin_projet(params["training"]["metrics_path"])
        with open(chemin_metriques, "w", encoding="utf-8") as f:
            json.dump({**metriques_test, "quality_gate_passed": gate_ok}, f, indent=2)

        mlflow.log_artifact(str(chemin_modele), artifact_path="local_model")
        mlflow.log_artifact(str(chemin_projet("params.yaml")))
        mlflow.sklearn.log_model(modele, artifact_path="model")

    print(
        f"RMSE={metriques_test['rmse']:.2f}  R2={metriques_test['r2']:.3f}  "
        f"Quality Gate={'OK' if gate_ok else 'ECHEC'}"
    )
    return {**metriques_test, "quality_gate_passed": gate_ok}


if __name__ == "__main__":
    resultat = entrainer()
    raise SystemExit(0 if resultat["quality_gate_passed"] else 1)