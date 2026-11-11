# src/ml/model.py
"""Construction et evaluation du modele."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def construire_modele(params):
    """Pipeline StandardScaler + RandomForestRegressor."""
    m = params["model"]
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=m["n_estimators"],
                    max_depth=m["max_depth"],
                    min_samples_leaf=m["min_samples_leaf"],
                    random_state=m["random_state"],
                    n_jobs=m["n_jobs"],
                ),
            ),
        ]
    )


def evaluer(modele, X, y):
    """Retourne RMSE et R2 sur (X, y)."""
    y_pred = modele.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "r2": float(r2_score(y, y_pred)),
    }


def verifier_quality_gate(metriques, params):
    """True si les metriques respectent les seuils de params.yaml."""
    gate = params["quality_gate"]
    return metriques["r2"] > gate["min_r2"] and metriques["rmse"] < gate["max_rmse"]