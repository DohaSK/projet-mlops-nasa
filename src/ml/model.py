# src/ml/model.py
"""Construction, evaluation et Quality Gate des modeles.

Trois configurations sont comparees (section `comparaison` de params.yaml) :
- ridge : regression lineaire regularisee, baseline simple et rapide ;
- random_forest : modele de reference du projet ;
- hist_gradient_boosting : boosting d'arbres, rapide sur ~20 000 lignes.
Chaque pipeline commence par un StandardScaler nomme "scaler" : l'assistant
l'utilise pour reperer les capteurs inhabituels.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NOMS_ETAPES = {
    "ridge": "ridge",
    "random_forest": "rf",
    "hist_gradient_boosting": "hgb",
}


def _estimateur(type_modele, params):
    if type_modele == "random_forest":
        m = params["model"]
        return RandomForestRegressor(
            n_estimators=m["n_estimators"],
            max_depth=m["max_depth"],
            min_samples_leaf=m["min_samples_leaf"],
            random_state=m["random_state"],
            n_jobs=m["n_jobs"],
        )
    if type_modele == "ridge":
        return Ridge(**params["comparaison"]["ridge"])
    if type_modele == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**params["comparaison"]["hist_gradient_boosting"])
    raise ValueError(f"Type de modele inconnu : {type_modele}")


def construire_modele(params, type_modele="random_forest"):
    """Pipeline StandardScaler + estimateur (RandomForest par defaut)."""
    estimateur = _estimateur(type_modele, params)  # ValueError si type inconnu
    return Pipeline(
        steps=[("scaler", StandardScaler()), (NOMS_ETAPES[type_modele], estimateur)]
    )


def evaluer(modele, X, y):
    """Retourne RMSE et R2 sur (X, y)."""
    y_pred = modele.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "r2": float(r2_score(y, y_pred)),
    }


def metriques_securite(y, y_pred, seuil_critique):
    """Erreurs qui comptent pour la maintenance, au seuil CRITIQUE.

    - taux_alertes_manquees : cycles reellement critiques (RUL < seuil) que le
      modele classe non critiques. C'est l'erreur dangereuse.
    - taux_fausses_alertes : cycles non critiques classes critiques (cout
      d'une inspection inutile).
    - surestimation_moyenne_critique : erreur moyenne (prediction - realite)
      sur les cycles critiques ; positive = le modele est trop optimiste.
    """
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    critique = y < seuil_critique
    predit_critique = y_pred < seuil_critique
    return {
        "taux_alertes_manquees": float((critique & ~predit_critique).sum() / max(critique.sum(), 1)),
        "taux_fausses_alertes": float((~critique & predit_critique).sum() / max((~critique).sum(), 1)),
        "surestimation_moyenne_critique": float((y_pred[critique] - y[critique]).mean())
        if critique.any()
        else 0.0,
    }


def verifier_quality_gate(metriques, params, baseline=None):
    """True si les metriques respectent les seuils de params.yaml.

    `baseline` : metriques de la baseline Ridge. Si fourni et si
    `exiger_mieux_que_baseline` est vrai, le modele doit aussi la battre en RMSE.
    """
    return not raisons_echec_gate(metriques, params, baseline)


def raisons_echec_gate(metriques, params, baseline=None):
    """Liste des criteres non respectes (vide si le Quality Gate est passe)."""
    gate = params["quality_gate"]
    raisons = []
    if not metriques["r2"] > gate["min_r2"]:
        raisons.append(f"r2 {metriques['r2']:.3f} <= {gate['min_r2']}")
    if not metriques["rmse"] < gate["max_rmse"]:
        raisons.append(f"rmse {metriques['rmse']:.2f} >= {gate['max_rmse']}")
    seuil_manquees = gate.get("max_taux_alertes_manquees")
    if seuil_manquees is not None and "taux_alertes_manquees" in metriques:
        if metriques["taux_alertes_manquees"] > seuil_manquees:
            raisons.append(
                f"alertes manquees {metriques['taux_alertes_manquees']:.3f} > {seuil_manquees}"
            )
    if baseline is not None and gate.get("exiger_mieux_que_baseline", False):
        if not metriques["rmse"] < baseline["rmse"]:
            raisons.append(
                f"rmse {metriques['rmse']:.2f} ne bat pas la baseline {baseline['rmse']:.2f}"
            )
    return raisons
