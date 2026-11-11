# tests/test_ml.py
import copy
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from src.config import charger_params, chemin_projet
from src.ml.data import (
    COLONNES,
    charger_donnees_brutes,
    preparer_donnees,
)
from src.ml.model import construire_modele, evaluer

PARAMS = charger_params()
FICHIER_REEL = chemin_projet(PARAMS["data"]["train_path"])
donnees_reelles = pytest.mark.skipif(
    not Path(FICHIER_REEL).exists(), reason="Donnees FD001 absentes"
)


# Tests unitaires
def test_charger_params_cles():
    for section in ["data", "model", "quality_gate", "training", "rag", "api"]:
        assert section in PARAMS
    assert "min_r2" in PARAMS["quality_gate"]
    assert "max_rmse" in PARAMS["quality_gate"]
    assert PARAMS["rag"]["pdf_path"] == "docs/nasa_cmapss_paper.pdf"


def test_construire_modele_est_un_pipeline():
    modele = construire_modele(PARAMS)
    assert isinstance(modele, Pipeline)
    assert [nom for nom, _ in modele.steps] == ["scaler", "rf"]


# Tests de donnees
def test_pas_de_valeurs_nulles(donnees_synthetiques):
    X_train, X_test, y_train, y_test = preparer_donnees(
        PARAMS, df=donnees_synthetiques
    )
    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()
    assert not y_train.isnull().any()
    assert not y_test.isnull().any()


def test_rul_positive(donnees_synthetiques):
    _, _, y_train, y_test = preparer_donnees(PARAMS, df=donnees_synthetiques)
    assert (y_train >= 0).all()
    assert (y_test >= 0).all()


def test_split_par_moteur(donnees_synthetiques):
    X_train, X_test, _, _ = preparer_donnees(PARAMS, df=donnees_synthetiques)
    moteurs_train = set(donnees_synthetiques.loc[X_train.index, "unit_number"])
    moteurs_test = set(donnees_synthetiques.loc[X_test.index, "unit_number"])
    assert moteurs_train.isdisjoint(moteurs_test)


@donnees_reelles
def test_donnees_reelles_sans_nulls():
    df = charger_donnees_brutes(FICHIER_REEL)
    assert list(df.columns) == COLONNES
    assert not df.isnull().any().any()


# Quality Gate
def _verifier_gate(params, df=None):
    X_train, X_test, y_train, y_test = preparer_donnees(params, df=df)
    modele = construire_modele(params).fit(X_train, y_train)
    m = evaluer(modele, X_test, y_test)
    gate = params["quality_gate"]  # seuils lus dynamiquement dans params.yaml
    assert m["r2"] > gate["min_r2"], f"R2 {m['r2']:.3f} <= {gate['min_r2']}"
    assert m["rmse"] < gate["max_rmse"], f"RMSE {m['rmse']:.2f} >= {gate['max_rmse']}"


def test_quality_gate_sous_ensemble(donnees_synthetiques):
    """Quality Gate en CI : petit modele sur un sous-ensemble synthetique."""
    params = copy.deepcopy(PARAMS)
    params["model"]["n_estimators"] = 20
    _verifier_gate(params, df=donnees_synthetiques)


@donnees_reelles
def test_quality_gate_donnees_reelles():
    """Quality Gate sur le vrai FD001, execute seulement si le fichier existe."""
    _verifier_gate(PARAMS)