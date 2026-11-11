# tests/test_comparaison.py
"""Comparaison des modeles, Quality Gate, Model Registry, analyse des
erreurs, simulation et reproductibilite (donnees synthetiques, sans reseau)."""

import copy

import numpy as np
import pytest

import src.ml.train as train
from src.config import charger_params
from src.ml.analyse_erreurs import erreurs_par_tranche, matrice_statuts
from src.ml.data import preparer_donnees
from src.ml.model import (
    construire_modele,
    metriques_securite,
    raisons_echec_gate,
    verifier_quality_gate,
)

PARAMS = charger_params()


@pytest.mark.parametrize("type_modele", PARAMS["comparaison"]["candidats"])
def test_chaque_candidat_a_un_scaler(type_modele):
    modele = construire_modele(PARAMS, type_modele)
    assert modele.steps[0][0] == "scaler"


def test_type_inconnu():
    with pytest.raises(ValueError):
        construire_modele(PARAMS, "svm")


def test_metriques_securite():
    y = [10, 20, 60, 80]
    y_pred = [15, 70, 40, 90]  # 20 -> 70 : alerte manquee ; 60 -> 40 : fausse alerte
    m = metriques_securite(y, y_pred, seuil_critique=50)
    assert m["taux_alertes_manquees"] == 0.5
    assert m["taux_fausses_alertes"] == 0.5
    assert m["surestimation_moyenne_critique"] == pytest.approx(27.5)


def test_quality_gate_baseline_et_alertes():
    bon = {"rmse": 18.0, "r2": 0.8, "taux_alertes_manquees": 0.1}
    baseline = {"rmse": 21.0, "r2": 0.74}
    assert verifier_quality_gate(bon, PARAMS, baseline)
    # Ne bat pas la baseline
    assert any("baseline" in r for r in raisons_echec_gate(bon, PARAMS, {"rmse": 17.0}))
    # Trop d'alertes critiques manquees
    risque = {**bon, "taux_alertes_manquees": 0.4}
    assert any("alertes" in r for r in raisons_echec_gate(risque, PARAMS, baseline))


def test_seuils_coherents_avec_la_baseline():
    """Les seuils documentes restent plus exigeants qu'une prediction constante."""
    gate = PARAMS["quality_gate"]
    assert gate["max_rmse"] <= PARAMS["api"]["seuil_critique"] / 2
    assert 0 < gate["min_r2"] < 1
    assert PARAMS["comparaison"]["baseline"] in PARAMS["comparaison"]["candidats"]


def test_entrainement_comparaison_et_registry(donnees_synthetiques, tmp_path, monkeypatch):
    params = copy.deepcopy(PARAMS)
    params["model"]["n_estimators"] = 10
    params["training"].update(
        mlruns_dir=str(tmp_path / "mlruns"),
        model_path=str(tmp_path / "model.joblib"),
        metrics_path=str(tmp_path / "metrics.json"),
        comparaison_path=str(tmp_path / "comparaison.json"),
    )
    monkeypatch.setattr(train, "preparer_donnees",
                        lambda p: preparer_donnees(p, df=donnees_synthetiques))
    monkeypatch.setattr(train, "_metriques_rag", lambda p: {})
    monkeypatch.setattr(train, "_test_officiel_disponible", lambda p: False)

    resultat = train.entrainer(params)
    assert resultat["champion"] in params["comparaison"]["candidats"]
    assert resultat["quality_gate_passed"], resultat["quality_gate_raisons"]
    assert resultat["registry_version"] == 1
    assert (tmp_path / "model.joblib").exists()

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    version = MlflowClient().get_model_version_by_alias(
        params["registry"]["nom_modele"], params["registry"]["alias_champion"]
    )
    assert version.tags["quality_gate"] == "passed"


def test_erreurs_par_tranche_et_statuts():
    y = np.array([10, 30, 60, 90, 120])
    y_pred = np.array([20, 30, 40, 90, 100])
    tranches = erreurs_par_tranche(y, y_pred)
    assert [t["tranche_rul_reelle"] for t in tranches] == ["0-24", "25-49", "50-74", "75-99", "100-125"]
    assert tranches[0]["biais"] == 10.0
    m = matrice_statuts(y, y_pred, PARAMS)
    assert m["SURVEILLANCE"]["CRITIQUE"] == 1  # 60 predit 40
    assert sum(sum(v.values()) for v in m.values()) == 5


def test_simulation_prepare_les_phases(donnees_synthetiques, monkeypatch):
    import src.simulation as sim

    monkeypatch.setattr(sim, "charger_donnees_brutes", lambda chemin: donnees_synthetiques)
    requetes = sim.preparer_requetes(PARAMS, n_nominal=40, n_derive=10, taux_invalide=0.1)
    phases = [r[0] for r in requetes]
    assert phases.count("nominal") == 40 and phases.count("derive") == 10
    assert phases.count("invalide") == 4
    assert phases.count("assistant") == len(sim.QUESTIONS)
    nominal = requetes[0][2]
    assert set(nominal) == set(requetes[40][2])  # meme schema en phase de derive


def test_fiche_reproductibilite(tmp_path):
    import src.reproductibilite as repro

    params = copy.deepcopy(PARAMS)
    params["training"]["mlruns_dir"] = str(tmp_path / "mlruns_vide")
    fiche = repro.generer(params, dossier=tmp_path)
    assert (tmp_path / "reproductibilite.json").exists()
    assert fiche["paquets"]["scikit-learn"]
    assert len(fiche["donnees_dvc"]) == 3
    assert len(fiche["params_yaml_sha256"]) == 64
