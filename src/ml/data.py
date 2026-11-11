# src/ml/data.py
"""Chargement et preparation des donnees NASA CMAPSS FD001.

Dataset : "Turbofan Engine Degradation Simulation Data Set" (CMAPSS),
NASA Prognostics Center of Excellence, section Data Set Repository.
Telecharger l'archive, puis placer train_FD001.txt, test_FD001.txt et
RUL_FD001.txt dans data/raw/.
Format : 26 colonnes separees par des espaces (unit, cycle, 3 reglages,
21 capteurs), sans en-tete. RUL_FD001.txt contient une valeur par moteur de test.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import charger_params, chemin_projet

COLONNES_REGLAGES = [f"setting{i}" for i in range(1, 4)]
COLONNES_CAPTEURS = [f"s{i}" for i in range(1, 22)]
COLONNES_FEATURES = COLONNES_REGLAGES + COLONNES_CAPTEURS
COLONNES = ["unit_number", "time_in_cycles"] + COLONNES_FEATURES


def charger_donnees_brutes(chemin):
    """Lit un fichier texte CMAPSS et nomme les colonnes."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}. Placez les fichiers FD001 "
            "dans data/raw/ (ou lancez 'dvc pull')."
        )
    return pd.read_csv(
        chemin,
        sep=r"\s+",
        header=None,
        names=COLONNES,
        usecols=range(len(COLONNES)),
    )


def calculer_rul(df, plafond=None):
    """RUL = dernier cycle du moteur - cycle courant, plafonnee si demande."""
    dernier_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    rul = dernier_cycle - df["time_in_cycles"]
    if plafond is not None:
        rul = rul.clip(upper=plafond)
    return rul.astype(float)


def separer_par_moteur(df, test_size):
    """Split par unit_number : un moteur est soit en train, soit en validation.

    Les derniers moteurs (par numero) vont en validation, ce qui evite toute
    fuite entre les cycles d'un meme moteur.
    """
    moteurs = np.sort(df["unit_number"].unique())
    n_test = max(1, int(round(len(moteurs) * test_size)))
    moteurs_test = moteurs[-n_test:]
    masque_test = df["unit_number"].isin(moteurs_test)
    return df[~masque_test], df[masque_test]


def preparer_donnees(params=None, df=None):
    """Retourne X_train, X_test, y_train, y_test.

    `df` permet d'injecter un DataFrame (tests) au lieu de lire le fichier.
    """
    params = params or charger_params()
    if df is None:
        df = charger_donnees_brutes(chemin_projet(params["data"]["train_path"]))

    manquantes = [c for c in COLONNES if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes : {manquantes}")

    df = df.copy()
    df["RUL"] = calculer_rul(df, params["data"].get("rul_cap"))

    train, test = separer_par_moteur(df, params["data"]["test_size"])
    return (
        train[COLONNES_FEATURES],
        test[COLONNES_FEATURES],
        train["RUL"],
        test["RUL"],
    )


def charger_test_officiel(params=None):
    """Jeu de test officiel NASA : dernier cycle de chaque moteur de test_FD001
    et RUL reelle fournie par RUL_FD001.txt (plafonnee comme l'entrainement)."""
    params = params or charger_params()
    df = charger_donnees_brutes(chemin_projet(params["data"]["test_path"]))
    dernier = (
        df.sort_values(["unit_number", "time_in_cycles"])
        .groupby("unit_number")
        .tail(1)
    )
    chemin_rul = chemin_projet(params["data"]["rul_path"])
    if not chemin_rul.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin_rul}")
    rul = pd.read_csv(chemin_rul, header=None, sep=r"\s+")[0].to_numpy(dtype=float)
    if len(rul) != len(dernier):
        raise ValueError(
            f"{len(dernier)} moteurs de test mais {len(rul)} valeurs de RUL"
        )
    plafond = params["data"].get("rul_cap")
    if plafond is not None:
        rul = np.minimum(rul, plafond)
    return dernier[COLONNES_FEATURES], pd.Series(rul, index=dernier.index)