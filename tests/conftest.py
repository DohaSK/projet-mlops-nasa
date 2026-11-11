# tests/conftest.py
"""Donnees synthetiques au format CMAPSS pour tester sans le vrai fichier."""

import numpy as np
import pandas as pd
import pytest

from src.ml.data import COLONNES


def generer_donnees_synthetiques(n_moteurs=30, seed=0):
    rng = np.random.default_rng(seed)
    coeffs = rng.uniform(1.0, 3.0, size=21) * rng.choice([-1, 1], size=21)
    bases = rng.uniform(10, 500, size=21)
    blocs = []
    for unit in range(1, n_moteurs + 1):
        duree = int(rng.integers(150, 250))
        cycles = np.arange(1, duree + 1)
        rul = np.minimum(duree - cycles, 125)
        usure = 1 - rul / 125
        capteurs = bases + np.outer(usure, coeffs)
        capteurs += rng.normal(0, 0.3, size=(duree, 21))
        reglages = rng.normal(0, 0.01, size=(duree, 3))
        blocs.append(np.column_stack([np.full(duree, unit), cycles, reglages, capteurs]))
    df = pd.DataFrame(np.vstack(blocs), columns=COLONNES)
    df["unit_number"] = df["unit_number"].astype(int)
    df["time_in_cycles"] = df["time_in_cycles"].astype(int)
    return df


@pytest.fixture(scope="session")
def donnees_synthetiques():
    return generer_donnees_synthetiques()