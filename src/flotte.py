# src/flotte.py
"""Donnees de la flotte de test (test_FD001) pour l'interface web.

Pour chaque moteur de test on connait tous ses cycles jusqu'a l'arret de
l'enregistrement, et la RUL reelle au dernier cycle (RUL_FD001.txt). La RUL
reelle d'un cycle anterieur vaut donc : RUL finale + (dernier cycle - cycle),
plafonnee comme a l'entrainement pour etre comparable a la prediction.
"""

import numpy as np
import pandas as pd

from src.config import chemin_projet
from src.ml.data import COLONNES_FEATURES, charger_donnees_brutes


def statut(rul, params):
    if rul < params["api"]["seuil_critique"]:
        return "CRITIQUE"
    if rul < params["api"]["seuil_attention"]:
        return "SURVEILLANCE"
    return "NORMAL"


class Flotte:
    def __init__(self, params, modele, df=None, rul_finales=None):
        """`df` et `rul_finales` permettent d'injecter des donnees (tests)."""
        self.params = params
        self.modele = modele
        if df is None:
            df = charger_donnees_brutes(chemin_projet(params["data"]["test_path"]))
        if rul_finales is None:
            rul_finales = pd.read_csv(
                chemin_projet(params["data"]["rul_path"]), header=None, sep=r"\s+"
            )[0].to_numpy(dtype=float)
        self.df = df.copy()
        rul_finales = np.asarray(rul_finales, dtype=float)
        self.unites = np.sort(self.df["unit_number"].unique())
        if len(rul_finales) != len(self.unites):
            raise ValueError("RUL_FD001.txt ne correspond pas a test_FD001.txt")
        self.rul_finale = dict(zip(self.unites.tolist(), rul_finales.tolist()))
        self.plafond = params["data"].get("rul_cap")

        # Predictions de tous les cycles en une fois (environ 13 000 lignes)
        self.df = self.df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)
        self.df["rul_predite"] = np.maximum(modele.predict(self.df[COLONNES_FEATURES]), 0.0)
        dernier = self.df.groupby("unit_number")["time_in_cycles"].transform("max")
        reelle = self.df["unit_number"].map(self.rul_finale) + (dernier - self.df["time_in_cycles"])
        self.df["rul_reelle"] = reelle.clip(upper=self.plafond) if self.plafond else reelle

    def resume(self):
        """Une ligne par moteur, au dernier cycle enregistre."""
        derniers = self.df.groupby("unit_number").tail(1)
        return [
            {
                "moteur": int(r.unit_number),
                "nb_cycles": int(r.time_in_cycles),
                "rul_predite": round(float(r.rul_predite), 1),
                "rul_reelle": round(float(r.rul_reelle), 1),
                "statut_predit": statut(r.rul_predite, self.params),
                "statut_reel": statut(r.rul_reelle, self.params),
            }
            for r in derniers.itertuples()
        ]

    def detail(self, moteur):
        lignes = self.df[self.df["unit_number"] == moteur]
        if lignes.empty:
            return None
        dernier = lignes.iloc[-1]
        return {
            "moteur": int(moteur),
            "cycles": lignes["time_in_cycles"].astype(int).tolist(),
            "rul_predite_par_cycle": lignes["rul_predite"].round(1).tolist(),
            "rul_reelle_par_cycle": lignes["rul_reelle"].round(1).tolist(),
            "capteurs": {c: float(dernier[c]) for c in COLONNES_FEATURES},
            "rul_predite": round(float(dernier["rul_predite"]), 1),
            "rul_reelle": round(float(dernier["rul_reelle"]), 1),
        }
