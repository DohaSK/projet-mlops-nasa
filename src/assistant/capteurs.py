# src/assistant/capteurs.py
"""Capteurs qui s'ecartent le plus des donnees d'entrainement.

On reutilise le StandardScaler du pipeline : z = (x - moyenne) / ecart-type,
avec la moyenne et l'ecart-type appris sur les moteurs d'entrainement. Un
|z| eleve signifie que la mesure est inhabituelle par rapport a la flotte.
C'est une indication pour l'humain, pas une explication causale du modele.
"""

import numpy as np

from src.ml.data import COLONNES_CAPTEURS, COLONNES_FEATURES

# Table 2 du papier NASA (Saxena et al., 2008), dans l'ordre s1 a s21
DESCRIPTION_CAPTEURS = {
    "s1": ("T2", "temperature totale en entree du fan"),
    "s2": ("T24", "temperature totale en sortie du LPC"),
    "s3": ("T30", "temperature totale en sortie du HPC"),
    "s4": ("T50", "temperature totale en sortie de la LPT"),
    "s5": ("P2", "pression en entree du fan"),
    "s6": ("P15", "pression totale dans le conduit de derivation"),
    "s7": ("P30", "pression totale en sortie du HPC"),
    "s8": ("Nf", "vitesse physique du fan"),
    "s9": ("Nc", "vitesse physique du coeur"),
    "s10": ("epr", "rapport de pression moteur (P50/P2)"),
    "s11": ("Ps30", "pression statique en sortie du HPC"),
    "s12": ("phi", "rapport debit carburant / Ps30"),
    "s13": ("NRf", "vitesse corrigee du fan"),
    "s14": ("NRc", "vitesse corrigee du coeur"),
    "s15": ("BPR", "taux de dilution (bypass ratio)"),
    "s16": ("farB", "rapport carburant-air du bruleur"),
    "s17": ("htBleed", "enthalpie de prelevement"),
    "s18": ("Nf_dmd", "vitesse du fan demandee"),
    "s19": ("PCNfR_dmd", "vitesse corrigee du fan demandee"),
    "s20": ("W31", "prelevement de refroidissement HPT"),
    "s21": ("W32", "prelevement de refroidissement LPT"),
}

# En dessous de cette variance, le capteur est considere constant (s1, s5,
# s10, s16, s18 et s19 sur FD001) : un z-score n'y a pas de sens.
VARIANCE_MIN = 1e-8


def _scaler(modele):
    """Retourne le StandardScaler du pipeline, ou None s'il n'existe pas."""
    etapes = getattr(modele, "named_steps", {})
    scaler = etapes.get("scaler")
    if scaler is None or not hasattr(scaler, "mean_"):
        return None
    return scaler


def capteurs_anormaux(modele, capteurs, nb=3):
    """Les `nb` capteurs avec le plus grand |z-score|.

    `capteurs` : dictionnaire {setting1..3, s1..s21: valeur}.
    Retourne une liste de dictionnaires tries par |z| decroissant.
    """
    scaler = _scaler(modele)
    if scaler is None:
        return []

    resultats = []
    for i, colonne in enumerate(COLONNES_FEATURES):
        if colonne not in COLONNES_CAPTEURS or scaler.var_[i] < VARIANCE_MIN:
            continue
        z = (float(capteurs[colonne]) - scaler.mean_[i]) / np.sqrt(scaler.var_[i])
        symbole, description = DESCRIPTION_CAPTEURS[colonne]
        resultats.append(
            {
                "capteur": colonne,
                "symbole": symbole,
                "description": description,
                "valeur": round(float(capteurs[colonne]), 4),
                "moyenne_entrainement": round(float(scaler.mean_[i]), 4),
                "z_score": round(float(z), 2),
                "sens": "au-dessus" if z > 0 else "en dessous",
            }
        )
    resultats.sort(key=lambda r: abs(r["z_score"]), reverse=True)
    return resultats[:nb]


def decrire_capteurs(anomalies):
    """Texte court pour le prompt, une ligne par capteur."""
    if not anomalies:
        return "Aucune information sur les capteurs."
    return "\n".join(
        f"- {a['capteur']} ({a['symbole']}, {a['description']}) : "
        f"{a['valeur']}, {a['sens']} de la moyenne d'entrainement "
        f"(z = {a['z_score']})"
        for a in anomalies
    )
