# src/config.py
"""Chargement de la configuration et chemins du projet."""

from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parent.parent


def chemin_projet(relatif):
    """Resout un chemin relatif a la racine du projet (un chemin absolu reste tel quel)."""
    return RACINE / relatif


def charger_params(chemin=None):
    """Charge params.yaml et retourne un dictionnaire."""
    chemin = Path(chemin) if chemin else RACINE / "params.yaml"
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)