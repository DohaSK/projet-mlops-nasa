# src/journal.py
"""Journal structure des requetes de l'API (une ligne JSON par requete).

Le fichier logs/requetes.jsonl sert de source au monitoring (monitor.py) :
disponibilite, erreurs, latence, statuts predits, mode de l'assistant,
tokens consommes et valeurs des capteurs recues (pour la derive).
Aucune donnee personnelle n'est enregistree.
"""

import json
import threading
from datetime import datetime, timezone

from src.config import chemin_projet

_verrou = threading.Lock()


def chemin_journal(params):
    return chemin_projet(params["monitoring"]["journal_path"])


def journaliser(evenement, params):
    """Ajoute un evenement au journal. Ne fait jamais echouer une requete."""
    ligne = {"horodatage": datetime.now(timezone.utc).isoformat(), **evenement}
    try:
        chemin = chemin_journal(params)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        with _verrou, open(chemin, "a", encoding="utf-8") as f:
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    except OSError:
        pass


def lire_journal(params):
    chemin = chemin_journal(params)
    if not chemin.exists():
        return []
    with open(chemin, encoding="utf-8") as f:
        return [json.loads(ligne) for ligne in f if ligne.strip()]
