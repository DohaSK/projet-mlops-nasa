# monitor.py
"""Tableau de bord simple a partir du journal de l'API (logs/requetes.jsonl).

Indicateurs : disponibilite et erreurs, latence moyenne et P95 par endpoint,
repartition des statuts predits, comportement de l'assistant (refus, mode de
secours, tokens et cout), qualite des entrees et derive des capteurs par
rapport aux donnees d'entrainement (moyenne et ecart-type du StandardScaler
du modele), globalement et par phase quand le journal en contient.

Usage : python monitor.py            (resume JSON dans reports/monitoring.json)
        python monitor.py --figure   (ajoute reports/fig_monitoring_derive.png)
"""

import argparse
import json

import joblib
import numpy as np

from src.config import charger_params, chemin_projet
from src.journal import lire_journal
from src.ml.data import COLONNES_CAPTEURS, COLONNES_FEATURES

# Decalage moyen (en ecarts-types d'entrainement) au-dela duquel on signale
# une derive. 0.5 ecart-type correspond a un decalage "modere" (d de Cohen).
SEUIL_DERIVE = 0.5


def _latences(evenements):
    latences = [e["latence_ms"] for e in evenements]
    return {
        "n": len(latences),
        "moyenne_ms": round(float(np.mean(latences)), 1),
        "p95_ms": round(float(np.percentile(latences, 95)), 1),
    }


def resume_requetes(evenements):
    if not evenements:
        return {"nb_requetes": 0}
    erreurs = [e for e in evenements if e["code_http"] >= 500]
    invalides = [e for e in evenements if 400 <= e["code_http"] < 500]
    ok = [e for e in evenements if e["code_http"] == 200]
    statuts = {}
    for e in ok:
        statuts[e.get("statut", "?")] = statuts.get(e.get("statut", "?"), 0) + 1

    resume = {
        "nb_requetes": len(evenements),
        "taux_erreur_serveur": round(len(erreurs) / len(evenements), 3),
        "taux_entrees_invalides": round(len(invalides) / len(evenements), 3),
        "latence": _latences(evenements),
        "latence_par_endpoint": {
            ep: _latences([e for e in evenements if e["endpoint"] == ep])
            for ep in sorted({e["endpoint"] for e in evenements})
        },
        "repartition_statuts": statuts,
    }

    assistant = [e for e in ok if e["endpoint"] == "/assistant"]
    if assistant:
        modes = {}
        for e in assistant:
            modes[e["mode_assistant"]] = modes.get(e["mode_assistant"], 0) + 1
        resume["assistant"] = {
            "nb_questions": len(assistant),
            "taux_refus": round(sum(e["refus"] for e in assistant) / len(assistant), 3),
            "modes": modes,
            "tokens_total": sum(e["tokens_entree"] + e["tokens_sortie"] for e in assistant),
            "cout_total_usd": round(sum(e["cout_estime_usd"] for e in assistant), 6),
        }
    return resume


def derive_capteurs(evenements, modele):
    """Decalage moyen de chaque capteur, en ecarts-types d'entrainement."""
    valeurs = [e["capteurs"] for e in evenements if e.get("capteurs")]
    scaler = getattr(modele, "named_steps", {}).get("scaler")
    if not valeurs or scaler is None:
        return {}
    derive = {}
    for i, col in enumerate(COLONNES_FEATURES):
        if col not in COLONNES_CAPTEURS or scaler.var_[i] < 1e-8:
            continue
        moyenne = np.mean([v[col] for v in valeurs])
        derive[col] = round(float((moyenne - scaler.mean_[i]) / np.sqrt(scaler.var_[i])), 2)
    return {
        "nb_observations": len(valeurs),
        "decalage_par_capteur": derive,
        "capteurs_en_derive": [c for c, d in derive.items() if abs(d) > SEUIL_DERIVE],
    }


def figure_derive(par_phase, chemin):
    """Barres horizontales : decalage moyen par capteur, une couleur par phase."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    couleurs = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    phases = [p for p in par_phase if par_phase[p]]
    capteurs = list(par_phase[phases[0]]["decalage_par_capteur"])
    y = np.arange(len(capteurs))
    hauteur = 0.8 / len(phases)
    fig, ax = plt.subplots(figsize=(7, 5.6), dpi=150)
    for i, phase in enumerate(phases):
        d = par_phase[phase]["decalage_par_capteur"]
        ax.barh(y + (i - (len(phases) - 1) / 2) * hauteur, [d[c] for c in capteurs],
                hauteur * 0.9, color=couleurs[i % len(couleurs)], label=phase)
    for s in (-SEUIL_DERIVE, SEUIL_DERIVE):
        ax.axvline(s, color="#8a8984", linestyle="--", linewidth=1)
    ax.axvline(0, color="#c9c8c2", linewidth=1)
    ax.set_yticks(y, capteurs)
    ax.invert_yaxis()
    ax.set_xlabel("Decalage moyen (ecarts-types d'entrainement)", color="#52514e")
    ax.set_title(f"Derive des capteurs par phase (alerte au-dela de +/-{SEUIL_DERIVE})",
                 color="#0b0b0b", fontsize=11, loc="left")
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


def main(figure=False):
    params = charger_params()
    evenements = lire_journal(params)
    rapport = resume_requetes(evenements)
    chemin_modele = chemin_projet(params["training"]["model_path"])
    if evenements and chemin_modele.exists():
        modele = joblib.load(chemin_modele)
        ok = [e for e in evenements if e["code_http"] == 200 and e.get("capteurs")]
        rapport["derive"] = derive_capteurs(ok, modele)
        phases = sorted({e.get("phase", "production") for e in ok})
        if len(phases) > 1:
            rapport["derive_par_phase"] = {
                p: derive_capteurs([e for e in ok if e.get("phase", "production") == p], modele)
                for p in phases
            }
    dossier = chemin_projet("reports")
    dossier.mkdir(exist_ok=True)
    (dossier / "monitoring.json").write_text(
        json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if figure and "derive" in rapport:
        par_phase = rapport.get("derive_par_phase", {"production": rapport["derive"]})
        ordre = ["production", "nominal", "derive"]
        par_phase = {p: par_phase[p] for p in ordre if p in par_phase} or par_phase
        figure_derive(par_phase, dossier / "fig_monitoring_derive.png")
    return rapport


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", action="store_true")
    print(json.dumps(main(parser.parse_args().figure), indent=2, ensure_ascii=False))
