# src/ml/analyse_erreurs.py
"""Analyse des erreurs des modeles compares (a lancer apres src.ml.train).

Produit dans reports/ :
- analyse_erreurs.json : erreur par tranche de RUL reelle et par modele,
  matrice des statuts du champion, moteurs de test les plus mal predits ;
- fig_predit_vs_reel.png : RUL predite contre RUL reelle (test officiel) ;
- fig_rmse_par_tranche.png : RMSE par tranche de RUL reelle (validation).
Les fichiers sont aussi enregistres dans le run MLflow du champion.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import charger_params, chemin_projet  # noqa: E402
from src.ml.data import charger_test_officiel, preparer_donnees  # noqa: E402

TRANCHES = [0, 25, 50, 75, 100, 126]
ETIQUETTES = ["0-24", "25-49", "50-74", "75-99", "100-125"]
COULEURS = {"ridge": "#2a78d6", "random_forest": "#eb6834", "hist_gradient_boosting": "#1baf7a"}
GRIS, CRITIQUE, TEXTE = "#8a8984", "#d03b3b", "#52514e"


def statut(rul, params):
    rul = np.asarray(rul, dtype=float)
    return np.where(
        rul < params["api"]["seuil_critique"], "CRITIQUE",
        np.where(rul < params["api"]["seuil_attention"], "SURVEILLANCE", "NORMAL"),
    )


def erreurs_par_tranche(y, y_pred):
    """RMSE, biais (prediction - realite) et effectif par tranche de RUL reelle."""
    df = pd.DataFrame({"y": np.asarray(y, float), "erreur": np.asarray(y_pred, float) - np.asarray(y, float)})
    df["tranche"] = pd.cut(df["y"], TRANCHES, right=False, labels=ETIQUETTES)
    lignes = []
    for tranche, g in df.groupby("tranche", observed=False):
        if g.empty:
            continue
        lignes.append({
            "tranche_rul_reelle": str(tranche),
            "n": int(len(g)),
            "rmse": round(float(np.sqrt((g["erreur"] ** 2).mean())), 2),
            "biais": round(float(g["erreur"].mean()), 2),
        })
    return lignes


def matrice_statuts(y, y_pred, params):
    ordre = ["CRITIQUE", "SURVEILLANCE", "NORMAL"]
    m = pd.crosstab(
        pd.Categorical(statut(y, params), ordre),
        pd.Categorical(statut(y_pred, params), ordre),
        rownames=["reel"], colnames=["predit"], dropna=False,
    )
    return {r: {c: int(m.loc[r, c]) for c in ordre} for r in ordre}


def _style(ax):
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color("#c9c8c2")
    ax.tick_params(colors=TEXTE, labelsize=9)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def figure_predit_vs_reel(y, y_pred, params, chemin):
    y, y_pred = np.asarray(y, float), np.asarray(y_pred, float)
    seuil = params["api"]["seuil_critique"]
    manquees = (y < seuil) & (y_pred >= seuil)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    ax.plot([0, 130], [0, 130], color=GRIS, linewidth=1.2, linestyle="--", label="Prediction parfaite")
    ax.axvline(seuil, color="#c9c8c2", linewidth=1)
    ax.axhline(seuil, color="#c9c8c2", linewidth=1)
    ax.scatter(y[~manquees], y_pred[~manquees], s=28, color="#2a78d6",
               edgecolor="white", linewidth=0.6, label="Moteur de test")
    ax.scatter(y[manquees], y_pred[manquees], s=40, marker="D", color=CRITIQUE,
               edgecolor="white", linewidth=0.6, label="Alerte critique manquee")
    ax.text(seuil + 1, 2, f"seuil critique ({seuil})", color=TEXTE, fontsize=8)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 130)
    ax.set_xlabel("RUL reelle (cycles)", color=TEXTE)
    ax.set_ylabel("RUL predite (cycles)", color=TEXTE)
    ax.set_title("RUL predite et reelle, jeu de test officiel FD001", color="#0b0b0b", fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


def figure_rmse_par_tranche(tableau, chemin):
    modeles = list(tableau)
    x = np.arange(len(ETIQUETTES))
    largeur = 0.8 / len(modeles)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for i, nom in enumerate(modeles):
        valeurs = {t["tranche_rul_reelle"]: t["rmse"] for t in tableau[nom]}
        ax.bar(x + (i - (len(modeles) - 1) / 2) * largeur,
               [valeurs.get(e, 0) for e in ETIQUETTES], largeur * 0.92,
               color=COULEURS.get(nom, GRIS), label=nom, edgecolor="white", linewidth=1)
    ax.set_xticks(x, ETIQUETTES)
    ax.set_xlabel("Tranche de RUL reelle (cycles)", color=TEXTE)
    ax.set_ylabel("RMSE (cycles)", color=TEXTE)
    ax.set_title("Erreur par tranche de RUL, validation", color="#0b0b0b", fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


def analyser(params=None):
    params = params or charger_params()
    dossier = chemin_projet("reports")
    dossier.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(chemin_projet(params["training"]["mlruns_dir"]).as_uri())

    _, X_val, _, y_val = preparer_donnees(params)
    comparaison = json.loads(
        chemin_projet(params["training"]["comparaison_path"]).read_text(encoding="utf-8")
    )
    par_modele = {}
    for c in comparaison:
        modele = mlflow.sklearn.load_model(f"runs:/{c['run_id']}/model")
        par_modele[c["type_modele"]] = erreurs_par_tranche(y_val, modele.predict(X_val))

    champion = joblib.load(chemin_projet(params["training"]["model_path"]))
    nom_champion = min(comparaison, key=lambda c: c["rmse"])["type_modele"]
    X_test, y_test = charger_test_officiel(params)
    pred_test = champion.predict(X_test)
    pred_val = champion.predict(X_val)

    # Une ligne par moteur de test, dans l'ordre des numeros (1 a 100)
    pires = pd.DataFrame({
        "moteur": np.arange(1, len(pred_test) + 1),
        "rul_reelle": np.asarray(y_test, float),
        "rul_predite": pred_test,
    })
    pires["erreur"] = pires["rul_predite"] - pires["rul_reelle"]
    pires = pires.reindex(pires["erreur"].abs().sort_values(ascending=False).index).head(5)

    resultat = {
        "champion": nom_champion,
        "erreur_par_tranche_validation": par_modele,
        "matrice_statuts_validation_champion": matrice_statuts(y_val, pred_val, params),
        "matrice_statuts_test_officiel_champion": matrice_statuts(y_test, pred_test, params),
        "pires_moteurs_test_officiel": [
            {"moteur": int(r.moteur), "rul_reelle": round(float(r.rul_reelle), 1),
             "rul_predite": round(float(r.rul_predite), 1), "erreur": round(float(r.erreur), 1)}
            for r in pires.itertuples()
        ],
    }
    chemin_json = dossier / "analyse_erreurs.json"
    chemin_json.write_text(json.dumps(resultat, indent=2, ensure_ascii=False), encoding="utf-8")
    figure_predit_vs_reel(y_test, pred_test, params, dossier / "fig_predit_vs_reel.png")
    figure_rmse_par_tranche(par_modele, dossier / "fig_rmse_par_tranche.png")

    run_champion = next(c["run_id"] for c in comparaison if c["type_modele"] == nom_champion)
    with mlflow.start_run(run_id=run_champion):
        for f in ("analyse_erreurs.json", "fig_predit_vs_reel.png", "fig_rmse_par_tranche.png"):
            mlflow.log_artifact(str(dossier / f), artifact_path="analyse_erreurs")
    return resultat


if __name__ == "__main__":
    print(json.dumps(analyser(), indent=2, ensure_ascii=False))
