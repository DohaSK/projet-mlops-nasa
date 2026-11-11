# src/assistant/evaluation.py
"""Evaluation de l'assistant et Quality Gate, suivie dans MLflow.

Chaque question du jeu data/eval/questions_assistant.yaml est posee sur trois
moteurs du jeu de test officiel (RUL predite la plus faible, mediane et la
plus elevee), pour couvrir les statuts CRITIQUE, SURVEILLANCE et NORMAL.

Metriques :
- taux_refus_hors_perimetre : questions hors sujet ou d'injection refusees ;
- taux_reponse_dans_perimetre : questions de maintenance non refusees ;
- taux_citation : reponses (dans le perimetre) avec au moins une reference ;
- taux_secours : reponses ou un garde-fou a force le mode extractif ;
- latence moyenne et P95, tokens et cout estime.
Usage : python -m src.assistant.evaluation (code 1 si le Quality Gate echoue).
"""

import json
import time
from collections import deque

import joblib
import mlflow
import numpy as np
import yaml
from dotenv import load_dotenv

from src.assistant.assistant import repondre
from src.assistant.capteurs import capteurs_anormaux
from src.assistant.llm import creer_client
from src.config import charger_params, chemin_projet
from src.ml.data import charger_test_officiel


def charger_questions(params):
    chemin = chemin_projet(params["assistant"]["questions_eval_path"])
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def statut_de(rul, params):
    if rul < params["api"]["seuil_critique"]:
        return "CRITIQUE"
    if rul < params["api"]["seuil_attention"]:
        return "SURVEILLANCE"
    return "NORMAL"


def choisir_moteurs(modele, X):
    """Trois moteurs : RUL predite minimale, mediane et maximale."""
    predictions = modele.predict(X)
    ordre = np.argsort(predictions)
    choix = [ordre[0], ordre[len(ordre) // 2], ordre[-1]]
    return [(X.iloc[i].to_dict(), round(max(float(predictions[i]), 0.0), 1)) for i in choix]


def calculer_metriques(resultats, questions, quality_gate):
    """Metriques agregees a partir des resultats de `repondre`."""
    attendu = {q["id"]: q["dans_perimetre"] for q in questions}
    dedans = [r for r in resultats if attendu[r["id"]]]
    dehors = [r for r in resultats if not attendu[r["id"]]]
    repondues = [r for r in dedans if not r["refus"]]
    latences = [r["latence_ms"] for r in resultats]

    def taux(liste, condition):
        return round(sum(1 for r in liste if condition(r)) / len(liste), 3) if liste else 0.0

    m = {
        "nb_requetes": len(resultats),
        "taux_refus_hors_perimetre": taux(dehors, lambda r: r["refus"]),
        "taux_reponse_dans_perimetre": taux(dedans, lambda r: not r["refus"]),
        "taux_citation": taux(repondues, lambda r: bool(r["references"])),
        "taux_secours": taux(repondues, lambda r: r["mode"] == "extractif_secours"),
        "latence_moyenne_ms": round(float(np.mean(latences)), 1) if latences else 0.0,
        "latence_p95_ms": round(float(np.percentile(latences, 95)), 1) if latences else 0.0,
        "tokens_total": sum(r["tokens_entree"] + r["tokens_sortie"] for r in resultats),
        "cout_total_usd": round(sum(r["cout_estime_usd"] for r in resultats), 6),
    }
    gate = quality_gate
    m["quality_gate_passed"] = bool(
        m["taux_refus_hors_perimetre"] >= gate["min_taux_refus_hors_perimetre"]
        and m["taux_reponse_dans_perimetre"] >= gate["min_taux_reponse_dans_perimetre"]
        and m["taux_citation"] >= gate["min_taux_citation"]
        and m["latence_p95_ms"] <= gate["max_latence_p95_ms"]
    )
    return m


def attendre_debit(historique, limite, estimation, horloge=time.monotonic, dormir=time.sleep):
    """Attend que la prochaine question tienne dans la limite de tokens/minute.

    `historique` : deque de (instant, tokens) des appels de la derniere minute.
    Retourne le temps attendu en secondes (exclu de la latence mesuree).
    """
    attente = 0.0
    while True:
        maintenant = horloge()
        while historique and maintenant - historique[0][0] >= 60:
            historique.popleft()
        if sum(t for _, t in historique) + estimation <= limite or not historique:
            return attente
        delai = 60 - (maintenant - historique[0][0]) + 0.1
        dormir(delai)
        attente += delai


def evaluer_assistant(params=None, client="auto", rechercher_fn=None):
    load_dotenv()
    params = params or charger_params()
    config = params["assistant"]
    if client == "auto":
        client = creer_client(params)

    modele = joblib.load(chemin_projet(params["training"]["model_path"]))
    X_test, _ = charger_test_officiel(params)
    moteurs = choisir_moteurs(modele, X_test)
    questions = charger_questions(params)

    resultats = []
    historique = deque()
    attente_totale = 0.0
    for q in questions:
        for capteurs, rul in moteurs:
            if client is not None:
                attente_totale += attendre_debit(
                    historique, config["limite_tokens_par_minute"],
                    config["tokens_estimes_par_question"],
                )
            anomalies = capteurs_anormaux(modele, capteurs, config["nb_capteurs_anormaux"])
            r = repondre(
                q["question"], rul, statut_de(rul, params), anomalies, params,
                client=client, rechercher_fn=rechercher_fn,
            )
            historique.append((time.monotonic(), r["tokens_entree"] + r["tokens_sortie"]))
            resultats.append({"id": q["id"], **r})

    metriques = calculer_metriques(resultats, questions, config["quality_gate"])
    metriques["attente_debit_s"] = round(attente_totale, 1)
    mode = "llm" if client is not None else "extractif"

    mlflow.set_tracking_uri(chemin_projet(params["training"]["mlruns_dir"]).as_uri())
    mlflow.set_experiment(params["training"]["experiment_name"])
    with mlflow.start_run(run_name=f"assistant-eval-{mode}"):
        mlflow.log_params(
            {
                "assistant_mode": mode,
                "assistant_modele": config["modele"] if client else "aucun",
                "assistant_top_k": config["top_k"],
                "assistant_temperature": config["temperature"],
                "assistant_nb_questions": len(questions),
            }
        )
        mlflow.log_metrics(
            {k: float(v) for k, v in metriques.items() if k != "quality_gate_passed"}
        )
        mlflow.set_tag("quality_gate_assistant", "passed" if metriques["quality_gate_passed"] else "failed")
        details = chemin_projet("models/assistant_eval_details.json")
        details.parent.mkdir(parents=True, exist_ok=True)
        details.write_text(json.dumps(resultats, ensure_ascii=False, indent=1), encoding="utf-8")
        mlflow.log_artifact(str(details), artifact_path="assistant")
        mlflow.log_artifact(str(chemin_projet(config["questions_eval_path"])), artifact_path="assistant")

    sortie = {"mode": mode, **metriques}
    with open(chemin_projet(config["metrics_path"]), "w", encoding="utf-8") as f:
        json.dump(sortie, f, indent=2)
    return sortie


if __name__ == "__main__":
    res = evaluer_assistant()
    print(json.dumps(res, indent=2))
    raise SystemExit(0 if res["quality_gate_passed"] else 1)
