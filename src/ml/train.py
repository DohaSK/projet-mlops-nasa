# src/ml/train.py
"""Entrainement et comparaison des modeles, suivi MLflow unifie (ML et RAG),
Quality Gate et enregistrement dans le MLflow Model Registry.

Un run parent "comparaison-fd001" contient un run enfant par modele. Le
champion est le modele de plus faible RMSE de validation. Il est enregistre
dans le registre "rul-fd001" avec l'alias "candidat" ; l'alias "champion"
ne lui est donne que s'il passe le Quality Gate.
"""

import json
import time

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.config import charger_params, chemin_projet
from src.ml.data import charger_test_officiel, preparer_donnees
from src.ml.model import (
    construire_modele,
    evaluer,
    metriques_securite,
    raisons_echec_gate,
)
from src.rag.ingestion import construire_index
from src.rag.retrieval import rechercher


def _test_officiel_disponible(params):
    return (
        chemin_projet(params["data"]["test_path"]).exists()
        and chemin_projet(params["data"]["rul_path"]).exists()
    )


def _metriques_test_officiel(modele, params):
    """Evalue sur le dernier cycle de test_FD001 si les fichiers sont presents."""
    if not _test_officiel_disponible(params):
        return {}
    X, y = charger_test_officiel(params)
    m = evaluer(modele, X, y)
    s = metriques_securite(y, modele.predict(X), params["api"]["seuil_critique"])
    return {
        "rmse_test_officiel": m["rmse"],
        "r2_test_officiel": m["r2"],
        "taux_alertes_manquees_test_officiel": s["taux_alertes_manquees"],
    }


def _metriques_rag(params):
    """Nombre de chunks, duree d'ingestion et temps de retrieval du RAG."""
    r = params["rag"]
    chemin_index = chemin_projet(r["index_path"])
    if not chemin_index.exists():
        if not chemin_projet(r["pdf_path"]).exists():
            return {}
        construire_index(params)

    chemin_stats = chemin_index.parent / "stats.json"
    stats = {}
    if chemin_stats.exists():
        stats = json.loads(chemin_stats.read_text(encoding="utf-8"))

    question = r["question_critique"]
    rechercher(question, params)  # premier appel : chargement du modele d'embeddings
    debut = time.perf_counter()
    rechercher(question, params)
    duree_ms = (time.perf_counter() - debut) * 1000
    return {
        "rag_nb_chunks": float(stats.get("nb_chunks", 0)),
        "rag_duree_ingestion_s": float(stats.get("duree_ingestion_s", 0.0)),
        "rag_retrieval_ms": round(duree_ms, 2),
    }


def _entrainer_candidat(type_modele, params, X_train, X_val, y_train, y_val):
    """Entraine un modele dans un run MLflow enfant et retourne ses resultats."""
    modele = construire_modele(params, type_modele)
    with mlflow.start_run(run_name=type_modele, nested=True) as run:
        estimateur = modele.steps[-1][1]
        mlflow.log_param("type_modele", type_modele)
        mlflow.log_params({f"hp_{k}": v for k, v in estimateur.get_params().items()
                           if isinstance(v, (int, float, str, bool)) or v is None})

        debut = time.perf_counter()
        modele.fit(X_train, y_train)
        duree = time.perf_counter() - debut

        metriques = evaluer(modele, X_val, y_val)
        metriques.update(
            metriques_securite(y_val, modele.predict(X_val), params["api"]["seuil_critique"])
        )
        train = evaluer(modele, X_train, y_train)
        metriques.update({"rmse_train": train["rmse"], "r2_train": train["r2"]})
        metriques.update(_metriques_test_officiel(modele, params))
        metriques["duree_entrainement_s"] = round(duree, 2)
        mlflow.log_metrics(metriques)

        exemple = X_val.head(5)
        mlflow.sklearn.log_model(
            modele,
            artifact_path="model",
            signature=infer_signature(exemple, modele.predict(exemple)),
            input_example=exemple,
        )
    return {"type_modele": type_modele, "run_id": run.info.run_id,
            "modele": modele, "metriques": metriques}


def _enregistrer_registry(champion, gate_ok, params):
    """Enregistre le champion dans le Model Registry et pose les alias."""
    reg = params["registry"]
    version = mlflow.register_model(f"runs:/{champion['run_id']}/model", reg["nom_modele"])
    client = MlflowClient()
    client.set_registered_model_alias(reg["nom_modele"], reg["alias_candidat"], version.version)
    client.set_model_version_tag(reg["nom_modele"], version.version, "quality_gate",
                                 "passed" if gate_ok else "failed")
    client.set_model_version_tag(reg["nom_modele"], version.version, "type_modele",
                                 champion["type_modele"])
    if gate_ok:
        client.set_registered_model_alias(reg["nom_modele"], reg["alias_champion"], version.version)
    return int(version.version)


def entrainer(params=None, enregistrer=True):
    """Compare les modeles, choisit le champion, applique le Quality Gate."""
    params = params or charger_params()
    X_train, X_val, y_train, y_val = preparer_donnees(params)

    # Tracking local dans ./mlruns (le registre utilise le meme stockage)
    mlflow.set_tracking_uri(chemin_projet(params["training"]["mlruns_dir"]).as_uri())
    mlflow.set_experiment(params["training"]["experiment_name"])

    with mlflow.start_run(run_name="comparaison-fd001"):
        mlflow.log_params({f"data_{k}": v for k, v in params["data"].items()})
        mlflow.log_params({f"gate_{k}": v for k, v in params["quality_gate"].items()})
        mlflow.log_params({f"rag_{k}": v for k, v in params["rag"].items()
                           if k in ("chunk_size", "chunk_overlap", "top_k", "embedding_model")})

        resultats = [
            _entrainer_candidat(t, params, X_train, X_val, y_train, y_val)
            for t in params["comparaison"]["candidats"]
        ]
        baseline = next(
            (r["metriques"] for r in resultats
             if r["type_modele"] == params["comparaison"]["baseline"]),
            None,
        )
        champion = min(resultats, key=lambda r: r["metriques"]["rmse"])
        raisons = raisons_echec_gate(champion["metriques"], params, baseline)
        gate_ok = not raisons

        # Vue unifiee dans le run parent : metriques du champion et du RAG
        mlflow.log_metrics({k: v for k, v in champion["metriques"].items()})
        rag = _metriques_rag(params)
        if rag:
            mlflow.log_metrics(rag)
            mlflow.log_artifact(
                str(chemin_projet(params["rag"]["index_path"])), artifact_path="rag"
            )
        mlflow.set_tag("champion", champion["type_modele"])
        mlflow.set_tag("quality_gate", "passed" if gate_ok else "failed")
        if raisons:
            mlflow.set_tag("quality_gate_raisons", " ; ".join(raisons))

        # Tableau de comparaison
        comparaison = [
            {"type_modele": r["type_modele"], "run_id": r["run_id"],
             **{k: round(v, 4) for k, v in r["metriques"].items()}}
            for r in resultats
        ]
        chemin_comp = chemin_projet(params["training"]["comparaison_path"])
        chemin_comp.parent.mkdir(parents=True, exist_ok=True)
        chemin_comp.write_text(json.dumps(comparaison, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(chemin_comp))

        # Sauvegarde locale du champion (utilisee par l'API et DVC)
        chemin_modele = chemin_projet(params["training"]["model_path"])
        chemin_modele.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(champion["modele"], chemin_modele)
        mlflow.log_artifact(str(chemin_projet("params.yaml")))

        version = _enregistrer_registry(champion, gate_ok, params) if enregistrer else None

    m = champion["metriques"]
    resultat = {
        "champion": champion["type_modele"],
        "rmse": m["rmse"],
        "r2": m["r2"],
        "taux_alertes_manquees": m["taux_alertes_manquees"],
        **{k: m[k] for k in m if k.endswith("_test_officiel")},
        "baseline_rmse": baseline["rmse"] if baseline else None,
        **rag,
        "registry_version": version,
        "quality_gate_passed": gate_ok,
        "quality_gate_raisons": raisons,
    }
    with open(chemin_projet(params["training"]["metrics_path"]), "w", encoding="utf-8") as f:
        json.dump(resultat, f, indent=2)

    print("Comparaison (validation) :")
    for r in resultats:
        print(f"  {r['type_modele']:<24} RMSE={r['metriques']['rmse']:.2f}  "
              f"R2={r['metriques']['r2']:.3f}  "
              f"alertes manquees={r['metriques']['taux_alertes_manquees']:.1%}")
    print(f"Champion : {champion['type_modele']}  "
          f"Quality Gate={'OK' if gate_ok else 'ECHEC ' + str(raisons)}")
    return resultat


if __name__ == "__main__":
    sortie = entrainer()
    raise SystemExit(0 if sortie["quality_gate_passed"] else 1)
