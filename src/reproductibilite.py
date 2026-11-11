# src/reproductibilite.py
"""Fiche de reproductibilite generee automatiquement.

Rassemble dans reports/reproductibilite.json : version de Python et des
paquets, commit Git, empreintes DVC des donnees, empreinte de params.yaml,
identifiants des runs MLflow (comparaison, evaluation de l'assistant) et
version du modele dans le registre. A lancer apres `dvc repro`.
Usage : python -m src.reproductibilite
"""

import hashlib
import json
import platform
import subprocess
from importlib import metadata

import yaml

from src.config import charger_params, chemin_projet

PAQUETS = [
    "numpy", "pandas", "scikit-learn", "mlflow", "fastapi", "uvicorn", "pypdf",
    "sentence-transformers", "faiss-cpu", "dvc", "groq", "pytest", "ruff",
]


def _commande(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True,
                              cwd=chemin_projet(".")).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def versions_paquets():
    versions = {}
    for p in PAQUETS:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return versions


def empreintes_dvc(params):
    empreintes = {}
    for cle in ("train_path", "test_path", "rul_path"):
        fichier_dvc = chemin_projet(params["data"][cle] + ".dvc")
        if fichier_dvc.exists():
            sortie = yaml.safe_load(fichier_dvc.read_text(encoding="utf-8"))["outs"][0]
            empreintes[params["data"][cle]] = {"md5": sortie["md5"], "taille": sortie.get("size")}
    return empreintes


def infos_mlflow(params):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(chemin_projet(params["training"]["mlruns_dir"]).as_uri())
    client = MlflowClient()
    experience = client.get_experiment_by_name(params["training"]["experiment_name"])
    if experience is None:
        return {}

    def dernier_run(nom):
        runs = client.search_runs(
            [experience.experiment_id],
            filter_string=f"attributes.run_name = '{nom}'",
            order_by=["attributes.start_time DESC"], max_results=1,
        )
        return runs[0] if runs else None

    infos = {"experience": experience.name, "experience_id": experience.experiment_id}
    parent = dernier_run("comparaison-fd001")
    if parent:
        enfants = client.search_runs(
            [experience.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
        )
        infos["run_comparaison"] = parent.info.run_id
        infos["runs_modeles"] = {r.info.run_name: r.info.run_id for r in enfants}
        infos["champion"] = parent.data.tags.get("champion")
        infos["quality_gate"] = parent.data.tags.get("quality_gate")
    for mode in ("llm", "extractif"):
        run = dernier_run(f"assistant-eval-{mode}")
        if run:
            infos[f"run_eval_assistant_{mode}"] = run.info.run_id
    reg = params["registry"]
    try:
        for alias in (reg["alias_candidat"], reg["alias_champion"]):
            v = client.get_model_version_by_alias(reg["nom_modele"], alias)
            infos[f"registry_{alias}"] = f"{reg['nom_modele']} v{v.version}"
    except Exception:
        pass
    return infos


def generer(params=None, dossier=None):
    params = params or charger_params()
    chemin_params = chemin_projet("params.yaml")
    fiche = {
        "python": platform.python_version(),
        "systeme": platform.platform(),
        "paquets": versions_paquets(),
        "git_commit": _commande(["git", "rev-parse", "HEAD"]),
        "git_modifications_non_commitees": bool(_commande(["git", "status", "--porcelain"])),
        "donnees_dvc": empreintes_dvc(params),
        "params_yaml_sha256": hashlib.sha256(chemin_params.read_bytes()).hexdigest(),
        "mlflow": infos_mlflow(params),
        "commandes": [
            "pip install -r requirements.txt",
            "dvc pull   # ou placer les fichiers FD001 dans data/raw/",
            "dvc repro  # ingestion RAG, comparaison et entrainement, analyse, evaluation assistant",
            "pytest -v",
            "python -m src.simulation --local",
            "python -m src.reproductibilite",
        ],
    }
    dossier = dossier or chemin_projet("reports")
    dossier.mkdir(exist_ok=True)
    (dossier / "reproductibilite.json").write_text(
        json.dumps(fiche, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return fiche


if __name__ == "__main__":
    print(json.dumps(generer(), indent=2, ensure_ascii=False))
