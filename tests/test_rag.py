# tests/test_rag.py
import copy

import faiss
import pytest

from src.config import charger_params, chemin_projet
from src.rag.generation import (
    extraire_references,
    generer_reponse,
    selectionner_phrases,
)
from src.rag.ingestion import construire_index, creer_chunks, decouper_texte
from src.rag.retrieval import rechercher

PARAMS = charger_params()
PDF = chemin_projet(PARAMS["rag"]["pdf_path"])
necessite_pdf = pytest.mark.skipif(not PDF.exists(), reason="PDF NASA absent")


# Tests sans PDF (toujours executes, y compris en CI)
def test_decouper_texte_chevauchement():
    texte = " ".join(f"mot{i}" for i in range(100))
    chunks = decouper_texte(texte, taille=40, chevauchement=10)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 40 for c in chunks)
    # les 10 derniers mots d'un chunk reviennent au debut du suivant
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]


def test_decouper_texte_parametres_invalides():
    with pytest.raises(ValueError):
        decouper_texte("a b c", taille=10, chevauchement=10)


def test_creer_chunks_garde_section_et_page():
    sections = [{"section": "IV. TEST", "page": 3, "texte": " ".join(["mot"] * 60)}]
    chunks = creer_chunks(sections, taille=30, chevauchement=5)
    assert chunks
    assert all(c["section"] == "IV. TEST" and c["page"] == 3 for c in chunks)


def test_generation_extractive_et_references():
    chunks = [
        {
            "texte": (
                "The HPC outlet temperature T30 increases as the compressor "
                "efficiency drops over time. Unrelated sentence about the weather "
                "in another city today."
            ),
            "section": "IV. DAMAGE PROPAGATION",
            "page": 4,
            "score": 0.8,
        }
    ]
    question = "Which sensors are affected by HPC degradation?"
    phrases = selectionner_phrases(question, chunks, nb_phrases=1)
    assert "HPC" in phrases[0]
    reponse = generer_reponse(question, chunks, PARAMS)
    assert reponse.startswith("D'apres la documentation NASA")
    assert extraire_references(chunks) == ["IV. DAMAGE PROPAGATION (p. 4)"]


def test_generation_sans_chunks():
    assert "Aucun passage" in generer_reponse("question", [], PARAMS)


# Tests avec le PDF (ignores s'il est absent)
@pytest.fixture(scope="module")
def params_index_temporaire(tmp_path_factory):
    dossier = tmp_path_factory.mktemp("rag")
    params = copy.deepcopy(PARAMS)
    params["rag"]["index_path"] = str(dossier / "faiss_index.bin")
    params["rag"]["chunks_path"] = str(dossier / "chunks.json")
    construire_index(params)
    return params


@necessite_pdf
def test_ingestion_cree_un_index_non_vide(params_index_temporaire):
    index = faiss.read_index(params_index_temporaire["rag"]["index_path"])
    assert index.ntotal > 0


@necessite_pdf
def test_retrieval_retourne_des_mots_cles(params_index_temporaire):
    question = (
        "Which engine modules and sensors are affected by HPC degradation "
        "in the C-MAPSS model?"
    )
    chunks = rechercher(question, params_index_temporaire)
    assert len(chunks) > 0
    texte = " ".join(c["texte"] for c in chunks).lower()
    assert any(mot in texte for mot in ["hpc", "health index", "c-mapss"])