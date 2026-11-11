# src/rag/ingestion.py
"""Ingestion du papier NASA : extraction PDF, chunking, embeddings, index FAISS.

Le texte est extrait avec pypdf puis nettoye : les lignes qui ressemblent a des
tableaux, des formules ou du bruit (moins de 50 % de lettres) sont ignorees, et
les mots coupes en fin de ligne sont recolles. Le texte est decoupe par section
(titres detectes par regex), puis en chunks de taille fixe avec chevauchement.
Chaque chunk garde sa section et sa page pour fournir des references.
"""

import json
import re
import time

import faiss

from src.config import charger_params, chemin_projet
from src.rag.retrieval import charger_encodeur

TITRE_SECTION = re.compile(r"^(?:[IVX]{1,5}|\d{1,2})\.\s+[A-Za-z][A-Za-z ,\-:]{3,80}$")
MIN_MOTS_CHUNK = 10


def ligne_utile(ligne):
    """Garde les lignes de texte, ecarte nombres, formules et debris de tableaux."""
    if len(ligne) < 3:
        return False
    lettres = sum(1 for c in ligne if c.isalpha())
    return lettres / len(ligne) >= 0.5


def extraire_sections(chemin_pdf):
    """Retourne (sections, nb_pages). Une section = {section, page, texte}."""
    from pypdf import PdfReader

    lecteur = PdfReader(str(chemin_pdf))
    sections = []
    courante = {"section": "En-tete et resume", "page": 1, "lignes": []}

    def cloturer(sec):
        texte = " ".join(sec["lignes"]).strip()
        if texte:
            sections.append(
                {"section": sec["section"], "page": sec["page"], "texte": texte}
            )

    for numero, page in enumerate(lecteur.pages, start=1):
        texte = page.extract_text() or ""
        texte = re.sub(r"-\n(?=[a-z])", "", texte)  # mots coupes en fin de ligne
        for ligne in texte.splitlines():
            ligne = re.sub(r"\s+", " ", ligne).strip()
            if not ligne_utile(ligne):
                continue
            if TITRE_SECTION.match(ligne):
                cloturer(courante)
                courante = {"section": ligne, "page": numero, "lignes": []}
            else:
                courante["lignes"].append(ligne)
    cloturer(courante)
    return sections, len(lecteur.pages)


def decouper_texte(texte, taille, chevauchement):
    """Decoupe un texte en chunks de `taille` mots avec chevauchement."""
    if chevauchement >= taille:
        raise ValueError("chunk_overlap doit etre inferieur a chunk_size")
    mots = texte.split()
    pas = taille - chevauchement
    chunks = []
    for debut in range(0, len(mots), pas):
        chunks.append(" ".join(mots[debut : debut + taille]))
        if debut + taille >= len(mots):
            break
    return chunks


def creer_chunks(sections, taille, chevauchement):
    """Chunks avec metadonnees (id, texte, section, page)."""
    chunks = []
    for sec in sections:
        for morceau in decouper_texte(sec["texte"], taille, chevauchement):
            if len(morceau.split()) < MIN_MOTS_CHUNK:
                continue
            chunks.append(
                {
                    "id": len(chunks),
                    "texte": morceau,
                    "section": sec["section"],
                    "page": sec["page"],
                }
            )
    return chunks


def construire_index(params=None):
    """Construit l'index FAISS et le sauvegarde. Retourne des statistiques."""
    params = params or charger_params()
    r = params["rag"]
    chemin_pdf = chemin_projet(r["pdf_path"])
    if not chemin_pdf.exists():
        raise FileNotFoundError(
            f"PDF introuvable : {chemin_pdf}. Placez le papier NASA dans "
            "docs/nasa_cmapss_paper.pdf."
        )

    debut = time.perf_counter()
    sections, nb_pages = extraire_sections(chemin_pdf)
    chunks = creer_chunks(sections, r["chunk_size"], r["chunk_overlap"])
    if not chunks:
        raise ValueError("Aucun texte extrait du PDF (document scanne ?).")

    encodeur = charger_encodeur(r["embedding_model"])
    vecteurs = encodeur.encode(
        [c["texte"] for c in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    # Vecteurs normalises : le produit scalaire equivaut a la similarite cosinus
    index = faiss.IndexFlatIP(vecteurs.shape[1])
    index.add(vecteurs)

    chemin_index = chemin_projet(r["index_path"])
    chemin_chunks = chemin_projet(r["chunks_path"])
    chemin_index.parent.mkdir(parents=True, exist_ok=True)
    chemin_chunks.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(chemin_index))
    chemin_chunks.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    stats = {
        "nb_pages": nb_pages,
        "nb_sections": len(sections),
        "nb_chunks": len(chunks),
        "duree_ingestion_s": round(time.perf_counter() - debut, 2),
    }
    (chemin_index.parent / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats


if __name__ == "__main__":
    print(construire_index())