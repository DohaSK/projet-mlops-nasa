# src/rag/retrieval.py
"""Recherche des passages les plus pertinents dans l'index FAISS."""

import json
from functools import lru_cache

import faiss

from src.config import charger_params, chemin_projet


@lru_cache(maxsize=2)
def charger_encodeur(nom_modele):
    """Charge le modele d'embeddings une seule fois par processus."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nom_modele)


def rechercher(question, params=None, top_k=None):
    """Retourne les top_k chunks (texte, section, page, score) pour une question."""
    params = params or charger_params()
    r = params["rag"]
    chemin_index = chemin_projet(r["index_path"])
    chemin_chunks = chemin_projet(r["chunks_path"])
    if not (chemin_index.exists() and chemin_chunks.exists()):
        raise FileNotFoundError(
            "Index RAG introuvable. Lancez : python -m src.rag.ingestion"
        )

    index = faiss.read_index(str(chemin_index))
    chunks = json.loads(chemin_chunks.read_text(encoding="utf-8"))

    k = min(top_k or r["top_k"], index.ntotal)
    encodeur = charger_encodeur(r["embedding_model"])
    vecteur = encodeur.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    scores, ids = index.search(vecteur, k)

    return [
        {**chunks[int(i)], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i >= 0
    ]