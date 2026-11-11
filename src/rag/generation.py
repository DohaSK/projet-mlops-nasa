# src/rag/generation.py
"""Generation de l'explication a partir des passages recuperes.

Choix technique : par defaut (mode "template"), la reponse est extractive. On
decoupe les passages en phrases, on garde celles qui recouvrent le mieux la
question et le vocabulaire du domaine (HPC, T30, P30, efficiency...), puis on
les insere dans une phrase d'introduction fixe. Aucune phrase n'est inventee,
ce qui limite le risque d'hallucination sur un sujet critique.
Option "flan_t5" : reformulation par google/flan-t5-small en local (necessite
sentencepiece). En cas d'erreur, on retombe sur le template.
"""

import re
from functools import lru_cache

MOTS_CLES_DOMAINE = {
    "hpc", "hpt", "lpc", "lpt", "fan", "efficiency", "flow", "health", "index",
    "degradation", "temperature", "pressure", "t24", "t30", "t50", "p30",
    "nf", "nc", "c-mapss", "module", "modules", "sensor", "sensors", "fault",
}
MOTS_VIDES = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "which", "by",
    "for", "on", "with", "that", "this", "as", "be", "at", "from", "what", "how",
    "does", "do", "were", "was", "it", "its", "most",
}


def _mots(texte):
    return set(re.findall(r"[a-z0-9][a-z0-9\-]*", texte.lower())) - MOTS_VIDES


def decouper_phrases(texte):
    """Phrases de 6 a 60 mots (on ecarte les fragments et les pavés)."""
    phrases = re.split(r"(?<=[.!?])\s+", texte)
    return [p.strip() for p in phrases if 6 <= len(p.split()) <= 60]


def selectionner_phrases(question, chunks, nb_phrases):
    """Phrases les plus pertinentes : recouvrement avec la question et le domaine."""
    mots_question = _mots(question)
    candidates = []
    vues = set()
    for rang, chunk in enumerate(chunks):
        for phrase in decouper_phrases(chunk["texte"]):
            cle = phrase.lower()
            if cle in vues:
                continue
            vues.add(cle)
            mots = _mots(phrase)
            score = (
                len(mots & mots_question)
                + 0.5 * len(mots & MOTS_CLES_DOMAINE)
                + chunk.get("score", 0.0)
            )
            candidates.append((score, -rang, phrase))
    candidates.sort(reverse=True)
    return [phrase for _, _, phrase in candidates[:nb_phrases]]


@lru_cache(maxsize=1)
def _charger_flan(nom_modele):
    from transformers import pipeline

    return pipeline("text2text-generation", model=nom_modele)


def _generer_flan_t5(question, chunks, g):
    contexte = " ".join(c["texte"] for c in chunks[:2])[:1500]
    prompt = (
        "Answer the question using only the context.\n"
        f"Context: {contexte}\nQuestion: {question}"
    )
    sortie = _charger_flan(g["flan_t5_model"])(prompt, max_new_tokens=120)
    return (
        "D'apres la documentation NASA (modele C-MAPSS, reformule par "
        f"flan-t5-small) : {sortie[0]['generated_text']}"
    )


def generer_reponse(question, chunks, params):
    """Construit la reponse a partir de la question et des chunks recuperes."""
    if not chunks:
        return "Aucun passage pertinent trouve dans la documentation NASA."

    g = params.get("generation", {})
    if g.get("mode") == "flan_t5":
        try:
            return _generer_flan_t5(question, chunks, g)
        except Exception:
            pass  # retombe sur le template

    phrases = selectionner_phrases(question, chunks, g.get("nb_phrases", 4))
    contexte = " ".join(phrases) if phrases else chunks[0]["texte"][:400]
    return f"D'apres la documentation NASA (modele C-MAPSS) : {contexte}"


def extraire_references(chunks):
    """Liste dedoublonnee 'Section (p. N)' dans l'ordre de pertinence."""
    references = []
    for c in chunks:
        ref = f"{c['section']} (p. {c['page']})"
        if ref not in references:
            references.append(ref)
    return references