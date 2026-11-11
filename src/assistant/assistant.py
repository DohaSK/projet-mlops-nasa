# src/assistant/assistant.py
"""Assistant de maintenance : combine la prediction ML, les capteurs
inhabituels et les passages RAG du papier NASA dans une reponse en francais.

Etapes :
1. garde-fous d'entree (injection, longueur) ;
2. analyse par le LLM : perimetre et requete de recherche en anglais
   (le modele d'embeddings et le papier sont anglophones) ;
3. recherche des passages dans l'index FAISS ;
4. generation par le LLM avec citations [n] ;
5. verification de la reponse ; en cas de probleme, on revient a la
   reponse extractive du module RAG (aucune phrase inventee).
Sans cle Groq, l'assistant fonctionne en mode extractif.
"""

import time

from src.assistant.capteurs import decrire_capteurs
from src.assistant.garde_fous import (
    citations,
    dans_perimetre_mots_cles,
    detecter_injection,
    lire_json,
    verifier_reponse,
)
from src.assistant.prompts import (
    PROMPT_ANALYSE,
    PROMPT_SYSTEME,
    construire_message_utilisateur,
)
from src.rag.generation import extraire_references, generer_reponse

MESSAGE_HORS_PERIMETRE = (
    "Je reponds uniquement aux questions sur l'etat de ce moteur, sa duree de "
    "vie restante, ses capteurs et sa maintenance."
)
MESSAGE_INJECTION = (
    "Cette demande ne peut pas etre traitee : elle cherche a modifier les "
    "regles de l'assistant."
)


def _entete(rul, statut):
    return f"RUL predite par le modele : {rul} cycles (statut {statut})."


def _reponse_extractive(requete, passages, anomalies, rul, statut, params):
    """Reponse sans LLM : en-tete, capteurs et phrases du papier NASA."""
    parties = [_entete(rul, statut)]
    if anomalies:
        noms = ", ".join(
            f"{a['capteur']} ({a['symbole']}, {a['sens']} de la moyenne)"
            for a in anomalies
        )
        parties.append(f"Capteurs les plus inhabituels : {noms}.")
    if passages:
        parties.append(generer_reponse(requete, passages, params))
    parties.append("Decision finale a valider par un expert de maintenance.")
    return " ".join(parties)


def _cout(usage, config):
    return round(
        usage["tokens_entree"] * config["prix_entree_usd_1m"] / 1e6
        + usage["tokens_sortie"] * config["prix_sortie_usd_1m"] / 1e6,
        6,
    )


def repondre(question, rul, statut, anomalies, params, client=None, rechercher_fn=None):
    """Repond a la question du technicien pour un moteur deja predit.

    `client` : objet avec `completer(messages)` (Groq ou faux client de test).
    `rechercher_fn(requete, params, top_k)` : recherche RAG (FAISS par defaut).
    """
    if rechercher_fn is None:
        from src.rag.retrieval import rechercher as rechercher_fn

    debut = time.perf_counter()
    config = params["assistant"]
    seuils = (params["api"]["seuil_critique"], params["api"]["seuil_attention"])
    usage = {"tokens_entree": 0, "tokens_sortie": 0}
    resultat = {
        "question": question,
        "rul_predite": rul,
        "statut": statut,
        "capteurs_anormaux": anomalies,
        "reponse": "",
        "references": [],
        "mode": "extractif" if client is None else "llm",
        "refus": False,
        "garde_fous": [],
    }

    def terminer():
        resultat.update(usage)
        resultat["cout_estime_usd"] = _cout(usage, config)
        resultat["latence_ms"] = round((time.perf_counter() - debut) * 1000, 1)
        return resultat

    def refuser(motif, message):
        resultat.update(refus=True, reponse=message, mode="refus")
        resultat["garde_fous"].append(motif)
        return terminer()

    # 1. Garde-fous d'entree
    if detecter_injection(question):
        return refuser("injection", MESSAGE_INJECTION)

    # 2. Perimetre et requete de recherche
    requete = params["rag"]["question_critique"]
    if client is None:
        if not dans_perimetre_mots_cles(question):
            return refuser("hors_perimetre", MESSAGE_HORS_PERIMETRE)
    else:
        try:
            texte, u = client.completer(
                [
                    {"role": "system", "content": PROMPT_ANALYSE},
                    {"role": "user", "content": question},
                ]
            )
            usage["tokens_entree"] += u["tokens_entree"]
            usage["tokens_sortie"] += u["tokens_sortie"]
            analyse = lire_json(texte)
            if analyse.get("dans_perimetre") is False:
                return refuser("hors_perimetre", MESSAGE_HORS_PERIMETRE)
            requete = (analyse.get("requete_anglais") or "").strip() or requete
        except Exception:
            # Analyse illisible : on garde la question par defaut du RAG
            resultat["garde_fous"].append("analyse_invalide")

    # 3. Recherche des passages
    try:
        passages = rechercher_fn(requete, params, config["top_k"])
    except FileNotFoundError:
        passages = []
        resultat["garde_fous"].append("index_rag_absent")
    resultat["requete_recherche"] = requete

    def extractif(mode):
        resultat["mode"] = mode
        resultat["reponse"] = _reponse_extractive(
            requete, passages, anomalies, rul, statut, params
        )
        resultat["references"] = extraire_references(passages)
        return terminer()

    if client is None:
        return extractif("extractif")

    # 4. Generation
    contexte = {
        "rul_predite": rul,
        "statut": statut,
        "seuil_critique": seuils[0],
        "seuil_attention": seuils[1],
    }
    message = construire_message_utilisateur(
        question, contexte, passages, decrire_capteurs(anomalies)
    )
    try:
        texte, u = client.completer(
            [
                {"role": "system", "content": PROMPT_SYSTEME},
                {"role": "user", "content": message},
            ]
        )
        usage["tokens_entree"] += u["tokens_entree"]
        usage["tokens_sortie"] += u["tokens_sortie"]
    except Exception:
        resultat["garde_fous"].append("erreur_llm")
        return extractif("extractif_secours")

    # 5. Verification
    problemes = verifier_reponse(texte, len(passages), rul, seuils)
    if problemes:
        resultat["garde_fous"].extend(problemes)
        return extractif("extractif_secours")

    cites = sorted(citations(texte))
    resultat["reponse"] = f"{_entete(rul, statut)} {texte}"
    resultat["references"] = [
        f"[{n}] {p['section']} (p. {p['page']})"
        for n, p in enumerate(passages, start=1)
        if n in cites
    ]
    return terminer()
