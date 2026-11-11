# src/assistant/garde_fous.py
"""Controles avant et apres l'appel au LLM.

Avant : tentative d'injection de prompt, question vide ou trop longue,
perimetre (filtre par mots-cles quand le LLM n'est pas disponible).
Apres : citations [n] valides et aucune duree de vie inventee par le LLM.
"""

import json
import re
import unicodedata

MOTIFS_INJECTION = [
    r"ignore[rz]? (les |toutes |all |the |previous |les instructions)",
    r"ignore (all|previous|the above)",
    r"oublie[rz]? (tes|les|vos) (instructions|regles|consignes)",
    r"system prompt",
    r"prompt syst[eè]me",
    r"tu es maintenant",
    r"you are now",
    r"r[ée]v[eè]le (ton|le|ta) (prompt|consigne)",
    r"jailbreak",
    r"mode d[ée]veloppeur",
]

# Utilise seulement en mode extractif (sans LLM) pour reconnaitre une
# question de maintenance. Avec le LLM, c'est lui qui classe la question.
MOTS_DOMAINE = {
    "moteur", "moteurs", "engine", "capteur", "capteurs", "sensor", "sensors",
    "panne", "pannes", "defaillance", "failure", "maintenance", "rul", "cycle",
    "cycles", "degradation", "usure", "temperature", "pression", "pressure",
    "compresseur", "compressor", "turbine", "hpc", "hpt", "lpc", "lpt", "fan",
    "ventilateur", "inspection", "inspecter", "intervention", "reparer",
    "reparation", "statut", "critique", "surveillance", "risque", "vie",
    "efficacite", "efficiency", "debit", "flow", "c-mapss", "cmapss", "nasa",
    "vitesse", "carburant", "prediction", "predite", "remplacer", "alerte",
    "recommandation", "recommandations", "action", "actions", "urgent",
}
CAPTEUR_OU_SYMBOLE = re.compile(r"\b(s([1-9]|1\d|2[01])|t2|t24|t30|t50|p30|ps30|nf|nc)\b")


def normaliser(texte):
    """Minuscules sans accents, pour comparer des mots."""
    texte = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in texte if not unicodedata.combining(c))


def detecter_injection(question):
    q = normaliser(question)
    return any(re.search(motif, q) for motif in MOTIFS_INJECTION)


def dans_perimetre_mots_cles(question):
    q = normaliser(question)
    mots = set(re.findall(r"[a-z0-9\-]+", q))
    return bool(mots & MOTS_DOMAINE) or bool(CAPTEUR_OU_SYMBOLE.search(q))


def lire_json(texte):
    """Extrait le premier objet JSON du texte du LLM (il ajoute parfois du texte)."""
    texte = re.sub(r"```(?:json)?", "", texte)
    debut, fin = texte.find("{"), texte.rfind("}")
    if debut == -1 or fin <= debut:
        raise ValueError("Pas de JSON dans la reponse du LLM")
    return json.loads(texte[debut : fin + 1])


def citations(texte):
    """Numeros cites sous la forme [1], [2] ou [1, 3]."""
    nums = set()
    for groupe in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", texte):
        nums.update(int(n) for n in groupe.split(","))
    return nums


def verifier_reponse(texte, nb_passages, rul, seuils):
    """Liste des problemes detectes (vide si la reponse est acceptable)."""
    problemes = []
    if not texte.strip():
        return ["reponse_vide"]

    cites = citations(texte)
    if nb_passages and not cites:
        problemes.append("citation_manquante")
    if any(n < 1 or n > nb_passages for n in cites):
        problemes.append("citation_invalide")

    # Le LLM ne doit pas inventer une autre duree de vie que celle du modele
    autorises = {round(rul), round(rul, 1), *seuils}
    for nombre in re.findall(r"(\d+(?:[.,]\d+)?)\s*cycles?", texte.lower()):
        valeur = float(nombre.replace(",", "."))
        if not any(abs(valeur - a) < 0.6 for a in autorises):
            problemes.append("rul_contradictoire")
            break
    return problemes
