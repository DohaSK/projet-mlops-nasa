# src/assistant/llm.py
"""Client LLM (Groq). La cle est lue dans la variable d'environnement
GROQ_API_KEY (fichier .env en local, jamais dans Git).

Tout client doit exposer `completer(messages) -> (texte, usage)` ; les tests
utilisent un faux client avec la meme interface, sans appel reseau.
"""

import os


class ClientGroq:
    def __init__(self, config, cle_api):
        from groq import Groq

        self.modele = config["modele"]
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 900)
        self.reasoning_effort = config.get("reasoning_effort")
        self._client = Groq(api_key=cle_api, timeout=config.get("timeout_s", 30))

    def completer(self, messages):
        options = {}
        if self.reasoning_effort and self.modele.startswith("openai/gpt-oss"):
            # gpt-oss est un modele de raisonnement : "low" limite les tokens
            options["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        reponse = self._client.chat.completions.create(
            model=self.modele,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **options,
        )
        texte = reponse.choices[0].message.content or ""
        usage = {
            "tokens_entree": getattr(reponse.usage, "prompt_tokens", 0) or 0,
            "tokens_sortie": getattr(reponse.usage, "completion_tokens", 0) or 0,
        }
        return texte.strip(), usage


def creer_client(params):
    """Client Groq si GROQ_API_KEY est definie, sinon None (mode extractif)."""
    cle = os.getenv("GROQ_API_KEY", "").strip()
    if not cle:
        return None
    return ClientGroq(params["assistant"], cle)
