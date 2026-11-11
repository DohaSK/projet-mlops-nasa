# model_card.md

# Model Card : RUL et explication physique, NASA CMAPSS FD001

## 1. Systeme ML

Trois pipelines scikit-learn sont compares (`StandardScaler` puis Ridge,
RandomForestRegressor ou HistGradientBoostingRegressor, hyperparametres dans
`params.yaml`). Le champion, HistGradientBoosting, est enregistre dans le MLflow
Model Registry (`rul-fd001`). Entree : 3 reglages operationnels et 21 capteurs
d'un cycle. Sortie : nombre de cycles restants avant defaillance.

Donnees : NASA CMAPSS FD001, 100 moteurs de train, une condition
operationnelle, un mode de defaillance (degradation du compresseur haute
pression, HPC). La RUL est calculee comme (dernier cycle moins cycle courant),
plafonnee a 125. Le split se fait par moteur (80 / 20) pour eviter toute fuite.

Metriques hors ligne du champion : RMSE 18.99 et R2 0.788 sur les moteurs de
validation ; RMSE 17.34 et R2 0.813 sur le dernier cycle des 100 moteurs de
test officiels ; 18.8 % d'alertes critiques manquees en validation, 10 % sur le
test officiel. Le Quality Gate (seuils justifies dans le README) exige
RMSE < 25, R2 > 0.7, au plus 25 % d'alertes manquees et un RMSE inferieur a
celui de la baseline Ridge.

Metriques hors ligne et qualite observable : apres deploiement, la vraie RUL
n'est connue qu'a la panne ou a l'inspection. En production on ne mesure donc
directement que les entrees (valeurs manquantes, derive), les sorties
(repartition des statuts) et la latence ; la qualite reelle doit etre
recalculee a posteriori quand les RUL reelles sont connues.

## 2. Systeme RAG

Source unique : le papier NASA "Damage Propagation Modeling for Aircraft
Engine Run-to-Failure Simulation". Le texte est extrait avec pypdf, nettoye
(tableaux et formules ignores), decoupe par section puis en chunks de 150 mots
avec chevauchement. Embeddings all-MiniLM-L6-v2, index FAISS local. La reponse
est extractive : des phrases du papier, jamais du texte genere.

## 3. Assistant de maintenance (LLM)

Endpoint `/assistant` : un LLM (`openai/gpt-oss-20b` via Groq, temperature 0)
redige une reponse en francais a partir de la RUL predite, des capteurs les
plus inhabituels (z-score par rapport a l'entrainement) et des passages RAG
numerotes. Garde-fous : detection d'injection avant tout appel, classement
du perimetre par le LLM, verification des citations [n] et de l'absence de
toute autre duree de vie que celle du modele. En cas d'echec d'un controle ou
d'erreur de l'API, la reponse extractive (sans texte genere) est renvoyee.
Evaluation : 14 questions x 3 moteurs, Quality Gate dans `params.yaml`.

## 4. Limites

- Donnees simulees : les performances reelles peuvent differer.
- FD001 couvre une seule condition operationnelle et un seul mode de panne.
- Le plafond de RUL a 125 cycles sous-estime la duree de vie des moteurs sains.
- Le modele ne voit qu'un cycle a la fois, sans historique temporel.
- Les erreurs les plus fortes (RMSE 26 a 29 cycles, biais optimiste) se
  situent entre 25 et 75 cycles, autour du seuil critique.
- Le trafic de test "normal" montre deja un decalage d'environ 0.4 ecart-type
  sur plusieurs capteurs (moteurs de test plus jeunes que la moyenne
  d'entrainement) : le seuil de derive de 0.5 est donc proche du bruit naturel.
- L'explication RAG decrit le modele de simulation C-MAPSS en general. Elle
  n'est pas une analyse causale de la prediction pour un moteur donne.
- L'extraction PDF peut perdre ou deformer des tableaux et des formules.
- Le modele d'embeddings est anglophone : les questions doivent etre en anglais.

- Les capteurs signales par l'assistant sont inhabituels par rapport a la
  flotte d'entrainement, ce n'est pas une explication causale du modele.
- Le LLM peut reformuler un passage de facon inexacte : les controles
  automatiques verifient la presence des citations, pas leur fidelite exacte.
- La question est envoyee a un service externe (Groq). Elle ne doit contenir
  aucune donnee personnelle ou confidentielle.
- Le jeu d'evaluation de l'assistant est petit (14 questions) : il detecte
  les regressions evidentes mais ne mesure pas la qualite fine des reponses.

## 5. Risques et considerations ethiques

La maintenance d'avion est un systeme critique pour la securite.
- Une surestimation de la RUL peut retarder une intervention necessaire.
- Un RAG peut remonter un passage hors sujet, ou un modele generatif peut
  halluciner sur des grandeurs physiques. Le mode par defaut est extractif pour
  limiter ce risque, et chaque reponse cite sa section et sa page pour
  permettre une verification.
- Le systeme est un outil d'aide a la decision, jamais un decideur autonome.
  Toute decision de maintenance doit etre validee par un expert humain, avec
  des marges de securite et une surveillance de la derive des donnees.
- L'assistant ne recalcule jamais la RUL, ne declare jamais un moteur apte au
  vol et rappelle que la decision revient a un expert. La cle API est lue
  depuis `.env` (ignore par Git), un fichier `.env.example` sans valeur est
  fourni.