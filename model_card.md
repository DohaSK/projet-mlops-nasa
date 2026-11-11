# model_card.md

# Model Card : RUL et explication physique, NASA CMAPSS FD001

## 1. Systeme ML

Pipeline scikit-learn : `StandardScaler` suivi d'un `RandomForestRegressor`
(hyperparametres dans `params.yaml`). Entree : 3 reglages operationnels et
21 capteurs d'un cycle. Sortie : nombre de cycles restants avant defaillance.

Donnees : NASA CMAPSS FD001, 100 moteurs de train, une condition
operationnelle, un mode de defaillance (degradation du compresseur haute
pression, HPC). La RUL est calculee comme (dernier cycle moins cycle courant),
plafonnee a 125. Le split se fait par moteur pour eviter toute fuite.

Metriques : RMSE et R2 sur des moteurs de validation, puis sur le dernier
cycle des moteurs de test_FD001 avec les RUL de RUL_FD001.txt. Le Quality Gate
exige R2 superieur a `min_r2` et RMSE inferieur a `max_rmse`.

## 2. Systeme RAG

Source unique : le papier NASA "Damage Propagation Modeling for Aircraft
Engine Run-to-Failure Simulation". Le texte est extrait avec pypdf, nettoye
(tableaux et formules ignores), decoupe par section puis en chunks de 150 mots
avec chevauchement. Embeddings all-MiniLM-L6-v2, index FAISS local. La reponse
est extractive : des phrases du papier, jamais du texte genere.

## 3. Limites

- Donnees simulees : les performances reelles peuvent differer.
- FD001 couvre une seule condition operationnelle et un seul mode de panne.
- Le plafond de RUL a 125 cycles sous-estime la duree de vie des moteurs sains.
- Le modele ne voit qu'un cycle a la fois, sans historique temporel.
- L'explication RAG decrit le modele de simulation C-MAPSS en general. Elle
  n'est pas une analyse causale de la prediction pour un moteur donne.
- L'extraction PDF peut perdre ou deformer des tableaux et des formules.
- Le modele d'embeddings est anglophone : les questions doivent etre en anglais.

## 4. Risques et considerations ethiques

La maintenance d'avion est un systeme critique pour la securite.
- Une surestimation de la RUL peut retarder une intervention necessaire.
- Un RAG peut remonter un passage hors sujet, ou un modele generatif peut
  halluciner sur des grandeurs physiques. Le mode par defaut est extractif pour
  limiter ce risque, et chaque reponse cite sa section et sa page pour
  permettre une verification.
- Le systeme est un outil d'aide a la decision, jamais un decideur autonome.
  Toute decision de maintenance doit etre validee par un expert humain, avec
  des marges de securite et une surveillance de la derive des donnees.