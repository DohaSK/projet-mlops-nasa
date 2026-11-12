#!/bin/sh
# Demarrage du conteneur : construit l'index RAG et le modele s'ils manquent,
# puis lance l'API (interface sur http://localhost:8000/ui).
set -e

if [ ! -f data/raw/train_FD001.txt ] && [ ! -f models/model.joblib ]; then
  echo "Donnees FD001 absentes : placez train_FD001.txt, test_FD001.txt et RUL_FD001.txt dans data/raw/"
  exit 1
fi

if [ ! -f models/rag/faiss_index.bin ]; then
  echo "Index RAG absent : construction (premier lancement, quelques minutes)"
  python -m src.rag.ingestion
fi

if [ ! -f models/model.joblib ]; then
  echo "Modele absent : comparaison et entrainement"
  python -m src.ml.train || echo "Attention : Quality Gate non respecte, modele sauvegarde quand meme"
fi

exec uvicorn app:app --host 0.0.0.0 --port 8000
