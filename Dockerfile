# Dockerfile : API + interface web + assistant (une seule image)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Dependances d'abord : cette couche reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Fins de ligne Windows (CRLF) retirees du script de demarrage
RUN sed -i 's/\r$//' scripts/demarrer.sh && chmod +x scripts/demarrer.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["./scripts/demarrer.sh"]
