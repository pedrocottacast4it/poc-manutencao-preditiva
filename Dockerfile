# syntax=docker/dockerfile:1
#
# Template de imagem única: frontend estático + backend HTTP na porta 8080.
# Pensado para AWS App Runner (sem WebSockets) e CI com --target runtime.
#
# Build de entrega (ECR / App Runner):
#   docker build --target runtime -t demo-template .
# Execução local:
#   docker run --rm -p 8080:8080 demo-template

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/ /app/backend/
COPY data/cmapss/test_FD001.txt data/cmapss/RUL_FD001.txt /app/data/cmapss/
COPY index.html logo_cast.png favicon.ico /app/frontend/
COPY vendor/ /app/frontend/vendor/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

RUN adduser --disabled-password --gecos "" --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER 10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]

CMD ["/app/docker-entrypoint.sh"]

# Último estágio = desenvolvimento local (docker compose build sem target).
FROM runtime AS development

USER root
RUN pip install --no-cache-dir watchfiles==0.24.0
USER 10001

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
