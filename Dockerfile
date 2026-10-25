# Flask service: recorded results, replay, external validation and its hardware subset,
# charts, and live classical inference.
# Trained models are not baked in; mount artifacts/benchmark read-only to enable prediction.
# The service never trains models, fetches URLs, or submits quantum jobs.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv/vigilantia

COPY requirements-app.txt .
RUN python -m pip install -r requirements-app.txt

COPY app/ app/
COPY reports/ reports/
COPY src/__init__.py src/__init__.py
COPY src/features/ src/features/
COPY src/inference/ src/inference/
COPY configs/feature-scaler.json configs/feature-scaler.json
COPY docs/benchmark-results.svg docs/benchmark-results.svg
COPY docs/model-comparison.svg docs/model-comparison.svg

RUN useradd --system --uid 10001 --no-create-home vigilantia
USER 10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--access-logfile", "-", "app:create_app()"]
