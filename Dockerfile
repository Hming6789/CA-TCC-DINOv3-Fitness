FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 MODEL_DIR=/models INFERENCE_THREADS=2
WORKDIR /app
COPY requirements-web-lock.txt ./
RUN pip install --no-cache-dir -r requirements-web-lock.txt \
    && useradd --uid 10001 --create-home app \
    && mkdir /models && chown app:app /models
COPY webapp/__init__.py webapp/app.py webapp/inference.py ./webapp/
COPY webapp/static ./webapp/static
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["python", "-m", "uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
