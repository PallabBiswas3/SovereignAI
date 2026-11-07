FROM python:3.11-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf AS dependencies
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY backend/requirements.txt backend/constraints-docker.txt ./backend/
COPY Time-Series-Diagnostic-Agent/pyproject.toml ./Time-Series-Diagnostic-Agent/pyproject.toml
COPY Time-Series-Diagnostic-Agent/src ./Time-Series-Diagnostic-Agent/src
COPY Controlplane.ai/pyproject.toml ./Controlplane.ai/pyproject.toml
COPY Controlplane.ai/src ./Controlplane.ai/src
# Build-time installation only; local modules are packaged without editable links.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.2 && \
    sed '/^-e /d' backend/requirements.txt > /tmp/runtime-requirements.txt && \
    pip install -c backend/constraints-docker.txt -r /tmp/runtime-requirements.txt \
      ./Time-Series-Diagnostic-Agent './Controlplane.ai[api]' \
      sentence-transformers==3.0.1 sentencepiece==0.2.1
RUN python -c "import controlplane; import adaptivefact.data.schema; import tsdiag.api" && pip check

FROM python:3.11-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH
WORKDIR /app
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get -o Acquire::Retries=3 -o APT::Update::Error-Mode=any update && \
    apt-get install -y --no-install-recommends tesseract-ocr libgomp1 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=dependencies /opt/venv /opt/venv
COPY backend/app ./backend/app
COPY backend/scripts ./backend/scripts
COPY config ./config
COPY deploy/tools.production.yaml ./deploy/tools.production.yaml
COPY workcells ./workcells
COPY Controlplane.ai/configs/policies ./Controlplane.ai/configs/policies
COPY Controlplane.ai/models ./Controlplane.ai/models
RUN useradd --uid 10001 --create-home sovereign && \
    mkdir -p backend/data workspace/evidence_capsules workspace/onboarding_reports knowledge_base organizations Controlplane.ai/results && \
    chown -R sovereign:sovereign backend/data workspace knowledge_base organizations Controlplane.ai/results
ENV HF_HOME=/models/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
USER sovereign
WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
