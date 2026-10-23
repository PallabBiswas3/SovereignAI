FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get -o Acquire::Retries=3 -o APT::Update::Error-Mode=any update && apt-get install -y --no-install-recommends tesseract-ocr libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt ./backend/requirements.txt
COPY Time-Series-Diagnostic-Agent/pyproject.toml ./Time-Series-Diagnostic-Agent/pyproject.toml
COPY Time-Series-Diagnostic-Agent/src ./Time-Series-Diagnostic-Agent/src
COPY Controlplane.ai/pyproject.toml ./Controlplane.ai/pyproject.toml
COPY Controlplane.ai/src ./Controlplane.ai/src
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.1.2 && \
    pip install --no-cache-dir -r backend/requirements.txt './Controlplane.ai[api]' sentence-transformers==3.0.1 sentencepiece==0.2.1
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
ENV HF_HOME=/models/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
USER sovereign
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
