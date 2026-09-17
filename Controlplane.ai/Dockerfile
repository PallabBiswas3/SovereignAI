FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    CONTROLPLANE_ADAPTIVE_VERIFICATION=1 \
    CONTROLPLANE_NLI_MODEL=cross-encoder/nli-deberta-v3-small \
    CONTROLPLANE_NLI_DEVICE=cpu

WORKDIR /app

COPY pyproject.toml requirements-hf.txt ./
COPY src ./src

# Install the CPU-only PyTorch wheel first to avoid bundling unused CUDA
# libraries, then install the application and its inference dependencies.
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1" && \
    python -m pip install -r requirements-hf.txt

# Cache model weights in the image so the first user request does not need to
# download DeBERTa from the Hub.
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; model='cross-encoder/nli-deberta-v3-small'; AutoTokenizer.from_pretrained(model); AutoModelForSequenceClassification.from_pretrained(model)"

COPY configs ./configs
COPY demo ./demo
COPY models ./models

RUN mkdir -p results/controlplane

EXPOSE 7860

CMD ["streamlit", "run", "demo/app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
