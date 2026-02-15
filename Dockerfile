FROM python:3.12-slim

WORKDIR /app


#step 1: install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*


# step 2: install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir --extra-index-url https://pypi.org/simple \
    -r requirements.txt


# step 3: download model (BUILD time) dengan retry
ARG MODEL_ID=qrizan/emotion-classifier-indonesia
ARG HF_TOKEN
ARG MAX_RETRIES=3
ARG RETRY_DELAY=10
ENV MODEL_ID=${MODEL_ID}
ENV HF_HOME=/app/.cache/huggingface

COPY scripts/download_model.py /tmp/download_model.py

# download dengan retry mechanism
RUN echo "============================================" && \
    echo "[BUILD] Downloading model: ${MODEL_ID}" && \
    echo "[BUILD] Max retries: ${MAX_RETRIES}" && \
    if [ -n "${HF_TOKEN}" ]; then \
        echo "[BUILD] Using HF_TOKEN for faster download"; \
    else \
        echo "[BUILD] No HF_TOKEN - using unauthenticated (slower)"; \
        echo "[BUILD] Consider setting HF_TOKEN for better reliability"; \
    fi && \
    echo "[BUILD] This may take 2-5 minutes..." && \
    echo "============================================" && \
    MODEL_ID=${MODEL_ID} \
    HF_TOKEN=${HF_TOKEN} \
    MAX_RETRIES=${MAX_RETRIES} \
    RETRY_DELAY=${RETRY_DELAY} \
    python /tmp/download_model.py && \
    echo "============================================" && \
    echo "[BUILD] ✓ Model downloaded and validated" && \
    echo "============================================" || \
    (echo "============================================" && \
     echo "[BUILD] ✗ Model download failed" && \
     echo "[BUILD] Check logs above for details" && \
     echo "============================================" && \
     exit 1)

# step 4: copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
RUN chmod +x scripts/download_model.py

# step 5: configure runtime
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=${MODEL_ID}
ENV CONFIDENCE_THRESHOLD=0.60
ENV EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
ENV MAX_LENGTH=128

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# step 6: Start application
# model sudah di-download di BUILD time, langsung start
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
