# GraphQA-Sentinel - single container serving both the FastAPI backend
# (agentic + baseline RAG) and the static frontend chat UI.
#
# Python 3.12 (not 3.14, which local dev used) is a deliberate choice for
# Docker: several dependencies here - notably ragas's scikit-network
# dependency - had no prebuilt wheels for 3.14 at build time, requiring a
# C++ compiler to build from source (the exact problem solved locally on
# Windows via installing Build Tools). 3.12 has much broader prebuilt-
# wheel coverage, avoiding that risk here. build-essential is still
# installed below as a defensive fallback regardless.

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application code, plus the pre-built vector store. chroma_store/ was
# populated locally via ingestion/*.py before building this image (see
# README) - baking it in avoids depending on live web scraping
# succeeding at deploy time.
COPY api/ ./api/
COPY agent/ ./agent/
COPY frontend/ ./frontend/
COPY chroma_store/ ./chroma_store/

ENV CHROMA_PERSIST_DIR=/app/chroma_store

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]