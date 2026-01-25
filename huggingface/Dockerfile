# Hugging Face Spaces Dockerfile for BioBERT Attention Analyzer
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    transformers>=4.30.0 \
    "gradio[mcp]>=5.32.0" \
    numpy>=1.23.0 \
    requests>=2.28.0

# Pre-download BioBERT model to cache it in the image
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1'); \
    AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')"

# Copy application
COPY app.py .

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 user
USER user

# Environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV GRADIO_MCP_SERVER="true"
ENV TRANSFORMERS_CACHE="/app/.cache"
ENV HF_HOME="/app/.cache"

EXPOSE 7860

CMD ["python", "app.py"]
