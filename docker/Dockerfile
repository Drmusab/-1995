# Multi-stage build for AI Assistant System
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies required for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    pkg-config \
    libpq-dev \
    libsndfile1-dev \
    ffmpeg \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install poetry for dependency management
RUN pip install --no-cache-dir poetry

# Copy only dependency definition files first for better caching
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Second stage: Runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libsndfile1 \
    ffmpeg \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories for data persistence
RUN mkdir -p /data/models /data/cache /data/user_data /data/knowledge_base /data/logs

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV ENVIRONMENT=production
ENV CONFIG_PATH=/app/configs/environments/production.yaml
ENV MODELS_PATH=/data/models
ENV CACHE_PATH=/data/cache
ENV USER_DATA_PATH=/data/user_data
ENV KNOWLEDGE_BASE_PATH=/data/knowledge_base
ENV LOG_PATH=/data/logs

# Expose necessary ports (adjust based on your API needs)
EXPOSE 8000 8080 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Set up entrypoint
ENTRYPOINT ["python", "-m", "src.main"]

# Default command (can be overridden)
CMD ["--config", "${CONFIG_PATH}"]
