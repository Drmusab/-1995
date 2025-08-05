# Multi-stage build for AI Assistant System
FROM python:3.10-slim AS builder

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
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure pip to use HTTPS with proper certificates
RUN pip install --no-cache-dir --upgrade pip setuptools wheel --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

# Copy requirements and install dependencies directly
COPY requirements/base.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

# Copy the entire source code
COPY . ./

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
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories for data persistence and set permissions
RUN mkdir -p /data/models /data/cache /data/user_data /data/knowledge_base /data/logs \
    && chmod -R 755 /data

# Create non-root user for security
RUN groupadd -r aiassistant && useradd -r -g aiassistant aiassistant \
    && chown -R aiassistant:aiassistant /app /data
USER aiassistant

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

# Health check - simplified for containerized environment
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Set up entrypoint
ENTRYPOINT ["python", "-m", "src.main"]

# Default command (can be overridden)
CMD ["--config", "/app/configs/environments/production.yaml"]
