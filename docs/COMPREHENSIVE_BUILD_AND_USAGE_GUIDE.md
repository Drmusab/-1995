# AI Assistant - Comprehensive Build and Usage Guide

## Table of Contents
1. [System Analysis & Overview](#system-analysis--overview)
2. [Requirements & Prerequisites](#requirements--prerequisites)
3. [Installation & Build Guide](#installation--build-guide)
4. [Configuration & Setup](#configuration--setup)
5. [Usage & Activation Guide](#usage--activation-guide)
6. [Feature Activation](#feature-activation)
7. [Deployment Options](#deployment-options)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tuning](#performance-tuning)
10. [Security Configuration](#security-configuration)

---

## System Analysis & Overview

### Architecture Overview
The AI Assistant is a sophisticated multimodal AI system built with a modular, event-driven architecture that supports:

- **Multimodal Processing**: Text, speech, and vision input/output
- **Workflow Orchestration**: Complex task automation and execution
- **Plugin System**: Extensible architecture with dynamic plugin loading
- **Memory Management**: Advanced storage, retrieval, and consolidation
- **Learning & Adaptation**: Continuous learning from user interactions
- **Real-time Communication**: WebSocket, REST, and GraphQL APIs
- **Security**: Built-in authentication, authorization, and encryption
- **Observability**: Comprehensive monitoring, logging, and tracing

### Core Components

#### 1. **Core Framework** (`src/core/`)
- Configuration management with hot-reloading
- Event-driven architecture with EventBus
- Dependency injection container
- Error handling and resilience
- Security components (authentication/authorization)

#### 2. **Assistant Engine** (`src/assistant/`)
- Core assistant logic and session management
- Workflow orchestration
- Plugin management and discovery
- Interaction handling

#### 3. **Processing Modules** (`src/processing/`)
- Natural Language Processing
- Speech processing and emotion detection
- Vision and multimodal processing
- Text analysis and generation

#### 4. **Memory System** (`src/memory/`)
- Session memory management
- Knowledge graph integration
- Context retrieval and consolidation
- Caching strategies

#### 5. **Skills Framework** (`src/skills/`)
- Extensible skill system
- Skill discovery and execution
- Custom skill development

#### 6. **Integrations** (`src/integrations/`)
- LLM providers (OpenAI, Ollama, DeepSeek)
- External APIs and services
- Storage backends (Redis, PostgreSQL, Qdrant)

#### 7. **APIs** (`src/api/`)
- REST API endpoints
- WebSocket real-time communication
- GraphQL query interface

#### 8. **User Interfaces** (`src/ui/`)
- Command-line interface (CLI)
- Web interface components

#### 9. **Observability** (`src/observability/`)
- Structured logging
- Metrics collection (Prometheus)
- Distributed tracing (OpenTelemetry)
- Health checks

---

## Requirements & Prerequisites

### System Requirements

#### Minimum Hardware
- **CPU**: 4+ cores (Intel i5/AMD Ryzen 5 or equivalent)
- **RAM**: 8GB (16GB recommended for AI models)
- **Storage**: 20GB free space (50GB+ for models and data)
- **Network**: Stable internet connection for model downloads

#### Recommended Hardware
- **CPU**: 8+ cores with AVX support
- **RAM**: 32GB+ for large language models
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: SSD with 100GB+ free space

### Software Requirements

#### Core Dependencies
- **Python**: 3.10 or 3.11 (3.12 supported but may have compatibility issues)
- **pip**: Latest version (24.0+)
- **Git**: For repository management

#### Optional Dependencies
- **Docker**: For containerized deployment
- **Docker Compose**: For multi-service deployment
- **Node.js**: For web interface development
- **PostgreSQL**: For production database
- **Redis**: For caching and session storage

### Python Dependencies

#### Core Libraries
```
pyyaml>=6.0
python-dotenv>=1.0.0
pydantic>=2.0.0
numpy>=1.24.0
torch>=2.0.0
asyncio>=3.4.3
uvicorn>=0.22.0
fastapi>=0.95.0
sqlalchemy>=2.0.0
alembic>=1.11.0
redis>=4.5.0
aiohttp>=3.8.0
websockets>=11.0.0
```

#### Development Dependencies
```
black>=23.3.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.3.0
pytest>=7.3.0
pytest-cov>=4.1.0
pre-commit>=3.3.0
```

#### Optional Dependencies
```
asyncpg  # PostgreSQL async driver
qdrant-client  # Vector database
openai  # OpenAI API client
transformers  # Hugging Face models
```

---

## Installation & Build Guide

### Quick Start Installation

#### 1. Clone Repository
```bash
git clone https://github.com/Drmusab/-1995.git
cd -1995
```

#### 2. Set Up Python Environment
```bash
# Using virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n ai-assistant python=3.11
conda activate ai-assistant
```

#### 3. Install Core Dependencies
```bash
# Basic installation
pip install -e .

# Development installation
pip install -e ".[dev,test]"

# Full installation with all features
pip install -e ".[all]"
```

### Step-by-Step Build Process

#### 1. **Environment Setup**
```bash
# Create and activate virtual environment
python3 -m venv ai-assistant-env
source ai-assistant-env/bin/activate

# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel
```

#### 2. **Core Installation**
```bash
# Install main package
pip install -e .

# Verify installation
python -c "import src; print('Installation successful')"
```

#### 3. **Optional Dependencies**
```bash
# For PostgreSQL support
pip install asyncpg psycopg2-binary

# For advanced NLP features
pip install transformers sentence-transformers

# For vector database support
pip install qdrant-client

# For LLM integrations
pip install openai anthropic ollama
```

#### 4. **Development Tools** (Optional)
```bash
pip install -e ".[dev]"
pre-commit install
```

### Alternative Installation Methods

#### Using Docker
```bash
# Build and run with Docker
docker build -f docker/Dockerfile -t ai-assistant:latest .
docker run -p 8000:8000 -v ai_data:/data ai-assistant:latest
```

#### Using Docker Compose
```bash
cd docker
docker-compose up -d
```

#### Development with Hot Reload
```bash
cd docker
docker-compose -f docker-compose.dev.yml up -d
```

---

## Configuration & Setup

### 1. **Environment Configuration**

#### Create Environment File
```bash
cp .env.example .env
```

#### Configure Environment Variables
```bash
# Basic configuration
AI_ASSISTANT_ENV=development
AI_ASSISTANT_DEBUG=true
AI_ASSISTANT_LOG_LEVEL=INFO

# API Configuration
AI_ASSISTANT_HOST=0.0.0.0
AI_ASSISTANT_PORT=8000

# Database Configuration
DATABASE_URL=sqlite:///data/ai_assistant.db
REDIS_URL=redis://localhost:6379

# LLM Provider Configuration
OPENAI_API_KEY=your_openai_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

### 2. **Configuration Files**

#### Main Configuration (`configs/environments/development.yaml`)
```yaml
app:
  name: "AI Assistant"
  version: "1.0.0"
  debug: true
  
logging:
  level: "INFO"
  format: "json"
  
database:
  url: "sqlite:///data/ai_assistant.db"
  
cache:
  redis_url: "redis://localhost:6379"
  
llm:
  default_provider: "openai"
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4"
```

### 3. **Directory Structure Setup**
```bash
# Create necessary directories
mkdir -p data/{logs,models,cache,sessions}
mkdir -p configs/{environments,models,skills}
mkdir -p plugins
```

---

## Usage & Activation Guide

### Command Line Interface (CLI)

#### Basic Usage
```bash
# Interactive mode (default)
ai-assistant

# Or with Python module
python -m src.cli
```

#### CLI Modes

##### 1. **Interactive Mode**
```bash
ai-assistant
# Enters interactive chat mode
```

##### 2. **Command Mode**
```bash
# Execute single command
ai-assistant -c "What's the weather like?"

# Execute with specific skill
ai-assistant -c "status" --skill system
```

##### 3. **REPL Mode**
```bash
# Python-like REPL for advanced users
ai-assistant -r
```

##### 4. **Monitor Mode**
```bash
# System monitoring dashboard
ai-assistant -m
```

#### Advanced CLI Options
```bash
# Specify configuration
ai-assistant --config configs/production.yaml

# Set log level
ai-assistant --log-level DEBUG

# Enable specific features
ai-assistant --enable-speech --enable-vision

# Load specific plugins
ai-assistant --plugins nlp,memory,api_extensions
```

### Server Mode

#### Start the Server
```bash
# Basic server
ai-assistant-server

# Or with Python module
python -m src.main

# With custom configuration
python -m src.main --config configs/production.yaml
```

#### Server Endpoints

##### REST API
- Base URL: `http://localhost:8000/api/v1/`
- Documentation: `http://localhost:8000/docs`
- API Explorer: `http://localhost:8000/redoc`

##### WebSocket
- Real-time communication: `ws://localhost:8000/ws`

##### GraphQL
- GraphQL endpoint: `http://localhost:8000/graphql`
- GraphiQL explorer: `http://localhost:8000/graphiql`

##### Monitoring
- Health check: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`

---

## Feature Activation

### 1. **Core Features**

#### Enable Basic AI Chat
```bash
# Start with basic configuration
ai-assistant --mode interactive
```

#### API Example
```python
import aiohttp
import asyncio

async def chat_with_assistant():
    async with aiohttp.ClientSession() as session:
        data = {
            "message": "Hello, how can you help me?",
            "session_id": "user123"
        }
        async with session.post(
            "http://localhost:8000/api/v1/chat",
            json=data
        ) as response:
            result = await response.json()
            print(result["response"])

asyncio.run(chat_with_assistant())
```

### 2. **Advanced Features**

#### Enable Speech Processing
```bash
# Install speech dependencies
pip install pyaudio speechrecognition pyttsx3

# Start with speech enabled
ai-assistant --enable-speech

# Configure speech in environment
export AI_ASSISTANT_SPEECH_ENABLED=true
```

#### Enable Vision Processing
```bash
# Install vision dependencies
pip install opencv-python pillow

# Start with vision enabled
ai-assistant --enable-vision

# Use vision in API
curl -X POST http://localhost:8000/api/v1/vision/analyze \
  -F "image=@image.jpg" \
  -F "task=describe"
```

#### Enable Memory and Learning
```bash
# Memory requires vector database
docker run -p 6333:6333 qdrant/qdrant

# Configure memory backend
export QDRANT_URL=http://localhost:6333

# Start with memory enabled
ai-assistant --enable-memory
```

### 3. **Plugin System**

#### Discover Available Plugins
```bash
ai-assistant -c "list plugins"
```

#### Load Specific Plugins
```bash
# Load single plugin
ai-assistant --plugin core_skills

# Load multiple plugins
ai-assistant --plugins "nlp_processor,memory_enhancer,api_extensions"
```

#### Custom Plugin Development
```python
# Create plugin in plugins/my_plugin.py
from src.assistant.core import PluginBase

class MyPlugin(PluginBase):
    def __init__(self):
        super().__init__("my_plugin", "1.0.0")
    
    async def initialize(self):
        print("My plugin initialized!")
    
    async def process(self, input_data):
        return f"Processed: {input_data}"
```

### 4. **LLM Provider Configuration**

#### OpenAI Configuration
```bash
export OPENAI_API_KEY=your_key_here
ai-assistant --llm-provider openai --model gpt-4
```

#### Ollama Configuration
```bash
# Start Ollama server
ollama serve

# Pull models
ollama pull llama2

# Configure assistant
export OLLAMA_BASE_URL=http://localhost:11434
ai-assistant --llm-provider ollama --model llama2
```

#### Multiple Provider Setup
```yaml
# In config file
llm:
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      models: ["gpt-4", "gpt-3.5-turbo"]
    ollama:
      base_url: "http://localhost:11434"
      models: ["llama2", "codellama"]
    
  routing:
    default: "openai"
    fallback: "ollama"
    rules:
      - pattern: "code*"
        provider: "ollama"
        model: "codellama"
```

---

## Deployment Options

### 1. **Development Deployment**
```bash
# Local development
ai-assistant --env development --debug

# With hot reload
python -m src.main --reload
```

### 2. **Production Deployment**

#### Single Server
```bash
# Production configuration
ai-assistant-server --env production --workers 4
```

#### Docker Production
```bash
# Build production image
docker build -f docker/Dockerfile -t ai-assistant:prod .

# Run with production config
docker run -d \
  --name ai-assistant \
  -p 8000:8000 \
  -v /data/ai-assistant:/data \
  -e AI_ASSISTANT_ENV=production \
  ai-assistant:prod
```

#### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-assistant
  template:
    metadata:
      labels:
        app: ai-assistant
    spec:
      containers:
      - name: ai-assistant
        image: ai-assistant:latest
        ports:
        - containerPort: 8000
        env:
        - name: AI_ASSISTANT_ENV
          value: "production"
```

### 3. **Cloud Deployment**

#### AWS ECS
```bash
# Create ECS task definition
aws ecs register-task-definition --cli-input-json file://ecs-task.json

# Deploy service
aws ecs create-service --cluster ai-assistant --service-name ai-assistant --task-definition ai-assistant
```

#### Docker Compose for Production
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  ai-assistant:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AI_ASSISTANT_ENV=production
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ai_assistant
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
  
  redis:
    image: redis:7-alpine
```

---

## Troubleshooting

### Common Issues

#### 1. **Installation Problems**

**Issue**: `ModuleNotFoundError: No module named 'src'`
```bash
# Solution: Set PYTHONPATH
export PYTHONPATH=/path/to/-1995:$PYTHONPATH
# Or install in development mode
pip install -e .
```

**Issue**: Missing dependencies during installation
```bash
# Solution: Install optional dependencies
pip install asyncpg psycopg2-binary
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 2. **Runtime Issues**

**Issue**: Cannot connect to database
```bash
# Check database URL
echo $DATABASE_URL

# Create database directory
mkdir -p data

# Initialize database
python -c "
from src.core.config.loader import ConfigLoader
from sqlalchemy import create_engine
# Database initialization code
"
```

**Issue**: LLM provider not responding
```bash
# Check API key
echo $OPENAI_API_KEY

# Test connection
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

#### 3. **Performance Issues**

**Issue**: Slow response times
```bash
# Enable performance monitoring
ai-assistant --enable-metrics --log-level DEBUG

# Check system resources
top
df -h
```

**Issue**: Memory leaks
```bash
# Monitor memory usage
python -m memory_profiler src/main.py

# Configure garbage collection
export PYTHONMALLOC=debug
```

### Debugging Tools

#### Enable Debug Mode
```bash
ai-assistant --debug --log-level DEBUG
```

#### Health Checks
```bash
# Check system health
curl http://localhost:8000/health

# Detailed diagnostics
ai-assistant -c "system diagnose"
```

#### Log Analysis
```bash
# View logs
tail -f data/logs/application/app.log

# Filter by component
grep "core.engine" data/logs/application/app.log
```

---

## Performance Tuning

### 1. **Memory Optimization**
```yaml
# In configuration
memory:
  cache_size: 1000  # Reduce for limited RAM
  session_timeout: 3600
  cleanup_interval: 300
```

### 2. **CPU Optimization**
```yaml
processing:
  worker_threads: 4  # Match CPU cores
  batch_size: 32
  async_workers: 8
```

### 3. **Model Optimization**
```yaml
llm:
  model_cache_size: 2  # Number of models in memory
  inference_timeout: 30
  batch_requests: true
```

### 4. **Database Tuning**
```yaml
database:
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
```

---

## Security Configuration

### 1. **Authentication Setup**
```yaml
security:
  auth:
    jwt_secret: "your-secret-key"
    token_expire: 24  # hours
    require_auth: true
```

### 2. **HTTPS Configuration**
```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Start with HTTPS
ai-assistant-server --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### 3. **API Security**
```yaml
api:
  rate_limit:
    requests_per_minute: 60
    burst: 10
  
  cors:
    origins: ["https://yourdomain.com"]
    methods: ["GET", "POST"]
```

### 4. **Environment Security**
```bash
# Secure environment file
chmod 600 .env

# Use secret management
export OPENAI_API_KEY=$(vault kv get -field=api_key secret/openai)
```

---

This comprehensive guide provides everything needed to analyze, build, configure, and use the AI Assistant system. For additional help, consult the [Architecture Documentation](docs/architecture/) or create an issue on the GitHub repository.