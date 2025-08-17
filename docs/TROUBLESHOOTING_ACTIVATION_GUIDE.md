# Troubleshooting & Activation Guide

This guide helps you resolve common issues and properly activate all features of the AI Assistant system.

## 🚨 Common Issues & Solutions

### Installation Issues

#### Issue: `ModuleNotFoundError: No module named 'src'`
**Cause**: Python cannot find the source modules
**Solutions**:
```bash
# Option 1: Set PYTHONPATH environment variable
export PYTHONPATH=/path/to/-1995:$PYTHONPATH

# Option 2: Install in development mode
pip install -e .

# Option 3: Add to Python path programmatically
import sys
sys.path.append('/path/to/-1995')
```

#### Issue: `pip install` fails with timeout errors
**Cause**: Network connectivity or PyPI server issues
**Solutions**:
```bash
# Use alternative index
pip install -e . -i https://pypi.python.org/simple/

# Increase timeout
pip install -e . --timeout 1000

# Install dependencies individually
pip install pyyaml python-dotenv pydantic numpy
pip install torch uvicorn fastapi sqlalchemy redis
```

#### Issue: PyTorch installation fails
**Cause**: Platform-specific installation required
**Solutions**:
```bash
# For CPU-only installation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Runtime Issues

#### Issue: Cannot start the assistant
**Symptoms**: Error when running `ai-assistant` or `python -m src.cli`
**Diagnostic**:
```bash
# Check Python path
echo $PYTHONPATH

# Test imports
python -c "import src; print('OK')"

# Check for missing dependencies
python -c "
try:
    import torch, numpy, pydantic, fastapi
    print('Core dependencies OK')
except ImportError as e:
    print(f'Missing: {e}')
"
```

**Solutions**:
```bash
# Install missing dependencies
pip install asyncpg psycopg2-binary  # Database drivers
pip install transformers sentence-transformers  # NLP components

# Create required directories
mkdir -p data/{logs,models,cache,sessions}
mkdir -p configs/{environments,models,skills}
```

#### Issue: Database connection errors
**Error**: `sqlalchemy.exc.OperationalError: could not connect to server`
**Solutions**:
```bash
# For SQLite (default)
mkdir -p data
export DATABASE_URL="sqlite:///data/ai_assistant.db"

# For PostgreSQL
sudo systemctl start postgresql
createdb ai_assistant
export DATABASE_URL="postgresql://username:password@localhost/ai_assistant"

# Initialize database
python -c "
from sqlalchemy import create_engine
engine = create_engine('sqlite:///data/ai_assistant.db')
engine.execute('SELECT 1')
print('Database OK')
"
```

#### Issue: Redis connection failed
**Error**: `redis.exceptions.ConnectionError: Error connecting to Redis`
**Solutions**:
```bash
# Install and start Redis
sudo apt-get install redis-server  # Ubuntu/Debian
brew install redis  # macOS
sudo systemctl start redis

# Test connection
redis-cli ping

# Alternative: Use in-memory cache
export REDIS_URL=""  # Disables Redis, uses memory cache
```

### Configuration Issues

#### Issue: LLM providers not working
**Symptoms**: API errors or no responses from AI models
**Diagnostic**:
```bash
# Check API keys
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Test API connection
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

**Solutions**:
```bash
# Set API keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Add to .env file
echo "OPENAI_API_KEY=your-key-here" >> .env
echo "ANTHROPIC_API_KEY=your-key-here" >> .env

# Configure fallback to local models
export OLLAMA_BASE_URL="http://localhost:11434"
ollama pull llama2  # Install local model
```

#### Issue: Permission denied errors
**Symptoms**: Cannot write to data directories or read config files
**Solutions**:
```bash
# Fix permissions
chmod -R 755 data/
chmod -R 644 configs/
chmod 600 .env  # Secure environment file

# Create with correct permissions
mkdir -p data/{logs,models,cache}
chown -R $USER:$USER data/
```

### Feature Activation Issues

#### Issue: Speech features not working
**Error**: `ModuleNotFoundError: No module named 'pyaudio'`
**Solutions**:
```bash
# Install system dependencies
sudo apt-get install portaudio19-dev python3-pyaudio  # Ubuntu
brew install portaudio  # macOS

# Install Python packages
pip install pyaudio speechrecognition pyttsx3

# Test audio
python -c "
import pyaudio
p = pyaudio.PyAudio()
print(f'Audio devices: {p.get_device_count()}')
p.terminate()
"
```

#### Issue: Vision features not working
**Error**: `ModuleNotFoundError: No module named 'cv2'`
**Solutions**:
```bash
# Install OpenCV
pip install opencv-python pillow

# Test installation
python -c "
import cv2
print(f'OpenCV version: {cv2.__version__}')
"
```

#### Issue: GPU not detected
**Symptoms**: Slow AI inference, CUDA not available
**Diagnostic**:
```bash
# Check CUDA availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
"

# Check NVIDIA drivers
nvidia-smi
```

**Solutions**:
```bash
# Install CUDA toolkit
# Follow instructions at https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## ✅ Feature Activation Checklist

### Core Features
- [ ] **Basic Chat**: `python -m src.cli` works without errors
- [ ] **Web API**: `python -m src.main` starts server at localhost:8000
- [ ] **Configuration**: Environment variables loaded from .env
- [ ] **Database**: SQLite or PostgreSQL connection established
- [ ] **Caching**: Redis connection or memory cache working

### AI Features
- [ ] **LLM Integration**: At least one provider (OpenAI/Ollama) configured
- [ ] **Text Processing**: Natural language understanding works
- [ ] **Response Generation**: AI responses are coherent and relevant
- [ ] **Context Memory**: Conversation history maintained
- [ ] **Plugin System**: Core plugins load successfully

### Advanced Features
- [ ] **Speech Input**: Microphone input processing
- [ ] **Speech Output**: Text-to-speech synthesis
- [ ] **Vision Processing**: Image analysis capabilities
- [ ] **Memory System**: Long-term memory and knowledge graph
- [ ] **Workflow Orchestration**: Multi-step task execution

### Monitoring & Health
- [ ] **Health Checks**: `/health` endpoint returns 200
- [ ] **Metrics**: Prometheus metrics available at `/metrics`
- [ ] **Logging**: Structured logs written to data/logs/
- [ ] **Error Handling**: Graceful degradation on failures
- [ ] **Performance**: Response times within acceptable limits

## 🔧 Diagnostic Commands

### System Health Check
```bash
# Quick health check
python -c "
import src
from src.core.health_check import HealthCheck
hc = HealthCheck()
print('System health: OK')
"

# Detailed diagnostics
PYTHONPATH=/path/to/-1995 python -m src.cli -c "system diagnose"
```

### Component Testing
```bash
# Test database connection
python -c "
from sqlalchemy import create_engine
from src.core.config.loader import ConfigLoader
config = ConfigLoader()
engine = create_engine(config.get('database.url', 'sqlite:///data/ai_assistant.db'))
engine.execute('SELECT 1')
print('Database: OK')
"

# Test Redis connection
python -c "
import redis
r = redis.Redis.from_url('redis://localhost:6379')
r.ping()
print('Redis: OK')
"

# Test LLM provider
python -c "
import os
if os.getenv('OPENAI_API_KEY'):
    print('OpenAI: Configured')
if os.getenv('ANTHROPIC_API_KEY'):
    print('Anthropic: Configured')
"
```

### Performance Testing
```bash
# Memory usage
python -c "
import psutil
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# Response time test
time curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "session_id": "test"}'
```

## 🚀 Performance Optimization

### Memory Optimization
```bash
# Set memory limits
export MALLOC_MMAP_THRESHOLD_=1048576
export MALLOC_TRIM_THRESHOLD_=134217728

# Optimize Python garbage collection
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

# Configure torch for CPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### CPU Optimization
```bash
# Set process priority
nice -n -10 python -m src.main

# Use multiple workers
python -m src.main --workers 4

# Enable async processing
export AI_ASSISTANT_ASYNC_WORKERS=8
```

### I/O Optimization
```bash
# Use faster file system
export TMPDIR=/dev/shm  # Use RAM disk for temp files

# Optimize database
export DATABASE_URL="postgresql://user:pass@localhost/db?pool_size=20"

# Configure logging
export AI_ASSISTANT_LOG_LEVEL=WARNING  # Reduce log volume
```

## 📊 Monitoring Setup

### Basic Monitoring
```bash
# Monitor logs
tail -f data/logs/application/app.log

# Monitor system resources
htop
iotop
nethogs
```

### Advanced Monitoring
```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Health check endpoint
curl http://localhost:8000/health

# System diagnostics
curl http://localhost:8000/api/v1/system/diagnostics
```

### Alerting
```bash
# Set up log monitoring
tail -f data/logs/application/app.log | grep -i error

# Monitor response times
while true; do
  time curl -s http://localhost:8000/health > /dev/null
  sleep 10
done
```

## 🔐 Security Hardening

### Environment Security
```bash
# Secure environment file
chmod 600 .env
chown $USER:$USER .env

# Generate secure JWT secret
python -c "
import secrets
print(f'JWT_SECRET={secrets.token_urlsafe(32)}')
" >> .env
```

### Network Security
```bash
# Configure firewall
sudo ufw allow 8000/tcp  # API port
sudo ufw enable

# Use HTTPS in production
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
```

### API Security
```bash
# Enable rate limiting
export AI_ASSISTANT_RATE_LIMIT=60  # requests per minute

# Require authentication
export AI_ASSISTANT_REQUIRE_AUTH=true

# Set CORS origins
export AI_ASSISTANT_CORS_ORIGINS="https://yourdomain.com"
```

## 🆘 Emergency Recovery

### Service Recovery
```bash
# Restart all services
sudo systemctl restart redis
sudo systemctl restart postgresql
pkill -f "python -m src.main"
python -m src.main &

# Clear cache and restart
redis-cli FLUSHALL
rm -rf data/cache/*
python -m src.main
```

### Data Recovery
```bash
# Backup current state
cp -r data/ data_backup_$(date +%Y%m%d_%H%M%S)

# Restore from backup
cp -r data_backup_YYYYMMDD_HHMMSS/ data/

# Database recovery
pg_dump ai_assistant > backup.sql
dropdb ai_assistant
createdb ai_assistant
psql ai_assistant < backup.sql
```

### Configuration Recovery
```bash
# Reset to default configuration
cp configs/environments/development.yaml.example configs/environments/development.yaml
cp .env.example .env

# Validate configuration
python -c "
from src.core.config.loader import ConfigLoader
config = ConfigLoader()
print('Configuration: OK')
"
```

---

## 📞 Getting Help

### Self-Service Resources
1. Check the [Comprehensive Guide](COMPREHENSIVE_BUILD_AND_USAGE_GUIDE.md) for detailed instructions
2. Review the [Requirements Analysis](REQUIREMENTS_ANALYSIS.md) for system requirements
3. Consult the [Architecture Documentation](architecture/) for technical details

### Community Support
1. Search existing [GitHub Issues](https://github.com/Drmusab/-1995/issues)
2. Create a new issue with detailed error information
3. Join community discussions for peer support

### Professional Support
For enterprise deployments or complex issues, consider professional support services.

---

**Remember**: Most issues can be resolved by carefully following the installation steps and ensuring all requirements are met. When in doubt, start with the [Quick Start Guide](QUICK_START.md) for a clean installation.