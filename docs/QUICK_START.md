# Quick Start Guide - AI Assistant

## 🚀 Get Started in 5 Minutes

This guide gets you up and running with the AI Assistant quickly.

### Prerequisites
- Python 3.10 or 3.11
- 8GB+ RAM
- Internet connection

### Step 1: Install
```bash
# Clone the repository
git clone https://github.com/Drmusab/-1995.git
cd -1995

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the assistant
pip install -e .
```

### Step 2: Configure
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional for basic usage)
# Add your OpenAI API key if you have one:
# OPENAI_API_KEY=your_key_here
```

### Step 3: Validate & Run
```bash
# Validate installation (recommended)
python validate_and_demo.py

# Start interactive assistant
PYTHONPATH=/path/to/-1995 python -m src.cli

# Or start server mode
PYTHONPATH=/path/to/-1995 python -m src.main
```

## 💡 Common Use Cases

### 1. Interactive Chat
```bash
# Start interactive mode
python -m src.cli

# Type your questions
> Hello, how can you help me?
> What's the weather like today?
> Explain quantum computing
```

### 2. API Usage
```bash
# Start server
python -m src.main

# Test with curl
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "session_id": "test"}'
```

### 3. Enable Advanced Features

#### Speech Processing
```bash
# Install speech dependencies
pip install pyaudio speechrecognition pyttsx3

# Start with speech
python -m src.cli --enable-speech
```

#### Vision Processing
```bash
# Install vision dependencies
pip install opencv-python pillow

# Start with vision
python -m src.cli --enable-vision
```

## 🔧 Troubleshooting

### Validate Your Installation
```bash
# Run the validation script to check everything is working
python validate_and_demo.py

# Run full demo with advanced features
python validate_and_demo.py --full-demo
```

### Can't find modules?
```bash
# Set Python path
export PYTHONPATH=/path/to/-1995:$PYTHONPATH

# Or reinstall in development mode
pip install -e .
```

### Missing dependencies?
```bash
# Install optional dependencies as needed
pip install asyncpg psycopg2-binary  # For PostgreSQL
pip install transformers  # For advanced NLP
```

### Need help?
```bash
# Check system status
python -m src.cli -c "status"

# Get help
python -m src.cli --help
```

## 📖 Next Steps

1. Read the [Comprehensive Guide](COMPREHENSIVE_BUILD_AND_USAGE_GUIDE.md) for detailed setup
2. Check the [Architecture Documentation](architecture/) to understand the system
3. Explore the [API Documentation](http://localhost:8000/docs) when running the server
4. Join the community discussions for support

## 🐳 Docker Quick Start

If you prefer Docker:

```bash
# Build and run
docker build -f docker/Dockerfile -t ai-assistant .
docker run -p 8000:8000 ai-assistant

# Or use docker-compose
cd docker
docker-compose up -d
```

Access the assistant at `http://localhost:8000`

---

**Need more details?** See the [full documentation](COMPREHENSIVE_BUILD_AND_USAGE_GUIDE.md) for advanced configuration, deployment options, and troubleshooting.