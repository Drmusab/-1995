#!/bin/bash
# ==============================================================================
# AI Assistant System Setup Script
# Author: Drmusab
# Created: 2025-07-05
# Last Modified: 2025-07-05 14:28:26
#
# This script sets up the complete environment for the AI Assistant system,
# including dependencies, directory structure, configurations, and initial
# data. It supports both development and production environments.
# ==============================================================================

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/data/logs/setup_$(date +%Y%m%d_%H%M%S).log"
PYTHON_MIN_VERSION="3.10"
PYTHON_MAX_VERSION="3.11"

# Define color codes for console output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Setup environment selection
ENV_TYPE="development" # Default to development
FORCE_MODE=false
SKIP_DOWNLOADS=false
USE_GPU=false
NO_INTERACTIVE=false
VERBOSE=false

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    echo -e "${timestamp} - $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo -e "\n${BLUE}====================================================================${NC}"
    echo -e "${BLUE}= $1${NC}"
    echo -e "${BLUE}====================================================================${NC}"
    log "SECTION: $1"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
    log "SUCCESS: $1"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    log "WARNING: $1"
}

log_error() {
    echo -e "${RED}✗ ERROR: $1${NC}" >&2
    log "ERROR: $1"
}

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -e, --environment <env>   Set environment type (development, production, staging)"
    echo "  -f, --force               Force setup even if already initialized"
    echo "  -s, --skip-downloads      Skip downloading large files like models"
    echo "  -g, --gpu                 Configure for GPU usage"
    echo "  -y, --yes                 Non-interactive mode, assume yes to all prompts"
    echo "  -v, --verbose             Enable verbose output"
    echo "  -h, --help                Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--environment)
            ENV_TYPE="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_MODE=true
            shift
            ;;
        -s|--skip-downloads)
            SKIP_DOWNLOADS=true
            shift
            ;;
        -g|--gpu)
            USE_GPU=true
            shift
            ;;
        -y|--yes)
            NO_INTERACTIVE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate environment type
if [[ ! "$ENV_TYPE" =~ ^(development|production|staging)$ ]]; then
    log_error "Invalid environment type: $ENV_TYPE. Must be one of: development, production, staging"
    exit 1
fi

# Set config file based on environment
CONFIG_FILE="${PROJECT_ROOT}/configs/environments/${ENV_TYPE}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Print setup information
log_section "AI Assistant Setup - ${ENV_TYPE} Environment"
log "Setup started by: $(whoami)"
log "Project root: ${PROJECT_ROOT}"
log "Environment: ${ENV_TYPE}"
log "Configuration: ${CONFIG_FILE}"
log "GPU support: $([ "$USE_GPU" == true ] && echo 'enabled' || echo 'disabled')"

# Check for existing setup
if [[ -f "${PROJECT_ROOT}/.setup_complete" ]] && [[ "$FORCE_MODE" == false ]]; then
    log_warning "Previous setup detected. Use --force to override."
    if [[ "$NO_INTERACTIVE" == false ]]; then
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Setup aborted by user."
            exit 0
        fi
    else
        log "Setup aborted due to previous installation."
        exit 0
    fi
fi

# ==============================================================================
# System Requirements Check
# ==============================================================================
log_section "Checking System Requirements"

# Check OS
OS="$(uname -s)"
log "Detected OS: $OS"
case "$OS" in
    Linux*)
        ;;
    Darwin*)
        ;;
    *)
        log_warning "Unsupported OS: $OS. This script is primarily tested on Linux and macOS."
        ;;
esac

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    log_error "Python not found. Please install Python ${PYTHON_MIN_VERSION} or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log "Detected Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < $PYTHON_MIN_VERSION" | bc -l) -eq 1 ]]; then
    log_error "Python version too old. Required minimum: ${PYTHON_MIN_VERSION}"
    exit 1
fi

if [[ $(echo "$PYTHON_VERSION > $PYTHON_MAX_VERSION" | bc -l) -eq 1 ]]; then
    log_warning "Python version ${PYTHON_VERSION} is newer than the recommended maximum (${PYTHON_MAX_VERSION})."
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    log_error "pip3 not found. Please install pip."
    exit 1
fi

# Check for poetry
if command -v poetry &> /dev/null; then
    log "Poetry found, version: $(poetry --version)"
else
    log "Poetry not found, will install..."
    pip3 install poetry
    log_success "Poetry installed"
fi

# Check for Docker if needed
if [[ "$ENV_TYPE" == "production" ]]; then
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not found. Docker is recommended for production deployments."
    else
        log "Docker found, version: $(docker --version)"
        
        if ! command -v docker-compose &> /dev/null; then
            log_warning "Docker Compose not found. Required for production deployment."
        else
            log "Docker Compose found, version: $(docker-compose --version)"
        fi
    fi
fi

# Check for GPU support if requested
if [[ "$USE_GPU" == true ]]; then
    log "Checking GPU capabilities..."
    
    # Check for NVIDIA GPU and CUDA
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
        log "CUDA version: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}')"
    else
        log_warning "NVIDIA GPU not detected or nvidia-smi not found. Falling back to CPU mode."
        USE_GPU=false
    fi
    
    # Check for PyTorch CUDA support
    if [[ "$USE_GPU" == true ]]; then
        TORCH_CUDA=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
        if [[ "$TORCH_CUDA" == "True" ]]; then
            log_success "PyTorch CUDA support confirmed"
        else
            log_warning "PyTorch CUDA support not available. Will install CUDA-enabled PyTorch."
        fi
    fi
fi

# ==============================================================================
# Create Directory Structure
# ==============================================================================
log_section "Creating Directory Structure"

# Define directory paths
DATA_DIRS=(
    "data/models/checkpoints"
    "data/models/fine_tuned"
    "data/datasets/training"
    "data/datasets/validation"
    "data/datasets/test"
    "data/cache/vector_cache"
    "data/cache/response_cache"
    "data/user_data/preferences"
    "data/user_data/history"
    "data/user_data/personalization"
    "data/knowledge_base/documents"
    "data/knowledge_base/embeddings"
    "data/knowledge_base/graphs"
    "data/logs/application"
    "data/logs/access"
    "data/logs/error"
    "data/logs/audit"
)

# Create directories
for dir in "${DATA_DIRS[@]}"; do
    mkdir -p "${PROJECT_ROOT}/$dir"
    log "Created directory: $dir"
done

# Create empty placeholder files to preserve directory structure in git
for dir in "${DATA_DIRS[@]}"; do
    touch "${PROJECT_ROOT}/$dir/.gitkeep"
done

log_success "Directory structure created"

# ==============================================================================
# Environment Setup
# ==============================================================================
log_section "Setting Up Python Environment"

# Setup virtual environment if not using system Python
if [[ "$ENV_TYPE" == "development" ]]; then
    log "Setting up Poetry environment..."
    cd "$PROJECT_ROOT"
    
    # Configure poetry for in-project virtual environments
    poetry config virtualenvs.in-project true
    
    # Install dependencies based on environment
    if [[ "$USE_GPU" == true ]]; then
        # Install with GPU support
        log "Installing dependencies with GPU support..."
        poetry install --extras "gpu"
    else
        # Install regular dependencies
        log "Installing dependencies..."
        poetry install
    fi
    
    # Activate virtual environment
    log "Activating virtual environment..."
    . "$(poetry env info --path)/bin/activate"
    
    log_success "Poetry environment setup complete"
else
    # For production, just ensure requirements are installed
    log "Installing production requirements..."
    if [[ "$USE_GPU" == true ]]; then
        pip3 install -r "${PROJECT_ROOT}/requirements/production.txt" -r "${PROJECT_ROOT}/requirements/gpu.txt"
    else
        pip3 install -r "${PROJECT_ROOT}/requirements/production.txt"
    fi
    log_success "Production requirements installed"
fi

# ==============================================================================
# Configuration Setup
# ==============================================================================
log_section "Setting Up Configuration"

# Create .env file from template if it doesn't exist
if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
    log "Creating .env file from template..."
    cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
    
    # Generate random secret key
    SECRET_KEY=$(openssl rand -hex 32)
    ENCRYPTION_KEY=$(openssl rand -hex 16)
    NODE_ID="node_$(openssl rand -hex 4)"
    
    # Update .env file with generated values
    sed -i.bak "s/SECRET_KEY=.*/SECRET_KEY=${SECRET_KEY}/" "${PROJECT_ROOT}/.env"
    sed -i.bak "s/ENCRYPTION_KEY=.*/ENCRYPTION_KEY=${ENCRYPTION_KEY}/" "${PROJECT_ROOT}/.env"
    sed -i.bak "s/NODE_ID=.*/NODE_ID=${NODE_ID}/" "${PROJECT_ROOT}/.env"
    sed -i.bak "s/ENVIRONMENT=.*/ENVIRONMENT=${ENV_TYPE}/" "${PROJECT_ROOT}/.env"
    
    if [[ "$USE_GPU" == true ]]; then
        sed -i.bak "s/USE_GPU=.*/USE_GPU=true/" "${PROJECT_ROOT}/.env"
    fi
    
    # Remove backup file
    rm -f "${PROJECT_ROOT}/.env.bak"
    
    log_success ".env file created and configured"
else
    log ".env file already exists, keeping existing configuration"
fi

# Update configuration file if needed
log "Validating configuration file: ${CONFIG_FILE}"
$PYTHON_CMD -m src.core.config.validators.config_validator "${CONFIG_FILE}"
log_success "Configuration validated"

# ==============================================================================
# Database Setup
# ==============================================================================
if [[ "$ENV_TYPE" != "development" ]]; then
    log_section "Setting Up Databases"
    
    log "Note: For production environments, databases should be configured separately."
    log "This script will only check connections to existing databases."
    
    # Check database connections
    if [[ "$NO_INTERACTIVE" == false ]]; then
        read -p "Would you like to check database connections? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # This would run a database connection check script
            log "Checking database connections..."
            $PYTHON_CMD -m src.cli check-connections
        fi
    fi
else
    log_section "Setting Up Development Databases"
    
    log "Setting up development databases..."
    
    # Check if docker-compose is available
    if command -v docker-compose &> /dev/null; then
        log "Starting development databases using docker-compose..."
        
        # Start only the database services for development
        cd "$PROJECT_ROOT"
        docker-compose -f docker/docker-compose.dev.yml up -d redis postgres vector-db
        
        log "Waiting for databases to be ready..."
        sleep 5
        
        log_success "Development databases started"
    else
        log_warning "Docker Compose not found. Skipping automatic database setup."
        log "Please set up development databases manually."
    fi
    
    # Run database migrations
    log "Running database migrations..."
    $PYTHON_CMD -m src.cli migrate
    log_success "Database migrations complete"
fi

# ==============================================================================
# Model Downloads
# ==============================================================================
if [[ "$SKIP_DOWNLOADS" == false ]]; then
    log_section "Downloading Required Models"
    
    # Create model registry if it doesn't exist
    if [[ ! -f "${PROJECT_ROOT}/data/models/model_registry.json" ]]; then
        echo '{"models": {}}' > "${PROJECT_ROOT}/data/models/model_registry.json"
    fi
    
    # Download required models based on configuration
    log "Downloading required models based on configuration..."
    $PYTHON_CMD -m src.cli download-models
    
    log_success "Model downloads complete"
else
    log_section "Skipping Model Downloads"
    log "Model downloads skipped. You will need to download models manually."
    log "Use: python -m src.cli download-models"
fi

# ==============================================================================
# Development Tools Setup
# ==============================================================================
if [[ "$ENV_TYPE" == "development" ]]; then
    log_section "Setting Up Development Tools"
    
    # Set up pre-commit hooks
    log "Setting up pre-commit hooks..."
    cd "$PROJECT_ROOT"
    pre-commit install
    
    # Install development tools
    log "Installing development tools..."
    pip install -r "${PROJECT_ROOT}/requirements/development.txt"
    
    log_success "Development tools setup complete"
fi

# ==============================================================================
# Finalization
# ==============================================================================
log_section "Setup Finalization"

# Mark setup as complete
touch "${PROJECT_ROOT}/.setup_complete"
echo "$(date -u +"%Y-%m-%d %H:%M:%S UTC")" > "${PROJECT_ROOT}/.setup_complete"
echo "Environment: ${ENV_TYPE}" >> "${PROJECT_ROOT}/.setup_complete"
echo "Setup by: $(whoami)" >> "${PROJECT_ROOT}/.setup_complete"

# Create initial health status
log "Creating initial health status..."
mkdir -p "${PROJECT_ROOT}/data/health"
cat > "${PROJECT_ROOT}/data/health/status.json" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "status": "initializing",
    "version": "$(cat ${PROJECT_ROOT}/VERSION 2>/dev/null || echo '1.0.0')",
    "environment": "${ENV_TYPE}",
    "components": {
        "core_engine": "not_started",
        "component_manager": "not_started",
        "session_manager": "not_started",
        "workflow_orchestrator": "not_started",
        "interaction_handler": "not_started",
        "plugin_manager": "not_started"
    }
}
EOF

# Print success message
log_success "AI Assistant setup completed successfully!"
log "Environment: ${ENV_TYPE}"
log "Log file: ${LOG_FILE}"

# Print next steps
cat << EOF

${GREEN}=========================================================${NC}
${GREEN}                   SETUP COMPLETE                        ${NC}
${GREEN}=========================================================${NC}

${BLUE}Next steps:${NC}

1. Review the configuration in configs/environments/${ENV_TYPE}.yaml
2. Set up API keys in your .env file for external services
3. Start the system:
   - Development: python -m src.main --dev
   - Production: Use docker-compose or the deploy script

${BLUE}Documentation:${NC}
- Development guide: docs/development/
- API documentation: docs/api/
- Deployment guide: docs/deployment/

For more information, see the README.md file.

EOF

exit 0
