#!/bin/bash
# ==============================================================================
# AI Assistant Deployment Script
# Author: Drmusab
# Created: 2025-07-05
# Last Modified: 2025-07-05 20:05:03
#
# This script handles deployment of the AI Assistant system to various
# environments using multiple deployment methods (Docker, Kubernetes, direct).
# It integrates with the core system components to ensure proper functioning
# after deployment.
# ==============================================================================

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/data/logs"
DEPLOY_LOG="${LOG_DIR}/deploy_$(date +%Y%m%d_%H%M%S).log"
TODAY="$(date +%Y-%m-%d)"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
DEPLOY_DIR="${PROJECT_ROOT}/data/deployments/${TIMESTAMP}"
BACKUP_DIR="${PROJECT_ROOT}/data/backups/$(date +%Y%m%d)"

# Default values
ENVIRONMENT="development"  # Environment: development, staging, production
DEPLOY_METHOD="docker"     # Deployment method: docker, kubernetes, direct
TARGET="local"             # Deployment target: local, remote, cloud
VERSION="latest"           # Version to deploy
SKIP_TESTS=false           # Skip running tests
SKIP_LINT=false            # Skip linting
SKIP_MIGRATIONS=false      # Skip database migrations
SKIP_BACKUP=false          # Skip backups
ROLLBACK_ON_FAILURE=true   # Automatically rollback on failure
VERBOSE=false              # Verbose output
FORCE=false                # Force deployment without confirmation
DRY_RUN=false              # Dry run (don't actually deploy)
NOTIFY=true                # Send notifications
REMOTE_HOST=""             # Remote host for direct deployment
REMOTE_PATH=""             # Remote path for direct deployment
REMOTE_USER=""             # Remote user for direct deployment
K8S_NAMESPACE="default"    # Kubernetes namespace
K8S_CONTEXT=""             # Kubernetes context
DOCKER_REGISTRY=""         # Docker registry

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Ensure log and deployment directories exist
mkdir -p "$LOG_DIR"
mkdir -p "$DEPLOY_DIR"
mkdir -p "$BACKUP_DIR"

# Logging functions
log() {
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    echo -e "${timestamp} - $1" | tee -a "$DEPLOY_LOG"
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

# Usage information
usage() {
    echo "Usage: $0 [options]"
    echo 
    echo "Options:"
    echo "  -e, --environment ENV    Environment: development, staging, production"
    echo "                           (default: development)"
    echo "  -m, --method METHOD      Deployment method: docker, kubernetes, direct"
    echo "                           (default: docker)"
    echo "  -t, --target TARGET      Deployment target: local, remote, cloud"
    echo "                           (default: local)"
    echo "  -v, --version VERSION    Version to deploy (default: latest)"
    echo "  --skip-tests             Skip running tests"
    echo "  --skip-lint              Skip linting"
    echo "  --skip-migrations        Skip database migrations"
    echo "  --skip-backup            Skip backups"
    echo "  --no-rollback            Don't automatically rollback on failure"
    echo "  --verbose                Show verbose output"
    echo "  -f, --force              Force deployment without confirmation"
    echo "  --dry-run                Don't actually deploy"
    echo "  --no-notify              Don't send notifications"
    echo 
    echo "Docker-specific options:"
    echo "  --registry REGISTRY      Docker registry to use"
    echo 
    echo "Kubernetes-specific options:"
    echo "  --namespace NAMESPACE    Kubernetes namespace (default: default)"
    echo "  --context CONTEXT        Kubernetes context"
    echo 
    echo "Direct deployment options:"
    echo "  --host HOST              Remote host for direct deployment"
    echo "  --path PATH              Remote path for direct deployment"
    echo "  --user USER              Remote user for direct deployment"
    echo 
    echo "  -h, --help               Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -m|--method)
            DEPLOY_METHOD="$2"
            shift 2
            ;;
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-lint)
            SKIP_LINT=true
            shift
            ;;
        --skip-migrations)
            SKIP_MIGRATIONS=true
            shift
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-notify)
            NOTIFY=false
            shift
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --namespace)
            K8S_NAMESPACE="$2"
            shift 2
            ;;
        --context)
            K8S_CONTEXT="$2"
            shift 2
            ;;
        --host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        --path)
            REMOTE_PATH="$2"
            shift 2
            ;;
        --user)
            REMOTE_USER="$2"
            shift 2
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

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be one of: development, staging, production"
    exit 1
fi

# Validate deployment method
if [[ ! "$DEPLOY_METHOD" =~ ^(docker|kubernetes|direct)$ ]]; then
    log_error "Invalid deployment method: $DEPLOY_METHOD. Must be one of: docker, kubernetes, direct"
    exit 1
fi

# Validate target
if [[ ! "$TARGET" =~ ^(local|remote|cloud)$ ]]; then
    log_error "Invalid target: $TARGET. Must be one of: local, remote, cloud"
    exit 1
fi

# Validate method-specific requirements
if [ "$DEPLOY_METHOD" = "kubernetes" ]; then
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Required for Kubernetes deployment."
        exit 1
    fi
elif [ "$DEPLOY_METHOD" = "direct" ] && [ "$TARGET" = "remote" ]; then
    if [ -z "$REMOTE_HOST" ]; then
        log_error "Remote host is required for direct remote deployment."
        exit 1
    fi
fi

# Function to activate virtual environment if exists
activate_venv() {
    if [ -d "${PROJECT_ROOT}/.venv" ]; then
        source "${PROJECT_ROOT}/.venv/bin/activate"
        log "Activated virtual environment: $(which python)"
    elif [ -n "$VIRTUAL_ENV" ]; then
        log "Using existing virtual environment: $VIRTUAL_ENV"
    else
        log_warning "No virtual environment detected. Using system Python."
    fi
}

# Function to load environment variables
load_env_vars() {
    log_section "Loading Environment Configuration"
    
    # Load specific environment file if it exists
    ENV_FILE="${PROJECT_ROOT}/.env.${ENVIRONMENT}"
    if [ ! -f "$ENV_FILE" ]; then
        ENV_FILE="${PROJECT_ROOT}/.env"
    fi
    
    if [ -f "$ENV_FILE" ]; then
        log "Loading environment variables from $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a
    else
        log_warning "No environment file found at $ENV_FILE"
    fi
    
    # Set environment variable for Python scripts
    export ENVIRONMENT="$ENVIRONMENT"
    export DEPLOYMENT_TARGET="$TARGET"
    export DEPLOYMENT_METHOD="$DEPLOY_METHOD"
    export DEPLOYMENT_VERSION="$VERSION"
    
    # Load specific config based on environment
    export CONFIG_PATH="${PROJECT_ROOT}/configs/environments/${ENVIRONMENT}.yaml"
    
    if [ ! -f "$CONFIG_PATH" ]; then
        log_error "Configuration file not found: $CONFIG_PATH"
        exit 1
    fi
    
    log_success "Environment configuration loaded: $ENVIRONMENT"
}

# Function to validate git status
validate_git_status() {
    log_section "Validating Git Status"
    
    # Check if we're in a git repository
    if [ ! -d "${PROJECT_ROOT}/.git" ]; then
        log_warning "Not a git repository. Skipping git validation."
        return 0
    fi
    
    # Check for uncommitted changes
    if [ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]; then
        log_warning "There are uncommitted changes in the repository"
        
        if [ "$FORCE" = false ]; then
            read -p "Continue deployment with uncommitted changes? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Deployment cancelled due to uncommitted changes"
                exit 0
            fi
        fi
    else
        log_success "No uncommitted changes in the repository"
    fi
    
    # Get current branch and commit
    CURRENT_BRANCH=$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD)
    CURRENT_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
    
    log "Current branch: $CURRENT_BRANCH"
    log "Current commit: $CURRENT_COMMIT"
    
    # Check if we're on the correct branch for the environment
    case "$ENVIRONMENT" in
        production)
            if [ "$CURRENT_BRANCH" != "main" ] && [ "$CURRENT_BRANCH" != "master" ]; then
                log_warning "Production deployments should be done from main/master branch"
                
                if [ "$FORCE" = false ]; then
                    read -p "Continue deployment from $CURRENT_BRANCH branch? [y/N] " -n 1 -r
                    echo
                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                        log "Deployment cancelled due to branch mismatch"
                        exit 0
                    fi
                fi
            fi
            ;;
        staging)
            if [ "$CURRENT_BRANCH" != "staging" ] && [ "$CURRENT_BRANCH" != "develop" ]; then
                log_warning "Staging deployments are typically done from staging/develop branch"
                
                if [ "$FORCE" = false ]; then
                    read -p "Continue deployment from $CURRENT_BRANCH branch? [y/N] " -n 1 -r
                    echo
                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                        log "Deployment cancelled due to branch mismatch"
                        exit 0
                    fi
                fi
            fi
            ;;
    esac
    
    log_success "Git validation passed"
}

# Function to run code quality checks
run_code_quality_checks() {
    log_section "Running Code Quality Checks"
    
    # Skip if requested
    if [ "$SKIP_LINT" = true ]; then
        log "Skipping code quality checks as requested"
        return 0
    fi
    
    # Run linting
    log "Running linter..."
    if "${SCRIPT_DIR}/lint.sh" --ci; then
        log_success "Linting passed"
    else
        log_warning "Linting found issues"
        
        if [ "$FORCE" = false ]; then
            read -p "Continue deployment despite linting issues? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Deployment cancelled due to linting issues"
                exit 0
            fi
        fi
    fi
    
    log_success "Code quality checks completed"
}

# Function to run tests
run_tests() {
    log_section "Running Tests"
    
    # Skip if requested
    if [ "$SKIP_TESTS" = true ]; then
        log "Skipping tests as requested"
        return 0
    fi
    
    # Determine which tests to run based on environment
    local test_type="all"
    case "$ENVIRONMENT" in
        production)
            test_type="smoke"  # Only smoke tests for production deployment
            ;;
        staging)
            test_type="integration"  # Integration tests for staging
            ;;
        development)
            test_type="unit"  # Unit tests for development
            ;;
    esac
    
    log "Running $test_type tests..."
    if "${SCRIPT_DIR}/test.sh" --type "$test_type" --ci; then
        log_success "Tests passed"
    else
        log_error "Tests failed"
        
        if [ "$FORCE" = false ]; then
            read -p "Continue deployment despite test failures? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Deployment cancelled due to test failures"
                exit 0
            fi
        fi
    fi
    
    log_success "Test suite completed"
}

# Function to run database migrations
run_migrations() {
    log_section "Running Database Migrations"
    
    # Skip if requested
    if [ "$SKIP_MIGRATIONS" = true ]; then
        log "Skipping database migrations as requested"
        return 0
    fi
    
    log "Running database migrations..."
    if "${SCRIPT_DIR}/migrate.sh" --environment "$ENVIRONMENT"; then
        log_success "Database migrations completed successfully"
    else
        log_error "Database migrations failed"
        
        if [ "$FORCE" = false ]; then
            read -p "Continue deployment despite migration failures? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Deployment cancelled due to migration failures"
                exit 0
            fi
        fi
    fi
    
    log_success "Database migrations completed"
}

# Function to create backup
create_backup() {
    log_section "Creating Backup"
    
    # Skip if requested
    if [ "$SKIP_BACKUP" = true ]; then
        log "Skipping backup as requested"
        return 0
    fi
    
    # Skip backup for dry run
    if [ "$DRY_RUN" = true ]; then
        log "Skipping backup for dry run"
        return 0
    }
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    log "Backing up database..."
    if python -m src.cli backup-db --environment "$ENVIRONMENT" --output "${BACKUP_DIR}/db_${TIMESTAMP}.sql"; then
        log_success "Database backup created"
    else
        log_warning "Database backup failed"
        
        if [ "$FORCE" = false ]; then
            read -p "Continue deployment without database backup? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Deployment cancelled due to backup failure"
                exit 0
            fi
        fi
    fi
    
    # Backup configuration
    log "Backing up configuration..."
    cp -r "${PROJECT_ROOT}/configs" "${BACKUP_DIR}/configs_${TIMESTAMP}"
    
    # Backup user data
    if [ -d "${PROJECT_ROOT}/data/user_data" ]; then
        log "Backing up user data..."
        cp -r "${PROJECT_ROOT}/data/user_data" "${BACKUP_DIR}/user_data_${TIMESTAMP}"
    fi
    
    log_success "Backup completed: ${BACKUP_DIR}"
}

# Function to check core components
check_core_components() {
    log_section "Checking Core System Components"
    
    # Core components that must be present
    CORE_COMPONENTS=(
        "src/assistant/core_engine.py"
        "src/assistant/component_manager.py"
        "src/assistant/workflow_orchestrator.py"
        "src/assistant/interaction_handler.py"
        "src/assistant/session_manager.py"
        "src/assistant/plugin_manager.py"
        "src/core/config/loader.py"
        "src/core/dependency_injection.py"
        "src/core/events/event_bus.py"
        "src/core/health_check.py"
    )
    
    # Check if core components exist
    MISSING=0
    for component in "${CORE_COMPONENTS[@]}"; do
        if [ ! -f "${PROJECT_ROOT}/$component" ]; then
            log_warning "Core component not found: $component"
            MISSING=$((MISSING + 1))
        fi
    done
    
    if [ $MISSING -gt 0 ]; then
        log_error "$MISSING core components are missing"
        
        if [ "$FORCE" = false ]; then
            read -p "Continue deployment with missing core components? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Deployment cancelled due to missing core components"
                exit 0
            fi
        fi
    else
        log_success "All core components exist"
    fi
    
    # Verify core component integrity
    log "Verifying core component integrity..."
    
    if python -m src.cli verify-core-components --environment "$ENVIRONMENT"; then
        log_success "Core component integrity verified"
    else
        log_warning "Core component integrity check failed"
        
        if [ "$FORCE" = false ]; then
            read -p "Continue deployment despite integrity check failure? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Deployment cancelled due to integrity check failure"
                exit 0
            fi
        fi
    fi
}

# Function to build Docker image
build_docker_image() {
    log_section "Building Docker Image"
    
    # Skip for dry run
    if [ "$DRY_RUN" = true ]; then
        log "Skipping Docker image build for dry run"
        return 0
    }
    
    # Determine Docker image name and tag
    local image_name="ai_assistant"
    local image_tag="${VERSION}"
    local full_image_name="${image_name}:${image_tag}"
    
    if [ -n "$DOCKER_REGISTRY" ]; then
        full_image_name="${DOCKER_REGISTRY}/${full_image_name}"
    fi
    
    log "Building Docker image: $full_image_name"
    
    # Build the Docker image
    docker build -t "$full_image_name" -f "${PROJECT_ROOT}/docker/Dockerfile" "${PROJECT_ROOT}"
    
    if [ $? -eq 0 ]; then
        log_success "Docker image built successfully: $full_image_name"
        
        # Push image if registry is specified
        if [ -n "$DOCKER_REGISTRY" ]; then
            log "Pushing Docker image to registry: $DOCKER_REGISTRY"
            docker push "$full_image_name"
            
            if [ $? -eq 0 ]; then
                log_success "Docker image pushed successfully"
            else
                log_error "Failed to push Docker image"
                return 1
            fi
        fi
        
        # Save image info for deployment
        echo "$full_image_name" > "${DEPLOY_DIR}/docker_image.txt"
        return 0
    else
        log_error "Failed to build Docker image"
        return 1
    fi
}

# Function to deploy using Docker Compose
deploy_docker_compose() {
    log_section "Deploying with Docker Compose"
    
    # Skip for dry run
    if [ "$DRY_RUN" = true ]; then
        log "Skipping Docker Compose deployment for dry run"
        return 0
    }
    
    # Determine Docker Compose file
    local compose_file="${PROJECT_ROOT}/docker/docker-compose.yml"
    if [ "$ENVIRONMENT" = "production" ]; then
        compose_file="${PROJECT_ROOT}/docker/docker-compose.prod.yml"
    elif [ "$ENVIRONMENT" = "development" ]; then
        compose_file="${PROJECT_ROOT}/docker/docker-compose.dev.yml"
    fi
    
    if [ ! -f "$compose_file" ]; then
        log_error "Docker Compose file not found: $compose_file"
        return 1
    }
    
    # Set deployment environment variables
    export TAG="${VERSION}"
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null; then
        log_warning "docker-compose command not found, using 'docker compose' instead"
        DOCKER_COMPOSE_CMD="docker compose"
    else
        DOCKER_COMPOSE_CMD="docker-compose"
    fi
    
    log "Deploying using Docker Compose: $compose_file"
    
    # Stop existing containers
    log "Stopping existing containers..."
    $DOCKER_COMPOSE_CMD -f "$compose_file" down || true
    
    # Start new containers
    log "Starting new containers..."
    $DOCKER_COMPOSE_CMD -f "$compose_file" up -d
    
    if [ $? -eq 0 ]; then
        log_success "Docker Compose deployment completed successfully"
        
        # Copy compose file to deployment directory
        cp "$compose_file" "${DEPLOY_DIR}/docker-compose.yml"
        
        # Save deployment environment variables
        env | grep -E '^(TAG|ENVIRONMENT|DOCKER_REGISTRY)' > "${DEPLOY_DIR}/deployment_env.txt"
        
        return 0
    else
        log_error "Docker Compose deployment failed"
        return 1
    fi
}

# Function to deploy to Kubernetes
deploy_kubernetes() {
    log_section "Deploying to Kubernetes"
    
    # Skip for dry run
    if [ "$DRY_RUN" = true ]; then
        log "Skipping Kubernetes deployment for dry run"
        return 0
    }
    
    # Set Kubernetes context if specified
    if [ -n "$K8S_CONTEXT" ]; then
        log "Setting Kubernetes context: $K8S_CONTEXT"
        kubectl config use-context "$K8S_CONTEXT"
        
        if [ $? -ne 0 ]; then
            log_error "Failed to set Kubernetes context: $K8S_CONTEXT"
            return 1
        fi
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$K8S_NAMESPACE" &> /dev/null; then
        log "Creating namespace: $K8S_NAMESPACE"
        kubectl create namespace "$K8S_NAMESPACE"
    fi
    
    # Determine manifest directory
    local manifest_dir="${PROJECT_ROOT}/infrastructure/kubernetes"
    if [ ! -d "$manifest_dir" ]; then
        log_error "Kubernetes manifest directory not found: $manifest_dir"
        return 1
    fi
    
    # Apply ConfigMap with environment-specific configuration
    log "Applying ConfigMap for environment: $ENVIRONMENT"
    kubectl create configmap ai-assistant-config \
        --from-file="${PROJECT_ROOT}/configs/environments/${ENVIRONMENT}.yaml" \
        --namespace="$K8S_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Set image version in deployments
    log "Setting image version in deployment manifests..."
    
    for deployment_file in "${manifest_dir}/deployments"/*.yaml; do
        if [ -f "$deployment_file" ]; then
            # Create a temporary file with the image version updated
            sed "s|image: ai_assistant:.*|image: ai_assistant:${VERSION}|g" "$deployment_file" > "${DEPLOY_DIR}/$(basename "$deployment_file")"
            
            # Apply the updated deployment
            log "Applying deployment: $(basename "$deployment_file")"
            kubectl apply -f "${DEPLOY_DIR}/$(basename "$deployment_file")" --namespace="$K8S_NAMESPACE"
        fi
    done
    
    # Apply other Kubernetes resources
    for resource_type in services configmaps secrets; do
        if [ -d "${manifest_dir}/${resource_type}" ]; then
            log "Applying $resource_type..."
            kubectl apply -f "${manifest_dir}/${resource_type}" --namespace="$K8S_NAMESPACE"
        fi
    done
    
    # Wait for deployments to be ready
    log "Waiting for deployments to be ready..."
    kubectl rollout status deployment/ai-assistant --namespace="$K8S_NAMESPACE" --timeout=300s
    
    if [ $? -eq 0 ]; then
        log_success "Kubernetes deployment completed successfully"
        
        # Save deployment info
        kubectl get deployments --namespace="$K8S_NAMESPACE" -o wide > "${DEPLOY_DIR}/kubernetes_deployments.txt"
        kubectl get services --namespace="$K8S_NAMESPACE" -o wide > "${DEPLOY_DIR}/kubernetes_services.txt"
        kubectl get pods --namespace="$K8S_NAMESPACE" -o wide > "${DEPLOY_DIR}/kubernetes_pods.txt"
        
        return 0
    else
        log_error "Kubernetes deployment failed"
        return 1
    fi
}

# Function to deploy directly to server
deploy_direct() {
    log_section "Deploying Directly to Server"
    
    # Skip for dry run
    if [ "$DRY_RUN" = true ]; then
        log "Skipping direct deployment for dry run"
        return 0
    }
    
    # Check if it's a local or remote deployment
    if [ "$TARGET" = "local" ]; then
        # Local deployment
        log "Performing local deployment..."
        
        # Create deployment directory
        DEPLOY_PATH="${PROJECT_ROOT}/local_deploy"
        mkdir -p "$DEPLOY_PATH"
        
        # Copy necessary files
        log "Copying files to deployment directory..."
        rsync -av --exclude=".git" --exclude=".venv" --exclude="__pycache__" \
            "${PROJECT_ROOT}/" "${DEPLOY_PATH}/"
        
        # Create virtual environment and install requirements
        log "Setting up virtual environment..."
        python -m venv "${DEPLOY_PATH}/.venv"
        source "${DEPLOY_PATH}/.venv/bin/activate"
        pip install -r "${DEPLOY_PATH}/requirements/production.txt"
        
        # Set up configuration
        log "Setting up configuration..."
        cp "${PROJECT_ROOT}/configs/environments/${ENVIRONMENT}.yaml" "${DEPLOY_PATH}/configs/environments/current.yaml"
        
        # Start the application
        log "Starting the application..."
        nohup python -m src.main --environment "$ENVIRONMENT" > "${DEPLOY_PATH}/nohup.out" 2>&1 &
        
        # Save PID
        echo $! > "${DEPLOY_PATH}/app.pid"
        
        # Save deployment info
        echo "Local deployment path: $DEPLOY_PATH" > "${DEPLOY_DIR}/direct_deployment.txt"
        echo "Started at: $(date)" >> "${DEPLOY_DIR}/direct_deployment.txt"
        echo "PID: $(cat "${DEPLOY_PATH}/app.pid")" >> "${DEPLOY_DIR}/direct_deployment.txt"
        
        log_success "Local deployment completed successfully"
        return 0
    else
        # Remote deployment
        log "Performing remote deployment to $REMOTE_HOST..."
        
        if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_USER" ]; then
            log_error "Remote host and user are required for remote deployment"
            return 1
        fi
        
        if [ -z "$REMOTE_PATH" ]; then
            REMOTE_PATH="/opt/ai_assistant"
        fi
        
        # Create deployment package
        log "Creating deployment package..."
        DEPLOY_PACKAGE="${DEPLOY_DIR}/deploy_package.tar.gz"
        tar --exclude=".git" --exclude=".venv" --exclude="__pycache__" \
            -czf "$DEPLOY_PACKAGE" -C "${PROJECT_ROOT}" .
        
        # Copy package to remote server
        log "Copying package to remote server..."
        scp "$DEPLOY_PACKAGE" "${REMOTE_USER}@${REMOTE_HOST}:/tmp/"
        
        # Execute deployment commands on remote server
        log "Executing deployment commands on remote server..."
        ssh "${REMOTE_USER}@${REMOTE_HOST}" "
            mkdir -p ${REMOTE_PATH} && \
            tar -xzf /tmp/deploy_package.tar.gz -C ${REMOTE_PATH} && \
            cd ${REMOTE_PATH} && \
            python -m venv .venv && \
            source .venv/bin/activate && \
            pip install -r requirements/production.txt && \
            cp configs/environments/${ENVIRONMENT}.yaml configs/environments/current.yaml && \
            systemctl restart ai-assistant || \
            (nohup python -m src.main --environment ${ENVIRONMENT} > nohup.out 2>&1 &)
        "
        
        if [ $? -eq 0 ]; then
            log_success "Remote deployment completed successfully"
            
            # Save deployment info
            echo "Remote deployment host: $REMOTE_HOST" > "${DEPLOY_DIR}/direct_deployment.txt"
            echo "Remote deployment path: $REMOTE_PATH" >> "${DEPLOY_DIR}/direct_deployment.txt"
            echo "Deployed at: $(date)" >> "${DEPLOY_DIR}/direct_deployment.txt"
            
            return 0
        else
            log_error "Remote deployment failed"
            return 1
        fi
    fi
}

# Function to verify deployment
verify_deployment() {
    log_section "Verifying Deployment"
    
    # Skip for dry run
    if [ "$DRY_RUN" = true ]; then
        log "Skipping deployment verification for dry run"
        return 0
    }
    
    # Wait a bit for everything to start up
    log "Waiting for system to initialize..."
    sleep 10
    
    # Determine how to verify based on deployment method
    case "$DEPLOY_METHOD" in
        docker)
            log "Verifying Docker deployment..."
            
            # Check if containers are running
            if ! docker ps | grep -q "ai_assistant"; then
                log_error "AI Assistant containers not found or not running"
                return 1
            fi
            
            # Check health status
            log "Checking container health status..."
            if docker ps --format "{{.Names}}: {{.Status}}" | grep "ai_assistant" | grep -q "(healthy)"; then
                log_success "Containers are healthy"
            else
                log_warning "Containers may not be fully healthy yet"
            fi
            
            # Check application health endpoint
            log "Checking application health endpoint..."
            HEALTH_URL="http://localhost:8000/health"
            if curl -s "$HEALTH_URL" | grep -q "healthy"; then
                log_success "Application health check passed"
            else
                log_warning "Application health check failed or not available yet"
            fi
            ;;
            
        kubernetes)
            log "Verifying Kubernetes deployment..."
            
            # Check if pods are running
            if ! kubectl get pods --namespace="$K8S_NAMESPACE" | grep -q "ai-assistant"; then
                log_error "AI Assistant pods not found or not running"
                return 1
            fi
            
            # Check pod status
            log "Checking pod status..."
            if kubectl get pods --namespace="$K8S_NAMESPACE" -l app=ai-assistant | grep -q "Running"; then
                log_success "Pods are running"
            else
                log_warning "Pods may not be fully running yet"
            fi
            
            # Check application health endpoint through service
            log "Checking application health through service..."
            if kubectl port-forward service/ai-assistant 8000:8000 --namespace="$K8S_NAMESPACE" &
                PF_PID=$!
                sleep 5
                HEALTH_CHECK=$(curl -s http://localhost:8000/health || echo "failed")
                kill $PF_PID
                echo "$HEALTH_CHECK" | grep -q "healthy"
            then
                log_success "Application health check passed"
            else
                log_warning "Application health check failed or not available yet"
            fi
            ;;
            
        direct)
            log "Verifying direct deployment..."
            
            if [ "$TARGET" = "local" ]; then
                # Check if process is running
                DEPLOY_PATH="${PROJECT_ROOT}/local_deploy"
                if [ -f "${DEPLOY_PATH}/app.pid" ] && ps -p $(cat "${DEPLOY_PATH}/app.pid") > /dev/null; then
                    log_success "Application process is running"
                else
                    log_error "Application process not found or not running"
                    return 1
                fi
                
                # Check application health endpoint
                log "Checking application health endpoint..."
                HEALTH_URL="http://localhost:8000/health"
                if curl -s "$HEALTH_URL" | grep -q "healthy"; then
                    log_success "Application health check passed"
                else
                    log_warning "Application health check failed or not available yet"
                fi
            else
                # Remote verification
                log "Verifying remote deployment..."
                
                # Check if process is running on remote server
                if ssh "${REMOTE_USER}@${REMOTE_HOST}" "pgrep -f 'python -m src.main'" > /dev/null; then
                    log_success "Application process is running on remote server"
                else
                    log_error "Application process not found or not running on remote server"
                    return 1
                fi
                
                # Check application health endpoint on remote server
                log "Checking application health endpoint on remote server..."
                if ssh "${REMOTE_USER}@${REMOTE_HOST}" "curl -s http://localhost:8000/health" | grep -q "healthy"; then
                    log_success "Remote application health check passed"
                else
                    log_warning "Remote application health check failed or not available yet"
                fi
            fi
            ;;
    esac
    
    # Verify core functionality
    log "Verifying core functionality..."
    python -m src.cli verify-deployment --environment "$ENVIRONMENT" --method "$DEPLOY_METHOD" --target "$TARGET"
    
    if [ $? -eq 0 ]; then
        log_success "Core functionality verification passed"
        return 0
    else
        log_warning "Core functionality verification failed or not available yet"
        return 1
    fi
}

# Function to send notification
send_notification() {
    log_section "Sending Deployment Notification"
    
    # Skip if disabled
    if [ "$NOTIFY" = false ]; then
        log "Skipping notification as requested"
        return 0
    }
    
    # Skip for dry run
    if [ "$DRY_RUN" = true ]; then
        log "Skipping notification for dry run"
        return 0
    }
    
    log "Sending deployment notification..."
    
    # Get deployment summary
    local summary="AI Assistant deployment to ${ENVIRONMENT} environment"
    local details="
Deployment completed by: $(whoami)
Date: $(date)
Environment: ${ENVIRONMENT}
Method: ${DEPLOY_METHOD}
Target: ${TARGET}
Version: ${VERSION}
Deployment directory: ${DEPLOY_DIR}
"
    
    # Use CLI notification module if available
    if python -m src.cli send-notification \
        --type deployment \
        --environment "$ENVIRONMENT" \
        --subject "$summary" \
        --message "$details"; then
        log_success "Notification sent successfully"
    else
        log_warning "Failed to send notification"
    fi
}

# Function to rollback deployment
rollback_deployment() {
    log_section "Rolling Back Deployment"
    
    log_error "Deployment failed. Rolling back..."
    
    case "$DEPLOY_METHOD" in
        docker)
            log "Rolling back Docker deployment..."
            
            # Check if we have previous image information
            if [ -f "${DEPLOY_DIR}/../previous/docker_image.txt" ]; then
                PREVIOUS_IMAGE=$(cat "${DEPLOY_DIR}/../previous/docker_image.txt")
                log "Rolling back to previous image: $PREVIOUS_IMAGE"
                
                # Set previous image tag
                export TAG=$(echo "$PREVIOUS_IMAGE" | cut -d':' -f2)
                
                # Determine Docker Compose file
                local compose_file="${PROJECT_ROOT}/docker/docker-compose.yml"
                if [ "$ENVIRONMENT" = "production" ]; then
                    compose_file="${PROJECT_ROOT}/docker/docker-compose.prod.yml"
                elif [ "$ENVIRONMENT" = "development" ]; then
                    compose_file="${PROJECT_ROOT}/docker/docker-compose.dev.yml"
                fi
                
                # Stop current containers and start with previous image
                docker-compose -f "$compose_file" down
                docker-compose -f "$compose_file" up -d
                
                log_success "Rollback to previous image completed"
            else
                log_warning "No previous deployment information found. Manual rollback may be required."
            fi
            ;;
            
        kubernetes)
            log "Rolling back Kubernetes deployment..."
            
            # Use Kubernetes rollback feature
            kubectl rollout undo deployment/ai-assistant --namespace="$K8S_NAMESPACE"
            
            if [ $? -eq 0 ]; then
                log_success "Kubernetes rollback completed"
            else
                log_error "Kubernetes rollback failed. Manual intervention required."
            fi
            ;;
            
        direct)
            log "Rolling back direct deployment..."
            
            if [ "$TARGET" = "local" ]; then
                # Check if we have a backup for local deployment
                if [ -d "${BACKUP_DIR}" ]; then
                    log "Restoring from backup..."
                    
                    # Stop current process
                    DEPLOY_PATH="${PROJECT_ROOT}/local_deploy"
                    if [ -f "${DEPLOY_PATH}/app.pid" ]; then
                        kill $(cat "${DEPLOY_PATH}/app.pid") || true
                    fi
                    
                    # Restore database if available
                    LATEST_DB_BACKUP=$(ls -t "${BACKUP_DIR}"/*.sql | head -1)
                    if [ -n "$LATEST_DB_BACKUP" ]; then
                        log "Restoring database from $LATEST_DB_BACKUP"
                        python -m src.cli restore-db --environment "$ENVIRONMENT" --input "$LATEST_DB_BACKUP"
                    fi
                    
                    log_success "Rollback completed"
                else
                    log_warning "No backup found for rollback. Manual restoration may be required."
                fi
            else
                # Remote rollback
                log "Rolling back remote deployment..."
                
                if [ -n "$REMOTE_HOST" ] && [ -n "$REMOTE_USER" ]; then
                    # Check if there's a backup on remote server
                    if ssh "${REMOTE_USER}@${REMOTE_HOST}" "ls -d /opt/ai_assistant.bak 2>/dev/null"; then
                        log "Restoring from remote backup..."
                        
                        ssh "${REMOTE_USER}@${REMOTE_HOST}" "
                            systemctl stop ai-assistant || pkill -f 'python -m src.main' || true
                            if [ -d /opt/ai_assistant.bak ]; then
                                rm -rf /opt/ai_assistant
                                mv /opt/ai_assistant.bak /opt/ai_assistant
                                cd /opt/ai_assistant
                                source .venv/bin/activate
                                systemctl start ai-assistant || \
                                (nohup python -m src.main --environment ${ENVIRONMENT} > nohup.out 2>&1 &)
                            fi
                        "
                        
                        log_success "Remote rollback completed"
                    else
                        log_warning "No remote backup found. Manual restoration may be required."
                    fi
                else
                    log_warning "Insufficient information for remote rollback. Manual intervention required."
                fi
            fi
            ;;
    esac
    
    # Send rollback notification
    if [ "$NOTIFY" = true ]; then
        log "Sending rollback notification..."
        
        local summary="AI Assistant deployment ROLLBACK for ${ENVIRONMENT} environment"
        local details="
Deployment FAILED and was rolled back
Rolled back by: $(whoami)
Date: $(date)
Environment: ${ENVIRONMENT}
Method: ${DEPLOY_METHOD}
Target: ${TARGET}
Version: ${VERSION}
Deployment directory: ${DEPLOY_DIR}
"
        
        python -m src.cli send-notification \
            --type deployment \
            --environment "$ENVIRONMENT" \
            --subject "$summary" \
            --message "$details" || true
    fi
}

# Function to save deployment record
save_deployment_record() {
    log_section "Saving Deployment Record"
    
    # Save deployment information
    local status="$1"
    local deployment_record="${DEPLOY_DIR}/deployment_record.txt"
    
    echo "AI Assistant Deployment Record" > "$deployment_record"
    echo "=============================" >> "$deployment_record"
    echo "Date: $(date)" >> "$deployment_record"
    echo "User: $(whoami)" >> "$deployment_record"
    echo "Environment: $ENVIRONMENT" >> "$deployment_record"
    echo "Method: $DEPLOY_METHOD" >> "$deployment_record"
    echo "Target: $TARGET" >> "$deployment_record"
    echo "Version: $VERSION" >> "$deployment_record"
    echo "Status: $status" >> "$deployment_record"
    echo "Git Branch: $(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD)" >> "$deployment_record"
    echo "Git Commit: $(git -C "$PROJECT_ROOT" rev-parse HEAD)" >> "$deployment_record"
    echo "Deployment Directory: $DEPLOY_DIR" >> "$deployment_record"
    echo "Log File: $DEPLOY_LOG" >> "$deployment_record"
    
    # Create symlink to latest deployment
    rm -f "${PROJECT_ROOT}/data/deployments/latest" || true
    ln -s "$DEPLOY_DIR" "${PROJECT_ROOT}/data/deployments/latest"
    
    # Copy deployment record to a central location
    cp "$deployment_record" "${PROJECT_ROOT}/data/deployments/deployment_history.txt"
    
    log_success "Deployment record saved: $deployment_record"
}

# Main function
main() {
    log_section "Starting AI Assistant Deployment"
    log "User: $(whoami)"
    log "Date: $TODAY"
    log "Environment: $ENVIRONMENT"
    log "Method: $DEPLOY_METHOD"
    log "Target: $TARGET"
    log "Version: $VERSION"
    
    # Create symbolic link to previous deployment if it exists
    if [ -d "${PROJECT_ROOT}/data/deployments/latest" ]; then
        rm -f "${DEPLOY_DIR}/../previous" || true
        ln -s "$(readlink -f "${PROJECT_ROOT}/data/deployments/latest")" "${DEPLOY_DIR}/../previous"
    fi
    
    # Activate virtual environment
    activate_venv
    
    # Load environment variables
    load_env_vars
    
    # Validate git status
    validate_git_status
    
    # Check core components
    check_core_components
    
    # Run code quality checks
    run_code_quality_checks
    
    # Run tests
    run_tests
    
    # Create backup
    create_backup
    
    # Run database migrations
    run_migrations
    
    # Print deployment confirmation
    log_section "Deployment Confirmation"
    echo -e "${YELLOW}You are about to deploy the AI Assistant with the following settings:${NC}"
    echo -e "  Environment: ${CYAN}${ENVIRONMENT}${NC}"
    echo -e "  Method: ${CYAN}${DEPLOY_METHOD}${NC}"
    echo -e "  Target: ${CYAN}${TARGET}${NC}"
    echo -e "  Version: ${CYAN}${VERSION}${NC}"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        echo -e "${RED}WARNING: This is a PRODUCTION deployment!${NC}"
    fi
    
    if [ "$FORCE" = false ] && [ "$DRY_RUN" = false ]; then
        read -p "Do you want to proceed with the deployment? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Perform deployment based on method
    case "$DEPLOY_METHOD" in
        docker)
            build_docker_image && deploy_docker_compose
            DEPLOY_RESULT=$?
            ;;
        kubernetes)
            build_docker_image && deploy_kubernetes
            DEPLOY_RESULT=$?
            ;;
        direct)
            deploy_direct
            DEPLOY_RESULT=$?
            ;;
    esac
    
    if [ $DEPLOY_RESULT -ne 0 ]; then
        log_error "Deployment failed"
        
        if [ "$ROLLBACK_ON_FAILURE" = true ]; then
            rollback_deployment
        else
            log_warning "Automatic rollback is disabled. Manual intervention may be required."
        fi
        
        save_deployment_record "FAILED"
        exit 1
    fi
    
    # Verify deployment
    verify_deployment
    VERIFY_RESULT=$?
    
    if [ $VERIFY_RESULT -ne 0 ]; then
        log_warning "Deployment verification had issues"
    else
        log_success "Deployment verification passed"
    fi
    
    # Send notification
    send_notification
    
    # Save deployment record
    save_deployment_record "SUCCESS"
    
    log_section "Deployment Complete"
    log "Deployment log saved to: $DEPLOY_LOG"
    
    echo -e "\n${GREEN}Deployment completed successfully!${NC}"
}

# Run main function
main
