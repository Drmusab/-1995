#!/bin/bash
# ==============================================================================
# AI Assistant Test Runner Script
# Author: Drmusab
# Created: 2025-07-05
# Last Modified: 2025-07-05 14:32:57
#
# This script runs tests for the AI Assistant system, supporting various
# test types and configurations while integrating with the core components.
# ==============================================================================

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/data/logs"
REPORT_DIR="${PROJECT_ROOT}/test-reports"
TEST_LOG="${LOG_DIR}/test_$(date +%Y%m%d_%H%M%S).log"
COVERAGE_DIR="${REPORT_DIR}/coverage"
JUNIT_DIR="${REPORT_DIR}/junit"
TODAY="$(date +%Y-%m-%d)"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"

# Default values
TEST_TYPE="all"
VERBOSITY=1
PARALLEL=true
PARALLEL_WORKERS=4
COVERAGE=false
FAIL_FAST=false
REUSE_DB=true
USE_DOCKER=false
TEST_PATTERN=""
SKIP_SLOW=false
CI_MODE=false
MOCK_EXTERNAL=true
MOCK_LLM=true

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Ensure log and report directories exist
mkdir -p "$LOG_DIR"
mkdir -p "$REPORT_DIR"
mkdir -p "$COVERAGE_DIR"
mkdir -p "$JUNIT_DIR"

# Logging functions
log() {
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    echo -e "${timestamp} - $1" | tee -a "$TEST_LOG"
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
    echo 
    echo "Options:"
    echo "  -t, --type TYPE          Test type to run: unit, integration, e2e, smoke,"
    echo "                           performance, security, resilience, all (default: all)"
    echo "  -m, --module MODULE      Specific module to test (e.g., core, processing)"
    echo "  -p, --pattern PATTERN    Filter tests by pattern"
    echo "  -v, --verbose            Increase verbosity"
    echo "  -q, --quiet              Decrease verbosity"
    echo "  -x, --no-parallel        Disable parallel test execution"
    echo "  -w, --workers N          Number of parallel workers (default: 4)"
    echo "  -c, --with-coverage      Generate coverage report"
    echo "  -f, --fail-fast          Stop on first failure"
    echo "  --no-db-reuse            Don't reuse test database"
    echo "  --docker                 Run tests in Docker container"
    echo "  --skip-slow              Skip slow tests"
    echo "  --ci                     Run in CI mode (non-interactive, full reports)"
    echo "  --no-mocks               Don't mock external services"
    echo "  --real-llm               Use real LLM services instead of mocks"
    echo "  -h, --help               Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -m|--module)
            TEST_MODULE="$2"
            shift 2
            ;;
        -p|--pattern)
            TEST_PATTERN="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSITY=$((VERBOSITY + 1))
            shift
            ;;
        -q|--quiet)
            VERBOSITY=$((VERBOSITY - 1))
            shift
            ;;
        -x|--no-parallel)
            PARALLEL=false
            shift
            ;;
        -w|--workers)
            PARALLEL_WORKERS="$2"
            shift 2
            ;;
        -c|--with-coverage)
            COVERAGE=true
            shift
            ;;
        -f|--fail-fast)
            FAIL_FAST=true
            shift
            ;;
        --no-db-reuse)
            REUSE_DB=false
            shift
            ;;
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --skip-slow)
            SKIP_SLOW=true
            shift
            ;;
        --ci)
            CI_MODE=true
            COVERAGE=true
            VERBOSITY=2
            shift
            ;;
        --no-mocks)
            MOCK_EXTERNAL=false
            shift
            ;;
        --real-llm)
            MOCK_LLM=false
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

# Validate test type
valid_test_types=("unit" "integration" "e2e" "smoke" "performance" "security" "resilience" "all")
if ! [[ " ${valid_test_types[*]} " =~ " ${TEST_TYPE} " ]]; then
    log_error "Invalid test type: $TEST_TYPE"
    usage
fi

# Set verbosity flags for pytest
case $VERBOSITY in
    0) VERBOSITY_FLAG="-q" ;;
    1) VERBOSITY_FLAG="" ;;
    2) VERBOSITY_FLAG="-v" ;;
    3) VERBOSITY_FLAG="-vv" ;;
    *) VERBOSITY_FLAG="-vvs" ;;
esac

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

# Function to check required dependencies
check_dependencies() {
    log_section "Checking Dependencies"
    
    # Check pytest
    if ! python -c "import pytest" &>/dev/null; then
        log_warning "pytest not found. Installing..."
        pip install pytest pytest-cov pytest-xdist pytest-timeout pytest-mock
    fi
    
    # Check pytest extensions
    for pkg in pytest-cov pytest-xdist pytest-timeout pytest-mock pytest-benchmark; do
        if ! python -c "import ${pkg//-/_}" &>/dev/null; then
            log_warning "$pkg not found. Installing..."
            pip install $pkg
        fi
    fi
    
    # Check specific dependencies for test types
    if [ "$TEST_TYPE" = "performance" ] || [ "$TEST_TYPE" = "all" ]; then
        if ! python -c "import locust" &>/dev/null; then
            log_warning "locust not found. Installing for performance tests..."
            pip install locust
        fi
    fi
    
    log_success "All required dependencies are installed"
}

# Function to prepare test environment
prepare_environment() {
    log_section "Preparing Test Environment"
    
    # Create test configuration
    log "Setting up test environment variables..."
    
    # Set environment to testing
    export ENVIRONMENT="testing"
    export TESTING="1"
    export TEST_MODE="1"
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    
    # Load test environment variables if exists
    if [ -f "${PROJECT_ROOT}/.env.test" ]; then
        log "Loading test environment variables from .env.test"
        set -a
        source "${PROJECT_ROOT}/.env.test"
        set +a
    elif [ -f "${PROJECT_ROOT}/.env" ]; then
        log "Loading environment variables from .env"
        set -a
        source "${PROJECT_ROOT}/.env"
        set +a
    fi
    
    # Set up mocking configuration
    if [ "$MOCK_EXTERNAL" = true ]; then
        export MOCK_EXTERNAL_SERVICES="1"
        log "External services will be mocked"
    else
        export MOCK_EXTERNAL_SERVICES="0"
        log "Using real external services"
    fi
    
    if [ "$MOCK_LLM" = true ]; then
        export MOCK_LLM_SERVICES="1"
        log "LLM services will be mocked"
    else
        export MOCK_LLM_SERVICES="0"
        log "Using real LLM services"
    fi
    
    # Set up test database
    if [ "$REUSE_DB" = true ]; then
        export REUSE_TEST_DB="1"
        log "Reusing test database if available"
    else
        export REUSE_TEST_DB="0"
        log "Creating fresh test database"
    fi
    
    # Check for test data
    if [ ! -d "${PROJECT_ROOT}/tests/fixtures/sample_models" ]; then
        log_warning "Test model fixtures not found. Some tests may fail."
    fi
    
    log_success "Test environment prepared"
}

# Function to build test command based on test type
build_test_command() {
    log_section "Building Test Command"
    
    # Base command
    CMD="python -m pytest"
    
    # Add verbosity
    CMD="$CMD $VERBOSITY_FLAG"
    
    # Add parallel execution if enabled
    if [ "$PARALLEL" = true ]; then
        CMD="$CMD -xvs"
        if [ "$TEST_TYPE" != "performance" ]; then  # Don't parallelize performance tests
            CMD="$CMD -n $PARALLEL_WORKERS"
        fi
    fi
    
    # Add coverage if enabled
    if [ "$COVERAGE" = true ]; then
        CMD="$CMD --cov=src --cov-report=term --cov-report=xml:${COVERAGE_DIR}/coverage.xml --cov-report=html:${COVERAGE_DIR}/html"
    fi
    
    # Add fail-fast if enabled
    if [ "$FAIL_FAST" = true ]; then
        CMD="$CMD -x"
    fi
    
    # Add JUnit report for CI
    if [ "$CI_MODE" = true ]; then
        CMD="$CMD --junitxml=${JUNIT_DIR}/test-results.xml"
    fi
    
    # Add timeout for tests
    CMD="$CMD --timeout=300"
    
    # Skip slow tests if requested
    if [ "$SKIP_SLOW" = true ]; then
        CMD="$CMD -k 'not slow'"
    fi
    
    # Determine test paths based on test type
    case $TEST_TYPE in
        unit)
            CMD="$CMD ${PROJECT_ROOT}/tests/unit/"
            ;;
        integration)
            CMD="$CMD ${PROJECT_ROOT}/tests/integration/"
            ;;
        e2e)
            CMD="$CMD ${PROJECT_ROOT}/tests/e2e/"
            ;;
        smoke)
            CMD="$CMD ${PROJECT_ROOT}/tests/smoke/"
            ;;
        performance)
            CMD="$CMD ${PROJECT_ROOT}/tests/performance/"
            ;;
        security)
            CMD="$CMD ${PROJECT_ROOT}/tests/security/"
            ;;
        resilience)
            CMD="$CMD ${PROJECT_ROOT}/tests/resilience/"
            ;;
        all)
            CMD="$CMD ${PROJECT_ROOT}/tests/"
            ;;
    esac
    
    # Add module filter if specified
    if [ -n "$TEST_MODULE" ]; then
        if [ "$TEST_TYPE" = "unit" ]; then
            CMD="$CMD -k test_${TEST_MODULE}"
        else
            CMD="$CMD -k ${TEST_MODULE}"
        fi
    fi
    
    # Add pattern filter if specified
    if [ -n "$TEST_PATTERN" ]; then
        CMD="$CMD -k '$TEST_PATTERN'"
    fi
    
    log "Test command: $CMD"
    echo "$CMD"
}

# Function to run core system check
check_core_system() {
    log_section "Checking Core System"
    
    # Check core component tests exist
    CORE_COMPONENTS=(
        "core_engine" 
        "component_manager" 
        "workflow_orchestrator" 
        "interaction_handler" 
        "session_manager" 
        "plugin_manager"
    )
    
    MISSING_TESTS=0
    for component in "${CORE_COMPONENTS[@]}"; do
        if [ ! -f "${PROJECT_ROOT}/tests/unit/test_assistant/test_${component}.py" ]; then
            log_warning "Missing tests for core component: ${component}"
            MISSING_TESTS=$((MISSING_TESTS + 1))
        fi
    done
    
    if [ $MISSING_TESTS -gt 0 ]; then
        log_warning "$MISSING_TESTS core component tests are missing"
    else
        log_success "All core component tests exist"
    fi
    
    # Run a quick sanity check on the core system
    log "Running core system sanity check..."
    SANITY_TEST="python -c \"from src.assistant.core_engine import CoreEngine; print('Core engine can be imported')\""
    if eval "$SANITY_TEST"; then
        log_success "Core system sanity check passed"
    else
        log_error "Core system sanity check failed"
        if [ "$CI_MODE" = true ]; then
            exit 1
        fi
    fi
}

# Function to run tests in Docker
run_docker_tests() {
    log_section "Running Tests in Docker"
    
    log "Building test Docker image..."
    docker build -f "${PROJECT_ROOT}/docker/Dockerfile.dev" -t ai-assistant-test "${PROJECT_ROOT}"
    
    # Prepare Docker command
    DOCKER_CMD="docker run --rm -v ${PROJECT_ROOT}:/app -w /app"
    
    # Add environment variables
    DOCKER_CMD="$DOCKER_CMD -e ENVIRONMENT=testing -e TESTING=1 -e TEST_MODE=1"
    DOCKER_CMD="$DOCKER_CMD -e MOCK_EXTERNAL_SERVICES=$MOCK_EXTERNAL_SERVICES"
    DOCKER_CMD="$DOCKER_CMD -e MOCK_LLM_SERVICES=$MOCK_LLM_SERVICES"
    
    # Add test command
    TEST_CMD=$(build_test_command)
    DOCKER_CMD="$DOCKER_CMD ai-assistant-test $TEST_CMD"
    
    log "Running tests in Docker container..."
    log "Docker command: $DOCKER_CMD"
    
    eval "$DOCKER_CMD"
    DOCKER_EXIT_CODE=$?
    
    if [ $DOCKER_EXIT_CODE -eq 0 ]; then
        log_success "Docker tests completed successfully"
    else
        log_error "Docker tests failed with exit code $DOCKER_EXIT_CODE"
    fi
    
    return $DOCKER_EXIT_CODE
}

# Function to run performance tests
run_performance_tests() {
    log_section "Running Performance Tests"
    
    # Create performance report directory
    PERF_DIR="${REPORT_DIR}/performance/${TIMESTAMP}"
    mkdir -p "$PERF_DIR"
    
    log "Running benchmark tests..."
    python -m pytest "${PROJECT_ROOT}/tests/performance/benchmarks/" \
        --benchmark-json="${PERF_DIR}/benchmark.json" \
        --benchmark-columns=mean,median,stddev,min,max,rounds \
        $VERBOSITY_FLAG
    
    # Optionally run load tests
    if [ "$CI_MODE" = false ]; then
        read -p "Do you want to run load tests? This may take a while. [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log "Running load tests..."
            python -m locust -f "${PROJECT_ROOT}/tests/performance/load_tests/locustfile.py" \
                --headless \
                --users 20 \
                --spawn-rate 2 \
                --run-time 2m \
                --html="${PERF_DIR}/load_test_report.html"
        fi
    fi
    
    # Optionally run memory tests
    if [ "$CI_MODE" = false ]; then
        read -p "Do you want to run memory profiling tests? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log "Running memory profiling tests..."
            python -m pytest "${PROJECT_ROOT}/tests/performance/memory_tests/" \
                --no-cov \
                -v
        fi
    fi
    
    log_success "Performance tests completed"
}

# Function to run security tests
run_security_tests() {
    log_section "Running Security Tests"
    
    # Run security test suite
    log "Running security test suite..."
    python -m pytest "${PROJECT_ROOT}/tests/security/" $VERBOSITY_FLAG
    
    # Optionally run dependency vulnerability check
    if command -v safety &> /dev/null; then
        log "Checking dependencies for vulnerabilities..."
        safety check -r "${PROJECT_ROOT}/requirements/production.txt"
    else
        log_warning "safety not installed. Skipping dependency vulnerability check."
        log_warning "Install with: pip install safety"
    fi
    
    log_success "Security tests completed"
}

# Function to process and print test results
process_results() {
    log_section "Processing Test Results"
    
    # Parse coverage if enabled
    if [ "$COVERAGE" = true ] && [ -f "${COVERAGE_DIR}/coverage.xml" ]; then
        COVERAGE_PCT=$(grep -o 'line-rate="[0-9.]*"' "${COVERAGE_DIR}/coverage.xml" | head -1 | grep -o '[0-9.]*')
        COVERAGE_PCT=$(echo "$COVERAGE_PCT * 100" | bc)
        
        echo -e "\n${CYAN}Test Coverage: ${COVERAGE_PCT}%${NC}"
        log "Test coverage: ${COVERAGE_PCT}%"
        
        # Print coverage by component
        echo -e "\n${CYAN}Coverage by Component:${NC}"
        python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('${COVERAGE_DIR}/coverage.xml')
root = tree.getroot()
for pkg in root.findall('.//package'):
    name = pkg.attrib.get('name', '')
    if name.startswith('src.'):
        line_rate = float(pkg.attrib.get('line-rate', 0)) * 100
        print(f\"  {name}: {line_rate:.1f}%\")
"
    fi
    
    echo -e "\n${GREEN}All tests completed!${NC}"
    echo -e "Test reports available at: ${REPORT_DIR}"
    echo -e "Test log: ${TEST_LOG}"
    
    log_success "Test execution completed"
}

# Main function
main() {
    log_section "Starting AI Assistant Test Runner"
    log "User: $(whoami)"
    log "Date: ${TODAY}"
    log "Test type: ${TEST_TYPE}"
    
    # Check and activate virtual environment
    activate_venv
    
    # Check dependencies
    check_dependencies
    
    # Prepare test environment
    prepare_environment
    
    # Check core system
    check_core_system
    
    # Run tests based on configuration
    if [ "$USE_DOCKER" = true ]; then
        run_docker_tests
        TEST_EXIT_CODE=$?
    else
        # Build and run test command
        TEST_CMD=$(build_test_command)
        
        # Run specific test types
        if [ "$TEST_TYPE" = "performance" ]; then
            run_performance_tests
            TEST_EXIT_CODE=$?
        elif [ "$TEST_TYPE" = "security" ]; then
            run_security_tests
            TEST_EXIT_CODE=$?
        else
            # Run standard tests
            log_section "Running ${TEST_TYPE} Tests"
            eval "$TEST_CMD"
            TEST_EXIT_CODE=$?
        fi
    fi
    
    # Process and display results
    process_results
    
    # Return test exit code
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        log_success "All tests passed!"
    else
        log_error "Tests failed with exit code $TEST_EXIT_CODE"
    fi
    
    exit $TEST_EXIT_CODE
}

# Run main function
main
