#!/bin/bash
# ==============================================================================
# AI Assistant Code Formatting Script
# Author: Drmusab
# Created: 2025-07-05
# Last Modified: 2025-07-05 19:52:10
#
# This script applies consistent code formatting across all components of the 
# AI Assistant system, ensuring coding standards are maintained throughout
# the codebase. It integrates with the core system architecture and supports
# various formatting tools.
# ==============================================================================

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/data/logs"
FORMAT_LOG="${LOG_DIR}/format_$(date +%Y%m%d_%H%M%S).log"
TODAY="$(date +%Y-%m-%d)"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"

# Default values
FORMAT_TARGET="all"     # Target to format: core, assistant, processing, etc., all
FORMAT_MODE="fix"       # Mode: check, fix, diff
CHECK_ONLY=false        # Only check formatting without fixing
SHOW_DIFF=false         # Show differences without fixing
VERBOSE=false           # Verbose output
CI_MODE=false           # CI mode (non-interactive, exit code reflects findings)
PARALLEL=true           # Run formatters in parallel
PARALLEL_JOBS=4         # Number of parallel jobs
FORMAT_PYTHON=true      # Format Python files
FORMAT_DOCS=true        # Format documentation files
FORMAT_CONFIG=true      # Format configuration files
IGNORE_PATTERNS=""      # Patterns to ignore

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging functions
log() {
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    echo -e "${timestamp} - $1" | tee -a "$FORMAT_LOG"
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
    echo "  -t, --target TARGET      Specific target to format: core, assistant, processing,"
    echo "                           etc., all (default: all)"
    echo "  -c, --check              Check formatting without making changes"
    echo "  -d, --diff               Show formatting differences without making changes"
    echo "  -m, --mode MODE          Format mode: check, fix, diff (default: fix)"
    echo "  -v, --verbose            Show verbose output"
    echo "  --ci                     Run in CI mode (non-interactive, exit with error on issues)"
    echo "  --no-parallel            Disable parallel formatting"
    echo "  -j, --jobs JOBS          Number of parallel jobs (default: 4)"
    echo "  --no-python              Skip Python file formatting"
    echo "  --no-docs                Skip documentation file formatting"
    echo "  --no-config              Skip configuration file formatting"
    echo "  --ignore PATTERN         Ignore pattern (can be used multiple times)"
    echo "  -h, --help               Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--target)
            FORMAT_TARGET="$2"
            shift 2
            ;;
        -c|--check)
            CHECK_ONLY=true
            FORMAT_MODE="check"
            shift
            ;;
        -d|--diff)
            SHOW_DIFF=true
            FORMAT_MODE="diff"
            shift
            ;;
        -m|--mode)
            FORMAT_MODE="$2"
            if [ "$FORMAT_MODE" = "check" ]; then
                CHECK_ONLY=true
            elif [ "$FORMAT_MODE" = "diff" ]; then
                SHOW_DIFF=true
            fi
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --ci)
            CI_MODE=true
            shift
            ;;
        --no-parallel)
            PARALLEL=false
            shift
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --no-python)
            FORMAT_PYTHON=false
            shift
            ;;
        --no-docs)
            FORMAT_DOCS=false
            shift
            ;;
        --no-config)
            FORMAT_CONFIG=false
            shift
            ;;
        --ignore)
            if [ -z "$IGNORE_PATTERNS" ]; then
                IGNORE_PATTERNS="$2"
            else
                IGNORE_PATTERNS="$IGNORE_PATTERNS,$2"
            fi
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

# Validate format mode
if [[ ! "$FORMAT_MODE" =~ ^(check|fix|diff)$ ]]; then
    log_error "Invalid format mode: $FORMAT_MODE. Must be one of: check, fix, diff"
    exit 1
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

# Function to check required dependencies
check_dependencies() {
    log_section "Checking Dependencies"
    
    local missing_tools=()
    
    # Define tools to check
    local tools_to_check=()
    
    if [ "$FORMAT_PYTHON" = true ]; then
        tools_to_check+=("black" "isort" "docformatter")
    fi
    
    if [ "$FORMAT_DOCS" = true ] || [ "$FORMAT_CONFIG" = true ]; then
        tools_to_check+=("prettier")
    fi
    
    for tool in "${tools_to_check[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_warning "Missing tools: ${missing_tools[*]}"
        
        if [ "$CI_MODE" = true ]; then
            log_error "Cannot install missing tools in CI mode"
            exit 1
        else
            read -p "Do you want to install missing tools? [Y/n] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Nn]$ ]]; then
                log_warning "Continuing without installing missing tools"
            else
                log "Installing missing tools..."
                
                # Install Python formatting tools
                if [[ " ${missing_tools[*]} " =~ " black " ]] || 
                   [[ " ${missing_tools[*]} " =~ " isort " ]] || 
                   [[ " ${missing_tools[*]} " =~ " docformatter " ]]; then
                    pip install black isort docformatter
                fi
                
                # Install prettier if needed
                if [[ " ${missing_tools[*]} " =~ " prettier " ]]; then
                    if command -v npm &> /dev/null; then
                        npm install -g prettier
                    else
                        log_warning "npm not found. Cannot install prettier. Docs and config formatting will be skipped."
                        FORMAT_DOCS=false
                        FORMAT_CONFIG=false
                    fi
                fi
                
                log_success "Tools installed"
            fi
        fi
    else
        log_success "All required tools are installed"
    fi
}

# Function to determine target paths
get_target_paths() {
    if [ "$FORMAT_TARGET" = "all" ]; then
        echo "${PROJECT_ROOT}/src ${PROJECT_ROOT}/tests"
    else
        # Handle special cases
        case "$FORMAT_TARGET" in
            core)
                echo "${PROJECT_ROOT}/src/core"
                ;;
            assistant)
                echo "${PROJECT_ROOT}/src/assistant"
                ;;
            processing)
                echo "${PROJECT_ROOT}/src/processing"
                ;;
            memory)
                echo "${PROJECT_ROOT}/src/memory"
                ;;
            api)
                echo "${PROJECT_ROOT}/src/api"
                ;;
            reasoning)
                echo "${PROJECT_ROOT}/src/reasoning"
                ;;
            skills)
                echo "${PROJECT_ROOT}/src/skills"
                ;;
            learning)
                echo "${PROJECT_ROOT}/src/learning"
                ;;
            integrations)
                echo "${PROJECT_ROOT}/src/integrations"
                ;;
            tests)
                echo "${PROJECT_ROOT}/tests"
                ;;
            docs)
                echo "${PROJECT_ROOT}/docs"
                ;;
            *)
                if [ -d "${PROJECT_ROOT}/src/$FORMAT_TARGET" ]; then
                    echo "${PROJECT_ROOT}/src/$FORMAT_TARGET"
                else
                    log_error "Invalid target module: $FORMAT_TARGET"
                    exit 1
                fi
                ;;
        esac
    fi
}

# Function to get file counts
get_file_counts() {
    local target_paths="$1"
    
    local py_count=0
    local md_count=0
    local yaml_count=0
    local json_count=0
    
    for path in $target_paths; do
        if [ -d "$path" ]; then
            py_count=$((py_count + $(find "$path" -name "*.py" | wc -l)))
            md_count=$((md_count + $(find "$path" -name "*.md" | wc -l)))
            yaml_count=$((yaml_count + $(find "$path" -name "*.yaml" -o -name "*.yml" | wc -l)))
            json_count=$((json_count + $(find "$path" -name "*.json" | wc -l)))
        elif [ -f "$path" ]; then
            case "$path" in
                *.py) py_count=$((py_count + 1)) ;;
                *.md) md_count=$((md_count + 1)) ;;
                *.yaml|*.yml) yaml_count=$((yaml_count + 1)) ;;
                *.json) json_count=$((json_count + 1)) ;;
            esac
        fi
    done
    
    echo "Python files: $py_count"
    echo "Markdown files: $md_count"
    echo "YAML files: $yaml_count"
    echo "JSON files: $json_count"
}

# Function to format Python files
format_python_files() {
    local target_paths="$1"
    local mode="$2"
    
    log_section "Formatting Python Files"
    
    # Skip if Python formatting is disabled
    if [ "$FORMAT_PYTHON" = false ]; then
        log_warning "Python formatting is disabled. Skipping."
        return 0
    fi
    
    # Black arguments
    local black_args="--config=${PROJECT_ROOT}/pyproject.toml"
    if [ "$VERBOSE" = true ]; then
        black_args="$black_args --verbose"
    fi
    if [ "$mode" = "check" ]; then
        black_args="$black_args --check"
    elif [ "$mode" = "diff" ]; then
        black_args="$black_args --diff"
    fi
    
    # isort arguments
    local isort_args="--settings-path=${PROJECT_ROOT}/pyproject.toml"
    if [ "$VERBOSE" = true ]; then
        isort_args="$isort_args --verbose"
    fi
    if [ "$mode" = "check" ]; then
        isort_args="$isort_args --check"
    elif [ "$mode" = "diff" ]; then
        isort_args="$isort_args --diff"
    fi
    
    # docformatter arguments
    local docformatter_args="--recursive"
    if [ "$mode" = "check" ]; then
        docformatter_args="$docformatter_args --check"
    elif [ "$mode" = "diff" ]; then
        docformatter_args="$docformatter_args --diff"
    else
        docformatter_args="$docformatter_args --in-place"
    fi
    
    # Add ignore patterns if any
    if [ -n "$IGNORE_PATTERNS" ]; then
        for pattern in ${IGNORE_PATTERNS//,/ }; do
            black_args="$black_args --exclude $pattern"
            isort_args="$isort_args --skip $pattern"
        done
    fi
    
    log "Running black..."
    local black_output
    if black_output=$(black $black_args $target_paths 2>&1); then
        log_success "Black formatting completed successfully"
        if [ "$VERBOSE" = true ]; then
            echo -e "${CYAN}$black_output${NC}"
        fi
    else
        log_warning "Black found formatting issues"
        echo -e "${YELLOW}$black_output${NC}"
    fi
    
    log "Running isort..."
    local isort_output
    if isort_output=$(isort $isort_args $target_paths 2>&1); then
        log_success "isort import sorting completed successfully"
        if [ "$VERBOSE" = true ]; then
            echo -e "${CYAN}$isort_output${NC}"
        fi
    else
        log_warning "isort found import sorting issues"
        echo -e "${YELLOW}$isort_output${NC}"
    fi
    
    log "Running docformatter..."
    local docformatter_output
    if docformatter_output=$(docformatter $docformatter_args $target_paths 2>&1); then
        log_success "docformatter completed successfully"
        if [ "$VERBOSE" = true ]; then
            echo -e "${CYAN}$docformatter_output${NC}"
        fi
    else
        log_warning "docformatter found docstring formatting issues"
        echo -e "${YELLOW}$docformatter_output${NC}"
    fi
}

# Function to format documentation files
format_doc_files() {
    local target_paths="$1"
    local mode="$2"
    
    log_section "Formatting Documentation Files"
    
    # Skip if documentation formatting is disabled
    if [ "$FORMAT_DOCS" = false ]; then
        log_warning "Documentation formatting is disabled. Skipping."
        return 0
    fi
    
    # Check if prettier is available
    if ! command -v prettier &> /dev/null; then
        log_warning "prettier not found. Skipping documentation formatting."
        return 0
    fi
    
    # Prettier arguments
    local prettier_args="--prose-wrap always"
    if [ "$mode" = "check" ]; then
        prettier_args="$prettier_args --check"
    elif [ "$mode" = "diff" ]; then
        prettier_args="$prettier_args --list-different"
    else
        prettier_args="$prettier_args --write"
    fi
    
    # Add ignore patterns if any
    if [ -n "$IGNORE_PATTERNS" ]; then
        for pattern in ${IGNORE_PATTERNS//,/ }; do
            prettier_args="$prettier_args --ignore-path $pattern"
        done
    fi
    
    # Find all Markdown files
    local md_files=()
    for path in $target_paths; do
        if [ -d "$path" ]; then
            while IFS= read -r file; do
                md_files+=("$file")
            done < <(find "$path" -name "*.md")
        elif [ -f "$path" ] && [[ "$path" == *.md ]]; then
            md_files+=("$path")
        fi
    done
    
    if [ ${#md_files[@]} -eq 0 ]; then
        log_warning "No Markdown files found to format"
        return 0
    fi
    
    log "Running prettier on ${#md_files[@]} Markdown files..."
    
    local prettier_output
    if prettier_output=$(prettier $prettier_args "${md_files[@]}" 2>&1); then
        log_success "prettier formatting of Markdown files completed successfully"
        if [ "$VERBOSE" = true ]; then
            echo -e "${CYAN}$prettier_output${NC}"
        fi
    else
        log_warning "prettier found formatting issues in Markdown files"
        echo -e "${YELLOW}$prettier_output${NC}"
    fi
}

# Function to format configuration files
format_config_files() {
    local target_paths="$1"
    local mode="$2"
    
    log_section "Formatting Configuration Files"
    
    # Skip if configuration formatting is disabled
    if [ "$FORMAT_CONFIG" = false ]; then
        log_warning "Configuration formatting is disabled. Skipping."
        return 0
    fi
    
    # Check if prettier is available
    if ! command -v prettier &> /dev/null; then
        log_warning "prettier not found. Skipping configuration formatting."
        return 0
    fi
    
    # Prettier arguments
    local prettier_args=""
    if [ "$mode" = "check" ]; then
        prettier_args="$prettier_args --check"
    elif [ "$mode" = "diff" ]; then
        prettier_args="$prettier_args --list-different"
    else
        prettier_args="$prettier_args --write"
    fi
    
    # Add ignore patterns if any
    if [ -n "$IGNORE_PATTERNS" ]; then
        for pattern in ${IGNORE_PATTERNS//,/ }; do
            prettier_args="$prettier_args --ignore-path $pattern"
        done
    fi
    
    # Find all YAML and JSON files
    local yaml_files=()
    local json_files=()
    
    for path in $target_paths; do
        if [ -d "$path" ]; then
            while IFS= read -r file; do
                yaml_files+=("$file")
            done < <(find "$path" -name "*.yaml" -o -name "*.yml")
            
            while IFS= read -r file; do
                json_files+=("$file")
            done < <(find "$path" -name "*.json")
        elif [ -f "$path" ]; then
            if [[ "$path" == *.yaml ]] || [[ "$path" == *.yml ]]; then
                yaml_files+=("$path")
            elif [[ "$path" == *.json ]]; then
                json_files+=("$path")
            fi
        fi
    done
    
    # Format YAML files
    if [ ${#yaml_files[@]} -gt 0 ]; then
        log "Running prettier on ${#yaml_files[@]} YAML files..."
        
        local yaml_output
        if yaml_output=$(prettier $prettier_args "${yaml_files[@]}" 2>&1); then
            log_success "prettier formatting of YAML files completed successfully"
            if [ "$VERBOSE" = true ]; then
                echo -e "${CYAN}$yaml_output${NC}"
            fi
        else
            log_warning "prettier found formatting issues in YAML files"
            echo -e "${YELLOW}$yaml_output${NC}"
        fi
    else
        log_warning "No YAML files found to format"
    fi
    
    # Format JSON files
    if [ ${#json_files[@]} -gt 0 ]; then
        log "Running prettier on ${#json_files[@]} JSON files..."
        
        local json_output
        if json_output=$(prettier $prettier_args "${json_files[@]}" 2>&1); then
            log_success "prettier formatting of JSON files completed successfully"
            if [ "$VERBOSE" = true ]; then
                echo -e "${CYAN}$json_output${NC}"
            fi
        else
            log_warning "prettier found formatting issues in JSON files"
            echo -e "${YELLOW}$json_output${NC}"
        fi
    else
        log_warning "No JSON files found to format"
    fi
}

# Function to check core component compliance
check_core_components() {
    log_section "Checking Core System Components"
    
    # Core components that must follow formatting standards
    CORE_COMPONENTS=(
        "src/assistant/core_engine.py"
        "src/assistant/component_manager.py"
        "src/assistant/workflow_orchestrator.py"
        "src/assistant/interaction_handler.py"
        "src/assistant/session_manager.py"
        "src/assistant/plugin_manager.py"
        "src/core/events/event_bus.py"
        "src/core/config/loader.py"
        "src/core/dependency_injection.py"
        "src/core/error_handling.py"
        "src/core/health_check.py"
    )
    
    # Check if each core component exists and follows formatting standards
    MISSING=0
    FAILING=0
    
    for component in "${CORE_COMPONENTS[@]}"; do
        if [ ! -f "${PROJECT_ROOT}/$component" ]; then
            log_warning "Core component not found: $component"
            MISSING=$((MISSING + 1))
            continue
        fi
        
        # Check formatting with black in check mode
        if ! black --quiet --check "${PROJECT_ROOT}/$component" &>/dev/null; then
            log_warning "Core component needs formatting: $component"
            FAILING=$((FAILING + 1))
        fi
    done
    
    if [ $MISSING -gt 0 ]; then
        log_warning "$MISSING core components are missing"
    fi
    
    if [ $FAILING -gt 0 ]; then
        log_warning "$FAILING core components need formatting"
        
        if [ "$FORMAT_MODE" = "fix" ]; then
            log "Formatting core components..."
            black --quiet "${PROJECT_ROOT}/src/assistant" "${PROJECT_ROOT}/src/core"
            isort --quiet "${PROJECT_ROOT}/src/assistant" "${PROJECT_ROOT}/src/core"
            log_success "Core components formatted"
        fi
    fi
    
    if [ $MISSING -eq 0 ] && [ $FAILING -eq 0 ]; then
        log_success "All core components exist and are properly formatted"
    fi
}

# Main function
main() {
    log_section "Starting AI Assistant Code Formatting"
    log "User: $(whoami)"
    log "Date: $TODAY"
    log "Target: $FORMAT_TARGET"
    log "Mode: $FORMAT_MODE"
    
    # Activate virtual environment
    activate_venv
    
    # Check dependencies
    check_dependencies
    
    # Get target paths
    TARGET_PATHS=$(get_target_paths)
    log "Target paths: $TARGET_PATHS"
    
    # Display file counts
    log_section "File Counts"
    get_file_counts "$TARGET_PATHS"
    
    # Check core components
    check_core_components
    
    # Format Python files
    format_python_files "$TARGET_PATHS" "$FORMAT_MODE"
    
    # Format documentation files
    format_doc_files "$TARGET_PATHS" "$FORMAT_MODE"
    
    # Format configuration files
    format_config_files "$TARGET_PATHS" "$FORMAT_MODE"
    
    # Print summary
    log_section "Formatting Summary"
    
    if [ "$FORMAT_MODE" = "check" ]; then
        log "Formatting check completed. See warnings above for any issues."
    elif [ "$FORMAT_MODE" = "diff" ]; then
        log "Formatting diff completed. See output above for required changes."
    else
        log_success "Code formatting completed successfully!"
    fi
    
    log "Format log saved to: $FORMAT_LOG"
    
    # Provide helpful commands
    if [ "$FORMAT_MODE" = "check" ] || [ "$FORMAT_MODE" = "diff" ]; then
        echo -e "\n${CYAN}To format all files, run:${NC}"
        echo "  ./scripts/format.sh"
    fi
    
    echo -e "\n${GREEN}Formatting complete!${NC}"
}

# Run main function
main
