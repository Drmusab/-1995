#!/bin/bash
# ==============================================================================
# AI Assistant Code Linting Script
# Author: Drmusab
# Created: 2025-07-05
# Last Modified: 2025-07-05 14:38:07
#
# This script runs various code quality tools on the AI Assistant codebase,
# integrating with the core system architecture to ensure consistent code
# quality across all components.
# ==============================================================================

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/data/logs"
REPORT_DIR="${PROJECT_ROOT}/lint-reports"
LINT_LOG="${LOG_DIR}/lint_$(date +%Y%m%d_%H%M%S).log"
TODAY="$(date +%Y-%m-%d)"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"

# Default values
LINT_TOOLS="all"       # Tools to run: flake8, pylint, mypy, bandit, black, isort, all
LINT_TARGET="all"      # Target to lint: core, assistant, processing, etc., all
FIX_ISSUES=false       # Whether to fix issues automatically where possible
VERBOSE=false          # Verbose output
CI_MODE=false          # CI mode (non-interactive, exit code reflects findings)
SHOW_STATS=true        # Show statistics
STRICT_MODE=false      # Strict mode (fail on any issue)
IGNORE_PATTERNS=""     # Patterns to ignore
SEVERITY="medium"      # Severity level: low, medium, high

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

# Logging functions
log() {
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    echo -e "${timestamp} - $1" | tee -a "$LINT_LOG"
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
    echo "  -t, --tools TOOLS       Linting tools to run: flake8, pylint, mypy, bandit,"
    echo "                          black, isort, all (default: all)"
    echo "  -m, --module MODULE     Specific module to lint (e.g., core, assistant, processing)"
    echo "  -f, --fix               Automatically fix issues where possible"
    echo "  -v, --verbose           Show verbose output"
    echo "  --ci                    Run in CI mode (non-interactive, exit with error on issues)"
    echo "  --no-stats              Don't show statistics"
    echo "  --strict                Strict mode (fail on any issue)"
    echo "  --ignore PATTERN        Ignore pattern (can be used multiple times)"
    echo "  --severity LEVEL        Severity level: low, medium, high (default: medium)"
    echo "  -h, --help              Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tools)
            LINT_TOOLS="$2"
            shift 2
            ;;
        -m|--module)
            LINT_TARGET="$2"
            shift 2
            ;;
        -f|--fix)
            FIX_ISSUES=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --ci)
            CI_MODE=true
            shift
            ;;
        --no-stats)
            SHOW_STATS=false
            shift
            ;;
        --strict)
            STRICT_MODE=true
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
        --severity)
            SEVERITY="$2"
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

# Validate tools
validate_tools() {
    if [ "$LINT_TOOLS" != "all" ]; then
        valid_tools=("flake8" "pylint" "mypy" "bandit" "black" "isort")
        IFS=',' read -ra selected_tools <<< "$LINT_TOOLS"
        
        for tool in "${selected_tools[@]}"; do
            if ! [[ " ${valid_tools[*]} " =~ " ${tool} " ]]; then
                log_error "Invalid tool: $tool"
                usage
            fi
        done
    fi
}

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
    
    # Define tools to check based on what we'll use
    local tools_to_check=()
    if [ "$LINT_TOOLS" = "all" ]; then
        tools_to_check=("flake8" "pylint" "mypy" "bandit" "black" "isort")
    else
        IFS=',' read -ra tools_to_check <<< "$LINT_TOOLS"
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
                pip install "${missing_tools[@]}"
                log_success "Tools installed"
            fi
        fi
    else
        log_success "All required tools are installed"
    fi
}

# Function to determine lint target path
get_target_path() {
    if [ "$LINT_TARGET" = "all" ]; then
        echo "${PROJECT_ROOT}/src"
    else
        # Handle special cases
        case "$LINT_TARGET" in
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
            *)
                if [ -d "${PROJECT_ROOT}/src/$LINT_TARGET" ]; then
                    echo "${PROJECT_ROOT}/src/$LINT_TARGET"
                else
                    log_error "Invalid target module: $LINT_TARGET"
                    exit 1
                fi
                ;;
        esac
    fi
}

# Function to run flake8
run_flake8() {
    log_section "Running Flake8"
    
    local target_path=$(get_target_path)
    local output_file="${REPORT_DIR}/flake8_${TIMESTAMP}.txt"
    local flake8_args="--config=${PROJECT_ROOT}/setup.cfg"
    
    if [ "$VERBOSE" = true ]; then
        flake8_args="$flake8_args --verbose"
    fi
    
    if [ -n "$IGNORE_PATTERNS" ]; then
        flake8_args="$flake8_args --extend-ignore=$IGNORE_PATTERNS"
    fi
    
    log "Running flake8 on: $target_path"
    log "Output will be saved to: $output_file"
    
    # Run flake8
    flake8 $flake8_args "$target_path" | tee "$output_file"
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        log_success "Flake8 found no issues"
    else
        issue_count=$(wc -l < "$output_file")
        log_warning "Flake8 found $issue_count issues"
        
        # Show statistics if requested
        if [ "$SHOW_STATS" = true ]; then
            echo -e "\n${CYAN}Top issues by frequency:${NC}"
            grep -o 'E[0-9]\+\|W[0-9]\+' "$output_file" | sort | uniq -c | sort -rn | head -10
        fi
    fi
    
    return $exit_code
}

# Function to run pylint
run_pylint() {
    log_section "Running Pylint"
    
    local target_path=$(get_target_path)
    local output_file="${REPORT_DIR}/pylint_${TIMESTAMP}.txt"
    local pylint_args="--rcfile=${PROJECT_ROOT}/pyproject.toml"
    
    # Set severity based on user input
    case "$SEVERITY" in
        low)
            pylint_args="$pylint_args --disable=C"
            ;;
        medium)
            pylint_args="$pylint_args --disable=C0111,C0103"
            ;;
        high)
            # Default is all checks enabled
            ;;
    esac
    
    if [ "$VERBOSE" = true ]; then
        pylint_args="$pylint_args -v"
    fi
    
    log "Running pylint on: $target_path"
    log "Output will be saved to: $output_file"
    
    # Run pylint
    pylint $pylint_args "$target_path" | tee "$output_file"
    local exit_code=${PIPESTATUS[0]}
    
    # Parse and display results
    if grep -q "Your code has been rated" "$output_file"; then
        rating=$(grep "Your code has been rated" "$output_file")
        echo -e "\n${CYAN}$rating${NC}"
        
        # Extract score
        score=$(echo "$rating" | grep -oP '\d+\.\d+/10')
        numeric_score=$(echo "$score" | cut -d'/' -f1)
        
        if (( $(echo "$numeric_score >= 9.0" | bc -l) )); then
            log_success "Pylint score is excellent: $score"
        elif (( $(echo "$numeric_score >= 7.0" | bc -l) )); then
            log_warning "Pylint score is good but could be improved: $score"
        else
            log_error "Pylint score is low: $score"
        fi
    else
        log_error "Failed to parse pylint results"
    fi
    
    # Show statistics if requested
    if [ "$SHOW_STATS" = true ] && [ -s "$output_file" ]; then
        echo -e "\n${CYAN}Top issues by category:${NC}"
        grep -E '^\S+.*: [CEFIRW][0-9]+' "$output_file" | cut -d':' -f3 | cut -d' ' -f2 | sort | uniq -c | sort -rn | head -10
    fi
    
    return $exit_code
}

# Function to run mypy
run_mypy() {
    log_section "Running MyPy"
    
    local target_path=$(get_target_path)
    local output_file="${REPORT_DIR}/mypy_${TIMESTAMP}.txt"
    local mypy_args="--config-file=${PROJECT_ROOT}/pyproject.toml"
    
    if [ "$VERBOSE" = true ]; then
        mypy_args="$mypy_args --verbose"
    fi
    
    log "Running mypy on: $target_path"
    log "Output will be saved to: $output_file"
    
    # Run mypy
    mypy $mypy_args "$target_path" | tee "$output_file"
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        log_success "MyPy found no type issues"
    else
        issue_count=$(grep -c "error:" "$output_file" || echo 0)
        log_warning "MyPy found $issue_count type issues"
        
        # Show statistics if requested
        if [ "$SHOW_STATS" = true ] && [ -s "$output_file" ]; then
            echo -e "\n${CYAN}Top file issues:${NC}"
            grep "error:" "$output_file" | cut -d':' -f1 | sort | uniq -c | sort -rn | head -10
        fi
    fi
    
    return $exit_code
}

# Function to run bandit
run_bandit() {
    log_section "Running Bandit Security Analysis"
    
    local target_path=$(get_target_path)
    local output_file="${REPORT_DIR}/bandit_${TIMESTAMP}.txt"
    local bandit_args="-r"
    
    # Set severity based on user input
    case "$SEVERITY" in
        low)
            bandit_args="$bandit_args -l"
            ;;
        medium)
            bandit_args="$bandit_args -i"
            ;;
        high)
            bandit_args="$bandit_args -iii"
            ;;
    esac
    
    if [ "$VERBOSE" = true ]; then
        bandit_args="$bandit_args -v"
    fi
    
    log "Running bandit on: $target_path"
    log "Output will be saved to: $output_file"
    
    # Run bandit
    bandit $bandit_args "$target_path" | tee "$output_file"
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        log_success "Bandit found no security issues"
    else
        issue_count=$(grep -c "Issue:" "$output_file" || echo 0)
        log_warning "Bandit found $issue_count security issues"
        
        # Show statistics if requested
        if [ "$SHOW_STATS" = true ] && [ -s "$output_file" ]; then
            echo -e "\n${CYAN}Security issues by severity:${NC}"
            grep "Severity: " "$output_file" | sort | uniq -c | sort -rn
            
            echo -e "\n${CYAN}Security issues by confidence:${NC}"
            grep "Confidence: " "$output_file" | sort | uniq -c | sort -rn
        fi
    fi
    
    return $exit_code
}

# Function to run black
run_black() {
    log_section "Running Black Code Formatter"
    
    local target_path=$(get_target_path)
    local output_file="${REPORT_DIR}/black_${TIMESTAMP}.txt"
    local black_args="--config=${PROJECT_ROOT}/pyproject.toml"
    
    if [ "$VERBOSE" = true ]; then
        black_args="$black_args --verbose"
    fi
    
    if [ "$FIX_ISSUES" = true ]; then
        log "Running black to format code: $target_path"
        black $black_args "$target_path" | tee "$output_file"
        local exit_code=${PIPESTATUS[0]}
        
        if [ $exit_code -eq 0 ]; then
            log_success "Black successfully formatted code"
        else
            log_error "Black encountered issues while formatting"
        fi
    else
        log "Running black in check mode: $target_path"
        black --check $black_args "$target_path" | tee "$output_file"
        local exit_code=${PIPESTATUS[0]}
        
        if [ $exit_code -eq 0 ]; then
            log_success "Black check passed - all files properly formatted"
        else
            file_count=$(grep -c "would reformat" "$output_file" || echo 0)
            log_warning "Black would reformat $file_count files"
            
            if [ "$SHOW_STATS" = true ] && [ -s "$output_file" ]; then
                echo -e "\n${CYAN}Files that would be reformatted:${NC}"
                grep "would reformat" "$output_file" | sed 's/would reformat //' | head -10
                
                if [ $file_count -gt 10 ]; then
                    echo -e "${CYAN}...and $(($file_count - 10)) more files${NC}"
                fi
            fi
        fi
    fi
    
    return $exit_code
}

# Function to run isort
run_isort() {
    log_section "Running isort Import Sorter"
    
    local target_path=$(get_target_path)
    local output_file="${REPORT_DIR}/isort_${TIMESTAMP}.txt"
    local isort_args="--settings-path=${PROJECT_ROOT}/pyproject.toml"
    
    if [ "$VERBOSE" = true ]; then
        isort_args="$isort_args --verbose"
    fi
    
    if [ "$FIX_ISSUES" = true ]; then
        log "Running isort to sort imports: $target_path"
        isort $isort_args "$target_path" | tee "$output_file"
        local exit_code=${PIPESTATUS[0]}
        
        if [ $exit_code -eq 0 ]; then
            log_success "isort successfully sorted imports"
        else
            log_error "isort encountered issues while sorting imports"
        fi
    else
        log "Running isort in check mode: $target_path"
        isort --check $isort_args "$target_path" | tee "$output_file"
        local exit_code=${PIPESTATUS[0]}
        
        if [ $exit_code -eq 0 ]; then
            log_success "isort check passed - all imports properly sorted"
        else
            file_count=$(grep -c "ERROR" "$output_file" || echo 0)
            log_warning "isort found $file_count files with unsorted imports"
            
            if [ "$SHOW_STATS" = true ] && [ -s "$output_file" ]; then
                echo -e "\n${CYAN}Files with unsorted imports:${NC}"
                grep -A 1 "ERROR" "$output_file" | grep -v "ERROR" | grep -v -- "--" | head -10
                
                if [ $file_count -gt 10 ]; then
                    echo -e "${CYAN}...and $(($file_count - 10)) more files${NC}"
                fi
            fi
        fi
    fi
    
    return $exit_code
}

# Check core system component coverage
check_core_components() {
    log_section "Checking Core System Components"
    
    # Core components that must have proper linting
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
    
    # Check if each core component exists and follows standards
    MISSING=0
    FAILING=0
    
    for component in "${CORE_COMPONENTS[@]}"; do
        if [ ! -f "${PROJECT_ROOT}/$component" ]; then
            log_warning "Core component not found: $component"
            MISSING=$((MISSING + 1))
            continue
        fi
        
        # Quick simple check for minimum documentation standards
        DOC_LINES=$(grep -c '"""' "${PROJECT_ROOT}/$component" || echo 0)
        if [ $DOC_LINES -lt 2 ]; then
            log_warning "Core component lacks proper documentation: $component"
            FAILING=$((FAILING + 1))
        fi
    done
    
    if [ $MISSING -gt 0 ]; then
        log_warning "$MISSING core components are missing"
    fi
    
    if [ $FAILING -gt 0 ]; then
        log_warning "$FAILING core components have documentation issues"
    fi
    
    if [ $MISSING -eq 0 ] && [ $FAILING -eq 0 ]; then
        log_success "All core components exist and have basic documentation"
    fi
}

# Run all linting tools
run_all_linters() {
    local success=true
    local tools=()
    
    # Determine which tools to run
    if [ "$LINT_TOOLS" = "all" ]; then
        tools=("flake8" "pylint" "mypy" "bandit" "black" "isort")
    else
        IFS=',' read -ra tools <<< "$LINT_TOOLS"
    fi
    
    # Initialize result tracking
    declare -A results
    
    # Run each tool
    for tool in "${tools[@]}"; do
        case $tool in
            flake8)
                run_flake8
                results["flake8"]=$?
                ;;
            pylint)
                run_pylint
                results["pylint"]=$?
                ;;
            mypy)
                run_mypy
                results["mypy"]=$?
                ;;
            bandit)
                run_bandit
                results["bandit"]=$?
                ;;
            black)
                run_black
                results["black"]=$?
                ;;
            isort)
                run_isort
                results["isort"]=$?
                ;;
        esac
        
        # Check if strict mode and tool failed
        if [ "$STRICT_MODE" = true ] && [ ${results["$tool"]} -ne 0 ]; then
            log_error "$tool failed and strict mode is enabled"
            success=false
            break
        fi
    done
    
    # Print summary
    log_section "Linting Summary"
    
    for tool in "${tools[@]}"; do
        if [ ${results["$tool"]} -eq 0 ]; then
            log_success "$tool: PASSED"
        else
            log_warning "$tool: FAILED"
            success=false
        fi
    done
    
    if [ "$success" = true ]; then
        log_success "All linters passed successfully!"
        return 0
    else
        log_warning "Some linters reported issues"
        if [ "$CI_MODE" = true ]; then
            return 1
        else
            return 0
        fi
    fi
}

# Main function
main() {
    log_section "Starting AI Assistant Code Linting"
    log "User: $(whoami)"
    log "Date: $TODAY"
    log "Tools: $LINT_TOOLS"
    log "Target: $LINT_TARGET"
    log "Fix issues: $FIX_ISSUES"
    log "Severity level: $SEVERITY"
    
    # Validate tools
    validate_tools
    
    # Activate virtual environment
    activate_venv
    
    # Check dependencies
    check_dependencies
    
    # Check core components
    check_core_components
    
    # Run linters
    run_all_linters
    result=$?
    
    # Provide recommendations
    if [ $result -ne 0 ]; then
        echo -e "\n${CYAN}Recommendations:${NC}"
        echo "- Run with --fix to automatically fix formatting issues"
        echo "- Check specific files mentioned in the reports"
        echo "- Focus on critical issues first (E and F level in flake8, error in mypy)"
    fi
    
    # Report location of logs
    echo -e "\n${GREEN}Lint reports saved to: ${REPORT_DIR}${NC}"
    echo -e "${GREEN}Lint log saved to: ${LINT_LOG}${NC}"
    
    exit $result
}

# Run main function
main
