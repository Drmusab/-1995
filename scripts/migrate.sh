#!/bin/bash
# ==============================================================================
# AI Assistant Database Migration Script
# Author: Drmusab
# Created: 2025-07-05
# Last Modified: 2025-07-05 20:02:23
#
# This script manages database migrations for the AI Assistant system.
# It integrates with the core system architecture to ensure database schema
# changes are properly synchronized with the application code.
# ==============================================================================

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/data/logs"
MIGRATION_LOG="${LOG_DIR}/migration_$(date +%Y%m%d_%H%M%S).log"
MIGRATIONS_DIR="${PROJECT_ROOT}/migrations"
ALEMBIC_CONFIG="${MIGRATIONS_DIR}/alembic.ini"
TODAY="$(date +%Y-%m-%d)"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"

# Default values
ENVIRONMENT="development"  # Environment: development, staging, production
OPERATION="upgrade"        # Operation: upgrade, downgrade, revision, status, history, current
REVISION="head"            # Revision: head, base, -1, +1, hash, etc.
MESSAGE=""                 # Message for new revisions
SQL_MODE=false             # Output SQL instead of executing
VERBOSE=false              # Verbose output
FORCE=false                # Force operation without confirmation
AUTO_GENERATE=true         # Auto-generate migrations based on models
CHECK_DB=true              # Check database connection before migrating
BACKUP=true                # Backup database before migrating (when possible)
BACKUP_DIR="${PROJECT_ROOT}/data/backups/db"
DRY_RUN=false              # Dry run (don't actually make changes)

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Ensure log and backup directories exist
mkdir -p "$LOG_DIR"
mkdir -p "$BACKUP_DIR"

# Logging functions
log() {
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    echo -e "${timestamp} - $1" | tee -a "$MIGRATION_LOG"
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
    echo "  -o, --operation OP       Operation: upgrade, downgrade, revision, status,"
    echo "                           history, current (default: upgrade)"
    echo "  -r, --revision REV       Revision: head, base, -1, +1, hash, etc."
    echo "                           (default: head for upgrade, -1 for downgrade)"
    echo "  -m, --message MSG        Message for new revisions (required for revision operation)"
    echo "  -s, --sql                Output SQL instead of executing"
    echo "  -v, --verbose            Show verbose output"
    echo "  -f, --force              Force operation without confirmation"
    echo "  --no-autogenerate        Disable auto-generating migrations based on models"
    echo "  --no-db-check            Skip database connection check"
    echo "  --no-backup              Skip database backup"
    echo "  --dry-run                Don't actually make changes"
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
        -o|--operation)
            OPERATION="$2"
            shift 2
            ;;
        -r|--revision)
            REVISION="$2"
            shift 2
            ;;
        -m|--message)
            MESSAGE="$2"
            shift 2
            ;;
        -s|--sql)
            SQL_MODE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --no-autogenerate)
            AUTO_GENERATE=false
            shift
            ;;
        --no-db-check)
            CHECK_DB=false
            shift
            ;;
        --no-backup)
            BACKUP=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
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

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be one of: development, staging, production"
    exit 1
fi

# Validate operation
if [[ ! "$OPERATION" =~ ^(upgrade|downgrade|revision|status|history|current)$ ]]; then
    log_error "Invalid operation: $OPERATION. Must be one of: upgrade, downgrade, revision, status, history, current"
    exit 1
fi

# Check if message is provided for revision operation
if [[ "$OPERATION" = "revision" && -z "$MESSAGE" && "$AUTO_GENERATE" = false ]]; then
    log_error "Message is required for revision operation without auto-generate"
    usage
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
    export MIGRATION_OPERATION="$OPERATION"
    
    # Load specific config based on environment
    export CONFIG_PATH="${PROJECT_ROOT}/configs/environments/${ENVIRONMENT}.yaml"
    
    if [ ! -f "$CONFIG_PATH" ]; then
        log_error "Configuration file not found: $CONFIG_PATH"
        exit 1
    fi
    
    log_success "Environment configuration loaded: $ENVIRONMENT"
}

# Function to check database connection
check_database() {
    if [ "$CHECK_DB" = false ]; then
        log "Skipping database connection check"
        return 0
    fi
    
    log_section "Checking Database Connection"
    
    log "Testing database connection..."
    
    # Use Python CLI module to check database connection
    if python -m src.cli check-db-connection --environment "$ENVIRONMENT" &> /dev/null; then
        log_success "Database connection successful"
        return 0
    else
        log_error "Failed to connect to database. Please check your database configuration."
        return 1
    fi
}

# Function to backup database
backup_database() {
    if [ "$BACKUP" = false ]; then
        log "Skipping database backup"
        return 0
    fi
    
    log_section "Backing Up Database"
    
    # Skip backup for certain operations
    if [[ "$OPERATION" =~ ^(status|history|current)$ ]]; then
        log "Skipping backup for read-only operation: $OPERATION"
        return 0
    fi
    
    # Skip backup for dry run
    if [ "$DRY_RUN" = true ]; then
        log "Skipping backup for dry run"
        return 0
    fi
    
    # Create backup filename with timestamp
    BACKUP_FILE="${BACKUP_DIR}/backup_${ENVIRONMENT}_${TIMESTAMP}.sql"
    
    log "Creating database backup: $BACKUP_FILE"
    
    # Use Python CLI module to backup database
    if python -m src.cli backup-db --environment "$ENVIRONMENT" --output "$BACKUP_FILE"; then
        log_success "Database backup created: $BACKUP_FILE"
        return 0
    else
        log_warning "Failed to create database backup. Proceeding without backup."
        
        # Ask for confirmation if not forced
        if [ "$FORCE" = false ]; then
            read -p "Continue without backup? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Operation cancelled by user"
                exit 0
            fi
        fi
        
        return 1
    fi
}

# Function to get database info
get_database_info() {
    log_section "Database Information"
    
    # Use Python CLI to get database info
    python -m src.cli db-info --environment "$ENVIRONMENT"
    
    if [ $? -ne 0 ]; then
        log_warning "Failed to retrieve database information"
    fi
}

# Function to build alembic command
build_alembic_command() {
    local cmd="alembic -c \"$ALEMBIC_CONFIG\""
    
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd --verbose"
    fi
    
    # Add operation
    case "$OPERATION" in
        upgrade)
            cmd="$cmd upgrade"
            if [ "$SQL_MODE" = true ]; then
                cmd="$cmd --sql"
            fi
            cmd="$cmd $REVISION"
            ;;
        downgrade)
            cmd="$cmd downgrade"
            if [ "$SQL_MODE" = true ]; then
                cmd="$cmd --sql"
            fi
            if [ "$REVISION" = "head" ]; then
                # If no revision specified for downgrade, use -1
                cmd="$cmd -1"
            else
                cmd="$cmd $REVISION"
            fi
            ;;
        revision)
            cmd="$cmd revision"
            if [ "$AUTO_GENERATE" = true ]; then
                cmd="$cmd --autogenerate"
            fi
            if [ -n "$MESSAGE" ]; then
                cmd="$cmd -m \"$MESSAGE\""
            fi
            ;;
        status)
            cmd="$cmd current"
            ;;
        history)
            cmd="$cmd history"
            ;;
        current)
            cmd="$cmd current"
            ;;
    esac
    
    echo "$cmd"
}

# Function to run migration
run_migration() {
    log_section "Running Migration: $OPERATION"
    
    # Build alembic command
    ALEMBIC_CMD=$(build_alembic_command)
    
    log "Migration command: $ALEMBIC_CMD"
    
    # Show warning for production migrations
    if [ "$ENVIRONMENT" = "production" ]; then
        log_warning "You are about to run migrations on the PRODUCTION environment!"
        if [ "$FORCE" = false ]; then
            read -p "Are you sure you want to continue? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Operation cancelled by user"
                exit 0
            fi
        fi
    fi
    
    # Show SQL only mode warning
    if [ "$SQL_MODE" = true ]; then
        log_warning "Running in SQL-only mode. No changes will be made to the database."
    fi
    
    # Show dry run warning
    if [ "$DRY_RUN" = true ]; then
        log_warning "Running in dry-run mode. No changes will be made to the database."
        return 0
    fi
    
    # Execute the command
    log "Executing migration..."
    
    if eval "$ALEMBIC_CMD"; then
        if [[ "$OPERATION" =~ ^(status|history|current)$ ]]; then
            log_success "Migration information retrieved successfully"
        else
            log_success "Migration completed successfully"
        fi
        return 0
    else
        log_error "Migration failed. See log for details."
        return 1
    fi
}

# Function to update system metadata
update_system_metadata() {
    log_section "Updating System Metadata"
    
    # Skip metadata update for certain operations
    if [[ "$OPERATION" =~ ^(status|history|current|revision)$ ]]; then
        log "Skipping metadata update for operation: $OPERATION"
        return 0
    fi
    
    # Skip metadata update for dry run
    if [ "$DRY_RUN" = true ]; then
        log "Skipping metadata update for dry run"
        return 0
    fi
    
    log "Updating system metadata with latest migration information..."
    
    # Use Python CLI to update system metadata
    if python -m src.cli update-system-metadata --component database --status migrated --environment "$ENVIRONMENT"; then
        log_success "System metadata updated successfully"
        return 0
    else
        log_warning "Failed to update system metadata. This may affect system health checks."
        return 1
    fi
}

# Function to verify migration
verify_migration() {
    log_section "Verifying Migration"
    
    # Skip verification for certain operations
    if [[ "$OPERATION" =~ ^(status|history|current|revision)$ ]]; then
        log "Skipping verification for operation: $OPERATION"
        return 0
    fi
    
    # Skip verification for dry run
    if [ "$DRY_RUN" = true ]; then
        log "Skipping verification for dry run"
        return 0
    fi
    
    log "Verifying database schema..."
    
    # Use Python CLI to verify database schema
    if python -m src.cli verify-db-schema --environment "$ENVIRONMENT"; then
        log_success "Database schema verified successfully"
        return 0
    else
        log_warning "Database schema verification failed. The migration may have issues."
        return 1
    fi
}

# Function to check core components
check_core_components() {
    log_section "Checking Core System Components"
    
    # Core storage components that might be affected by migrations
    CORE_COMPONENTS=(
        "src/integrations/storage/database.py"
        "src/memory/storage/vector_store.py"
        "src/memory/storage/memory_graph.py"
        "src/skills/skill_registry.py"
        "src/learning/continual_learning.py"
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
        log_warning "$MISSING core components are missing"
    else
        log_success "All core components exist"
    fi
    
    # Verify database integration
    if python -c "from src.integrations.storage.database import Database; print('Database integration verified')" &> /dev/null; then
        log_success "Database integration verified"
    else
        log_warning "Could not verify database integration"
    fi
}

# Function to display migration status
show_migration_status() {
    log_section "Migration Status"
    
    # Get current migration version
    local current_version=$(alembic -c "$ALEMBIC_CONFIG" current 2>/dev/null | grep -oP '(?<=\()\w+(?=\))' || echo "None")
    
    echo -e "${CYAN}Current Database Version: ${current_version}${NC}"
    
    # Count pending migrations
    local pending_count=$(alembic -c "$ALEMBIC_CONFIG" history --verbose 2>/dev/null | grep -c "not applied" || echo 0)
    
    if [ "$pending_count" -gt 0 ]; then
        echo -e "${YELLOW}Pending Migrations: ${pending_count}${NC}"
        
        # List pending migrations
        echo -e "\n${CYAN}Pending Migration Revisions:${NC}"
        alembic -c "$ALEMBIC_CONFIG" history --verbose 2>/dev/null | grep -B 1 "not applied" | grep -v "not applied" || echo "None"
    else
        echo -e "${GREEN}Pending Migrations: 0${NC}"
        echo -e "${GREEN}Database is up to date${NC}"
    fi
}

# Main function
main() {
    log_section "Starting AI Assistant Database Migration"
    log "User: $(whoami)"
    log "Date: $TODAY"
    log "Environment: $ENVIRONMENT"
    log "Operation: $OPERATION"
    log "Revision: $REVISION"
    
    # Activate virtual environment
    activate_venv
    
    # Load environment variables
    load_env_vars
    
    # Check core components
    check_core_components
    
    # Check database connection
    check_database || exit 1
    
    # Get database info
    get_database_info
    
    # Show migration status
    show_migration_status
    
    # Backup database if needed
    backup_database
    
    # Run migration
    run_migration
    
    # Verify migration
    verify_migration
    
    # Update system metadata
    update_system_metadata
    
    # Show final status
    if [[ ! "$OPERATION" =~ ^(status|history|current)$ ]]; then
        show_migration_status
    fi
    
    log_section "Migration Complete"
    log "Migration log saved to: $MIGRATION_LOG"
    
    echo -e "\n${GREEN}Migration operation completed successfully!${NC}"
}

# Run main function
main
