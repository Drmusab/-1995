#!/bin/bash
# ==============================================================================
# AI Assistant System Health Check Script
# Author: Drmusab
# Created: 2025-07-05
# Last Modified: 2025-07-05
#
# This script performs comprehensive health checks on the AI Assistant system
# including core components, dependencies, and resource availability.
# It integrates with the core system architecture to ensure all critical
# components are functioning properly.
# ==============================================================================

set -e

# Configuration
TIMEOUT=10
API_PORT=8000
GRPC_PORT=50051
LOG_FILE="/data/logs/health-check.log"
MAX_MEMORY_USAGE_PCT=90
MAX_DISK_USAGE_PCT=85
METRICS_FILE="/tmp/health_metrics.json"

# Ensure log directory exists
mkdir -p $(dirname "$LOG_FILE")

# Logging function
log() {
  local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
  echo "[$timestamp] $1" | tee -a "$LOG_FILE"
}

log "Starting health check..."

# ===============================
# REST API Health Check
# ===============================
check_rest_api() {
  log "Checking REST API health..."
  
  local health_endpoint="http://localhost:$API_PORT/health"
  local response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT "$health_endpoint")
  
  if [ "$response" -eq 200 ]; then
    log "REST API is healthy"
    return 0
  else
    log "ERROR: REST API health check failed with status $response"
    return 1
  fi
}

# ===============================
# Core System Health Check
# ===============================
check_core_system() {
  log "Checking core system health..."
  
  # Check core components via API
  local components_endpoint="http://localhost:$API_PORT/internal/health/components"
  local response=$(curl -s --connect-timeout $TIMEOUT "$components_endpoint")
  
  # Extract health status
  if echo "$response" | grep -q '"status":"healthy"'; then
    log "Core system components are healthy"
    return 0
  else
    log "ERROR: Core system health check failed"
    log "Response: $response"
    return 1
  fi
}

# ===============================
# Component Status Check
# ===============================
check_component_status() {
  log "Checking individual component status..."
  
  # Get component statuses
  local components_endpoint="http://localhost:$API_PORT/internal/components/status"
  local response=$(curl -s --connect-timeout $TIMEOUT "$components_endpoint")
  
  # Parse response and check critical components
  if echo "$response" | grep -q '"core_engine":.*"state":"running"' && 
     echo "$response" | grep -q '"component_manager":.*"state":"running"' && 
     echo "$response" | grep -q '"session_manager":.*"state":"running"' && 
     echo "$response" | grep -q '"plugin_manager":.*"state":"running"'; then
    log "Critical components are running"
    return 0
  else
    log "ERROR: Some critical components are not running"
    log "Response: $response"
    return 1
  fi
}

# ===============================
# Database Connection Check
# ===============================
check_database_connections() {
  log "Checking database connections..."
  
  # Check database connections via API
  local db_endpoint="http://localhost:$API_PORT/internal/health/database"
  local response=$(curl -s --connect-timeout $TIMEOUT "$db_endpoint")
  
  # Check if all connections are good
  if echo "$response" | grep -q '"postgres":"connected"' && 
     echo "$response" | grep -q '"redis":"connected"' && 
     echo "$response" | grep -q '"vector_db":"connected"'; then
    log "Database connections are healthy"
    return 0
  else
    log "ERROR: Some database connections are failing"
    log "Response: $response"
    return 1
  fi
}

# ===============================
# Memory System Check
# ===============================
check_memory_system() {
  log "Checking memory system..."
  
  # Check memory system health
  local memory_endpoint="http://localhost:$API_PORT/internal/health/memory"
  local response=$(curl -s --connect-timeout $TIMEOUT "$memory_endpoint")
  
  if echo "$response" | grep -q '"status":"healthy"'; then
    log "Memory system is healthy"
    return 0
  else
    log "ERROR: Memory system check failed"
    log "Response: $response"
    return 1
  fi
}

# ===============================
# Plugin System Check
# ===============================
check_plugin_system() {
  log "Checking plugin system..."
  
  # Check plugin system health
  local plugin_endpoint="http://localhost:$API_PORT/internal/plugins/status"
  local response=$(curl -s --connect-timeout $TIMEOUT "$plugin_endpoint")
  
  if echo "$response" | grep -q '"status":"healthy"'; then
    log "Plugin system is healthy"
    return 0
  else
    log "ERROR: Plugin system check failed"
    log "Response: $response"
    return 1
  fi
}

# ===============================
# System Resource Check
# ===============================
check_system_resources() {
  log "Checking system resources..."
  
  # Check memory usage
  local memory_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
  if [ "$memory_usage" -gt "$MAX_MEMORY_USAGE_PCT" ]; then
    log "WARNING: High memory usage: ${memory_usage}%"
    # Continue despite warning
  fi
  
  # Check disk usage
  local disk_usage=$(df -h / | grep / | awk '{print int($5)}')
  if [ "$disk_usage" -gt "$MAX_DISK_USAGE_PCT" ]; then
    log "WARNING: High disk usage: ${disk_usage}%"
    # Continue despite warning
  fi
  
  # Check for critical Python processes
  if ! pgrep -f "python -m src.main" > /dev/null; then
    log "ERROR: Main Python process not found"
    return 1
  fi
  
  log "System resources check passed"
  return 0
}

# ===============================
# Model Availability Check
# ===============================
check_model_availability() {
  log "Checking model availability..."
  
  # Check model status
  local model_endpoint="http://localhost:$API_PORT/internal/models/status"
  local response=$(curl -s --connect-timeout $TIMEOUT "$model_endpoint")
  
  # We need at least one model to be available
  if echo "$response" | grep -q '"loaded":true'; then
    log "Required models are available"
    return 0
  else
    log "ERROR: Required models are not available"
    log "Response: $response"
    return 1
  fi
}

# ===============================
# gRPC Service Check
# ===============================
check_grpc_service() {
  log "Checking gRPC service..."
  
  # Check if the gRPC port is listening
  if nc -z localhost $GRPC_PORT; then
    log "gRPC service is running"
    return 0
  else
    log "ERROR: gRPC service is not available"
    return 1
  fi
}

# ===============================
# API Key Verification
# ===============================
check_api_key() {
  log "Verifying API key configuration..."
  
  # Check if API key verification is working
  local auth_endpoint="http://localhost:$API_PORT/internal/auth/status"
  local response=$(curl -s --connect-timeout $TIMEOUT "$auth_endpoint")
  
  if echo "$response" | grep -q '"configured":true'; then
    log "API key configuration is valid"
    return 0
  else
    log "WARNING: API key configuration may have issues"
    # Continue despite warning
    return 0
  fi
}

# ===============================
# Collect Metrics
# ===============================
collect_metrics() {
  log "Collecting health metrics..."
  
  # Gather key metrics
  local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  local uptime=$(cat /proc/uptime | awk '{print $1}')
  local memory_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
  local disk_usage=$(df -h / | grep / | awk '{print int($5)}')
  local process_count=$(ps aux | grep python | wc -l)
  
  # Create metrics JSON
  cat > "$METRICS_FILE" << EOF
{
  "timestamp": "$timestamp",
  "uptime_seconds": $uptime,
  "memory_usage_percent": $memory_usage,
  "disk_usage_percent": $disk_usage,
  "python_process_count": $process_count,
  "health_check_run_by": "$(whoami)"
}
EOF

  log "Metrics collected and saved to $METRICS_FILE"
}

# ===============================
# Run all checks
# ===============================
run_all_checks() {
  local failed=false
  
  # Critical checks that must pass
  check_rest_api || failed=true
  check_core_system || failed=true
  check_component_status || failed=true
  check_database_connections || failed=true
  
  # Important but non-critical checks
  check_memory_system || log "WARNING: Memory system check failed but continuing"
  check_plugin_system || log "WARNING: Plugin system check failed but continuing"
  check_model_availability || log "WARNING: Model availability check failed but continuing"
  
  # Additional checks
  check_system_resources
  check_grpc_service || log "WARNING: gRPC service check failed but continuing"
  check_api_key
  
  # Always collect metrics regardless of check results
  collect_metrics
  
  if [ "$failed" = true ]; then
    log "HEALTH CHECK FAILED: One or more critical checks failed"
    return 1
  else
    log "HEALTH CHECK PASSED: All critical checks successful"
    return 0
  fi
}

# ===============================
# Main execution
# ===============================

# Create a unique file for this run
UNIQUE_RUN_ID=$(date +"%Y%m%d%H%M%S")
TEMP_STATUS_FILE="/tmp/health_check_${UNIQUE_RUN_ID}.status"

# Execute all health checks and capture the result
if run_all_checks; then
  echo "healthy" > "$TEMP_STATUS_FILE"
  exit_code=0
else
  echo "unhealthy" > "$TEMP_STATUS_FILE"
  exit_code=1
fi

# Move temp file to final status location for other processes to read
mv "$TEMP_STATUS_FILE" "/tmp/health_check_latest.status"

log "Health check completed with exit code: $exit_code"
exit $exit_code
