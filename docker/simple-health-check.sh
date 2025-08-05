#!/bin/bash
# ==============================================================================
# Simple Docker Health Check Script
# Author: Drmusab
# Created: 2025-01-14
# 
# Simplified health check script optimized for containerized environments.
# This performs basic health checks without complex dependencies.
# ==============================================================================

set -e

# Configuration
API_PORT=${API_PORT:-8000}
TIMEOUT=${TIMEOUT:-10}

# Simple logging
log() {
  echo "[$(date -u +"%Y-%m-%d %H:%M:%S UTC")] $1"
}

# Basic health check function
check_health() {
  local endpoint="http://localhost:${API_PORT}/health"
  
  # Try to connect to the health endpoint
  if curl -f -s --connect-timeout "$TIMEOUT" "$endpoint" > /dev/null 2>&1; then
    log "Health check passed"
    return 0
  else
    log "Health check failed - service not responding"
    return 1
  fi
}

# Check if the main process is running
check_process() {
  if pgrep -f "python -m src.main" > /dev/null 2>&1; then
    log "Main process is running"
    return 0
  else
    log "Main process not found"
    return 1
  fi
}

# Run health checks
main() {
  log "Starting simple health check..."
  
  # Check if the main process is running first
  if ! check_process; then
    exit 1
  fi
  
  # Check if the API is responding
  if ! check_health; then
    exit 1
  fi
  
  log "All health checks passed"
  exit 0
}

# Execute health check
main "$@"