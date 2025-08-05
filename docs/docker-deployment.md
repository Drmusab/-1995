# Docker Deployment Guide

This guide covers deploying the AI Assistant using Docker for both development and production environments.

## Quick Start

### Production Deployment

```bash
# Build and run with docker-compose
cd docker
docker-compose up -d

# Or build and run manually
docker build -f docker/Dockerfile -t ai-assistant:latest .
docker run -d -p 8000:8000 -v ai_data:/data ai-assistant:latest
```

### Development Environment

```bash
# Development with hot reload
cd docker
docker-compose -f docker-compose.dev.yml up -d

# Access the development container
docker exec -it ai_assistant_dev bash
```

## Docker Configuration

### Environment Files

- `docker/.env.production` - Production environment variables
- `docker/.env.development` - Development environment variables

### Health Checks

The containers include built-in health checks:
- **Production**: `curl -f http://localhost:8000/health`
- **Interval**: 30s with 60s startup period
- **Simplified health check**: `docker/simple-health-check.sh`

### Security Features

- Multi-stage builds for optimized image size
- Non-root user (`aiassistant`) for container security
- Separate development and production configurations
- Minimal runtime dependencies

### Volumes and Data Persistence

```yaml
volumes:
  - ai_data:/data              # Application data
  - redis_data:/var/lib/redis  # Redis persistence
  - postgres_data:/var/lib/postgresql/data  # Database
```

## Service Architecture

### Production Stack (`docker-compose.yml`)

- **ai_assistant**: Main application (Python 3.10)
- **redis**: Cache and session storage
- **postgres**: Primary database
- **vector_db** (Qdrant): Vector embeddings storage
- **monitoring** (Prometheus): Metrics collection

### Development Stack (`docker-compose.dev.yml`)

Similar to production but with:
- Hot reload enabled
- Debug logging
- Development database
- Source code mounted as volume

## Resource Limits

### Production
- **CPU**: 4 cores limit, 1 core reservation
- **Memory**: 8GB limit, 2GB reservation

### Development
- Default Docker limits apply
- Suitable for local development

## Environment Variables

### Core Configuration
```bash
ENVIRONMENT=production|development
LOG_LEVEL=INFO|DEBUG
CONFIG_PATH=/app/configs/environments/production.yaml
```

### Database Connections
```bash
REDIS_HOST=redis
POSTGRES_HOST=postgres
VECTOR_DB_HOST=vector_db
```

### Security
```bash
JWT_SECRET_KEY=your-jwt-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here
```

## Troubleshooting

### Common Issues

1. **Build Failures**: Check network connectivity and SSL certificates
2. **Container Won't Start**: Check logs with `docker logs ai_assistant`
3. **Health Check Failures**: Verify service dependencies are running
4. **Permission Issues**: Ensure proper volume permissions

### Debugging

```bash
# Check container logs
docker logs ai_assistant

# Access container shell
docker exec -it ai_assistant bash

# Check service health
docker exec ai_assistant curl -f http://localhost:8000/health

# Monitor resources
docker stats ai_assistant
```

### Performance Tuning

1. **Memory**: Adjust `deploy.resources.limits.memory` in docker-compose.yml
2. **CPU**: Adjust `deploy.resources.limits.cpus`
3. **Workers**: Set `MAX_WORKERS` environment variable
4. **Cache**: Configure Redis memory limits

## Security Considerations

- All containers run as non-root users
- Secrets should be managed via Docker secrets or external secret management
- Network isolation using Docker networks
- Regular security updates for base images

## Monitoring and Observability

- Health checks on all critical services
- Prometheus metrics exposure
- Structured logging with correlation IDs
- OpenTelemetry tracing support

## Backup and Recovery

```bash
# Backup volumes
docker run --rm -v ai_data:/data -v $(pwd):/backup alpine tar czf /backup/ai_data.tar.gz -C /data .

# Restore volumes
docker run --rm -v ai_data:/data -v $(pwd):/backup alpine tar xzf /backup/ai_data.tar.gz -C /data
```