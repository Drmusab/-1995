# AI Assistant

Advanced AI Assistant with multimodal processing, workflow orchestration, and extensible plugin system.

## Features

- **Multimodal Processing**: Support for text, speech, and vision inputs
- **Workflow Orchestration**: Complex task automation and execution
- **Plugin System**: Extensible architecture with custom plugins
- **Memory Management**: Advanced memory storage and retrieval
- **Learning & Adaptation**: Continuous learning from user interactions
- **Real-time Communication**: WebSocket, REST, and GraphQL APIs
- **Security**: Built-in authentication, authorization, and encryption
- **Monitoring**: Comprehensive observability and health checks

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Drmusab/-1995.git
cd -1995
```

2. Install dependencies:
```bash
pip install -e .
```

3. Start the assistant:
```bash
ai-assistant
```

### CLI Usage

```bash
# Interactive mode (default)
ai-assistant

# Command mode
ai-assistant -c "status"

# REPL mode
ai-assistant -r

# Monitor mode
ai-assistant -m
```

### Server Mode

```bash
ai-assistant-server
```

## Architecture

The AI Assistant is built with a modular architecture:

- **Core**: Configuration, events, security, dependency injection
- **Assistant**: Core engine, session management, workflows, plugins
- **Processing**: Natural language, speech, vision, multimodal
- **Reasoning**: Logic engine, knowledge graph, planning
- **Memory**: Storage, retrieval, consolidation
- **Skills**: Extensible skill system
- **Integrations**: LLM providers, caching, storage, external APIs
- **APIs**: REST, WebSocket, GraphQL interfaces
- **UI**: CLI and web interfaces
- **Observability**: Monitoring, logging, tracing

## Development

### Setup Development Environment

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest
```

4. Format code:
```bash
black src tests
isort src tests
```

5. Lint code:
```bash
flake8 src tests
mypy src
```

### Running in Docker

#### Production Deployment
```bash
# Using docker-compose (recommended)
cd docker
docker-compose up -d

# Using Docker directly
docker build -f docker/Dockerfile -t ai-assistant:latest .
docker run -d -p 8000:8000 -v ai_data:/data ai-assistant:latest
```

#### Development Environment
```bash
# Development with hot reload
cd docker
docker-compose -f docker-compose.dev.yml up -d

# Access development container
docker exec -it ai_assistant_dev bash
```

#### Docker Features
- **Multi-stage builds** for optimized production images
- **Security hardened** with non-root user and minimal attack surface
- **Health checks** for reliable container orchestration
- **Environment-specific** configurations for dev/prod
- **Persistent volumes** for data and model storage
- **Service mesh** with Redis, PostgreSQL, and Qdrant integration

For detailed Docker deployment instructions, see [Docker Deployment Guide](docs/docker-deployment.md).

## Configuration

The assistant uses YAML configuration files in the `configs/` directory:

- `configs/environments/development.yaml` - Development settings
- `configs/environments/production.yaml` - Production settings
- `configs/models/model_configs.yaml` - Model configurations
- `configs/skills/skill_configs.yaml` - Skill configurations

Environment variables can be set in `.env` file based on `.env.example`.

## API Documentation

### REST API

The REST API is available at `http://localhost:8000/api/v1/` with:
- OpenAPI documentation at `/docs`
- Interactive API explorer at `/redoc`

### WebSocket API

Real-time communication via WebSocket at `ws://localhost:8000/ws`

### GraphQL API

GraphQL endpoint at `http://localhost:8000/graphql` with GraphiQL explorer

## Security

- JWT-based authentication
- Role-based access control (RBAC)
- End-to-end encryption for sensitive data
- Input sanitization and validation
- Audit logging

## Monitoring

- Prometheus metrics at `/metrics`
- Health checks at `/health`
- Distributed tracing with OpenTelemetry
- Structured logging with correlation IDs

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: [https://github.com/Drmusab/-1995/docs](https://github.com/Drmusab/-1995/docs)
- Issues: [https://github.com/Drmusab/-1995/issues](https://github.com/Drmusab/-1995/issues)
- Discussions: [https://github.com/Drmusab/-1995/discussions](https://github.com/Drmusab/-1995/discussions)