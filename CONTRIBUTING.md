# Contributing to AI Assistant

We welcome contributions to the AI Assistant project! This document provides guidelines for contributing to ensure a smooth and productive experience for everyone.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Docker (optional, for containerized development)

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/-1995.git
   cd -1995
   ```

3. Install development dependencies:
   ```bash
   make install-dev
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Style

- Follow PEP 8 conventions
- Use Black for code formatting
- Use isort for import sorting
- Maximum line length: 100 characters
- Run linting before submitting: `make lint`
- Format code: `make format`

### Testing

- Write tests for new features and bug fixes
- Maintain or improve test coverage
- Run tests: `make test`
- Run specific test types:
  - `make test-unit` for unit tests
  - `make test-integration` for integration tests
  - `make test-e2e` for end-to-end tests

### Commit Messages

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for test additions/changes
- `chore:` for maintenance tasks

Example: `feat: add speech processing pipeline`

### Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Write/update tests
4. Run the full test suite
5. Update documentation if needed
6. Commit your changes with descriptive messages
7. Push to your fork
8. Create a Pull Request

### Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include screenshots for UI changes
- Ensure all tests pass
- Maintain code coverage
- Address review feedback promptly

## Architecture Guidelines

### Directory Structure

Follow the established directory structure:
- `src/` for source code
- `tests/` for all test files
- `docs/` for documentation
- `tools/` for development tools
- `infrastructure/` for deployment configs

### Code Organization

- Keep modules focused and cohesive
- Use dependency injection where appropriate
- Follow SOLID principles
- Write clear docstrings
- Use type hints consistently

### Performance Considerations

- Profile code for performance bottlenecks
- Use async/await for I/O operations
- Implement caching where beneficial
- Monitor memory usage

## Security

- Never commit secrets or API keys
- Use environment variables for configuration
- Follow security best practices
- Report security vulnerabilities privately

## Documentation

- Update README.md for significant changes
- Document new APIs and modules
- Include code examples
- Update CHANGELOG.md

## Community

### Code of Conduct

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

### Getting Help

- Check existing issues and documentation
- Ask questions in GitHub Discussions
- Join our community channels

### Recognition

Contributors will be recognized in:
- CHANGELOG.md
- GitHub contributors list
- Project documentation

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. Automated deployment to PyPI

Thank you for contributing to AI Assistant!