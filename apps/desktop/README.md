# Desktop Application

This is the desktop workspace for the AI Assistant productivity suite.

## Overview

The desktop application provides a native desktop interface for the AI Assistant, built with Node.js and React.

## Dependencies

- **@types/react** (18.3.27) - TypeScript definitions for React
- **redux** (5.0.1) - State management library
- **yaml** (2.8.2) - YAML parser and serializer

## Development

### Install Dependencies

```bash
# Clean install (recommended for consistency)
npm ci

# Or regular install
npm install
```

### Run the Application

```bash
npm start
```

### Build

```bash
npm run build
```

## Docker

### Production Build

The production Docker build uses `npm ci` for reproducible builds:

```bash
# From repository root
docker build -f docker/Dockerfile.desktop.prod -t desktop-app:prod .
```

### Run in Docker

```bash
docker run -d --name desktop-prod -p 3000:3000 desktop-app:prod
```

## Important Notes

### package-lock.json

- **Always commit** package-lock.json to version control
- The lockfile ensures consistent dependency versions across environments
- Docker production build uses `npm ci`, which requires package.json and package-lock.json to be in sync

### Adding Dependencies

```bash
# Add a new dependency (updates both package.json and package-lock.json)
npm install <package-name>

# Add a dev dependency
npm install --save-dev <package-name>
```

### Updating Dependencies

```bash
# Update a specific package
npm update <package-name>

# Update all packages (respecting semver ranges in package.json)
npm update
```

### Troubleshooting

If you encounter `npm ci` errors:

1. Delete node_modules and package-lock.json:
   ```bash
   rm -rf node_modules package-lock.json
   ```

2. Regenerate the lockfile:
   ```bash
   npm install
   ```

3. Verify npm ci works:
   ```bash
   npm ci
   ```

## Architecture

```
apps/desktop/
├── index.js           # Application entry point
├── package.json       # Dependencies and scripts
├── package-lock.json  # Lockfile for reproducible builds
└── README.md         # This file
```

## Testing

Run the verification script to ensure npm ci works correctly:

```bash
../../scripts/test-npm-ci-fix.sh
```

This script verifies:
- package.json has all required dependencies
- package-lock.json exists and is in sync
- npm ci completes successfully
- All dependencies are installed with correct versions

## Related Documentation

- [Docker Build Fix Documentation](../../docs/DOCKER_BUILD_FIX.md)
- [Main Repository README](../../README.md)
