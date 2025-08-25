# AI Assistant n8n Integration

This package provides n8n nodes for integrating with the AI Assistant system, enabling you to incorporate advanced AI capabilities into your n8n workflows.

## Features

The AI Assistant n8n integration includes four powerful nodes:

### 1. AI Assistant Chat Node
- **Send Message**: Send messages to the AI Assistant for conversational interactions
- **Get Chat History**: Retrieve conversation history for a session
- **Session Management**: Create, manage, and delete chat sessions

### 2. AI Assistant Memory Node
- **Store Facts**: Store important information in the AI's memory system
- **Retrieve Memory**: Search and retrieve relevant memories based on queries
- **Search Memory**: Advanced memory search with similarity scoring
- **Update/Delete Memory**: Manage existing memory entries

### 3. AI Assistant Skill Node
- **Execute Skills**: Run specific AI skills and capabilities
- **Note Taking**: Create, update, search, and manage notes
- **List Skills**: Discover available AI skills
- **Skill Information**: Get detailed information about specific skills

### 4. AI Assistant Workflow Node
- **Process Messages**: Advanced message processing with configurable options
- **Execute Tasks**: Run complex multi-step tasks
- **Health Monitoring**: Check system health and status
- **System Status**: Get detailed system information

## Installation

### Prerequisites

1. **AI Assistant Server**: You need a running AI Assistant server
2. **n8n Instance**: A working n8n installation (self-hosted or cloud)

### Step 1: Install the AI Assistant System

First, set up the AI Assistant system:

```bash
# Clone the AI Assistant repository
git clone https://github.com/Drmusab/-1995.git
cd -1995

# Install dependencies
pip install -e .

# Start the AI Assistant server
ai-assistant-server
```

The server will be available at `http://localhost:8000` by default.

### Step 2: Install n8n Nodes

#### Option A: Install from npm (when published)
```bash
npm install n8n-nodes-ai-assistant
```

#### Option B: Install from source
```bash
# Navigate to the n8n-ai-assistant directory
cd n8n-ai-assistant

# Install dependencies
npm install

# Build the package
npm run build

# Link for development
npm link

# In your n8n installation directory
npm link n8n-nodes-ai-assistant
```

### Step 3: Configure n8n

Add the package to your n8n configuration:

**For Docker installations**, add to your environment variables:
```bash
N8N_CUSTOM_EXTENSIONS="/path/to/n8n-nodes-ai-assistant"
```

**For npm installations**, restart n8n after installing the package.

## Configuration

### Setting up Credentials

1. In n8n, go to **Settings** > **Credentials**
2. Click **Create New** and search for "AI Assistant API"
3. Configure the connection:
   - **Base URL**: Your AI Assistant server URL (e.g., `http://localhost:8000`)
   - **API Key**: Your API key (if authentication is enabled)
   - **Username/Password**: Alternative authentication method

### Authentication Options

The AI Assistant supports multiple authentication methods:

1. **API Key Authentication** (Recommended)
   - Set the `API Key` field in credentials
   - Leave username/password empty

2. **Basic Authentication**
   - Set `Username` and `Password` fields
   - Leave API Key empty

3. **No Authentication** (Development only)
   - Leave all authentication fields empty

## Usage Examples

### Example 1: Simple Chat Workflow

Create a workflow that:
1. Receives a webhook with a message
2. Sends it to the AI Assistant
3. Returns the response

**Workflow Steps:**
1. **Webhook Node** - Trigger
2. **AI Assistant Chat Node** - Send Message
3. **Respond to Webhook Node** - Return response

**Configuration:**
- Chat Node: Set `Message` to `{{$json.message}}`
- Use the same `Session ID` for conversation context

### Example 2: Memory-Enhanced Assistant

Create a workflow that stores and retrieves information:

**Workflow Steps:**
1. **Webhook Node** - Receive query
2. **AI Assistant Memory Node** - Search relevant memories
3. **AI Assistant Chat Node** - Send message with memory context
4. **AI Assistant Memory Node** - Store important facts from conversation

### Example 3: Note-Taking Assistant

Automate note-taking from conversations:

**Workflow Steps:**
1. **AI Assistant Chat Node** - Process user message
2. **AI Assistant Skill Node** - Extract key information using note-taking skill
3. **AI Assistant Skill Node** - Create note with extracted information

### Example 4: Task Automation

Execute complex tasks using the AI Assistant:

**Workflow Steps:**
1. **Webhook Node** - Receive task description
2. **AI Assistant Workflow Node** - Execute task
3. **Multiple Action Nodes** - Perform task-specific actions based on AI response

## Node Reference

### AI Assistant Chat Node

#### Operations

**Send Message**
- `Message` (required): The message to send
- `Session ID` (optional): Session for context
- `User ID` (optional): User identifier
- `Context` (optional): Additional context as JSON

**Get Chat History**
- `Session ID` (required): Session to retrieve history for

#### Session Operations

**Create Session**
- `User ID` (optional): User for the session
- `Session Type`: Type of session (interactive, batch, api)

### AI Assistant Memory Node

#### Operations

**Store Fact**
- `Fact` (required): Information to store
- `Session ID` (required): Associated session
- `Importance` (0-1): Importance score
- `Tags`: Comma-separated tags

**Retrieve Memory**
- `Query` (required): Search query
- `Max Results`: Maximum number of results
- `Similarity Threshold`: Minimum similarity score

### AI Assistant Skill Node

#### Operations

**Execute Skill**
- `Skill Name` (required): Name of skill to execute
- `Skill Parameters`: Parameters as JSON
- `Session ID` (optional): Session context

**Note Taking Operations**
- Create, update, get, list, delete, search, export notes
- Various parameters depending on operation

### AI Assistant Workflow Node

#### Operations

**Process Message**
- `Message` (required): Message to process
- `Processing Options`: Memory, learning, tokens, temperature settings

**Execute Task**
- `Task Description` (required): Task to execute
- `Task Parameters`: Task-specific parameters
- `Priority`: Task priority level
- `Timeout`: Maximum execution time

## Error Handling

All nodes support n8n's error handling mechanisms:

1. **Continue on Fail**: Enable to continue workflow execution on errors
2. **Error Outputs**: Errors are returned as JSON with error details
3. **Retry Logic**: Configure retry attempts for transient failures

## Performance Considerations

### Timeouts
- Default timeout: 120 seconds
- Long-running tasks: Use appropriate timeout values
- Complex workflows: Consider breaking into smaller steps

### Rate Limiting
- The AI Assistant may have rate limits
- Implement delays between requests if needed
- Monitor response times and adjust accordingly

### Memory Usage
- Memory operations can be resource-intensive
- Use appropriate similarity thresholds
- Limit result counts for large datasets

## Troubleshooting

### Common Issues

**Connection Errors**
- Verify AI Assistant server is running
- Check base URL in credentials
- Ensure network connectivity

**Authentication Errors**
- Verify API key or username/password
- Check authentication method configuration
- Ensure user has required permissions

**Timeout Errors**
- Increase timeout values for complex operations
- Check AI Assistant server performance
- Consider breaking large requests into smaller parts

**Memory Errors**
- Verify session IDs are correct
- Check memory storage configuration
- Ensure sufficient storage space

### Debug Mode

Enable debug logging in n8n to troubleshoot issues:

1. Set `N8N_LOG_LEVEL=debug` in environment
2. Check n8n logs for detailed error information
3. Monitor AI Assistant server logs

## API Compatibility

This n8n integration is compatible with AI Assistant API version 1.0 and above.

Supported endpoints:
- `/api/v1/chat` - Chat operations
- `/api/v1/sessions` - Session management
- `/api/v1/memory` - Memory operations
- `/api/v1/skills` - Skill execution
- `/api/v1/notes` - Note management
- `/api/v1/process` - Advanced processing
- `/api/v1/health` - Health checks

## Contributing

We welcome contributions to improve the n8n integration:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Drmusab/-1995.git
cd -1995/n8n-ai-assistant

# Install dependencies
npm install

# Start development mode
npm run dev

# Build for production
npm run build

# Run linting
npm run lint

# Format code
npm run format
```

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Support

- **Documentation**: [GitHub Repository](https://github.com/Drmusab/-1995)
- **Issues**: [GitHub Issues](https://github.com/Drmusab/-1995/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Drmusab/-1995/discussions)

## Changelog

### Version 1.0.0
- Initial release
- AI Assistant Chat Node
- AI Assistant Memory Node
- AI Assistant Skill Node
- AI Assistant Workflow Node
- Complete API integration
- Comprehensive documentation