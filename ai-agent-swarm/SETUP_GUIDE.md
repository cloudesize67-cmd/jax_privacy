# AI Agent Swarm - Complete Setup Guide

This guide will walk you through setting up and using the AI Agent Swarm system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required

- Python 3.8 or higher
- GitHub account
- Anthropic API account (for Claude 4.5 Sonnet)

### Getting API Keys

#### 1. Anthropic API Key

1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (you'll need it for configuration)

#### 2. GitHub Personal Access Token

1. Go to [https://github.com/settings/tokens](https://github.com/settings/tokens)
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a descriptive name (e.g., "AI Agent Swarm")
4. Select the following scopes:
   - `repo` (full control of private repositories)
   - `workflow` (update GitHub Action workflows)
   - `admin:org` (if working with organization repos)
5. Click "Generate token"
6. **IMPORTANT**: Copy the token immediately (you won't be able to see it again)

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/jax_privacy/ai-agent-swarm
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### Step 1: Create Environment File

```bash
cp .env.example .env
```

### Step 2: Edit .env File

Open `.env` in your text editor and add your API keys:

```bash
# Required: Your Anthropic API key
ANTHROPIC_API_KEY=sk-ant-api03-...your-key-here...

# Required: Your GitHub Personal Access Token
GITHUB_TOKEN=ghp_...your-token-here...

# Optional: Rate limiting (seconds between requests)
RATE_LIMIT_DELAY=1.0

# Optional: Web interface settings
WEB_HOST=0.0.0.0
WEB_PORT=5000
WEB_DEBUG=false

# Optional: Logging level
LOG_LEVEL=INFO
```

**SECURITY WARNING**: Never commit your `.env` file to version control!

## Running the System

### Option 1: Interactive Mode (Recommended for First Use)

This mode allows you to interact with the orchestrator via command line:

```bash
python main.py interactive
```

You'll see a prompt where you can:
- Type requests for the orchestrator
- Use commands like `status`, `agents`, `save`, `quit`

### Option 2: Demo Mode

Run a demonstration workflow:

```bash
python main.py demo
```

This will demonstrate:
- Creating a GitHub repository
- Adding files to the repository
- Coordinating multiple agents

### Option 3: Web Interface

Start the web dashboard:

```bash
python web/app.py
```

Then open your browser to:
```
http://localhost:5000
```

The web interface provides:
- Real-time system status
- Agent monitoring
- Interactive request submission
- System metrics visualization

## Usage Examples

### Example 1: Create a GitHub Repository

**Interactive Mode:**
```
>>> Create a new repository called 'my-test-repo' with a README
```

**Python API:**
```python
from core.orchestrator import OrchestratorAgent

orchestrator = OrchestratorAgent()
result = orchestrator.process_user_request(
    "Create a new repository called 'my-test-repo' with a README file"
)
print(result)
```

### Example 2: Create Multi-File Project Structure

```python
request = """
Create a repository called 'my-web-app' with the following structure:
- README.md with project description
- src/index.html for the main page
- src/style.css for styles
- src/script.js for JavaScript
- .gitignore for Python projects
"""

orchestrator.process_user_request(request)
```

### Example 3: Data Collection and Storage

```python
request = """
Collect data from https://api.github.com/repos/python/cpython
and save it to a file called 'cpython-info.json' in repository 'api-data-collection'
"""

orchestrator.process_user_request(request)
```

### Example 4: Using Specialized Agents Directly

```python
from agents.github_agent import GitHubManagerAgent
from core.base_agent import Task

# Initialize agent
github_agent = GitHubManagerAgent()

# Create a task
task = Task(
    task_id="test_task_1",
    task_type="create_repository",
    payload={
        "name": "test-repo",
        "description": "A test repository",
        "private": False
    }
)

# Execute task
if github_agent.accept_task(task):
    result = github_agent.execute_task()
    print(result)
```

### Example 5: Check System Status

**Interactive Mode:**
```
>>> status
```

**Python API:**
```python
status = orchestrator.get_system_status()
print(f"Orchestrator Status: {status['orchestrator']['status']}")
print(f"Registered Agents: {len(status['agents'])}")
print(f"GitHub User: {status['github_user']}")
```

### Example 6: Save System State to GitHub

**Interactive Mode:**
```
>>> save
```

**Python API:**
```python
orchestrator.save_state_to_github(repo_name='agent-swarm-state')
```

This creates a timestamped JSON file with complete system state.

## Command Reference

### Interactive Mode Commands

| Command | Description |
|---------|-------------|
| `status` | Show complete system status |
| `agents` | List all registered agents |
| `save` | Save system state to GitHub |
| `quit` | Exit the program |
| `[any text]` | Send request to orchestrator |

### Web Interface Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/api/status` | GET | Get system status JSON |
| `/api/request` | POST | Submit user request |
| `/api/agents` | GET | Get all agents info |
| `/health` | GET | Health check |

## Architecture Overview

### Agent Hierarchy

```
OrchestratorAgent (Claude 4.5 Sonnet)
├── GitHubManagerAgent
│   ├── Create repositories
│   ├── Manage files
│   └── Handle branches
├── DataCollectorAgent
│   ├── Fetch URLs
│   ├── Collect API data
│   └── Validate data
└── [Add more agents as needed]
```

### Data Flow

1. **User Request** → Orchestrator Agent
2. **Orchestrator** uses Claude 4.5 to analyze and plan
3. **Orchestrator** delegates to specialized agents
4. **Agents** execute tasks and report back
5. **Results** saved to GitHub (optional)
6. **Response** returned to user

## Troubleshooting

### Issue: "ANTHROPIC_API_KEY not provided"

**Solution**: Make sure you've:
1. Created the `.env` file
2. Added your Anthropic API key
3. Saved the file

### Issue: "GitHub token not provided"

**Solution**: Add your GitHub Personal Access Token to `.env`:
```
GITHUB_TOKEN=ghp_your_token_here
```

### Issue: "Repository already exists"

**Solution**: The system will attempt to use the existing repository. If you want to create a new one, use a different name or delete the old repository first.

### Issue: "Rate limit exceeded"

**Solution**:
- Increase `RATE_LIMIT_DELAY` in `.env`
- Wait before making more requests
- Check GitHub API rate limits

### Issue: "Permission denied" for GitHub operations

**Solution**: Verify your GitHub token has the correct scopes:
- `repo` for repository access
- `workflow` for Actions
- `admin:org` for organization repos (if applicable)

### Issue: Web interface won't start

**Solution**:
```bash
# Check if port is already in use
lsof -i :5000

# Use a different port
WEB_PORT=8080 python web/app.py
```

### Issue: Module import errors

**Solution**:
```bash
# Make sure you're in the ai-agent-swarm directory
cd ai-agent-swarm

# Reinstall dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

## Advanced Configuration

### Adding Custom Agents

1. Create new agent file in `agents/` directory
2. Inherit from `BaseAgent`
3. Implement the `execute()` method
4. Register with orchestrator

Example:
```python
from core.base_agent import BaseAgent, Task

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("MyCustomAgent", "CustomType")

    def execute(self, task: Task):
        # Your custom logic here
        return {"result": "success"}

# Register with orchestrator
orchestrator.register_agent(MyCustomAgent())
```

### Customizing Claude's Behavior

Edit the `system_prompt` in `core/orchestrator.py`:

```python
self.system_prompt = """Your custom system prompt here..."""
```

### Persistent State Management

The system can save its state to GitHub:

```python
# Automatic state saving every hour
import schedule
schedule.every(1).hours.do(orchestrator.save_state_to_github)
```

## Security Best Practices

1. **Never commit `.env` file** - It's in `.gitignore` by default
2. **Use environment variables** for all secrets
3. **Rotate API keys regularly**
4. **Use private repositories** for sensitive data
5. **Enable GitHub 2FA** on your account
6. **Review token permissions** periodically
7. **Monitor API usage** to detect anomalies

## Next Steps

- Read the main [README.md](README.md) for architecture details
- Explore example code in `examples/` directory (if available)
- Check agent documentation for specific capabilities
- Join discussions or contribute to the project

## Support

For issues, questions, or contributions:
- Check existing issues
- Create a new issue with details
- Include error logs and system info

---

**Happy Agent Swarming!** 🤖🐝
