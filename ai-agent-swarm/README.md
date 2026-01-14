# AI Agent Swarm Orchestration System

A sophisticated multi-agent AI system where Claude 4.5 Sonnet acts as the orchestrator, managing GitHub repositories and coordinating specialized agents to accomplish complex tasks.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│     Orchestrator Agent (Claude 4.5)     │
│         (Main Controller)                │
└────────────┬────────────────────────────┘
             │
             ├──► GitHubManagerAgent
             ├──► DataCollectorAgent
             ├──► AnalysisAgent
             ├──► WebInterfaceAgent
             ├──► PersistenceAgent
             └──► ResearchAgent
```

## Features

- **Orchestrator Agent**: Claude 4.5 Sonnet controls all operations
- **GitHub Integration**: Automatic repository creation and file management
- **Persistent Storage**: All data saved to GitHub repositories
- **Specialized Agents**: Each agent has a specific role
- **Web Interface**: Dashboard for monitoring and interaction
- **Extensible**: Easy to add new agent types

## Agent Roles

### 1. OrchestratorAgent (Claude 4.5 Sonnet)
- Main controller and decision maker
- Delegates tasks to specialized agents
- Coordinates multi-agent workflows
- Manages GitHub account operations

### 2. GitHubManagerAgent
- Creates and manages repositories
- Handles file operations (create, update, delete)
- Manages branches and commits
- Syncs local data with GitHub

### 3. DataCollectorAgent
- Collects data from approved APIs
- Implements rate limiting and ethical scraping
- Validates and cleans data
- Stores results in structured format

### 4. AnalysisAgent
- Processes collected data
- Performs statistical analysis
- Generates insights and reports
- Creates visualizations

### 5. WebInterfaceAgent
- Manages the web dashboard
- Provides real-time agent monitoring
- Handles user interactions
- Displays system status

### 6. PersistenceAgent
- Manages data persistence to GitHub
- Handles versioning and backups
- Ensures data integrity
- Manages agent state

### 7. ResearchAgent
- Performs web searches
- Gathers information from public sources
- Validates information sources
- Compiles research reports

## Installation

```bash
cd ai-agent-swarm
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`
2. Add your GitHub personal access token
3. Configure agent settings in `config/agents.yaml`

```bash
cp .env.example .env
# Edit .env with your credentials
```

## Usage

### Start the Orchestrator

```bash
python main.py
```

### Start the Web Interface

```bash
python web/app.py
```

### Create a New Repository via Agent

```python
from core.orchestrator import OrchestratorAgent

orchestrator = OrchestratorAgent()
orchestrator.create_repository("my-agent-data", description="Data managed by AI agents")
```

## GitHub Integration

The system automatically:
- Creates repositories for data storage
- Commits files with descriptive messages
- Manages branches for different agent tasks
- Maintains a changelog of all operations

## Security

- Uses GitHub Personal Access Tokens (PAT)
- Tokens stored in `.env` (never committed)
- All API calls are authenticated
- Rate limiting implemented
- No unauthorized data access

## Architecture Details

### Agent Communication
Agents communicate through a message queue system:
- **Tasks**: Orchestrator sends tasks to agents
- **Results**: Agents return results to orchestrator
- **Status**: Real-time status updates

### Data Flow
```
User Request → Orchestrator → Specialized Agents → GitHub Storage → Web Interface
```

## Adding New Agents

1. Create a new agent class in `agents/`
2. Inherit from `BaseAgent`
3. Implement required methods
4. Register in `config/agents.yaml`

Example:
```python
from core.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("MyCustomAgent")

    def execute(self, task):
        # Your agent logic here
        return result
```

## Requirements

- Python 3.8+
- GitHub account with PAT
- Internet connection for API calls

## License

Apache 2.0 License (inherits from parent repository)

## Disclaimer

This is an educational and research tool. Users are responsible for:
- Complying with GitHub's terms of service
- Respecting API rate limits
- Following data collection ethics
- Proper use of automated systems
