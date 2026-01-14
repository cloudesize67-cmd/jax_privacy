# AI Agent Swarm - Usage Examples

This document contains practical examples of using the AI Agent Swarm system.

## Quick Start Examples

### 1. Basic Repository Creation

```python
from core.orchestrator import OrchestratorAgent

# Initialize orchestrator
orchestrator = OrchestratorAgent()

# Create a simple repository
request = "Create a new repository called 'hello-world' with a README"
result = orchestrator.process_user_request(request)

print(f"Success: {result['success']}")
print(f"Response: {result['result']['response']}")
```

### 2. Create Project Structure

```python
request = """
Create a Python project repository called 'my-python-app' with:
- README.md describing a Python application
- requirements.txt with common dependencies
- src/main.py with a hello world function
- tests/test_main.py with a basic test
- .gitignore for Python projects
"""

result = orchestrator.process_user_request(request)
```

### 3. Data Collection and Storage

```python
# Collect data from an API and store in GitHub
request = """
Use the DataCollector agent to fetch data from
https://api.github.com/repos/anthropics/anthropic-sdk-python
and store the results in a repository called 'api-data'
in a file named 'anthropic-sdk-info.json'
"""

result = orchestrator.process_user_request(request)
```

### 4. Multi-Agent Workflow

```python
request = """
1. Create a repository called 'research-data'
2. Collect information about AI agents from public APIs
3. Save the collected data in organized JSON files
4. Create a summary README with statistics
"""

result = orchestrator.process_user_request(request)
```

## Agent-Specific Examples

### Using GitHubManagerAgent Directly

```python
from agents.github_agent import GitHubManagerAgent
from core.base_agent import Task

# Initialize agent
github_agent = GitHubManagerAgent()

# Create repository
task = Task(
    task_id="repo_create_1",
    task_type="create_repository",
    payload={
        "name": "my-new-repo",
        "description": "Created by AI Agent",
        "private": False,
        "auto_init": True
    }
)

github_agent.accept_task(task)
result = github_agent.execute_task()
print(result)

# Create a file
task = Task(
    task_id="file_create_1",
    task_type="create_file",
    payload={
        "repo_name": "my-new-repo",
        "file_path": "hello.txt",
        "content": "Hello from AI Agent!",
        "commit_message": "Add greeting file"
    }
)

github_agent.accept_task(task)
result = github_agent.execute_task()
```

### Using DataCollectorAgent

```python
from agents.data_collector_agent import DataCollectorAgent
from core.base_agent import Task

# Initialize agent
collector = DataCollectorAgent(rate_limit_delay=2.0)

# Fetch URL
task = Task(
    task_id="fetch_1",
    task_type="fetch_url",
    payload={
        "url": "https://api.github.com/users/anthropics",
        "method": "GET"
    }
)

collector.accept_task(task)
result = collector.execute_task()
print(result['result']['content'])
```

## Web Interface Examples

### Starting the Web Interface

```bash
# Terminal 1: Start web server
python web/app.py

# Open browser to http://localhost:5000
```

### Interacting via Web API

```python
import requests

# Get system status
response = requests.get('http://localhost:5000/api/status')
status = response.json()
print(f"Orchestrator Status: {status['orchestrator']['status']}")

# Submit request
response = requests.post(
    'http://localhost:5000/api/request',
    json={
        "request": "Create a repository called 'web-api-test'"
    }
)
result = response.json()
print(result)
```

## Interactive Mode Examples

```bash
# Start interactive mode
python main.py interactive

# Then type commands:
>>> Create a repository called 'test-repo-1'
# ... system processes request ...

>>> status
# ... shows system status ...

>>> agents
# ... lists all agents ...

>>> save
# ... saves state to GitHub ...

>>> quit
```

## Advanced Examples

### Custom Agent Creation

```python
from core.base_agent import BaseAgent, Task

class AnalyzerAgent(BaseAgent):
    """Custom agent for analyzing data"""

    def __init__(self):
        super().__init__("AnalyzerAgent", "Analyzer")

    def execute(self, task: Task):
        data = task.payload.get('data', [])

        # Perform analysis
        analysis = {
            "count": len(data),
            "summary": f"Analyzed {len(data)} items"
        }

        return analysis

# Register with orchestrator
orchestrator = OrchestratorAgent()
analyzer = AnalyzerAgent()
orchestrator.register_agent(analyzer)

# Now the orchestrator can delegate to your custom agent
```

### Scheduled Tasks

```python
import schedule
import time

def scheduled_data_collection():
    """Collect data every hour"""
    orchestrator = OrchestratorAgent()
    request = "Collect latest data and save to repository 'scheduled-data'"
    orchestrator.process_user_request(request)

# Schedule the task
schedule.every(1).hours.do(scheduled_data_collection)

# Run forever
while True:
    schedule.run_pending()
    time.sleep(60)
```

### Batch Processing

```python
# Process multiple requests
requests = [
    "Create repository 'project-1'",
    "Create repository 'project-2'",
    "Create repository 'project-3'",
]

orchestrator = OrchestratorAgent()

for req in requests:
    print(f"Processing: {req}")
    result = orchestrator.process_user_request(req)
    print(f"Result: {result['success']}\n")
```

### Error Handling

```python
orchestrator = OrchestratorAgent()

try:
    result = orchestrator.process_user_request(
        "Create a repository with invalid!@#$%^name"
    )

    if not result['success']:
        print(f"Error: {result['error']}")
        # Handle error appropriately

except Exception as e:
    print(f"Exception occurred: {str(e)}")
    # Log error, retry, or alert user
```

### Monitoring Agent Performance

```python
orchestrator = OrchestratorAgent()

# Get detailed status
status = orchestrator.get_system_status()

# Check each agent
for agent_name, agent_status in status['agents'].items():
    print(f"\nAgent: {agent_name}")
    print(f"  Status: {agent_status['status']}")
    print(f"  Completed: {agent_status['metrics']['tasks_completed']}")
    print(f"  Failed: {agent_status['metrics']['tasks_failed']}")
    print(f"  Uptime: {agent_status['uptime']:.2f}s")

    # Check if agent needs attention
    if agent_status['metrics']['tasks_failed'] > 5:
        print(f"  ⚠️  WARNING: High failure rate!")
```

### Saving and Loading State

```python
# Save current state to GitHub
orchestrator.save_state_to_github(repo_name='agent-state-backup')

# State includes:
# - All agent statuses
# - Task history
# - Metrics
# - Timestamps

# You can retrieve this later to analyze system behavior
```

## Real-World Use Cases

### Use Case 1: Automated Documentation

```python
request = """
Create a documentation repository called 'project-docs' with:
1. README.md with project overview
2. docs/installation.md with setup instructions
3. docs/usage.md with usage examples
4. docs/api-reference.md with API documentation
5. docs/troubleshooting.md with common issues
"""

orchestrator.process_user_request(request)
```

### Use Case 2: Data Pipeline

```python
request = """
1. Collect data from multiple API endpoints
2. Validate and clean the data
3. Store in a structured format in repository 'data-pipeline'
4. Generate a daily report summarizing the collected data
"""

orchestrator.process_user_request(request)
```

### Use Case 3: Project Template Generator

```python
request = """
Create a full-stack web application template repository called 'webapp-template':
- Frontend: React app structure
- Backend: Python Flask API structure
- Database: SQL schema files
- Docker: Dockerfile and docker-compose.yml
- CI/CD: GitHub Actions workflow
- Documentation: Complete README with setup instructions
"""

orchestrator.process_user_request(request)
```

### Use Case 4: Automated Backup System

```python
import schedule

def backup_important_data():
    orchestrator = OrchestratorAgent()
    request = """
    Create a backup of important project files:
    1. Collect current system configuration
    2. Gather recent logs
    3. Export database schemas
    4. Save everything to 'daily-backup' repository with timestamp
    """
    orchestrator.process_user_request(request)

schedule.every().day.at("02:00").do(backup_important_data)
```

## Tips and Best Practices

1. **Start Simple**: Begin with basic requests and gradually increase complexity

2. **Be Specific**: Clear, detailed requests get better results
   - Good: "Create a Python Flask API with user authentication"
   - Poor: "Make an API"

3. **Check Status**: Regularly monitor agent status to catch issues early

4. **Save State**: Periodically save system state for analysis and recovery

5. **Handle Errors**: Always check `result['success']` before proceeding

6. **Rate Limiting**: Respect API rate limits, especially for GitHub

7. **Security**: Never hardcode API keys, always use environment variables

8. **Testing**: Test new workflows in demo mode before production use

9. **Logging**: Enable appropriate logging levels for debugging

10. **Documentation**: Document your custom agents and workflows

## Debugging Examples

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check agent history
agent = orchestrator.agents['GitHubManagerAgent']
history = agent.get_task_history(limit=10)
for task in history:
    print(f"Task: {task['task_type']}, Status: {task['status']}")

# Inspect failed tasks
for task in history:
    if task['status'] == 'error':
        print(f"Failed task: {task['task_id']}")
        print(f"Error: {task['error']}")
```

## More Examples Coming Soon!

Check the GitHub repository for additional examples and community contributions.
