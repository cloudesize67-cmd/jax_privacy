"""
Orchestrator Agent - Main controller powered by Claude 4.5 Sonnet

This agent coordinates all other agents and makes high-level decisions.
It acts as the "brain" of the agent swarm.
"""
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from anthropic import Anthropic

from core.base_agent import BaseAgent, Task, AgentStatus
from github_integration.github_manager import GitHubManager


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent - Powered by Claude 4.5 Sonnet

    This agent:
    - Receives high-level user requests
    - Breaks them down into subtasks
    - Delegates to specialized agents
    - Manages GitHub repositories
    - Coordinates multi-agent workflows
    - Makes strategic decisions
    """

    def __init__(self, anthropic_api_key: Optional[str] = None,
                 github_token: Optional[str] = None):
        """
        Initialize the Orchestrator Agent

        Args:
            anthropic_api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            github_token: GitHub token (defaults to GITHUB_TOKEN env var)
        """
        super().__init__(
            agent_name="OrchestratorAgent",
            agent_type="Orchestrator"
        )

        # Initialize Claude 4.5 Sonnet
        self.api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not provided")

        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-5-20250929"

        # Initialize GitHub Manager
        self.github = GitHubManager(access_token=github_token)

        # Registry of specialized agents
        self.agents: Dict[str, BaseAgent] = {}

        # Task queue
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []

        # System prompt for Claude
        self.system_prompt = """You are the Orchestrator Agent, the main controller of an AI agent swarm system.

Your capabilities:
1. Manage GitHub repositories and files
2. Coordinate specialized agents (GitHubManager, DataCollector, Analyst, etc.)
3. Break down complex tasks into subtasks
4. Make strategic decisions about workflow
5. Persist all data to GitHub repositories

When given a task:
1. Analyze what needs to be done
2. Determine which agents/resources are needed
3. Create a step-by-step plan
4. Delegate subtasks appropriately
5. Ensure results are saved to GitHub

Always respond in JSON format with this structure:
{
    "analysis": "Your analysis of the task",
    "plan": ["step 1", "step 2", ...],
    "delegations": [
        {"agent": "agent_name", "task": "task_description", "priority": 1-10}
    ],
    "github_operations": [
        {"operation": "create_repo|create_file|update_file", "params": {...}}
    ],
    "response": "Summary for the user"
}"""

        self.logger.info("OrchestratorAgent initialized with Claude 4.5 Sonnet")

    def register_agent(self, agent: BaseAgent):
        """
        Register a specialized agent with the orchestrator

        Args:
            agent: Agent instance to register
        """
        self.agents[agent.agent_name] = agent
        self.logger.info(f"Registered agent: {agent.agent_name}")

    def execute(self, task: Task) -> Any:
        """
        Execute a task using Claude 4.5 Sonnet for decision making

        Args:
            task: Task to execute

        Returns:
            Task execution result
        """
        self.logger.info(f"Orchestrator processing task: {task.task_type}")

        # Use Claude to analyze and plan
        decision = self._get_claude_decision(task)

        # Execute the plan
        result = self._execute_plan(decision, task)

        return result

    def _get_claude_decision(self, task: Task) -> Dict[str, Any]:
        """
        Use Claude 4.5 Sonnet to analyze task and create execution plan

        Args:
            task: Task to analyze

        Returns:
            Claude's decision and plan as dictionary
        """
        # Build context about available agents
        agent_context = "\n".join([
            f"- {name}: {agent.agent_type} (status: {agent.status.value})"
            for name, agent in self.agents.items()
        ])

        # Build prompt
        user_prompt = f"""Task Type: {task.task_type}
Task Details: {json.dumps(task.payload, indent=2)}

Available Agents:
{agent_context}

GitHub Account: {self.github.user.login}

Please analyze this task and provide a detailed execution plan."""

        try:
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Parse response
            response_text = message.content[0].text

            # Try to extract JSON from response
            try:
                # Handle markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.index("```json") + 7
                    json_end = response_text.index("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.index("```") + 3
                    json_end = response_text.index("```", json_start)
                    response_text = response_text[json_start:json_end].strip()

                decision = json.loads(response_text)

            except json.JSONDecodeError:
                # If JSON parsing fails, create a simple plan
                self.logger.warning("Failed to parse JSON response, using fallback")
                decision = {
                    "analysis": response_text[:500],
                    "plan": ["Execute task directly"],
                    "delegations": [],
                    "github_operations": [],
                    "response": "Task will be executed directly"
                }

            self.logger.info(f"Claude decision: {decision.get('response', 'N/A')}")
            return decision

        except Exception as e:
            self.logger.error(f"Error getting Claude decision: {str(e)}")
            return {
                "analysis": f"Error: {str(e)}",
                "plan": [],
                "delegations": [],
                "github_operations": [],
                "response": f"Failed to process task: {str(e)}"
            }

    def _execute_plan(self, decision: Dict[str, Any], original_task: Task) -> Dict[str, Any]:
        """
        Execute the plan created by Claude

        Args:
            decision: Decision dictionary from Claude
            original_task: Original task object

        Returns:
            Execution results
        """
        results = {
            "analysis": decision.get("analysis", ""),
            "plan": decision.get("plan", []),
            "delegations_results": [],
            "github_results": [],
            "response": decision.get("response", "")
        }

        # Execute GitHub operations
        for gh_op in decision.get("github_operations", []):
            try:
                gh_result = self._execute_github_operation(gh_op)
                results["github_results"].append({
                    "operation": gh_op,
                    "success": True,
                    "result": gh_result
                })
            except Exception as e:
                self.logger.error(f"GitHub operation failed: {str(e)}")
                results["github_results"].append({
                    "operation": gh_op,
                    "success": False,
                    "error": str(e)
                })

        # Delegate to other agents
        for delegation in decision.get("delegations", []):
            agent_name = delegation.get("agent")
            if agent_name in self.agents:
                try:
                    agent = self.agents[agent_name]

                    # Create subtask
                    subtask = Task(
                        task_id=f"{original_task.task_id}_sub_{len(results['delegations_results'])}",
                        task_type=delegation.get("task_type", "generic"),
                        payload=delegation.get("params", {}),
                        priority=delegation.get("priority", 5)
                    )

                    # Execute subtask
                    if agent.accept_task(subtask):
                        subtask_result = agent.execute_task()
                        results["delegations_results"].append({
                            "agent": agent_name,
                            "success": subtask_result.get("success", False),
                            "result": subtask_result
                        })

                except Exception as e:
                    self.logger.error(f"Delegation to {agent_name} failed: {str(e)}")
                    results["delegations_results"].append({
                        "agent": agent_name,
                        "success": False,
                        "error": str(e)
                    })

        return results

    def _execute_github_operation(self, operation: Dict[str, Any]) -> Any:
        """
        Execute a GitHub operation

        Args:
            operation: Operation dictionary with 'operation' and 'params'

        Returns:
            Operation result
        """
        op_type = operation.get("operation")
        params = operation.get("params", {})

        if op_type == "create_repo":
            return self.github.create_repository(**params)

        elif op_type == "create_file":
            return self.github.create_file(**params)

        elif op_type == "update_file":
            return self.github.update_file(**params)

        elif op_type == "get_file":
            return self.github.get_file_content(**params)

        elif op_type == "create_directory_structure":
            return self.github.create_directory_structure(**params)

        elif op_type == "create_branch":
            return self.github.create_branch(**params)

        elif op_type == "list_repos":
            return self.github.list_repositories()

        else:
            raise ValueError(f"Unknown GitHub operation: {op_type}")

    def process_user_request(self, request: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a high-level user request

        Args:
            request: User's request as string
            context: Optional context dictionary

        Returns:
            Processing results
        """
        # Create task from request
        task = Task(
            task_id=f"user_req_{datetime.utcnow().timestamp()}",
            task_type="user_request",
            payload={
                "request": request,
                "context": context or {}
            }
        )

        # Execute task
        if self.accept_task(task):
            result = self.execute_task()
            return result
        else:
            return {
                "success": False,
                "error": f"Orchestrator busy (status: {self.status.value})"
            }

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status

        Returns:
            System status dictionary
        """
        return {
            "orchestrator": self.get_status(),
            "agents": {
                name: agent.get_status()
                for name, agent in self.agents.items()
            },
            "github_user": self.github.user.login,
            "task_queue_length": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks)
        }

    def save_state_to_github(self, repo_name: str = "agent-swarm-state"):
        """
        Save the current system state to GitHub

        Args:
            repo_name: Repository name for state storage
        """
        try:
            # Create state document
            state = {
                "timestamp": datetime.utcnow().isoformat(),
                "orchestrator_status": self.get_status(),
                "agents_status": {
                    name: agent.get_status()
                    for name, agent in self.agents.items()
                },
                "task_queue": [task.to_dict() for task in self.task_queue],
                "completed_tasks": [task.to_dict() for task in self.completed_tasks[-50:]]
            }

            # Save to GitHub
            content = json.dumps(state, indent=2)
            filename = f"state_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

            self.github.create_file(
                repo_name=repo_name,
                file_path=f"states/{filename}",
                content=content,
                commit_message=f"Save system state at {datetime.utcnow().isoformat()}"
            )

            self.logger.info(f"System state saved to {repo_name}")

        except Exception as e:
            self.logger.error(f"Failed to save state to GitHub: {str(e)}")
            raise
