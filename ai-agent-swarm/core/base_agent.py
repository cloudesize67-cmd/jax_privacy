"""
Base Agent Class - Foundation for all specialized agents
"""
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class AgentStatus(Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


class Task:
    """Represents a task for an agent to execute"""

    def __init__(self, task_id: str, task_type: str, payload: Dict[str, Any],
                 priority: int = 5):
        self.task_id = task_id
        self.task_type = task_type
        self.payload = payload
        self.priority = priority
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.status = AgentStatus.IDLE
        self.result: Optional[Any] = None
        self.error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "result": self.result,
            "error": self.error
        }


class BaseAgent(ABC):
    """
    Base class for all agents in the swarm

    All specialized agents inherit from this class and implement
    the execute() method with their specific logic.
    """

    def __init__(self, agent_name: str, agent_type: str):
        self.agent_id = str(uuid4())
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.status = AgentStatus.IDLE
        self.current_task: Optional[Task] = None
        self.task_history: List[Task] = []
        self.created_at = datetime.utcnow()
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0
        }

        # Setup logging
        self.logger = logging.getLogger(f"Agent.{self.agent_name}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.agent_name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def accept_task(self, task: Task) -> bool:
        """
        Accept a task if agent is available

        Args:
            task: Task object to execute

        Returns:
            True if task was accepted, False otherwise
        """
        if self.status != AgentStatus.IDLE:
            self.logger.warning(f"Cannot accept task {task.task_id}, agent is {self.status.value}")
            return False

        self.current_task = task
        self.status = AgentStatus.WORKING
        task.status = AgentStatus.WORKING
        task.started_at = datetime.utcnow()
        self.logger.info(f"Accepted task {task.task_id} of type {task.task_type}")
        return True

    def execute_task(self) -> Dict[str, Any]:
        """
        Execute the current task

        Returns:
            Dictionary containing task result and metadata
        """
        if not self.current_task:
            return {"error": "No task assigned"}

        start_time = time.time()

        try:
            self.logger.info(f"Executing task {self.current_task.task_id}")

            # Call the agent-specific execution logic
            result = self.execute(self.current_task)

            # Update task status
            self.current_task.result = result
            self.current_task.status = AgentStatus.COMPLETED
            self.current_task.completed_at = datetime.utcnow()

            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["tasks_completed"] += 1
            self.metrics["total_execution_time"] += execution_time

            self.logger.info(
                f"Task {self.current_task.task_id} completed in {execution_time:.2f}s"
            )

            return {
                "success": True,
                "task_id": self.current_task.task_id,
                "result": result,
                "execution_time": execution_time
            }

        except Exception as e:
            self.logger.error(f"Task {self.current_task.task_id} failed: {str(e)}")

            self.current_task.error = str(e)
            self.current_task.status = AgentStatus.ERROR
            self.current_task.completed_at = datetime.utcnow()
            self.metrics["tasks_failed"] += 1

            return {
                "success": False,
                "task_id": self.current_task.task_id,
                "error": str(e),
                "execution_time": time.time() - start_time
            }

        finally:
            # Archive the task and reset status
            if self.current_task:
                self.task_history.append(self.current_task)
                self.current_task = None
            self.status = AgentStatus.IDLE

    @abstractmethod
    def execute(self, task: Task) -> Any:
        """
        Agent-specific execution logic

        This method must be implemented by all subclasses.

        Args:
            task: The task to execute

        Returns:
            The result of the task execution
        """
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "current_task": self.current_task.to_dict() if self.current_task else None,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "uptime": (datetime.utcnow() - self.created_at).total_seconds()
        }

    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task history"""
        return [task.to_dict() for task in self.task_history[-limit:]]

    def reset_metrics(self):
        """Reset agent metrics"""
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0
        }
        self.logger.info("Metrics reset")

    def __repr__(self) -> str:
        return f"<{self.agent_type}(name={self.agent_name}, status={self.status.value})>"
