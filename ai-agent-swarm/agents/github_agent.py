"""
GitHub Manager Agent - Specialized agent for GitHub operations
"""
from typing import Any, Dict
from core.base_agent import BaseAgent, Task
from github_integration.github_manager import GitHubManager


class GitHubManagerAgent(BaseAgent):
    """
    Specialized agent for managing GitHub repositories and files

    Handles:
    - Repository creation and management
    - File operations (CRUD)
    - Branch management
    - Directory structure creation
    """

    def __init__(self, github_token: str = None):
        super().__init__(
            agent_name="GitHubManagerAgent",
            agent_type="GitHubManager"
        )

        self.github = GitHubManager(access_token=github_token)
        self.logger.info("GitHubManagerAgent initialized")

    def execute(self, task: Task) -> Any:
        """
        Execute GitHub-related tasks

        Supported task types:
        - create_repository
        - create_file
        - update_file
        - get_file
        - delete_file
        - create_branch
        - list_repositories
        - create_structure

        Args:
            task: Task with GitHub operation details

        Returns:
            Operation result
        """
        task_type = task.task_type
        params = task.payload

        try:
            if task_type == "create_repository":
                return self._create_repository(params)

            elif task_type == "create_file":
                return self._create_file(params)

            elif task_type == "update_file":
                return self._update_file(params)

            elif task_type == "get_file":
                return self._get_file(params)

            elif task_type == "delete_file":
                return self._delete_file(params)

            elif task_type == "create_branch":
                return self._create_branch(params)

            elif task_type == "list_repositories":
                return self._list_repositories()

            elif task_type == "create_structure":
                return self._create_structure(params)

            elif task_type == "get_repo_stats":
                return self._get_repo_stats(params)

            else:
                raise ValueError(f"Unknown task type: {task_type}")

        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            raise

    def _create_repository(self, params: Dict) -> Dict[str, Any]:
        """Create a new repository"""
        repo = self.github.create_repository(
            name=params['name'],
            description=params.get('description', ''),
            private=params.get('private', False),
            auto_init=params.get('auto_init', True)
        )
        return {
            "name": repo.name,
            "full_name": repo.full_name,
            "url": repo.html_url,
            "created": True
        }

    def _create_file(self, params: Dict) -> Dict[str, Any]:
        """Create a new file in repository"""
        result = self.github.create_file(
            repo_name=params['repo_name'],
            file_path=params['file_path'],
            content=params['content'],
            commit_message=params.get('commit_message', f"Create {params['file_path']}"),
            branch=params.get('branch', 'main')
        )
        return {
            "file_path": params['file_path'],
            "created": True,
            "sha": result.sha
        }

    def _update_file(self, params: Dict) -> Dict[str, Any]:
        """Update existing file"""
        result = self.github.update_file(
            repo_name=params['repo_name'],
            file_path=params['file_path'],
            content=params['content'],
            commit_message=params.get('commit_message', f"Update {params['file_path']}"),
            branch=params.get('branch', 'main')
        )
        return {
            "file_path": params['file_path'],
            "updated": True,
            "sha": result.sha
        }

    def _get_file(self, params: Dict) -> Dict[str, Any]:
        """Get file content"""
        content = self.github.get_file_content(
            repo_name=params['repo_name'],
            file_path=params['file_path'],
            branch=params.get('branch', 'main')
        )
        return {
            "file_path": params['file_path'],
            "content": content
        }

    def _delete_file(self, params: Dict) -> Dict[str, Any]:
        """Delete file"""
        self.github.delete_file(
            repo_name=params['repo_name'],
            file_path=params['file_path'],
            commit_message=params.get('commit_message', f"Delete {params['file_path']}"),
            branch=params.get('branch', 'main')
        )
        return {
            "file_path": params['file_path'],
            "deleted": True
        }

    def _create_branch(self, params: Dict) -> Dict[str, Any]:
        """Create new branch"""
        result = self.github.create_branch(
            repo_name=params['repo_name'],
            branch_name=params['branch_name'],
            source_branch=params.get('source_branch', 'main')
        )
        return result

    def _list_repositories(self) -> Dict[str, Any]:
        """List all repositories"""
        repos = self.github.list_repositories()
        return {
            "count": len(repos),
            "repositories": repos
        }

    def _create_structure(self, params: Dict) -> Dict[str, Any]:
        """Create directory structure with multiple files"""
        created = self.github.create_directory_structure(
            repo_name=params['repo_name'],
            structure=params['structure'],
            branch=params.get('branch', 'main')
        )
        return {
            "files_created": len(created),
            "files": created
        }

    def _get_repo_stats(self, params: Dict) -> Dict[str, Any]:
        """Get repository statistics"""
        return self.github.get_repository_stats(params['repo_name'])
