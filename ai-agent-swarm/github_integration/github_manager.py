"""
GitHub Manager - Handles all GitHub API interactions
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from github import Github, GithubException, Repository
from github.ContentFile import ContentFile
import base64


class GitHubManager:
    """
    Manages all interactions with GitHub API

    Provides methods for repository management, file operations,
    and branch management.
    """

    def __init__(self, access_token: Optional[str] = None):
        """
        Initialize GitHub Manager

        Args:
            access_token: GitHub Personal Access Token
                         If not provided, looks for GITHUB_TOKEN in environment
        """
        self.access_token = access_token or os.getenv("GITHUB_TOKEN")

        if not self.access_token:
            raise ValueError(
                "GitHub token not provided. Set GITHUB_TOKEN environment variable "
                "or pass access_token to constructor."
            )

        self.github = Github(self.access_token)
        self.user = self.github.get_user()

        self.logger = logging.getLogger("GitHubManager")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - GitHubManager - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"GitHubManager initialized for user: {self.user.login}")

    def create_repository(self, name: str, description: str = "",
                          private: bool = False, auto_init: bool = True) -> Repository.Repository:
        """
        Create a new GitHub repository

        Args:
            name: Repository name
            description: Repository description
            private: Whether the repository should be private
            auto_init: Whether to initialize with README

        Returns:
            Repository object
        """
        try:
            repo = self.user.create_repo(
                name=name,
                description=description,
                private=private,
                auto_init=auto_init
            )
            self.logger.info(f"Created repository: {repo.full_name}")
            return repo

        except GithubException as e:
            if e.status == 422:  # Repository already exists
                self.logger.warning(f"Repository {name} already exists, fetching it")
                return self.get_repository(name)
            else:
                self.logger.error(f"Failed to create repository: {str(e)}")
                raise

    def get_repository(self, name: str) -> Repository.Repository:
        """
        Get an existing repository

        Args:
            name: Repository name

        Returns:
            Repository object
        """
        try:
            repo = self.user.get_repo(name)
            self.logger.info(f"Retrieved repository: {repo.full_name}")
            return repo
        except GithubException as e:
            self.logger.error(f"Failed to get repository {name}: {str(e)}")
            raise

    def list_repositories(self) -> List[Dict[str, Any]]:
        """
        List all repositories for the authenticated user

        Returns:
            List of repository information dictionaries
        """
        try:
            repos = self.user.get_repos()
            repo_list = []

            for repo in repos:
                repo_list.append({
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "private": repo.private,
                    "url": repo.html_url,
                    "created_at": repo.created_at.isoformat(),
                    "updated_at": repo.updated_at.isoformat(),
                    "size": repo.size,
                    "stars": repo.stargazers_count
                })

            self.logger.info(f"Retrieved {len(repo_list)} repositories")
            return repo_list

        except GithubException as e:
            self.logger.error(f"Failed to list repositories: {str(e)}")
            raise

    def create_file(self, repo_name: str, file_path: str, content: str,
                    commit_message: str, branch: str = "main") -> ContentFile:
        """
        Create a new file in a repository

        Args:
            repo_name: Repository name
            file_path: Path where file should be created
            content: File content
            commit_message: Commit message
            branch: Branch name (default: main)

        Returns:
            ContentFile object
        """
        try:
            repo = self.get_repository(repo_name)
            result = repo.create_file(
                path=file_path,
                message=commit_message,
                content=content,
                branch=branch
            )
            self.logger.info(f"Created file {file_path} in {repo_name}")
            return result['content']

        except GithubException as e:
            self.logger.error(f"Failed to create file {file_path}: {str(e)}")
            raise

    def update_file(self, repo_name: str, file_path: str, content: str,
                    commit_message: str, branch: str = "main") -> ContentFile:
        """
        Update an existing file in a repository

        Args:
            repo_name: Repository name
            file_path: Path to file to update
            content: New file content
            commit_message: Commit message
            branch: Branch name (default: main)

        Returns:
            ContentFile object
        """
        try:
            repo = self.get_repository(repo_name)

            # Get current file to obtain its SHA
            file_content = repo.get_contents(file_path, ref=branch)

            result = repo.update_file(
                path=file_path,
                message=commit_message,
                content=content,
                sha=file_content.sha,
                branch=branch
            )
            self.logger.info(f"Updated file {file_path} in {repo_name}")
            return result['content']

        except GithubException as e:
            self.logger.error(f"Failed to update file {file_path}: {str(e)}")
            raise

    def get_file_content(self, repo_name: str, file_path: str,
                         branch: str = "main") -> str:
        """
        Get content of a file from repository

        Args:
            repo_name: Repository name
            file_path: Path to file
            branch: Branch name (default: main)

        Returns:
            File content as string
        """
        try:
            repo = self.get_repository(repo_name)
            file_content = repo.get_contents(file_path, ref=branch)

            if isinstance(file_content, list):
                raise ValueError(f"{file_path} is a directory, not a file")

            content = base64.b64decode(file_content.content).decode('utf-8')
            self.logger.info(f"Retrieved content of {file_path} from {repo_name}")
            return content

        except GithubException as e:
            self.logger.error(f"Failed to get file content {file_path}: {str(e)}")
            raise

    def delete_file(self, repo_name: str, file_path: str,
                    commit_message: str, branch: str = "main") -> Dict[str, Any]:
        """
        Delete a file from repository

        Args:
            repo_name: Repository name
            file_path: Path to file to delete
            commit_message: Commit message
            branch: Branch name (default: main)

        Returns:
            Commit information dictionary
        """
        try:
            repo = self.get_repository(repo_name)

            # Get file to obtain its SHA
            file_content = repo.get_contents(file_path, ref=branch)

            result = repo.delete_file(
                path=file_path,
                message=commit_message,
                sha=file_content.sha,
                branch=branch
            )
            self.logger.info(f"Deleted file {file_path} from {repo_name}")
            return result

        except GithubException as e:
            self.logger.error(f"Failed to delete file {file_path}: {str(e)}")
            raise

    def create_branch(self, repo_name: str, branch_name: str,
                      source_branch: str = "main") -> Dict[str, Any]:
        """
        Create a new branch in repository

        Args:
            repo_name: Repository name
            branch_name: Name for new branch
            source_branch: Branch to branch from (default: main)

        Returns:
            Branch information dictionary
        """
        try:
            repo = self.get_repository(repo_name)

            # Get SHA of source branch
            source = repo.get_branch(source_branch)
            sha = source.commit.sha

            # Create new branch
            ref = repo.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=sha
            )

            self.logger.info(f"Created branch {branch_name} in {repo_name}")
            return {
                "branch": branch_name,
                "sha": sha,
                "ref": ref.ref
            }

        except GithubException as e:
            self.logger.error(f"Failed to create branch {branch_name}: {str(e)}")
            raise

    def list_branches(self, repo_name: str) -> List[str]:
        """
        List all branches in repository

        Args:
            repo_name: Repository name

        Returns:
            List of branch names
        """
        try:
            repo = self.get_repository(repo_name)
            branches = [branch.name for branch in repo.get_branches()]
            self.logger.info(f"Retrieved {len(branches)} branches from {repo_name}")
            return branches

        except GithubException as e:
            self.logger.error(f"Failed to list branches: {str(e)}")
            raise

    def create_directory_structure(self, repo_name: str, structure: Dict[str, str],
                                    branch: str = "main") -> List[str]:
        """
        Create multiple files/directories in repository

        Args:
            repo_name: Repository name
            structure: Dictionary mapping file paths to content
            branch: Branch name (default: main)

        Returns:
            List of created file paths
        """
        created_files = []

        for file_path, content in structure.items():
            try:
                self.create_file(
                    repo_name=repo_name,
                    file_path=file_path,
                    content=content,
                    commit_message=f"Create {file_path}",
                    branch=branch
                )
                created_files.append(file_path)

            except GithubException as e:
                # If file exists, try updating it
                if e.status == 422:
                    self.update_file(
                        repo_name=repo_name,
                        file_path=file_path,
                        content=content,
                        commit_message=f"Update {file_path}",
                        branch=branch
                    )
                    created_files.append(file_path)
                else:
                    self.logger.error(f"Failed to create {file_path}: {str(e)}")

        self.logger.info(f"Created/updated {len(created_files)} files in {repo_name}")
        return created_files

    def get_repository_stats(self, repo_name: str) -> Dict[str, Any]:
        """
        Get statistics for a repository

        Args:
            repo_name: Repository name

        Returns:
            Dictionary of repository statistics
        """
        try:
            repo = self.get_repository(repo_name)

            return {
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description,
                "url": repo.html_url,
                "size": repo.size,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "watchers": repo.watchers_count,
                "open_issues": repo.open_issues_count,
                "created_at": repo.created_at.isoformat(),
                "updated_at": repo.updated_at.isoformat(),
                "pushed_at": repo.pushed_at.isoformat(),
                "default_branch": repo.default_branch,
                "language": repo.language,
                "private": repo.private
            }

        except GithubException as e:
            self.logger.error(f"Failed to get stats for {repo_name}: {str(e)}")
            raise
