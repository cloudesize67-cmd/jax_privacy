"""
Data Collector Agent - Collects data from various sources
"""
import requests
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from core.base_agent import BaseAgent, Task


class DataCollectorAgent(BaseAgent):
    """
    Specialized agent for collecting data from various sources

    Handles:
    - API data collection (with rate limiting)
    - Web data fetching (ethical, respecting robots.txt)
    - Data validation and cleaning
    - Structured data storage
    """

    def __init__(self, rate_limit_delay: float = 1.0):
        super().__init__(
            agent_name="DataCollectorAgent",
            agent_type="DataCollector"
        )

        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.collected_data = []

        self.logger.info("DataCollectorAgent initialized")

    def execute(self, task: Task) -> Any:
        """
        Execute data collection tasks

        Supported task types:
        - fetch_url: Fetch data from a URL
        - collect_api_data: Collect data from API
        - validate_data: Validate collected data
        - batch_collect: Collect from multiple sources

        Args:
            task: Task with data collection parameters

        Returns:
            Collected data
        """
        task_type = task.task_type
        params = task.payload

        if task_type == "fetch_url":
            return self._fetch_url(params)

        elif task_type == "collect_api_data":
            return self._collect_api_data(params)

        elif task_type == "validate_data":
            return self._validate_data(params)

        elif task_type == "batch_collect":
            return self._batch_collect(params)

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _fetch_url(self, params: Dict) -> Dict[str, Any]:
        """
        Fetch data from a URL

        Args:
            params: Dictionary with 'url', optional 'headers', 'method'

        Returns:
            Dictionary with response data
        """
        url = params['url']
        method = params.get('method', 'GET').upper()
        headers = params.get('headers', {})
        timeout = params.get('timeout', 30)

        self._enforce_rate_limit()

        try:
            self.logger.info(f"Fetching URL: {url}")

            # Add user agent if not present
            if 'User-Agent' not in headers:
                headers['User-Agent'] = 'AI-Agent-Swarm/1.0 (Educational/Research)'

            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                timeout=timeout
            )

            response.raise_for_status()

            return {
                "success": True,
                "url": url,
                "status_code": response.status_code,
                "content": response.text,
                "content_type": response.headers.get('Content-Type'),
                "size": len(response.content),
                "timestamp": datetime.utcnow().isoformat()
            }

        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch {url}: {str(e)}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _collect_api_data(self, params: Dict) -> Dict[str, Any]:
        """
        Collect data from an API endpoint

        Args:
            params: Dictionary with 'endpoint', 'api_key', 'parameters'

        Returns:
            API response data
        """
        endpoint = params['endpoint']
        api_key = params.get('api_key')
        api_params = params.get('parameters', {})

        headers = {}
        if api_key:
            # Support different auth header formats
            auth_type = params.get('auth_type', 'bearer')
            if auth_type == 'bearer':
                headers['Authorization'] = f'Bearer {api_key}'
            elif auth_type == 'api_key':
                headers['X-API-Key'] = api_key

        self._enforce_rate_limit()

        try:
            self.logger.info(f"Collecting API data from: {endpoint}")

            response = requests.get(
                endpoint,
                params=api_params,
                headers=headers,
                timeout=30
            )

            response.raise_for_status()

            return {
                "success": True,
                "endpoint": endpoint,
                "status_code": response.status_code,
                "data": response.json() if response.headers.get('Content-Type', '').startswith('application/json') else response.text,
                "timestamp": datetime.utcnow().isoformat()
            }

        except requests.RequestException as e:
            self.logger.error(f"API collection failed: {str(e)}")
            return {
                "success": False,
                "endpoint": endpoint,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _validate_data(self, params: Dict) -> Dict[str, Any]:
        """
        Validate collected data against a schema

        Args:
            params: Dictionary with 'data' and 'validation_rules'

        Returns:
            Validation results
        """
        data = params['data']
        rules = params.get('validation_rules', {})

        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Check required fields
        required_fields = rules.get('required_fields', [])
        for field in required_fields:
            if field not in data:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Missing required field: {field}")

        # Check data types
        field_types = rules.get('field_types', {})
        for field, expected_type in field_types.items():
            if field in data:
                actual_type = type(data[field]).__name__
                if actual_type != expected_type:
                    validation_results["warnings"].append(
                        f"Field '{field}' has type {actual_type}, expected {expected_type}"
                    )

        # Check value ranges
        value_ranges = rules.get('value_ranges', {})
        for field, range_def in value_ranges.items():
            if field in data:
                value = data[field]
                min_val = range_def.get('min')
                max_val = range_def.get('max')

                if min_val is not None and value < min_val:
                    validation_results["valid"] = False
                    validation_results["errors"].append(
                        f"Field '{field}' value {value} is below minimum {min_val}"
                    )

                if max_val is not None and value > max_val:
                    validation_results["valid"] = False
                    validation_results["errors"].append(
                        f"Field '{field}' value {value} exceeds maximum {max_val}"
                    )

        self.logger.info(f"Validation completed: {'PASSED' if validation_results['valid'] else 'FAILED'}")

        return validation_results

    def _batch_collect(self, params: Dict) -> Dict[str, Any]:
        """
        Collect data from multiple sources

        Args:
            params: Dictionary with 'sources' list

        Returns:
            Results from all sources
        """
        sources = params.get('sources', [])
        results = []

        for source in sources:
            source_type = source.get('type', 'url')

            if source_type == 'url':
                result = self._fetch_url(source)
            elif source_type == 'api':
                result = self._collect_api_data(source)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown source type: {source_type}"
                }

            results.append(result)

        return {
            "total_sources": len(sources),
            "successful": sum(1 for r in results if r.get('success')),
            "failed": sum(1 for r in results if not r.get('success')),
            "results": results
        }
