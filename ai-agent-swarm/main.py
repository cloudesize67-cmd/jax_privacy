#!/usr/bin/env python3
"""
AI Agent Swarm - Main Entry Point

This is the main application that initializes and coordinates the agent swarm.
"""
import os
import sys
import logging
from dotenv import load_dotenv

from core.orchestrator import OrchestratorAgent
from agents.github_agent import GitHubManagerAgent
from agents.data_collector_agent import DataCollectorAgent


def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('agent_swarm.log')
        ]
    )


def initialize_system():
    """
    Initialize the AI agent swarm system

    Returns:
        OrchestratorAgent instance
    """
    logger = logging.getLogger("MainSystem")

    # Load environment variables
    load_dotenv()

    # Verify required environment variables
    required_vars = ['ANTHROPIC_API_KEY', 'GITHUB_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please create a .env file with the required variables")
        sys.exit(1)

    logger.info("Starting AI Agent Swarm System")
    logger.info("=" * 60)

    # Initialize orchestrator
    logger.info("Initializing Orchestrator Agent (Claude 4.5 Sonnet)...")
    orchestrator = OrchestratorAgent()

    # Initialize specialized agents
    logger.info("Initializing specialized agents...")

    github_agent = GitHubManagerAgent()
    orchestrator.register_agent(github_agent)
    logger.info("✓ GitHub Manager Agent registered")

    data_collector = DataCollectorAgent(rate_limit_delay=1.0)
    orchestrator.register_agent(data_collector)
    logger.info("✓ Data Collector Agent registered")

    logger.info("=" * 60)
    logger.info("System initialization complete")
    logger.info(f"Orchestrator ID: {orchestrator.agent_id}")
    logger.info(f"Registered agents: {len(orchestrator.agents)}")

    # Get system status
    status = orchestrator.get_system_status()
    logger.info(f"GitHub user: {status['github_user']}")

    return orchestrator


def demo_workflow(orchestrator: OrchestratorAgent):
    """
    Demonstrate the agent swarm capabilities

    Args:
        orchestrator: OrchestratorAgent instance
    """
    logger = logging.getLogger("Demo")

    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATION: Creating a repository and storing data")
    logger.info("=" * 60)

    # Example: Create a repository via orchestrator
    request = """
    Create a new GitHub repository called 'agent-swarm-demo' with the following structure:

    1. A README.md file explaining this is a demo
    2. A data/ directory with a sample.json file
    3. A logs/ directory with an initial.log file

    The repository should be public and contain information about the AI agent system.
    """

    logger.info("\nSending request to Orchestrator...")
    logger.info(f"Request: {request[:100]}...")

    result = orchestrator.process_user_request(request)

    logger.info("\n" + "-" * 60)
    logger.info("RESULT:")
    logger.info(f"Success: {result.get('success', False)}")

    if result.get('success'):
        task_result = result.get('result', {})
        logger.info(f"\nAnalysis: {task_result.get('analysis', 'N/A')[:200]}")
        logger.info(f"\nPlan: {task_result.get('plan', [])}")
        logger.info(f"\nResponse: {task_result.get('response', 'N/A')}")

        # Show GitHub operations performed
        github_ops = task_result.get('github_results', [])
        if github_ops:
            logger.info(f"\nGitHub operations completed: {len(github_ops)}")
            for op in github_ops:
                logger.info(f"  - {op.get('operation', {}).get('operation', 'N/A')}: "
                           f"{'✓' if op.get('success') else '✗'}")
    else:
        logger.error(f"Task failed: {result.get('error', 'Unknown error')}")

    logger.info("=" * 60)


def interactive_mode(orchestrator: OrchestratorAgent):
    """
    Run in interactive mode, accepting user commands

    Args:
        orchestrator: OrchestratorAgent instance
    """
    logger = logging.getLogger("Interactive")

    print("\n" + "=" * 60)
    print("AI AGENT SWARM - INTERACTIVE MODE")
    print("=" * 60)
    print("\nCommands:")
    print("  - Type your request and press Enter")
    print("  - 'status' - Show system status")
    print("  - 'agents' - List all agents")
    print("  - 'save' - Save system state to GitHub")
    print("  - 'quit' - Exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input(">>> ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                logger.info("Shutting down...")
                break

            elif user_input.lower() == 'status':
                status = orchestrator.get_system_status()
                print("\n--- SYSTEM STATUS ---")
                print(f"Orchestrator: {status['orchestrator']['status']}")
                print(f"GitHub User: {status['github_user']}")
                print(f"Registered Agents: {len(status['agents'])}")
                print(f"Task Queue: {status['task_queue_length']}")
                print(f"Completed Tasks: {status['completed_tasks']}")

                for agent_name, agent_status in status['agents'].items():
                    print(f"\n  {agent_name}:")
                    print(f"    Status: {agent_status['status']}")
                    print(f"    Tasks Completed: {agent_status['metrics']['tasks_completed']}")
                    print(f"    Tasks Failed: {agent_status['metrics']['tasks_failed']}")

            elif user_input.lower() == 'agents':
                print("\n--- REGISTERED AGENTS ---")
                for name, agent in orchestrator.agents.items():
                    print(f"  - {name} ({agent.agent_type}): {agent.status.value}")

            elif user_input.lower() == 'save':
                print("Saving system state to GitHub...")
                try:
                    orchestrator.save_state_to_github()
                    print("✓ State saved successfully")
                except Exception as e:
                    print(f"✗ Failed to save state: {str(e)}")

            else:
                # Process user request
                print("\nProcessing request...")
                result = orchestrator.process_user_request(user_input)

                if result.get('success'):
                    task_result = result.get('result', {})
                    print("\n--- RESPONSE ---")
                    print(task_result.get('response', 'Task completed'))

                    # Show execution details
                    if task_result.get('github_results'):
                        print(f"\nGitHub operations: {len(task_result['github_results'])}")

                    if task_result.get('delegations_results'):
                        print(f"Agent delegations: {len(task_result['delegations_results'])}")

                else:
                    print(f"\n✗ Error: {result.get('error', 'Unknown error')}")

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}")
            print(f"✗ Error: {str(e)}")


def main():
    """Main entry point"""
    setup_logging()

    # Initialize system
    orchestrator = initialize_system()

    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'demo':
            demo_workflow(orchestrator)
        elif command == 'interactive':
            interactive_mode(orchestrator)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python main.py [demo|interactive]")
    else:
        # Default to interactive mode
        interactive_mode(orchestrator)


if __name__ == "__main__":
    main()
