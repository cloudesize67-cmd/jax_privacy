"""
Web Interface for AI Agent Swarm

Provides a dashboard to monitor and interact with the agent swarm.
"""
import os
import sys
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import OrchestratorAgent
from agents.github_agent import GitHubManagerAgent
from agents.data_collector_agent import DataCollectorAgent

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WebInterface")

# Initialize orchestrator (global)
orchestrator = None


def initialize_orchestrator():
    """Initialize the orchestrator and agents"""
    global orchestrator

    if orchestrator is not None:
        return orchestrator

    logger.info("Initializing agent swarm...")

    orchestrator = OrchestratorAgent()

    # Register agents
    github_agent = GitHubManagerAgent()
    orchestrator.register_agent(github_agent)

    data_collector = DataCollectorAgent()
    orchestrator.register_agent(data_collector)

    logger.info("Agent swarm initialized")
    return orchestrator


@app.route('/')
def index():
    """Main dashboard page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Agent Swarm Dashboard</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            h1 {
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }
            h2 {
                color: #764ba2;
                margin-top: 30px;
            }
            .status-card {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
            .agent-card {
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                transition: all 0.3s;
            }
            .agent-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                transform: translateY(-2px);
            }
            .status-badge {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                text-transform: uppercase;
            }
            .status-idle { background: #e3f2fd; color: #1976d2; }
            .status-working { background: #fff3e0; color: #f57c00; }
            .status-error { background: #ffebee; color: #c62828; }
            .status-completed { background: #e8f5e9; color: #388e3c; }
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 5px;
                transition: background 0.3s;
            }
            button:hover {
                background: #764ba2;
            }
            .request-section {
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
            }
            textarea {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                font-family: inherit;
                font-size: 14px;
                resize: vertical;
            }
            #response {
                margin-top: 20px;
                padding: 15px;
                background: white;
                border-radius: 5px;
                border: 2px solid #e0e0e0;
                white-space: pre-wrap;
            }
            .metric {
                display: inline-block;
                margin: 10px 20px 10px 0;
                font-size: 14px;
            }
            .metric strong {
                color: #667eea;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AI Agent Swarm Dashboard</h1>
            <p>Orchestrator powered by Claude 4.5 Sonnet</p>

            <div class="status-card">
                <h3>System Status</h3>
                <div id="systemStatus">Loading...</div>
                <button onclick="refreshStatus()">🔄 Refresh Status</button>
            </div>

            <h2>📋 Registered Agents</h2>
            <div id="agentsList">Loading...</div>

            <div class="request-section">
                <h2>💬 Send Request to Orchestrator</h2>
                <textarea id="userRequest" rows="5" placeholder="Enter your request here...
Example: Create a new repository called 'test-repo' with a README file explaining it's a test."></textarea>
                <br>
                <button onclick="sendRequest()">🚀 Send Request</button>
                <button onclick="clearRequest()">🗑️ Clear</button>
                <div id="response"></div>
            </div>

            <h2>📊 System Metrics</h2>
            <div id="metrics">Loading...</div>
        </div>

        <script>
            async function refreshStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();

                    // Update system status
                    const orchestrator = data.orchestrator;
                    document.getElementById('systemStatus').innerHTML = `
                        <div class="metric"><strong>Status:</strong> <span class="status-badge status-${orchestrator.status}">${orchestrator.status}</span></div>
                        <div class="metric"><strong>GitHub User:</strong> ${data.github_user}</div>
                        <div class="metric"><strong>Task Queue:</strong> ${data.task_queue_length}</div>
                        <div class="metric"><strong>Completed:</strong> ${data.completed_tasks}</div>
                    `;

                    // Update agents list
                    let agentsHTML = '';
                    for (const [name, agent] of Object.entries(data.agents)) {
                        agentsHTML += `
                            <div class="agent-card">
                                <h3>${name}</h3>
                                <div class="metric"><strong>Type:</strong> ${agent.agent_type}</div>
                                <div class="metric"><strong>Status:</strong> <span class="status-badge status-${agent.status}">${agent.status}</span></div>
                                <div class="metric"><strong>Completed:</strong> ${agent.metrics.tasks_completed}</div>
                                <div class="metric"><strong>Failed:</strong> ${agent.metrics.tasks_failed}</div>
                                <div class="metric"><strong>Uptime:</strong> ${Math.round(agent.uptime)} seconds</div>
                            </div>
                        `;
                    }
                    document.getElementById('agentsList').innerHTML = agentsHTML;

                    // Update metrics
                    const metrics = data.orchestrator.metrics;
                    document.getElementById('metrics').innerHTML = `
                        <div class="status-card">
                            <div class="metric"><strong>Tasks Completed:</strong> ${metrics.tasks_completed}</div>
                            <div class="metric"><strong>Tasks Failed:</strong> ${metrics.tasks_failed}</div>
                            <div class="metric"><strong>Total Execution Time:</strong> ${metrics.total_execution_time.toFixed(2)}s</div>
                            <div class="metric"><strong>Orchestrator Uptime:</strong> ${Math.round(orchestrator.uptime)}s</div>
                        </div>
                    `;

                } catch (error) {
                    console.error('Error fetching status:', error);
                    document.getElementById('systemStatus').innerHTML = '<p style="color:red;">Error loading status</p>';
                }
            }

            async function sendRequest() {
                const request = document.getElementById('userRequest').value;
                if (!request.trim()) {
                    alert('Please enter a request');
                    return;
                }

                document.getElementById('response').innerHTML = '<p>⏳ Processing request...</p>';

                try {
                    const response = await fetch('/api/request', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ request: request })
                    });

                    const data = await response.json();

                    if (data.success) {
                        const result = data.result;
                        document.getElementById('response').innerHTML = `
                            <h3>✅ Response:</h3>
                            <p>${result.response || 'Task completed successfully'}</p>
                            ${result.github_results ? `<p><strong>GitHub Operations:</strong> ${result.github_results.length}</p>` : ''}
                            ${result.delegations_results ? `<p><strong>Agent Delegations:</strong> ${result.delegations_results.length}</p>` : ''}
                            <pre>${JSON.stringify(result, null, 2)}</pre>
                        `;
                    } else {
                        document.getElementById('response').innerHTML = `
                            <h3>❌ Error:</h3>
                            <p>${data.error}</p>
                        `;
                    }

                    // Refresh status after request
                    refreshStatus();

                } catch (error) {
                    console.error('Error sending request:', error);
                    document.getElementById('response').innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
                }
            }

            function clearRequest() {
                document.getElementById('userRequest').value = '';
                document.getElementById('response').innerHTML = '';
            }

            // Initial load
            refreshStatus();

            // Auto-refresh every 10 seconds
            setInterval(refreshStatus, 10000);
        </script>
    </body>
    </html>
    """


@app.route('/api/status')
def get_status():
    """Get current system status"""
    try:
        orch = initialize_orchestrator()
        status = orch.get_system_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/request', methods=['POST'])
def process_request():
    """Process a user request"""
    try:
        data = request.json
        user_request = data.get('request')

        if not user_request:
            return jsonify({"error": "No request provided"}), 400

        orch = initialize_orchestrator()
        result = orch.process_user_request(user_request)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/agents')
def get_agents():
    """Get list of all agents"""
    try:
        orch = initialize_orchestrator()
        agents = {
            name: agent.get_status()
            for name, agent in orch.agents.items()
        }
        return jsonify(agents)
    except Exception as e:
        logger.error(f"Error getting agents: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    })


def main():
    """Start the web server"""
    host = os.getenv('WEB_HOST', '0.0.0.0')
    port = int(os.getenv('WEB_PORT', 5000))
    debug = os.getenv('WEB_DEBUG', 'false').lower() == 'true'

    logger.info(f"Starting web interface on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
