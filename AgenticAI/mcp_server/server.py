from mcp.server import MCPServer


# ==========================================================
# MOCK ENTERPRISE DATA
# ==========================================================

INCIDENTS = {
    "INC-2045": {
        "incident_id": "INC-2045",
        "application": "Checkout-Service",
        "server": "APP-PROD-12",
        "severity": "P1",
        "description": "Checkout response time increased from 1.2 sec to 8.7 sec"
    }
}


SERVER_METRICS = {
    "APP-PROD-12": {
        "cpu": 96,
        "memory": 71,
        "db_connections": 100,
        "error_rate": 18,
        "status": "DEGRADED"
    }
}


HISTORICAL_INCIDENTS = [
    {
        "incident_id": "INC-1781",
        "symptom": "High latency and DB connections at 100%",
        "root_cause": "Database connection pool exhaustion",
        "resolution": "Restart application connection pool and increase pool size"
    },

    {
        "incident_id": "INC-1654",
        "symptom": "High CPU and slow checkout",
        "root_cause": "Expensive SQL query",
        "resolution": "Optimize SQL query and add missing database index"
    }
]


RUNBOOKS = {
    "connection_pool": """
Connection Pool Runbook

1. Check active DB connections.
2. Verify connection pool utilization.
3. Check database availability.
4. Review application logs.
5. Restart application connection pool if approved.
6. Monitor latency after remediation.
""",

    "high_cpu": """
High CPU Runbook

1. Identify high CPU process.
2. Check recent deployments.
3. Inspect expensive database queries.
4. Check application thread utilization.
5. Monitor CPU trend.
6. Escalate if CPU remains above threshold.
"""
}


# ==========================================================
# ORDINARY PYTHON BUSINESS FUNCTIONS
# ==========================================================

def get_incident(incident_id: str) -> dict:

    return INCIDENTS.get(
        incident_id,
        {
            "error": "Incident not found",
            "incident_id": incident_id
        }
    )


def get_server_metrics(server: str) -> dict:

    return SERVER_METRICS.get(
        server,
        {
            "error": "Server not found",
            "server": server
        }
    )


def search_previous_incidents(keyword: str) -> list:

    keyword = keyword.lower()

    results = []

    for incident in HISTORICAL_INCIDENTS:

        if keyword in str(incident).lower():

            results.append(incident)

    return results


def search_runbook(topic: str) -> str:

    topic = topic.lower()

    if "connection" in topic or "database" in topic:
        return RUNBOOKS["connection_pool"]

    if "cpu" in topic:
        return RUNBOOKS["high_cpu"]

    return "No matching runbook found."


# ==========================================================
# CREATE MCP SERVER
# ==========================================================

mcp = MCPServer(
    "Incident Investigation MCP Server"
)


# ==========================================================
# MCP TOOLS
# ==========================================================

@mcp.tool()
def mcp_get_incident(
    incident_id: str
) -> dict:

    """Retrieve an IT incident."""

    return get_incident(
        incident_id
    )


@mcp.tool()
def mcp_get_server_metrics(
    server: str
) -> dict:

    """Retrieve server operational metrics."""

    return get_server_metrics(
        server
    )


@mcp.tool()
def mcp_search_incidents(
    keyword: str
) -> list:

    """Search historical incidents."""

    return search_previous_incidents(
        keyword
    )


@mcp.tool()
def mcp_get_runbook(
    topic: str
) -> str:

    """Retrieve an operational runbook."""

    return search_runbook(
        topic
    )


# ==========================================================
# RUN MCP SERVER
# ==========================================================

if __name__ == "__main__":

    print(
        "Starting Incident Investigation MCP Server..."
    )

    print(
        "MCP endpoint:"
    )

    print(
        "http://localhost:8000/mcp"
    )

    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000
    )