import os
os.environ["OPENAI_API_KEY"] = "sk-proj-tV6ERv0reypQk322xucXx4A"


# ============================================================
# COMPLETE MULTI-AGENT MCP CLIENT
# ============================================================

import anyio
import json
from typing import Any

from openai import OpenAI
from mcp import Client


# ============================================================
# CONFIGURATION
# ============================================================

MCP_SERVER_URL = "http://localhost:8000/mcp"

MODEL = "gpt-5"

INCIDENT_ID = "INC-2045"


# ============================================================
# OPENAI CLIENT
# ============================================================

llm = OpenAI()


# ============================================================
# LLM HELPER
# ============================================================

def call_llm(
    instructions: str,
    user_input: str
) -> str:
    """
    Generic LLM helper used by specialist agents.
    """

    response = llm.responses.create(
        model=MODEL,
        instructions=instructions,
        input=user_input
    )

    return response.output_text


# ============================================================
# MCP RESULT EXTRACTION
# ============================================================

def extract_result(result) -> Any:
    """
    Convert MCP CallToolResult into normal Python data.

    Handles:
    - structured_content
    - {"result": ...} wrappers
    - JSON returned as MCP text
    - plain text such as runbooks
    - MCP tool errors
    """

    # --------------------------------------------------------
    # MCP ERROR
    # --------------------------------------------------------

    if result.is_error:

        return {
            "error": "MCP tool returned an error",
            "content": str(result.content)
        }


    # --------------------------------------------------------
    # STRUCTURED CONTENT
    # --------------------------------------------------------

    if result.structured_content is not None:

        data = result.structured_content

        # Some MCP tool results are wrapped:
        #
        # {"result": {...}}
        #
        if (
            isinstance(data, dict)
            and "result" in data
        ):
            data = data["result"]

        return data


    # --------------------------------------------------------
    # TEXT CONTENT
    # --------------------------------------------------------

    text_parts = []

    for item in result.content:

        if hasattr(item, "text"):
            text_parts.append(item.text)


    text = "\n".join(text_parts).strip()


    # --------------------------------------------------------
    # TRY JSON DESERIALIZATION
    # --------------------------------------------------------

    if not text:
        return None


    try:

        data = json.loads(text)

        if (
            isinstance(data, dict)
            and "result" in data
        ):
            data = data["result"]

        return data

    except json.JSONDecodeError:

        # Example:
        # operational runbook
        return text


# ============================================================
# TYPE NORMALIZATION HELPERS
# ============================================================

def ensure_dict(
    value: Any,
    name: str
) -> dict:
    """
    Ensure MCP result is a dictionary.
    """

    if isinstance(value, dict):
        return value

    raise TypeError(
        f"{name} was expected to be dict, "
        f"but MCP returned {type(value).__name__}: "
        f"{value}"
    )


def ensure_list(
    value: Any
) -> list:
    """
    Normalize an MCP result into a list.

    Useful because one historical result may occasionally
    arrive as a dictionary.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        return [value]

    return [value]


def ensure_text(
    value: Any
) -> str:
    """
    Convert MCP output into safe text.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        indent=2,
        default=str
    )


# ============================================================
# MCP TOOL HELPER
# ============================================================

async def call_mcp_tool(
    mcp_client,
    tool_name: str,
    arguments: dict
) -> Any:
    """
    Common helper for calling MCP tools.
    """

    print(
        f"\n[MCP] Calling {tool_name}"
    )

    result = await mcp_client.call_tool(
        tool_name,
        arguments
    )

    data = extract_result(result)

    return data


# ============================================================
# MONITORING AGENT
# ============================================================

async def monitoring_agent(
    mcp_client,
    incident_id: str
) -> dict:

    print("\n" + "=" * 70)
    print("MONITORING AGENT")
    print("=" * 70)


    # --------------------------------------------------------
    # GET INCIDENT
    # --------------------------------------------------------

    print(
        "\n[Monitoring] Retrieving incident..."
    )


    incident_raw = await call_mcp_tool(
        mcp_client,
        "mcp_get_incident",
        {
            "incident_id": incident_id
        }
    )


    incident = ensure_dict(
        incident_raw,
        "Incident"
    )


    print(
        "\nIncident:"
    )

    print(
        json.dumps(
            incident,
            indent=2
        )
    )


    if "error" in incident:

        return {
            "agent": "Monitoring Agent",
            "error": incident["error"]
        }


    # --------------------------------------------------------
    # IDENTIFY SERVER
    # --------------------------------------------------------

    server = incident.get(
        "server"
    )


    if not server:

        return {
            "agent": "Monitoring Agent",
            "error": "Incident does not contain a server name."
        }


    # --------------------------------------------------------
    # GET SERVER METRICS
    # --------------------------------------------------------

    print(
        f"\n[Monitoring] Collecting metrics for {server}..."
    )


    metrics_raw = await call_mcp_tool(
        mcp_client,
        "mcp_get_server_metrics",
        {
            "server": server
        }
    )


    metrics = ensure_dict(
        metrics_raw,
        "Server metrics"
    )


    print(
        "\nMetrics:"
    )

    print(
        json.dumps(
            metrics,
            indent=2
        )
    )


    # --------------------------------------------------------
    # LLM MONITORING ANALYSIS
    # --------------------------------------------------------

    instructions = """
You are the Monitoring Agent in an enterprise
IT incident investigation system.

Your responsibility is to analyse operational
incident information and server telemetry.

Identify important operational signals.

Do NOT determine the final root cause.

Only report evidence supported by the supplied
incident and metrics.

Return:

1. Critical metrics
2. Abnormal conditions
3. Incident severity
4. Important evidence requiring investigation
"""


    prompt = f"""
INCIDENT

{json.dumps(
    incident,
    indent=2
)}


SERVER METRICS

{json.dumps(
    metrics,
    indent=2
)}
"""


    assessment = call_llm(
        instructions,
        prompt
    )


    print(
        "\nMonitoring Assessment:"
    )

    print(
        assessment
    )


    return {

        "agent":
            "Monitoring Agent",

        "incident":
            incident,

        "metrics":
            metrics,

        "assessment":
            assessment
    }


# ============================================================
# KNOWLEDGE AGENT
# ============================================================

async def knowledge_agent(
    mcp_client,
    monitoring_result: dict
) -> dict:

    print("\n" + "=" * 70)
    print("KNOWLEDGE AGENT")
    print("=" * 70)


    metrics = monitoring_result[
        "metrics"
    ]


    # --------------------------------------------------------
    # DETERMINE KNOWLEDGE SEARCH
    # --------------------------------------------------------

    if metrics.get(
        "db_connections",
        0
    ) >= 95:

        keyword = "connection"
        runbook_topic = "connection pool"


    elif metrics.get(
        "cpu",
        0
    ) >= 90:

        keyword = "cpu"
        runbook_topic = "high cpu"


    else:

        keyword = "latency"
        runbook_topic = "general"


    print(
        "\n[Knowledge] Search keyword:",
        keyword
    )


    # --------------------------------------------------------
    # SEARCH HISTORICAL INCIDENTS
    # --------------------------------------------------------

    history_raw = await call_mcp_tool(
        mcp_client,
        "mcp_search_incidents",
        {
            "keyword": keyword
        }
    )


    history = ensure_list(
        history_raw
    )


    print(
        "\nHistorical Incidents:"
    )


    print(
        json.dumps(
            history,
            indent=2,
            default=str
        )
    )


    # --------------------------------------------------------
    # GET RUNBOOK
    # --------------------------------------------------------

    runbook_raw = await call_mcp_tool(
        mcp_client,
        "mcp_get_runbook",
        {
            "topic": runbook_topic
        }
    )


    runbook = ensure_text(
        runbook_raw
    )


    print(
        "\nRetrieved Runbook:"
    )

    print(
        runbook
    )


    # --------------------------------------------------------
    # LLM KNOWLEDGE ANALYSIS
    # --------------------------------------------------------

    instructions = """
You are the Knowledge Agent in an enterprise
IT incident investigation system.

Your responsibility is to analyse:

- previous incidents
- previous root causes
- previous resolutions
- approved operational runbooks

Compare the historical knowledge with the
current monitoring evidence.

Do NOT make the final diagnosis.

Return:

1. Relevant historical incident(s)
2. Similarity to the current incident
3. Historical root causes
4. Previous remediation
5. Relevant runbook guidance
"""


    prompt = f"""
CURRENT MONITORING ASSESSMENT

{monitoring_result["assessment"]}


HISTORICAL INCIDENTS

{json.dumps(
    history,
    indent=2,
    default=str
)}


RUNBOOK

{runbook}
"""


    assessment = call_llm(
        instructions,
        prompt
    )


    print(
        "\nKnowledge Assessment:"
    )

    print(
        assessment
    )


    return {

        "agent":
            "Knowledge Agent",

        "search_keyword":
            keyword,

        "runbook_topic":
            runbook_topic,

        "historical_incidents":
            history,

        "runbook":
            runbook,

        "assessment":
            assessment
    }


# ============================================================
# DIAGNOSIS AGENT
# ============================================================

async def diagnosis_agent(
    monitoring_result: dict,
    knowledge_result: dict
) -> dict:

    print("\n" + "=" * 70)
    print("DIAGNOSIS AGENT")
    print("=" * 70)


    instructions = """
You are the Diagnosis Agent in an enterprise
IT incident investigation system.

You receive evidence from:

1. Monitoring Agent
2. Knowledge Agent

Your responsibility is to determine the most
likely explanation for the incident.

Do not invent evidence.

Determine:

1. Most likely root cause
2. Supporting evidence
3. Confidence from 0 to 1
4. Recommended next action
5. Whether the recommendation could impact production

Production-impacting actions such as restart,
shutdown, delete, terminate or configuration
change require human approval.

Return exactly in this format:

ROOT CAUSE:
...

EVIDENCE:
...

CONFIDENCE:
...

RECOMMENDATION:
...

HUMAN APPROVAL:
REQUIRED or NOT_REQUIRED
"""


    prompt = f"""
MONITORING AGENT ASSESSMENT

{monitoring_result["assessment"]}


RAW INCIDENT

{json.dumps(
    monitoring_result["incident"],
    indent=2
)}


RAW METRICS

{json.dumps(
    monitoring_result["metrics"],
    indent=2
)}


KNOWLEDGE AGENT ASSESSMENT

{knowledge_result["assessment"]}


HISTORICAL INCIDENTS

{json.dumps(
    knowledge_result["historical_incidents"],
    indent=2,
    default=str
)}


RUNBOOK

{knowledge_result["runbook"]}
"""


    diagnosis = call_llm(
        instructions,
        prompt
    )


    print(
        "\nDiagnosis:"
    )

    print(
        diagnosis
    )


    return {

        "agent":
            "Diagnosis Agent",

        "diagnosis":
            diagnosis
    }


# ============================================================
# DETERMINISTIC RISK CHECK
# ============================================================

def risk_check(
    diagnosis_result: dict
) -> dict:

    print("\n" + "=" * 70)
    print("RISK CHECK")
    print("=" * 70)


    diagnosis = diagnosis_result.get(
        "diagnosis",
        ""
    )


    diagnosis_lower = diagnosis.lower()


    # --------------------------------------------------------
    # PRODUCTION-RISK KEYWORDS
    # --------------------------------------------------------

    high_risk_terms = [

        "restart",
        "shutdown",
        "delete",
        "terminate",
        "configuration change",
        "production change"
    ]


    detected_terms = [

        term

        for term in high_risk_terms

        if term in diagnosis_lower
    ]


    # --------------------------------------------------------
    # DETERMINE APPROVAL REQUIREMENT
    # --------------------------------------------------------

    if detected_terms:

        result = {

            "risk_level":
                "HIGH",

            "approval_status":
                "REQUIRED",

            "status":
                "WAITING_FOR_APPROVAL",

            "detected_risk_terms":
                detected_terms
        }


    else:

        result = {

            "risk_level":
                "LOW",

            "approval_status":
                "NOT_REQUIRED",

            "status":
                "READY_FOR_ACTION",

            "detected_risk_terms":
                []
        }


    print(
        "Risk Level:",
        result["risk_level"]
    )


    print(
        "Approval:",
        result["approval_status"]
    )


    if detected_terms:

        print(
            "Detected terms:",
            ", ".join(detected_terms)
        )


    return result


# ============================================================
# SUPERVISOR AGENT
# ============================================================

async def supervisor_agent(
    mcp_client,
    incident_id: str
) -> dict:

    print("\n" + "=" * 75)
    print("SUPERVISOR AGENT")
    print(
        f"Starting investigation: {incident_id}"
    )
    print("=" * 75)


    # --------------------------------------------------------
    # STEP 1 — MONITORING AGENT
    # --------------------------------------------------------

    monitoring_result = (
        await monitoring_agent(
            mcp_client,
            incident_id
        )
    )


    if "error" in monitoring_result:

        return {

            "incident_id":
                incident_id,

            "status":
                "FAILED",

            "reason":
                monitoring_result["error"]
        }


    # --------------------------------------------------------
    # STEP 2 — KNOWLEDGE AGENT
    # --------------------------------------------------------

    knowledge_result = (
        await knowledge_agent(
            mcp_client,
            monitoring_result
        )
    )


    # --------------------------------------------------------
    # STEP 3 — DIAGNOSIS AGENT
    # --------------------------------------------------------

    diagnosis_result = (
        await diagnosis_agent(
            monitoring_result,
            knowledge_result
        )
    )


    # --------------------------------------------------------
    # STEP 4 — RISK CHECK
    # --------------------------------------------------------

    risk_result = risk_check(
        diagnosis_result
    )


    # --------------------------------------------------------
    # FINAL STATE
    # --------------------------------------------------------

    final_state = {

        "incident_id":
            incident_id,

        "monitoring_agent":
            monitoring_result,

        "knowledge_agent":
            knowledge_result,

        "diagnosis_agent":
            diagnosis_result,

        "risk":
            risk_result,

        "approval_status":
            risk_result[
                "approval_status"
            ],

        "status":
            risk_result[
                "status"
            ]
    }


    return final_state


# ============================================================
# MAIN
# ============================================================

async def main():

    print(
        "Connecting to MCP Server..."
    )


    async with Client(
        MCP_SERVER_URL
    ) as mcp_client:


        print(
            "\nConnected to MCP Server"
        )


        # ----------------------------------------------------
        # MCP CAPABILITY DISCOVERY
        # ----------------------------------------------------

        tools_result = (
            await mcp_client.list_tools()
        )


        print(
            "\nAVAILABLE MCP TOOLS"
        )

        print(
            "=" * 60
        )


        for tool in tools_result.tools:

            print(
                f"- {tool.name}"
            )


        required_tools = {

            "mcp_get_incident",

            "mcp_get_server_metrics",

            "mcp_search_incidents",

            "mcp_get_runbook"
        }


        available_tools = {

            tool.name

            for tool in tools_result.tools
        }


        missing_tools = (
            required_tools
            - available_tools
        )


        if missing_tools:

            raise RuntimeError(
                "Required MCP tools are missing: "
                + ", ".join(
                    sorted(missing_tools)
                )
            )


        # ----------------------------------------------------
        # RUN MULTI-AGENT INVESTIGATION
        # ----------------------------------------------------

        final_state = (
            await supervisor_agent(
                mcp_client,
                INCIDENT_ID
            )
        )


        # ----------------------------------------------------
        # FINAL OUTPUT
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 75
        )

        print(
            "FINAL MULTI-AGENT STATE"
        )

        print(
            "=" * 75
        )


        print(
            json.dumps(
                final_state,
                indent=2,
                default=str
            )
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    anyio.run(
        main
    )