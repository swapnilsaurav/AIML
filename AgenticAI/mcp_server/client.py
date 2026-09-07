import anyio
from mcp import Client


async def main():

    print("Connecting to MCP Server...")

    async with Client(
        "http://localhost:8000/mcp"
    ) as client:

        print("\nConnected to MCP Server")


        # =====================================================
        # 1. LIST AVAILABLE TOOLS
        # =====================================================

        print("\nAVAILABLE TOOLS")
        print("=" * 60)

        tools_result = await client.list_tools()

        for tool in tools_result.tools:

            print("\nTool Name:")
            print(tool.name)

            print("Description:")
            print(tool.description)

            print("Input Schema:")
            print(tool.input_schema)


        # =====================================================
        # 2. CALL GET INCIDENT
        # =====================================================

        print("\n" + "=" * 60)
        print("CALLING: mcp_get_incident")
        print("=" * 60)

        incident_result = await client.call_tool(
            "mcp_get_incident",
            {
                "incident_id": "INC-2045"
            }
        )

        print("\nContent:")
        print(incident_result.content)

        print("\nStructured Content:")
        print(incident_result.structured_content)

        print("\nIs Error:")
        print(incident_result.is_error)


        # =====================================================
        # 3. CALL SERVER METRICS
        # =====================================================

        print("\n" + "=" * 60)
        print("CALLING: mcp_get_server_metrics")
        print("=" * 60)

        metrics_result = await client.call_tool(
            "mcp_get_server_metrics",
            {
                "server": "APP-PROD-12"
            }
        )

        print("\nContent:")
        print(metrics_result.content)

        print("\nStructured Content:")
        print(metrics_result.structured_content)


        # =====================================================
        # 4. SEARCH HISTORICAL INCIDENTS
        # =====================================================

        print("\n" + "=" * 60)
        print("CALLING: mcp_search_incidents")
        print("=" * 60)

        history_result = await client.call_tool(
            "mcp_search_incidents",
            {
                "keyword": "connection"
            }
        )

        print("\nContent:")
        print(history_result.content)

        print("\nStructured Content:")
        print(history_result.structured_content)


        # =====================================================
        # 5. GET RUNBOOK
        # =====================================================

        print("\n" + "=" * 60)
        print("CALLING: mcp_get_runbook")
        print("=" * 60)

        runbook_result = await client.call_tool(
            "mcp_get_runbook",
            {
                "topic": "connection pool"
            }
        )

        print("\nContent:")
        print(runbook_result.content)

        print("\nStructured Content:")
        print(runbook_result.structured_content)


        print("\n" + "=" * 60)
        print("MCP CLIENT TEST COMPLETED")
        print("=" * 60)


if __name__ == "__main__":

    anyio.run(main)