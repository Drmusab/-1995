"""GraphQL API Setup Module"""

from typing import Any, Dict, Optional

try:
    import graphene
    from graphene import Field, Float, Int
    from graphene import List as GrapheneList
    from graphene import ObjectType, String

    GRAPHENE_AVAILABLE = True
except ImportError:
    GRAPHENE_AVAILABLE = False

from src.observability.logging.config import get_logger

if GRAPHENE_AVAILABLE:

    class SystemStatus(ObjectType):
        """GraphQL type for system status."""

        status = String()
        version = String()
        uptime_seconds = Float()
        components_status = String()
        api_servers_running = Int()

    class TextResponse(ObjectType):
        """GraphQL type for text response."""

        response_text = String()
        response_id = String()
        session_id = String()
        interaction_id = String()
        processing_time = Float()
        confidence = Float()
        suggested_follow_ups = GrapheneList(String)

    class Query(ObjectType):
        """GraphQL Query root."""

        status = Field(SystemStatus)

        def resolve_status(self, info):
            """Resolve system status query."""
            assistant = getattr(info.context, "assistant", None)
            if assistant:
                status_data = assistant.get_status()
                return SystemStatus(
                    status=status_data.get("status"),
                    version=status_data.get("version"),
                    uptime_seconds=status_data.get("uptime_seconds"),
                    components_status=status_data.get("components_status"),
                    api_servers_running=status_data.get("api_servers_running"),
                )
            return SystemStatus(
                status="unavailable",
                version="unknown",
                uptime_seconds=0,
                components_status="unknown",
                api_servers_running=0,
            )

    class Mutation(ObjectType):
        """GraphQL Mutation root."""

        process_text = Field(TextResponse, text=String(required=True), session_id=String())

        async def resolve_process_text(self, info, text, session_id=None):
            """Resolve text processing mutation."""
            assistant = getattr(info.context, "assistant", None)
            if assistant:
                try:
                    result = await assistant.process_text_input(text=text, session_id=session_id)
                    return TextResponse(
                        response_text=result.get("response_text"),
                        response_id=result.get("response_id"),
                        session_id=result.get("session_id"),
                        interaction_id=result.get("interaction_id"),
                        processing_time=result.get("processing_time"),
                        confidence=result.get("confidence"),
                        suggested_follow_ups=result.get("suggested_follow_ups", []),
                    )
                except Exception as e:
                    return TextResponse(
                        response_text=f"Error: {str(e)}",
                        response_id="error",
                        session_id=session_id or "unknown",
                        interaction_id="error",
                        processing_time=0.0,
                        confidence=0.0,
                        suggested_follow_ups=[],
                    )
            return TextResponse(
                response_text="Assistant unavailable",
                response_id="unavailable",
                session_id=session_id or "unknown",
                interaction_id="unavailable",
                processing_time=0.0,
                confidence=0.0,
                suggested_follow_ups=[],
            )

    # Create the GraphQL schema
    schema = graphene.Schema(query=Query, mutation=Mutation)


class GraphQLContext:
    """Context object for GraphQL requests."""

    def __init__(self, assistant: Any):
        """Initialize GraphQL context."""
        self.assistant = assistant


class GraphQLServer:
    """GraphQL server implementation."""

    def __init__(self, assistant: Any):
        """Initialize GraphQL server."""
        self.assistant = assistant
        self.logger = get_logger(__name__)
        self.schema = schema if GRAPHENE_AVAILABLE else None

    async def execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """Execute a GraphQL query."""
        if not GRAPHENE_AVAILABLE:
            return {"errors": [{"message": "GraphQL not available - graphene not installed"}]}

        try:
            context = GraphQLContext(self.assistant)
            result = await self.schema.execute_async(
                query, variable_values=variables, context=context
            )

            response = {}
            if result.data:
                response["data"] = result.data
            if result.errors:
                response["errors"] = [{"message": str(error)} for error in result.errors]

            return response

        except Exception as e:
            self.logger.error(f"GraphQL execution error: {str(e)}")
            return {"errors": [{"message": f"Execution error: {str(e)}"}]}

    def get_schema_definition(self) -> str:
        """Get the GraphQL schema definition."""
        if not GRAPHENE_AVAILABLE:
            return "# GraphQL not available - graphene not installed"

        return str(self.schema)


async def setup_graphql_api(assistant: Any) -> GraphQLServer:
    """
    Setup and configure the GraphQL API.

    Args:
        assistant: AI Assistant instance

    Returns:
        GraphQL server instance
    """
    logger = get_logger(__name__)

    if not GRAPHENE_AVAILABLE:
        logger.warning("GraphQL setup failed - graphene package not available")
        return None

    server = GraphQLServer(assistant)

    logger.info("GraphQL API setup completed")
    return server
