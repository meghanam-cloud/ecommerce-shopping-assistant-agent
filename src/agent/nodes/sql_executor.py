"""
Node: the worker's tool-execution step. Runs the SQL produced by
sql_generator via the sql_execution tool (Databricks SQL Warehouse
connector) and records the outcome on state.
"""

import logging
from src.agent.state import AgentState
from src.agent.tools.sql_tools import execute_sql_query
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)


def sql_executor_node(state: AgentState) -> AgentState:
    if state.get("blocked"):
        return state

    trace = state.get("trace")
    sql_query = state["sql_query"]
    attempt = state.get("generation_attempts", 0)

    with trace_span(
        trace,
        "sql_execution",
        input=sql_query,
        metadata={"attempt": attempt},
    ) as span:
        result = execute_sql_query.invoke({"query": sql_query})

        if result.get("error"):
            logger.warning("SQL execution error on attempt %s: %s", attempt, result["error"])
            span.update(
                output={"error": result["error"]},
                level="ERROR",
                metadata={"attempt": attempt},
            )
            return {**state, "sql_error": result["error"], "sql_columns": [], "sql_rows": []}

        span.update(
            output={
                "columns": result["columns"],
                "row_count": result.get("row_count", len(result["rows"])),
            },
            metadata={
                "attempt": attempt,
                "row_count": len(result["rows"]),
                "columns": result["columns"],
            },
        )

    return {
        **state,
        "sql_columns": result["columns"],
        "sql_rows": result["rows"],
        "sql_error": None,
    }


def should_retry(state: AgentState) -> str:
    """Conditional edge: retry generation on error (up to the configured
    max), otherwise move on to response formatting."""
    max_attempts = state.get("max_generation_attempts", 3)
    if state.get("sql_error") and state.get("generation_attempts", 0) < max_attempts:
        return "retry"
    return "format"
