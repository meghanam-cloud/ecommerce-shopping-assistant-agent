"""
The worker's SQL tool. This is the only place in the agent that talks to
the SQL Warehouse for product queries. Wrapped as a LangChain @tool so it
can also be bound directly to an LLM if a future node wants a full
tool-calling ReAct loop instead of the fixed generate->execute pipeline.
"""

import logging
from langchain_core.tools import tool

from src.utils.config_loader import settings
from src.utils.sql_connection import run_query

logger = logging.getLogger(__name__)

MAX_ROWS = 50


def _is_safe_select(query: str) -> tuple[bool, str]:
    """Defense-in-depth on top of the guardrail node: only ever allow a
    single read-only SELECT against the configured product table."""
    stripped = query.strip().rstrip(";")
    lowered = stripped.lower()

    if ";" in stripped:
        return False, "Only a single statement is allowed."
    if not lowered.startswith("select"):
        return False, "Only SELECT statements are allowed."

    forbidden = ["insert", "update", "delete", "drop", "truncate", "alter", "merge", "create", "grant"]
    if any(f" {kw} " in f" {lowered} " for kw in forbidden):
        return False, "Query contains a disallowed write/DDL keyword."

    if settings.data.table.lower() not in lowered:
        return False, f"Query must reference the {settings.data.table} table."

    return True, ""


@tool
def execute_sql_query(query: str) -> dict:
    """Execute a read-only SELECT SQL query against the ecom product Delta
    table via the SQL Warehouse connector, and return the results.

    Args:
        query: A single SELECT statement targeting the configured product table.

    Returns:
        dict with keys: columns (list[str]), rows (list[list]), row_count (int),
        error (str | None).
    """
    is_safe, reason = _is_safe_select(query)
    if not is_safe:
        return {"columns": [], "rows": [], "row_count": 0, "error": reason}

    guarded_query = query.strip().rstrip(";")
    if " limit " not in guarded_query.lower():
        guarded_query = f"{guarded_query} LIMIT {MAX_ROWS}"

    try:
        columns, rows = run_query(guarded_query)
        rows = [list(r) for r in rows][:MAX_ROWS]
        return {"columns": columns, "rows": rows, "row_count": len(rows), "error": None}
    except Exception as e:
        logger.warning("SQL execution failed: %s", e)
        return {"columns": [], "rows": [], "row_count": 0, "error": str(e)}
