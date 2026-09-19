"""
LangGraph state definition shared by every node in the graph.
"""

from typing import TypedDict, Optional, List, Dict, Any


class ChatTurn(TypedDict):
    role: str          # "user" | "assistant"
    content: str


class AgentState(TypedDict, total=False):
    # --- input ---
    session_id: str
    user_id: str
    question: str
    chat_history: List[ChatTurn]

    # --- guardrails ---
    blocked: bool
    block_reason: str

    # --- schema retriever ---
    schema_context: str

    # --- worker: sql_generator + sql_executor ---
    sql_query: str
    sql_columns: List[str]
    sql_rows: List[Any]
    sql_error: Optional[str]
    generation_attempts: int
    max_generation_attempts: int

    # --- response formatter ---
    final_response: str

    # --- observability passthrough ---
    trace: Any                       # Langfuse trace object (or NoOp)
    trace_id: Optional[str]
