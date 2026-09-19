"""
Supervisor: builds and compiles the LangGraph state graph.

Flow:
    guardrail_node
        -> [blocked]  -> response_formatter -> END
        -> [allowed]  -> schema_retriever -> sql_generator -> sql_executor
                              ^                                    |
                              |___________ [sql error, retries left]
                                                                    |
                                                          [ok / out of retries]
                                                                    v
                                                          response_formatter -> END

sql_generator + sql_executor together are the "worker": sql_generator does
the LLM reasoning, sql_executor calls the bound SQL tool. The supervisor
(this graph's conditional edges) decides whether the worker retries or the
turn moves on to formatting.
"""

import time
import logging
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes.guardrail_node import guardrail_node
from src.agent.nodes.schema_retriever import schema_retriever_node
from src.agent.nodes.sql_generator import sql_generator_node
from src.agent.nodes.sql_executor import sql_executor_node, should_retry
from src.agent.nodes.response_formatter import response_formatter_node
from src.observability.mlflow_tracker import log_turn
from src.observability.langfuse_tracer import get_tracer
from src.utils.config_loader import settings

logger = logging.getLogger(__name__)

MAX_GENERATION_ATTEMPTS = 3


def _route_after_guardrail(state: AgentState) -> str:
    return "blocked" if state.get("blocked") else "allowed"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("schema_retriever", schema_retriever_node)
    graph.add_node("sql_generator", sql_generator_node)
    graph.add_node("sql_executor", sql_executor_node)
    graph.add_node("response_formatter", response_formatter_node)

    graph.set_entry_point("guardrail")

    graph.add_conditional_edges(
        "guardrail",
        _route_after_guardrail,
        {"blocked": "response_formatter", "allowed": "schema_retriever"},
    )

    graph.add_edge("schema_retriever", "sql_generator")
    graph.add_edge("sql_generator", "sql_executor")

    graph.add_conditional_edges(
        "sql_executor",
        should_retry,
        {"retry": "sql_generator", "format": "response_formatter"},
    )

    graph.add_edge("response_formatter", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(
    question: str,
    session_id: str,
    user_id: str,
    chat_history: list,
    trace=None,
) -> dict:
    """Entry point called by the Streamlit UI for a single chat turn.

    If *trace* is not provided a new Langfuse trace is created so the graph
    can be run standalone (e.g. in unit tests).  When called from app.py the
    caller passes an existing trace so memory operations are visible under the
    same trace.
    """
    start = time.time()
    graph = get_graph()

    # ------------------------------------------------------------------
    # Langfuse trace: either reuse the one passed in (from app.py so memory
    # ops share it) or create a fresh one.  The NoOp tracer makes this safe
    # even when Langfuse is disabled.
    # ------------------------------------------------------------------
    if trace is None:
        trace = get_tracer().trace(
            name="shopping_assistant_turn",
            user_id=user_id,
            session_id=session_id,
            input=question,
            metadata={"chat_history_len": len(chat_history or [])},
        )

    initial_state: AgentState = {
        "session_id": session_id,
        "user_id": user_id,
        "question": question,
        "chat_history": chat_history,
        "blocked": False,
        "block_reason": "",
        "generation_attempts": 0,
        "max_generation_attempts": MAX_GENERATION_ATTEMPTS,
        "trace": trace,
    }

    final_state = graph.invoke(initial_state)

    # Finalise the Langfuse trace with the answer and overall latency.
    try:
        trace.update(
            output=final_state.get("final_response", ""),
            metadata={
                "latency_seconds": time.time() - start,
                "blocked": final_state.get("blocked", False),
                "sql_error": final_state.get("sql_error"),
                "generation_attempts": final_state.get("generation_attempts", 0),
            },
        )
    except Exception:
        pass  # tracing must never break the app

    log_turn(
        question=question,
        system_prompt=final_state.get("schema_context", ""),
        sql_query=final_state.get("sql_query", ""),
        response=final_state.get("final_response", ""),
        model_name=settings.llm.endpoint_name,
        latency_seconds=time.time() - start,
        sql_row_count=len(final_state.get("sql_rows", []) or []),
        guardrail_blocked=final_state.get("blocked", False),
        error=final_state.get("sql_error"),
    )

    return final_state
