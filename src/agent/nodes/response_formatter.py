"""
Node: turns raw SQL results (or a guardrail/error condition) into a
natural-language, conversational response for the Streamlit UI.
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import AgentState
from src.utils.llm_client import get_llm
from src.utils.config_loader import settings
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful e-commerce shopping assistant. You are given the user's "
    "question and the raw SQL query results that answer it. Write a concise, "
    "friendly, conversational answer using ONLY the data provided - do not "
    "invent products, prices or facts not present in the results. If the "
    "results are empty, say no matching products were found and suggest "
    "broadening the search. Mention prices with their currency."
)


def response_formatter_node(state: AgentState) -> AgentState:
    trace = state.get("trace")

    if state.get("blocked"):
        with trace_span(trace, "response_formatting", input={"blocked": True}) as span:
            span.update(output=state["block_reason"], metadata={"type": "guardrail_block"})
        return {**state, "final_response": state["block_reason"]}

    if state.get("sql_error"):
        fallback_msg = (
            "I wasn't able to look that up right now - the query kept failing. "
            "Could you try rephrasing your question?"
        )
        with trace_span(trace, "response_formatting", input={"sql_error": True}) as span:
            span.update(output=fallback_msg, metadata={"type": "sql_error_fallback"})
        return {**state, "final_response": fallback_msg}

    columns = state.get("sql_columns", [])
    rows = state.get("sql_rows", [])
    results_text = "No rows returned." if not rows else "\n".join(
        ", ".join(f"{col}={val}" for col, val in zip(columns, row)) for row in rows[:50]
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question: {state['question']}\n\n"
                f"SQL used: {state.get('sql_query', '')}\n\n"
                f"Results:\n{results_text}"
            )
        ),
    ]

    # Trace the LLM response generation as a Langfuse generation.
    with trace_span(
        trace,
        "response_formatting",
        input={"question": state["question"], "row_count": len(rows)},
        metadata={"row_count": len(rows)},
    ) as span:
        gen = span.generation(
            name="llm_response_generation",
            input=[m.content for m in messages],
            model=settings.llm.endpoint_name,
            metadata={"temperature": settings.llm.temperature},
        )

        llm = get_llm()
        response = llm.invoke(messages)

        gen.update(output=response.content, usage=None)
        gen.end()
        span.update(
            output=response.content[:1000],
            metadata={"type": "llm_response", "row_count": len(rows)},
        )

    return {**state, "final_response": response.content.strip()}
