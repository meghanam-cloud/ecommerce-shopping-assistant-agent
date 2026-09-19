"""
Node: the worker's reasoning step. Given the schema context, chat history
and question (and, on retry, the previous SQL error), the LLM reasons
about the request and produces a single SQL SELECT statement.
"""

import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import AgentState
from src.utils.llm_client import get_llm
from src.utils.config_loader import settings
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)

SYSTEM_TEMPLATE = """You are a SQL reasoning engine for an e-commerce product catalog assistant.
Given the table schema below and the user's question, think step by step about
which columns and filters answer the question, then output ONE Databricks SQL
SELECT statement that answers it. Output ONLY the SQL - no explanation, no
markdown fences.

{schema_context}

Rules:
- SELECT only, single statement, must query the table above.
- Prefer selecting human-readable columns (brand/title, product_description,
  final_price, rating, category) over dumping raw JSON columns, unless the
  user specifically asked about details/specifications/sizes/offers.
- Use get_json_object(...) to read a specific field out of a JSON-string column.
- Always include a LIMIT (50 by default) unless the question is a count/aggregate.
"""


def _extract_sql(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    return text.strip().rstrip(";").strip()


def sql_generator_node(state: AgentState) -> AgentState:
    if state.get("blocked"):
        return state

    trace = state.get("trace")
    attempt = state.get("generation_attempts", 0) + 1
    is_retry = bool(state.get("sql_error"))

    system_prompt = SYSTEM_TEMPLATE.format(schema_context=state["schema_context"])

    messages = [SystemMessage(content=system_prompt)]
    for turn in state.get("chat_history", [])[-6:]:
        role_prefix = "User" if turn["role"] == "user" else "Assistant"
        messages.append(HumanMessage(content=f"{role_prefix}: {turn['content']}"))

    user_prompt = f"Question: {state['question']}"
    if is_retry:
        user_prompt += (
            f"\n\nThe previous SQL attempt failed with error:\n{state['sql_error']}\n"
            f"Previous SQL:\n{state.get('sql_query', '')}\n"
            "Fix the query and try again."
        )
    messages.append(HumanMessage(content=user_prompt))

    span_name = f"sql_generation_attempt_{attempt}" if is_retry else "sql_generation"

    with trace_span(
        trace,
        span_name,
        input={
            "question": state["question"],
            "schema_context": state["schema_context"][:500],
            "is_retry": is_retry,
        },
        metadata={
            "attempt": attempt,
            "is_retry": is_retry,
            "chat_history_len": len(state.get("chat_history", [])),
        },
    ) as span:
        # Trace the LLM call as a Langfuse generation.
        gen = span.generation(
            name="llm_sql_generation",
            input=[m.content for m in messages],
            model=settings.llm.endpoint_name,
            metadata={
                "temperature": settings.llm.temperature,
                "attempt": attempt,
                "is_retry": is_retry,
            },
        )

        llm = get_llm()
        response = llm.invoke(messages)
        sql_query = _extract_sql(response.content)

        gen.update(output=response.content)
        gen.end()
        span.update(
            output=sql_query,
            metadata={
                "attempt": attempt,
                "is_retry": is_retry,
                "sql_query": sql_query,
            },
        )

    logger.info("Generated SQL (attempt %s): %s", attempt, sql_query)

    return {
        **state,
        "sql_query": sql_query,
        "generation_attempts": attempt,
        "sql_error": None,
    }
