"""
Integration test for the full graph. Requires network access to a real SQL
Warehouse + model serving endpoint, so it's skipped by default in CI unless
RUN_INTEGRATION=1 is set.
"""

import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run against real Databricks resources.",
)


def test_blocked_question_short_circuits():
    from src.agent.graph import run_agent
    result = run_agent(
        question="what's the weather today?",
        session_id="test-session",
        user_id="test-user",
        chat_history=[],
    )
    assert result["blocked"] is True
    assert result["final_response"]


def test_product_question_returns_sql_and_response():
    from src.agent.graph import run_agent
    result = run_agent(
        question="show me backpacks under 3000 rupees",
        session_id="test-session",
        user_id="test-user",
        chat_history=[],
    )
    assert result["blocked"] is False
    assert "select" in result["sql_query"].lower()
    assert result["final_response"]
