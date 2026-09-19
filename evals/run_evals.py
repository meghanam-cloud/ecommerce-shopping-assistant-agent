"""
MLflow 3 evaluation pipeline: replays evals/eval_dataset.json through the
compiled graph and logs a pass/fail row + generated SQL per case as one
MLflow run, so eval history is comparable across model/prompt versions.
"""

import json
import logging
import uuid

import mlflow

from src.agent.graph import run_agent
from src.utils.config_loader import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(path: str = "evals/eval_dataset.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate_case(case: dict) -> dict:
    session_id = str(uuid.uuid4())
    result = run_agent(
        question=case["question"],
        session_id=session_id,
        user_id="eval-runner",
        chat_history=[],
    )

    if case.get("expect_blocked"):
        passed = bool(result.get("blocked"))
    else:
        sql = (result.get("sql_query") or "").lower()
        expected = case.get("expected_sql_contains", [])
        passed = all(term.lower() in sql for term in expected) and not result.get("sql_error")

    return {
        "question": case["question"],
        "passed": passed,
        "sql_query": result.get("sql_query", ""),
        "blocked": result.get("blocked", False),
        "response": result.get("final_response", ""),
    }


def main():
    mlflow.set_experiment(settings.mlflow_experiment_path)
    dataset = load_dataset()

    with mlflow.start_run(run_name="eval_suite"):
        results = [evaluate_case(c) for c in dataset]
        pass_count = sum(r["passed"] for r in results)

        mlflow.log_metric("total_cases", len(results))
        mlflow.log_metric("passed_cases", pass_count)
        mlflow.log_metric("pass_rate", pass_count / len(results) if results else 0)
        mlflow.log_text(json.dumps(results, indent=2), "eval_results.json")

        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            logger.info("[%s] %s", status, r["question"])

        print(f"\n{pass_count}/{len(results)} cases passed.")


if __name__ == "__main__":
    main()
