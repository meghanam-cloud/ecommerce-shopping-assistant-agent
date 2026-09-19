# my-sql-agent

Multi-turn SQL agent chatbot over an e-commerce product catalog, deployed as a
**Databricks App** (Streamlit), built with **LangGraph**.

## Architecture

```
User -> Streamlit (src/app/app.py)
          -> Supervisor graph (src/agent/graph.py)
               1. guardrail_node        - PII / prompt-injection / topical-scope checks
               2. schema_retriever      - builds table schema + column semantics for the prompt
               3. sql_generator (worker)- LLM reasons over schema + question -> SQL
               4. sql_executor  (worker)- runs SQL via databricks-sql-connector (SQL Warehouse)
                    -> on error, supervisor routes back to sql_generator (max 3 attempts)
               5. response_formatter    - LLM turns SQL results into a natural-language answer
          -> Delta tables (src/memory) - short-term (per-session) + long-term (per-user) memory
          -> Langfuse (OpenTelemetry tracing) + MLflow 3 (per-turn + eval logging)
```

Databricks Apps run in a serverless container with no
Spark runtime attached. All reads/writes (product queries + memory tables)
go through `databricks-sql-connector` against the configured SQL Warehouse
(`src/utils/sql_connection.py`).

## Setup

1. Create the memory tables:
   ```
   databricks sql execute --file src/memory/schema.sql
   ```
2. Populate the secret scope (see `bundle/resources/secrets.yml`):
   ```
   databricks secrets create-scope dbx-secret-scope
   databricks secrets put-secret dbx-secret-scope DATABRICKS_TOKEN
   databricks secrets put-secret dbx-secret-scope LANGFUSE_PUBLIC_KEY
   databricks secrets put-secret dbx-secret-scope LANGFUSE_SECRET_KEY
   databricks secrets put-secret dbx-secret-scope LANGFUSE_HOST
   ```
3. Edit `config/config.yaml` - set `databricks.host`, `databricks.warehouse_id`,
   and your `data.catalog` / `data.schema` / `data.table`.

## Local run

```
cp .env.example .env   # fill in real values
pip install -r requirements.txt
streamlit run src/app/app.py
```

## Deploy (Databricks Asset Bundles)

```
cd bundle
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run ecom_sql_agent_app -t dev
```
CI (`.github/workflows/ci.yml`) lints, unit-tests and validates the bundle on
every PR. Deploy (`.github/workflows/deploy.yml`) deploys + runs the app on
push to `main`.

## Testing

```
pytest tests/unit                 # fast, no external deps
RUN_INTEGRATION=1 pytest tests/integration   # needs real warehouse + endpoint
python evals/run_evals.py         # golden-question eval suite, logged to MLflow
```

## Extending

- Add columns: update `data.columns` in `config/config.yaml` and the semantic
  notes in `src/agent/nodes/schema_retriever.py:COLUMN_NOTES`.
- Add a guardrail: add a check function + call in `src/agent/guardrails.py`.
- Add a new node: implement in `src/agent/nodes/`, wire it into
  `src/agent/graph.py`.
