"""
Node: builds the schema context string injected into the SQL generation
prompt. Combines:
  1. Live column list/types from DESCRIBE TABLE (source of truth, catches
     schema drift), with a safe fallback to the static config.yaml mapping
     if the warehouse call fails (e.g. running offline/unit tests).
  2. Hand-written semantic notes on what each column means and how the
     semi-structured (JSON-as-string) columns are shaped, so the LLM knows
     HOW to query them (e.g. get_json_object) not just that they exist.
"""

import logging
from src.agent.state import AgentState
from src.utils.config_loader import settings
from src.utils.sql_connection import run_query
from src.observability.langfuse_tracer import trace_span

logger = logging.getLogger(__name__)

# Notes about what each column contains and, for JSON-shaped text columns,
# how to pull fields out of them in Databricks SQL.
COLUMN_NOTES = {
    "product_id": "Unique integer product identifier. Use for joins/dedup.",
    "title": "Brand name of the product (e.g. 'Tommy Hilfiger'). NOT the product title/description.",
    "product_description": "Short product name/title text (e.g. 'Women Navy Blue Solid Backpack'). Use LIKE/ILIKE for keyword search on product type, color, gender.",
    "rating": "Average star rating, DOUBLE, 0-5. 0 means no ratings yet.",
    "ratings_count": "Number of ratings received, INT.",
    "initial_price": "Original listed price before discount, DOUBLE, in `currency`.",
    "discount": "Discount percentage applied, INT (e.g. 58 means 58% off). Can be NULL/blank.",
    "final_price": "Actual selling price after discount, DOUBLE, in `currency`. Use this for 'price'/'cost' questions unless the user explicitly asks about original/MRP price.",
    "currency": "ISO-like currency code, STRING (e.g. 'INR').",
    "delivery_options": "Free-text, comma-separated list of delivery/return policy phrases. Use LIKE for keyword search (e.g. '%pay on delivery%').",
    "product_details": (
        "JSON-encoded STRING (not a native struct) with keys: description, "
        "material_and_care, size_and_fit. To query a sub-field in Databricks SQL use "
        "get_json_object(product_details, '$.description') etc. Cast/parse as needed."
    ),
    "product_specifications": (
        "JSON-encoded STRING representing an ARRAY of objects, each shaped "
        "{\"specification_name\": ..., \"specification_value\": ...} (e.g. Material, "
        "Occasion, Warranty). To search within it use "
        "product_specifications LIKE '%\"specification_name\":\"Material\"%\"specification_value\":\"Leather\"%' "
        "style matching, or from_json with a defined array<struct> schema for structured extraction."
    ),
    "amount_of_rating": (
        "JSON-encoded STRING object with keys 1_star..5_star giving counts per rating "
        "level, e.g. get_json_object(amount_of_rating, '$.5_star') for 5-star count."
    ),
    "sizes": "Free-text STRING listing available sizes / size measurements, comma-separated (e.g. 'S, M, L' or detailed per-size dimensions). Use LIKE for a specific size like '%\"M\"%' or '%M —%'.",
    "all_offers": "Free-text STRING of bank/coupon offers and EMI info, comma-separated. Use LIKE for keyword search (e.g. '%EMI%', '%Coupon%').",
    "category": "Product category slug, STRING, lowercase-hyphenated (e.g. 'backpacks', 'bedsheets', 'boots'). Use for filtering/grouping by product type.",
}


def _describe_table_live() -> list[str]:
    columns, rows = run_query(f"DESCRIBE TABLE {settings.data.full_table_name}")
    lines = []
    for row in rows:
        col_name, col_type = row[0], row[1]
        if not col_name or col_name.startswith("#"):
            continue
        note = COLUMN_NOTES.get(col_name, "")
        lines.append(f"  - {col_name} ({col_type}): {note}")
    return lines


def _describe_table_static() -> list[str]:
    lines = []
    for logical_name, physical_col in settings.data.columns.items():
        note = COLUMN_NOTES.get(physical_col, "")
        lines.append(f"  - {physical_col} [config alias: {logical_name}]: {note}")
    return lines


def build_schema_context() -> str:
    try:
        column_lines = _describe_table_live()
        source = "live DESCRIBE TABLE"
    except Exception as e:
        logger.warning("DESCRIBE TABLE failed, using static config schema: %s", e)
        column_lines = _describe_table_static()
        source = "static config.yaml mapping"

    return (
        f"Table: {settings.data.full_table_name}\n"
        f"(schema source: {source})\n"
        f"Columns:\n" + "\n".join(column_lines) + "\n\n"
        "Notes:\n"
        "  - Always filter/search using the exact column names above.\n"
        "  - Prices and rating are numeric; do not quote them in comparisons.\n"
        "  - product_details, product_specifications and amount_of_rating are "
        "JSON-encoded strings, not native structs - extract fields with "
        "get_json_object(column, '$.key') or LIKE-based text search.\n"
        "  - Always add a LIMIT (default 50) unless the user asks for a count/aggregate."
    )


def schema_retriever_node(state: AgentState) -> AgentState:
    if state.get("blocked"):
        return state

    trace = state.get("trace")
    with trace_span(
        trace,
        "schema_retrieval",
        input={"table": settings.data.full_table_name},
    ) as span:
        schema_context = build_schema_context()
        span.update(
            output=schema_context[:2000],
            metadata={"full_length": len(schema_context)},
        )

    return {**state, "schema_context": schema_context}
