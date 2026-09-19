from src.utils.config_loader import settings


def test_data_config_loaded():
    assert settings.data.table == "ecom_products_tbl"
    assert settings.data.full_table_name.endswith(".ecom_products_tbl")


def test_llm_config_loaded():
    assert settings.llm.endpoint_name == "databricks-meta-llama-3-3-70b-instruct"


def test_guardrails_config_loaded():
    assert settings.guardrails.pii_check is True
    assert "backpack" in settings.guardrails.allowed_topics_keywords
