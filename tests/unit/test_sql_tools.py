from src.agent.tools.sql_tools import _is_safe_select


def test_rejects_non_select():
    ok, _ = _is_safe_select("DELETE FROM ecom_products_tbl")
    assert not ok


def test_rejects_multi_statement():
    ok, _ = _is_safe_select("SELECT * FROM ecom_products_tbl; DROP TABLE ecom_products_tbl")
    assert not ok


def test_rejects_wrong_table():
    ok, _ = _is_safe_select("SELECT * FROM some_other_table")
    assert not ok


def test_accepts_valid_select():
    ok, reason = _is_safe_select("SELECT title, final_price FROM ecom_products_tbl WHERE category = 'boots'")
    assert ok, reason
