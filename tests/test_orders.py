import pytest
from tools import normalize_order_id, validate_order_id, lookup_order

def test_order_id_normalization():
    assert normalize_order_id("ord-1007") == "ORD-1007"
    assert normalize_order_id("ORD-1007") == "ORD-1007"
    assert normalize_order_id("1007") == "ORD-1007"
    assert normalize_order_id("  ord1007  ") == "ORD-1007"
    assert normalize_order_id("ORD - 1007") == "ORD-1007"

def test_order_id_validation():
    assert validate_order_id("ORD-1007") is True
    assert validate_order_id("ORD-1234") is True
    assert validate_order_id("1007") is False
    assert validate_order_id("ORD1007") is False
    assert validate_order_id("ord-1007") is False

def test_lookup_order_success():
    res = lookup_order("ORD-1007")
    assert res.found is True
    assert res.error is None
    # Exclude customer details and internal notes from returned data dictionary
    assert "customer" not in res.order_data
    assert "internal" not in res.order_data
    assert res.order_data["order_id"] == "ORD-1007"
    assert res.order_data["status"] == "shipped"
    assert res.order_data["carrier"] == "UPS"

def test_lookup_order_cancelled_nullifies_eta():
    # ORD-1004 is cancelled. Stale tracking/carrier/estimation fields should be set to None.
    res = lookup_order("ORD-1004")
    assert res.found is True
    assert res.order_data["status"] == "cancelled"
    assert res.order_data["estimated_delivery"] is None
    assert res.order_data["carrier"] is None
    assert res.order_data["tracking_number"] is None

def test_lookup_order_unknown():
    res = lookup_order("ORD-9999")
    assert res.found is False
    assert res.handoff_recommended is True
    assert "not found" in res.error.lower()

def test_lookup_order_exception():
    # ORD-1010 has status 'exception' in dataset, triggering immediate handoff
    res = lookup_order("ORD-1010")
    assert res.found is True
    assert res.order_data["status"] == "exception"
    assert res.handoff_recommended is True
    assert "exception" in res.handoff_reason.lower()
