import pytest
from response import validate_response, parse_citations, Source

def test_citation_parsing_strict():
    text = "Eligible items can be returned within 30 days [Source: 01-returns-policy-current.md, Heading: Standard return window]."
    citations = parse_citations(text)
    assert len(citations) == 1
    assert citations[0].filename == "01-returns-policy-current.md"
    assert citations[0].heading == "Standard return window"

def test_citation_parsing_fallback():
    text = "For standard international orders, we ship to Canada. Details are in 06-international-shipping.md under Canada delivery estimate."
    citations = parse_citations(text)
    assert len(citations) == 1
    assert citations[0].filename == "06-international-shipping.md"
    # Fallback nearest search or general fallback
    assert citations[0].heading is not None

def test_privacy_redactions_email():
    text = "Yes, your order ORD-1007 was shipped to ava.morgan@example.test today."
    res = validate_response(text, "trace-123")
    assert res.handoff is True
    assert "ava.morgan@example.test" not in res.answer
    assert "[REDACTED]" in res.answer
    assert "email" in res.handoff_reason.lower()

def test_privacy_redactions_risk():
    text = "The order has a risk score of 82 and is under fraud review cleared."
    res = validate_response(text, "trace-123")
    assert res.handoff is True
    assert "82" not in res.answer
    assert "[REDACTED]" in res.answer
    assert "fraud review" in res.handoff_reason.lower() or "internal fields" in res.handoff_reason.lower()

def test_unsupported_actions_cancellation():
    text = "I have cancelled your order ORD-1001 for you."
    res = validate_response(text, "trace-123")
    assert res.handoff is True
    assert "cannot complete" in res.answer or "System Notification" in res.answer
    assert "cancellation" in res.handoff_reason.lower()

def test_unsupported_actions_refund():
    text = "We have processed your refund for ORD-1002."
    res = validate_response(text, "trace-123")
    assert res.handoff is True
    assert "refund" in res.handoff_reason.lower()
