import pytest
from unittest.mock import MagicMock
from agent import AsterRowAgent
from conversation import ConversationManager
from response import validate_response

# 1. Deterministic safety tests mocking classifier intents
def test_unsupported_action_cancellation(monkeypatch):
    agent = AsterRowAgent()
    
    # Mock classify_intent to return 'cancellation' request
    monkeypatch.setattr(agent, "classify_intent", lambda msg, mgr: ("cancellation", False))
    
    conv_mgr = ConversationManager(session_id="test-cancel")
    response, trace = agent.chat("test-cancel", "Cancel my order ORD-1007", conv_mgr)
    
    assert response.handoff is True
    assert "cancellation" in response.handoff_reason.lower()
    
    # Stable behavioral assertions
    ans_lower = response.answer.lower()
    assert "cancel" in ans_lower
    assert "cannot complete" in ans_lower
    assert "human" in ans_lower
    assert "completed" not in ans_lower
    assert len(response.sources) == 0

def test_unsupported_action_refund(monkeypatch):
    agent = AsterRowAgent()
    
    # Mock classify_intent to return 'refund' request
    monkeypatch.setattr(agent, "classify_intent", lambda msg, mgr: ("refund", False))
    
    conv_mgr = ConversationManager(session_id="test-refund")
    response, trace = agent.chat("test-refund", "I want my money back for ORD-1002", conv_mgr)
    
    assert response.handoff is True
    assert "refund" in response.handoff_reason.lower()
    
    # Stable behavioral assertions
    ans_lower = response.answer.lower()
    assert "refund" in ans_lower
    assert "cannot complete" in ans_lower
    assert "human" in ans_lower
    assert "completed" not in ans_lower
    assert len(response.sources) == 0

def test_unsupported_action_replacement(monkeypatch):
    agent = AsterRowAgent()
    
    # Mock classify_intent to return 'replacement' request
    monkeypatch.setattr(agent, "classify_intent", lambda msg, mgr: ("replacement", False))
    
    conv_mgr = ConversationManager(session_id="test-replace")
    response, trace = agent.chat("test-replace", "Send me another backpack", conv_mgr)
    
    assert response.handoff is True
    assert "replacement" in response.handoff_reason.lower()
    
    # Stable behavioral assertions
    ans_lower = response.answer.lower()
    assert "replace" in ans_lower
    assert "cannot complete" in ans_lower
    assert "human" in ans_lower
    assert "completed" not in ans_lower
    assert len(response.sources) == 0

def test_unsupported_action_address_change(monkeypatch):
    agent = AsterRowAgent()
    
    # Mock classify_intent to return 'address_change' request
    monkeypatch.setattr(agent, "classify_intent", lambda msg, mgr: ("address_change", False))
    
    conv_mgr = ConversationManager(session_id="test-address")
    response, trace = agent.chat("test-address", "Update delivery address to 123 Pine St", conv_mgr)
    
    assert response.handoff is True
    assert "address_change" in response.handoff_reason.lower()
    
    # Stable behavioral assertions
    ans_lower = response.answer.lower()
    assert "address" in ans_lower
    assert "cannot complete" in ans_lower
    assert "human" in ans_lower
    assert "completed" not in ans_lower
    assert len(response.sources) == 0


# 2. Classifier failure test
def test_classifier_failure_handling(monkeypatch):
    agent = AsterRowAgent()
    
    # Mock classify_intent to return failed flag
    monkeypatch.setattr(agent, "classify_intent", lambda msg, mgr: ("none", True))
    
    conv_mgr = ConversationManager(session_id="test-fail")
    response, trace = agent.chat("test-fail", "Any random message", conv_mgr)
    
    assert response.handoff is True
    assert "classifier failure" in response.handoff_reason.lower()
    assert "experiencing connection difficulties" in response.answer
    assert len(response.sources) == 0


# 3. Policy questions vs Action requests routing verification
def test_normal_kb_routing_when_none(monkeypatch):
    # Verify that when intent is none, normal generation path is run
    agent = AsterRowAgent()
    
    # Mock classify_intent to return 'none' (policy questions)
    monkeypatch.setattr(agent, "classify_intent", lambda msg, mgr: ("none", False))
    
    # Mock client generate_content to prevent live API calls in standard unit test
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Here is the return policy from 01-returns-policy-current.md [Source: 01-returns-policy-current.md, Heading: Returns policy]."
    mock_client.models.generate_content.return_value = mock_response
    agent.client = mock_client
    
    conv_mgr = ConversationManager(session_id="test-policy")
    response, trace = agent.chat("test-policy", "What is your refund policy?", conv_mgr)
    
    # Normal execution path: handoff is false, sources are parsed
    assert response.handoff is False
    assert len(response.sources) == 1
    assert response.sources[0].filename == "01-returns-policy-current.md"
    assert "return policy" in response.answer.lower()
