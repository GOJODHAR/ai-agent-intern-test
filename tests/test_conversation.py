import pytest
from conversation import ConversationManager

def test_conversation_management():
    conv_mgr = ConversationManager(session_id="test-session")
    assert len(conv_mgr.history) == 0
    
    # Add messages
    conv_mgr.add_message("user", "Hello support", raw_query="Hello support")
    conv_mgr.add_message("model", "Hello customer, how can I help you today?")
    conv_mgr.add_message("user", "Where is ORD-1007?", raw_query="Where is ORD-1007?")
    
    assert len(conv_mgr.history) == 3
    assert conv_mgr.history[0].role == 'user'
    assert conv_mgr.history[1].role == 'model'
    assert conv_mgr.history[2].role == 'user'
    assert conv_mgr.history[2].raw_query == 'Where is ORD-1007?'
    
    # Get last query
    assert conv_mgr.get_last_user_query() == "Where is ORD-1007?"
    
    # Format for LLM
    llm_history = conv_mgr.get_history_for_llm()
    assert len(llm_history) == 3
    assert llm_history[0]["role"] == "user"
    assert llm_history[0]["parts"][0]["text"] == "Hello support"
    assert llm_history[1]["role"] == "model"
    assert llm_history[2]["role"] == "user"
    
    # Clear session
    conv_mgr.clear()
    assert len(conv_mgr.history) == 0
    assert conv_mgr.get_last_user_query() == ""
