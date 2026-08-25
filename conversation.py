from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ChatMessage:
    role: str       # 'user' or 'model'
    content: str    # Text content of the message
    raw_query: str = "" # Keep the original user query without context injected

class ConversationManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[ChatMessage] = []

    def add_message(self, role: str, content: str, raw_query: str = ""):
        """Add a message to the session history.
        role must be 'user' or 'model'.
        """
        # Ensure role is mapped to standard 'user' or 'model' for Gemini compatibility
        mapped_role = 'user' if role in ['user', 'system_context'] else 'model'
        self.history.append(ChatMessage(role=mapped_role, content=content, raw_query=raw_query))

    def get_history_for_llm(self) -> List[Dict[str, Any]]:
        """Format the history for the google-genai contents parameter."""
        formatted = []
        for msg in self.history:
            formatted.append({
                "role": msg.role,
                "parts": [{"text": msg.content}]
            })
        return formatted

    def get_last_user_query(self) -> str:
        """Retrieve the last raw user query from history to use for retrieval."""
        for msg in reversed(self.history):
            if msg.role == 'user':
                return msg.raw_query if msg.raw_query else msg.content
        return ""

    def clear(self):
        """Clear conversation history."""
        self.history.clear()
