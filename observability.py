import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Setup standard logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AsterRowAgent")

@dataclass
class ExecutionTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_message: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_passages: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    sanitized_tool_results: List[Dict[str, Any]] = field(default_factory=list)
    raw_llm_response: str = ""
    final_response: Optional[Any] = None
    errors: List[str] = field(default_factory=list)
    handoff_triggered: bool = False
    handoff_reason: Optional[str] = None

    def format_trace(self) -> str:
        """Formats the trace into the clean, readable format recommended by the user."""
        lines = []
        lines.append("=" * 50)
        lines.append(f"TRACE ID: {self.trace_id}")
        lines.append("=" * 50)
        
        lines.append("\nUSER MESSAGE:")
        lines.append(self.user_message)
        
        lines.append("\nCONVERSATION HISTORY:")
        if not self.history:
            lines.append("  (None)")
        else:
            for idx, msg in enumerate(self.history):
                role = msg.get('role', 'unknown')
                content = msg.get('parts', [{}])[0].get('text', '')
                # Summarize content if it contains context wrapper to keep trace clean
                clean_content = content
                if "[Retrieved Context]" in content:
                    clean_content = content.split("User Query:")[-1].strip()
                lines.append(f"  {idx + 1}. {role}: {clean_content}")
                
        lines.append("\nRETRIEVED PASSAGES:")
        if not self.retrieved_passages:
            lines.append("  (None)")
        else:
            for idx, passage in enumerate(self.retrieved_passages):
                lines.append(f"  {idx + 1}. File: {passage.get('filename')}")
                lines.append(f"     Heading: {passage.get('heading')}")
                lines.append(f"     Score: {passage.get('score'):.4f}")
                lines.append(f"     Status: {passage.get('status')} | Authority: {passage.get('policy_authority')}")
                
        lines.append("\nTOOL CALLS:")
        if not self.tool_calls:
            lines.append("  (None)")
        else:
            for idx, call in enumerate(self.tool_calls):
                lines.append(f"  {idx + 1}. Tool: {call.get('name')}")
                lines.append(f"     Args: {call.get('args')}")
                
        lines.append("\nSANITIZED TOOL RESULTS:")
        if not self.sanitized_tool_results:
            lines.append("  (None)")
        else:
            for idx, result in enumerate(self.sanitized_tool_results):
                lines.append(f"  {idx + 1}. Success: {result.get('success')}")
                lines.append(f"     Data: {result.get('data')}")
                lines.append(f"     Error: {result.get('error')}")
                
        lines.append("\nRAW LLM RESPONSE:")
        lines.append(self.raw_llm_response if self.raw_llm_response else "  (None)")
        
        lines.append("\nFINAL RESPONSE:")
        if self.final_response:
            lines.append(f"  Answer: {self.final_response.answer}")
            lines.append(f"  Sources: {[(s.filename, s.heading) for s in self.final_response.sources]}")
            lines.append(f"  Handoff: {self.final_response.handoff}")
            lines.append(f"  Handoff Reason: {self.final_response.handoff_reason}")
        else:
            lines.append("  (None)")
            
        if self.errors:
            lines.append("\nERRORS / WARNINGS:")
            for err in self.errors:
                lines.append(f"  - {err}")
                
        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

def set_log_level(level_name: str):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(level)
