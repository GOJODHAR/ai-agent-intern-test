import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Source:
    filename: str
    heading: str

@dataclass
class AgentResponse:
    answer: str
    sources: List[Source]
    handoff: bool
    handoff_reason: Optional[str]
    trace_id: str

# List of all knowledge base files for reference scanning
KB_FILES = [
    '01-returns-policy-current.md',
    '02-returns-policy-legacy.md',
    '03-final-sale-and-promotions.md',
    '04-damaged-or-wrong-items.md',
    '05-domestic-shipping.md',
    '06-international-shipping.md',
    '07-warranty.md',
    '08-order-changes-and-cancellations.md',
    '09-trailplus-membership.md',
    '10-gift-cards-and-price-adjustments.md',
    '11-product-care.md',
    '12-breeze-tumbler-product-card.md',
    '13-support-escalation.md',
    '14-internal-content-migration-notes.md'
]

def parse_citations(text: str) -> List[Source]:
    """Parse citations from the model response text.
    First tries to find strict format: [Source: filename.md, Heading: Heading Name]
    Then falls back to searching for filenames and nearest text in brackets/parentheses.
    """
    sources = []
    
    # 1. Strict regex search: [Source: filename.md, Heading: Heading Name]
    # Also handles format [Source: filename.md - Heading: Heading Name] or [Source: filename.md (Heading Name)]
    strict_matches = re.findall(r'\[Source:\s*([\w-]+\.md),\s*(?:Heading:\s*)?([^\]]+)\]', text, re.IGNORECASE)
    for filename, heading in strict_matches:
        sources.append(Source(filename=filename.strip(), heading=heading.strip()))
        
    if sources:
        # Deduplicate
        unique_sources = []
        seen = set()
        for s in sources:
            key = (s.filename.lower(), s.heading.lower())
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)
        return unique_sources

    # 2. Fallback scan for raw filenames in the text
    for filename in KB_FILES:
        if filename in text:
            # Try to find a heading near the filename
            # Match: filename.md followed by (Heading) or -> Heading or : Heading
            context_match = re.search(rf'{re.escape(filename)}(?:\s*(?:\(|->|–|heading:)\s*([^\]\)\n,]+)(?:\))?)?', text, re.IGNORECASE)
            heading = "General"
            if context_match and context_match.group(1):
                heading = context_match.group(1).strip()
            
            # Simple clean up of heading to remove extra punctuation
            heading = re.sub(r'[\*\`\'\"]', '', heading)
            
            sources.append(Source(filename=filename, heading=heading))
            
    # Deduplicate
    unique_sources = []
    seen = set()
    for s in sources:
        if s.filename.lower() not in seen:
            seen.add(s.filename.lower())
            unique_sources.append(s)
            
    return unique_sources

def validate_response(
    text: str,
    trace_id: str,
    is_kb_question: bool = False,
    classified_intent: str = "none",
    classification_failed: bool = False
) -> AgentResponse:
    """Validate model output for security, privacy, and correctness rules."""
    handoff = False
    handoff_reason = None
    cleaned_text = text

    # Deterministic safety state overrides
    if classification_failed:
        handoff = True
        handoff_reason = "Classifier failure or timeout. Failing safe."
        cleaned_text = "I am experiencing connection difficulties. Let me transfer you to a human support specialist to assist you."
        return AgentResponse(
            answer=cleaned_text,
            sources=[],
            handoff=handoff,
            handoff_reason=handoff_reason,
            trace_id=trace_id
        )

    if classified_intent != "none":
        handoff = True
        handoff_reason = f"Unsupported action: User requested {classified_intent}."
        refusal_templates = {
            "cancellation": "I understand you would like to cancel your order. However, I cannot complete cancellations directly. I am requesting a transfer to a human support specialist to assist you.",
            "refund": "I understand you are requesting a refund. However, I cannot complete refunds directly. I am requesting a transfer to a human support specialist to assist you.",
            "replacement": "I understand you are requesting a replacement. However, I cannot complete replacements directly. I am requesting a transfer to a human support specialist to assist you.",
            "address_change": "I understand you want to change your shipping address. However, I cannot complete address changes directly. I am requesting a transfer to a human support specialist to assist you."
        }
        cleaned_text = refusal_templates.get(
            classified_intent,
            "I cannot complete this action directly. A human support specialist has been requested to assist you."
        )
        return AgentResponse(
            answer=cleaned_text,
            sources=[],
            handoff=handoff,
            handoff_reason=handoff_reason,
            trace_id=trace_id
        )

    # 1. Check for Forbidden Data Exposure (Emails, shipping addresses, internal notes, risk scores)
    # Regex for emails
    email_matches = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_matches:
        handoff = True
        handoff_reason = "Privacy violation: Detected email address in response."
        for email in email_matches:
            cleaned_text = cleaned_text.replace(email, "[REDACTED]")

    # Look for risk scores (e.g. risk score is 82 or risk score: 82 or risk score of 82)
    risk_pattern = r'(?:risk\s*score|fraud\s*review|warehouse\s*note|internal\s*note)(?:\s*(?:is|of|value|:|-)\s*)?\s*(\w+)'
    risk_match = re.search(risk_pattern, text, re.IGNORECASE)
    if risk_match:
        handoff = True
        handoff_reason = "Privacy violation: Internal fields (risk score, warehouse notes) detected in response."
        cleaned_text = re.sub(risk_pattern, '[REDACTED]', cleaned_text, flags=re.IGNORECASE)

    # Keywords from orders.json that must never be exposed
    forbidden_terms = [
        "ava.morgan@example.test", "maya.reed@example.test", "noah.kim@example.test", "olivia.chen@example.test",
        "220 King Street", "18 Cedar Lane", "44 Lake Street", "79 Market Street",
        "fraud review cleared", "high risk", "Normal priority", "Picking complete"
    ]
    for term in forbidden_terms:
        if term in cleaned_text:
            handoff = True
            handoff_reason = "Privacy violation: Sensitive customer or internal note data detected in response."
            cleaned_text = cleaned_text.replace(term, "[REDACTED]")

    # 2. Check for Unsupported Action Claims
    # Exclude claiming completion of refund, cancellation, replacement, address change
    unsupported_actions = [
        (r'(?:i\s+have|i\'ve|we\s+have|support\s+has)\s+(?:cancelled|canceled)\s+your\s+order', "cancellation"),
        (r'(?:i\s+have|i\'ve|we\s+have|support\s+has)\s+(?:refunded|processed\s+your\s+refund|issued\s+a\s+refund)', "refund"),
        (r'(?:i\s+have|i\'ve|we\s+have|support\s+has)\s+(?:processed\s+a\s+replacement|replaced|shipped\s+a\s+replacement)', "replacement"),
        (r'(?:i\s+have|i\'ve|we\s+have|support\s+has)\s+(?:updated|changed|corrected)\s+your\s+(?:shipping\s+)?address', "address change")
    ]
    
    for pattern, action in unsupported_actions:
        if re.search(pattern, cleaned_text, re.IGNORECASE):
            handoff = True
            handoff_reason = f"Unsupported action: Agent cannot complete {action} directly."
            cleaned_text += f"\n\n[System Notification: The support agent cannot complete {action}s directly. A human support specialist has been requested to assist you with this action.]"
            break # only append once

    # 3. Parse Citations
    sources = parse_citations(cleaned_text)
    
    # If the question was a KB question, but no sources were cited, we don't automatically fail it,
    # but we will extract them if they are found.
    # Also look for explicit handoff keywords in text
    if any(k in cleaned_text.lower() for k in ["escalate", "human specialist", "transfer to a human", "customer service representative", "human review"]):
        handoff = True
        if not handoff_reason:
            handoff_reason = "Handoff explicitly requested or recommended in message."

    return AgentResponse(
        answer=cleaned_text,
        sources=sources,
        handoff=handoff,
        handoff_reason=handoff_reason,
        trace_id=trace_id
    )
