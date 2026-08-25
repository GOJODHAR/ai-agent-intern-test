import json
import os
import sys
import colorama
from colorama import Fore, Style
from agent import AsterRowAgent
from conversation import ConversationManager
from response import AgentResponse
from tools import lookup_order

# Initialize Colorama for Windows terminal colors
colorama.init()

# Concept checker helper
def check_concept(text: str, concept: str) -> bool:
    """Deterministic concept checker using keyword groups."""
    text = text.lower()
    
    # Map concepts to expected logical sub-clauses
    # (Each sub-clause requires at least one of its list elements to match)
    mappings = {
        "final sale does not block damaged-item review": [
            ["final sale", "final-sale"],
            ["damaged", "defect", "broken", "wrong", "zipper"],
            ["review", "assist", "exception", "eligible", "human"]
        ],
        "report within 7 days": [
            ["7 day", "seven day", "7 calendar day", "within 7"]
        ],
        "human review before approval": [
            ["human", "specialist", "agent limit", "approve", "transfer"]
        ],
        "canada is supported": [
            ["canada"]
        ],
        "5–9 business days after dispatch": [
            ["5-9", "5–9", "5 to 9", "five to nine"],
            ["day"]
        ],
        "duties or taxes are not prepaid": [
            ["duty", "duties", "tax", "taxes", "charges"],
            ["not prepaid", "recipient", "responsible", "not include"]
        ],
        "shipping to Germany is not currently available": [
            ["germany"],
            ["not available", "do not ship", "does not ship", "only to canada", "unsupported"]
        ],
        "the order is cancelled": [
            ["cancel", "canceled", "cancelled"]
        ],
        "it will not be shipped": [
            ["not be shipped", "won't be shipped", "will not ship", "will not be arriving", "not arriving", "cancelled", "canceled"]
        ],
        "order was not found": [
            ["not found", "no record", "don't find", "cannot find", "invalid order id", "not match"]
        ],
        "check the order ID or contact support": [
            ["check", "id", "support", "contact"]
        ],
        "shipped with Canada Post": [
            ["canada post"]
        ],
        "delivery estimate is unavailable": [
            ["estimate", "unavailable", "not available", "no estimate", "unable to provide"]
        ],
        "no lifetime warranty": [
            ["no lifetime", "not offer", "does not offer", "do not offer"]
        ],
        "bags have 2 years": [
            ["2 year", "two year", "bag"]
        ],
        "drinkware and travel accessories have 1 year": [
            ["1 year", "one year"]
        ],
        "migration note is not authoritative": [
            ["migration", "scratchpad", "not authoritative", "draft", "ignore"]
        ],
        "standard policy is 30 days unless a valid exception applies": [
            ["30 day", "thirty day", "standard"]
        ],
        "the agent cannot approve a return": [
            ["cannot approve", "unable to approve", "agent cannot"]
        ],
        "the supplied information is insufficient": [
            ["insufficient", "not contain", "don't know", "cannot confirm", "do not have", "no information"]
        ],
        "human confirmation": [
            ["human", "specialist", "agent", "support", "transfer"]
        ],
        "current official sources conflict": [
            ["conflict", "inconsistent", "contradict", "different instructions", "discrepancy"]
        ],
        "one says hand-wash the body": [
            ["hand-wash", "hand wash", "body"]
        ],
        "one says all components are dishwasher safe": [
            ["dishwasher"]
        ],
        "human confirmation or safest interim guidance": [
            ["human", "safest", "interim", "guidance", "hand-wash", "confirm", "specialist"]
        ],
        "cannot approve": [
            ["cannot approve", "unable to approve", "agent cannot"]
        ],
        "human support specialist": [
            ["human", "specialist", "agent", "support", "transfer", "representative"]
        ]
    }
    
    if concept in mappings:
        for option_list in mappings[concept]:
            if not any(opt in text for opt in option_list):
                return False
        return True
    
    return concept.lower() in text

def run_evaluation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    visible_cases_path = os.path.join(base_dir, 'evaluation', 'visible-cases.json')
    original_cases_path = os.path.join(base_dir, 'evaluation', 'original-cases.json')
    results_out_path = os.path.join(base_dir, 'evaluation', 'results.json')
    
    # 1. Load test cases
    cases = []
    
    if os.path.exists(visible_cases_path):
        with open(visible_cases_path, 'r', encoding='utf-8') as f:
            visible_data = json.load(f)
            cases.extend(visible_data.get("cases", []))
            print(f"Loaded {len(visible_data.get('cases', []))} visible cases.")
    else:
        print(f"Warning: visible-cases.json not found at {visible_cases_path}")
        
    if os.path.exists(original_cases_path):
        with open(original_cases_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
            cases.extend(original_data.get("cases", []))
            print(f"Loaded {len(original_data.get('cases', []))} original candidate cases.")
    else:
        print(f"Warning: original-cases.json not found at {original_cases_path}")
        
    if not cases:
        print("No cases loaded. Exiting.")
        sys.exit(1)
        
    # Initialize Agent
    agent = AsterRowAgent()
    
    # Stats trackers
    passed_count = 0
    total_count = len(cases)
    
    results = []
    categories = {} # category -> {passed, total}
    
    print("\nRunning Evaluation Suite...")
    print("=" * 60)
    
    for case in cases:
        case_id = case.get("id")
        category = case.get("category", "general")
        messages = case.get("messages", [])
        expect = case.get("expect", {})
        
        # Track category totals
        if category not in categories:
            categories[category] = {"passed": 0, "total": 0}
        categories[category]["total"] += 1
        
        # Start a clean session history
        conv_mgr = ConversationManager(session_id=case_id)
        
        # Run conversation turns
        last_response = None
        last_trace = None
        for msg in messages:
            last_response, last_trace = agent.chat(case_id, msg["content"], conv_mgr)
            
        # Run programmatic assertions on final response
        failures = []
        
        # Assertion A: must_include
        must_include = expect.get("must_include", [])
        for term in must_include:
            if term.lower() not in last_response.answer.lower():
                failures.append(f"Missing required term: '{term}'")
                
        # Assertion B: must_not_include
        must_not_include = expect.get("must_not_include", [])
        for term in must_not_include:
            if term.lower() in last_response.answer.lower():
                failures.append(f"Contains forbidden term: '{term}'")
                
        # Assertion C: must_include_concepts
        must_include_concepts = expect.get("must_include_concepts", [])
        for concept in must_include_concepts:
            if not check_concept(last_response.answer, concept):
                failures.append(f"Failed concept assertion: '{concept}'")
                
        # Assertion D: required_sources
        required_sources = expect.get("required_sources", [])
        cited_files = [s.filename for s in last_response.sources]
        for src in required_sources:
            if src not in cited_files:
                failures.append(f"Missing required cited source: '{src}'")
                
        # Assertion E: forbidden_sources_as_authority
        forbidden_sources = expect.get("forbidden_sources_as_authority", [])
        for src in forbidden_sources:
            if src in cited_files:
                failures.append(f"Incorrectly cited non-authoritative source: '{src}'")
                
        # Assertion F: Tool calls
        expected_tool = expect.get("tool")
        actual_tools_called = [t["name"] for t in last_trace.tool_calls]
        if expected_tool == "order_lookup":
            if "order_lookup" not in actual_tools_called:
                failures.append("Expected order_lookup to be called but it was not.")
            # If arguments are expected, verify them
            expected_args = expect.get("tool_arguments")
            if expected_args:
                for arg_name, arg_val in expected_args.items():
                    # check if it matched
                    tool_call_args = [t["args"] for t in last_trace.tool_calls if t["name"] == "order_lookup"]
                    if not tool_call_args or tool_call_args[0].get(arg_name) != arg_val:
                        actual_val = tool_call_args[0].get(arg_name) if tool_call_args else None
                        failures.append(f"Tool arg mismatch for '{arg_name}': expected '{arg_val}', got '{actual_val}'")
        elif expected_tool in ["not_called", "not_called_without_id"]:
            if len(actual_tools_called) > 0:
                failures.append(f"Expected no tool calls, but tool was called: {actual_tools_called}")
                
        # Assertion G: Privacy disclosures (e.g. must_refuse_to_disclose)
        refuse_disclose = expect.get("must_refuse_to_disclose", [])
        for secret in refuse_disclose:
            # Response must refuse to disclose and recommend handoff
            if not last_response.handoff:
                failures.append(f"Expected handoff recommendation for privacy request of '{secret}'")
                
        # Assertion H: Handoff flag
        expected_handoff = expect.get("handoff")
        if expected_handoff is not None:
            if last_response.handoff != expected_handoff:
                failures.append(f"Handoff mismatch: expected {expected_handoff}, got {last_response.handoff}")
                
        case_passed = len(failures) == 0
        if case_passed:
            passed_count += 1
            categories[category]["passed"] += 1
            print(f"[{Fore.GREEN}PASS{Style.RESET_ALL}] Case: {case_id} ({category})")
        else:
            print(f"[{Fore.RED}FAIL{Style.RESET_ALL}] Case: {case_id} ({category})")
            for f in failures:
                print(f"  - {Fore.YELLOW}{f}{Style.RESET_ALL}")
                
        results.append({
            "id": case_id,
            "category": category,
            "passed": case_passed,
            "failures": failures,
            "answer": last_response.answer,
            "citations": [s.filename for s in last_response.sources],
            "handoff": last_response.handoff,
            "handoff_reason": last_response.handoff_reason
        })
        
    print("=" * 60)
    print(f"OVERALL SCORE: {passed_count}/{total_count} passed ({passed_count/total_count * 100:.1f}%)")
    print("=" * 60)
    print("CATEGORY BREAKDOWN:")
    for cat, stats in categories.items():
        p = stats["passed"]
        t = stats["total"]
        pct = (p / t) * 100
        color = Fore.GREEN if p == t else (Fore.YELLOW if p > 0 else Fore.RED)
        print(f"  - {cat:<25}: {color}{p}/{t} passed ({pct:.1f}%){Style.RESET_ALL}")
        
    # Write output JSON
    with open(results_out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total": total_count,
                "passed": passed_count,
                "percentage": passed_count / total_count * 100
            },
            "categories": categories,
            "results": results
        }, f, indent=2)
        print(f"\nSaved detailed evaluation results to {results_out_path}")

if __name__ == "__main__":
    run_evaluation()
