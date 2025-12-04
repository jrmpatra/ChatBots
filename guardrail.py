# guarded_ai_agent.py
import re
import json
import logging
from datetime import datetime
from typing import Callable, Any

# Simulated LLM response (replace with your actual LLM API call)
class MockLLM:
    def __init__(self):
        pass
    
    def generate(self, prompt: str):
        # For demo purposes, returns fixed response + confidence
        return {
            "text": "This is a demo response. No sensitive info here.",
            "confidence": 0.92,  # Confidence score between 0-1
            "claims": ["Demo claim 1", "Demo claim 2"]
        }

# -------------------------------
# Setup Logging / Audit
# -------------------------------
logging.basicConfig(
    filename="agent_audit.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def audit_log(event: str, data: dict):
    log_entry = {"event": event, "timestamp": datetime.utcnow().isoformat(), "data": data}
    logging.info(json.dumps(log_entry))

# -------------------------------
# Input Layer Guardrails
# -------------------------------
def detect_prompt_injection(user_input: str) -> bool:
    injection_patterns = [
        r"ignore instructions",
        r"bypass safety",
        r"malicious code"
    ]
    for p in injection_patterns:
        if re.search(p, user_input, re.IGNORECASE):
            return True
    return False

def redact_pii(text: str) -> str:
    # Simple regex for emails, phone numbers, SSN
    patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{10}\b",             # phone numbers
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"  # email
    ]
    for pattern in patterns:
        text = re.sub(pattern, "[REDACTED]", text)
    return text

def dangerous_keywords(user_input: str) -> bool:
    keywords = ["hack", "illegal", "exploit", "weapon", "attack"]
    for word in keywords:
        if word.lower() in user_input.lower():
            return True
    return False

def sanitize_input(user_input: str) -> str:
    if detect_prompt_injection(user_input):
        audit_log("blocked_input", {"reason": "prompt_injection", "input": user_input})
        raise ValueError("Blocked input: prompt injection detected")
    if dangerous_keywords(user_input):
        audit_log("blocked_input", {"reason": "dangerous_keyword", "input": user_input})
        raise ValueError("Blocked input: dangerous keyword detected")
    sanitized = redact_pii(user_input)
    audit_log("input_sanitized", {"original": user_input, "sanitized": sanitized})
    return sanitized

# -------------------------------
# Agent Reasoning Layer
# -------------------------------
CONFIDENCE_THRESHOLD = 0.8

def agent_reasoning(sanitized_input: str, llm: MockLLM) -> dict:
    response = llm.generate(sanitized_input)
    text = response.get("text", "")
    confidence = response.get("confidence", 0.0)
    claims = response.get("claims", [])
    
    if confidence < CONFIDENCE_THRESHOLD:
        audit_log("blocked_output", {"reason": "low_confidence", "confidence": confidence, "text": text})
        raise ValueError("Blocked output: confidence too low")
    
    audit_log("agent_response", {"text": text, "confidence": confidence, "claims": claims})
    return response

# -------------------------------
# Output Layer Guardrails
# -------------------------------
def unsafe_content_check(text: str) -> bool:
    unsafe_keywords = ["hack", "illegal", "exploit", "attack"]
    return any(word.lower() in text.lower() for word in unsafe_keywords)

def detect_hallucinations(claims: list) -> list:
    # For demo, mark all claims as grounded = False
    flagged_claims = [{"claim": c, "grounded": False} for c in claims]
    return flagged_claims

def sanitize_output(response: dict) -> dict:
    text = redact_pii(response["text"])
    if unsafe_content_check(text):
        audit_log("blocked_output", {"reason": "unsafe_content", "text": text})
        raise ValueError("Blocked output: unsafe content detected")
    response["text"] = text
    response["claims"] = detect_hallucinations(response.get("claims", []))
    audit_log("output_sanitized", response)
    return response

# -------------------------------
# Action Layer (Optional)
# -------------------------------
def safe_execute(action: Callable, *args, **kwargs) -> Any:
    approval = input(f"Do you approve executing {action.__name__}? (yes/no): ")
    if approval.lower() != "yes":
        audit_log("action_blocked", {"action": action.__name__})
        print(f"Action {action.__name__} blocked by user")
        return None
    audit_log("action_executed", {"action": action.__name__, "args": args, "kwargs": kwargs})
    return action(*args, **kwargs)

# -------------------------------
# Full Agent Pipeline
# -------------------------------
def guarded_agent(user_input: str, llm: MockLLM) -> dict:
    try:
        sanitized = sanitize_input(user_input)
        response = agent_reasoning(sanitized, llm)
        final_response = sanitize_output(response)
        return final_response
    except ValueError as e:
        return {"error": str(e)}

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    llm = MockLLM()
    user_input = input("Enter your query: ")
    result = guarded_agent(user_input, llm)
    print("\nAgent Response:")
    print(json.dumps(result, indent=2))

    # Example critical action
    def delete_file_demo(filename):
        print(f"Deleted {filename} (demo)")
    
    safe_execute(delete_file_demo, "test.txt")
