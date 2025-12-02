import re
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# ===============================
# SAFETY GUARDRAILS
# ===============================

BLOCKED_WORDS = [
    "kill", "hack", "suicide", "bomb", "terrorist",
    "child abuse", "drug making"
]

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"disable safety guard",
    r"system prompt",
    r"you are now"
]

PII_PATTERNS = [
    (r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "email"),
    (r"\b\d{10}\b", "phone number"),
    (r"\b\d{16}\b", "credit card"),
]

HALLUCINATION_TRIGGERS = [
    "100% proven", "guaranteed truth",
    "officially confirmed", "without doubt"
]

def check_harmful(text: str):
    for w in BLOCKED_WORDS:
        if w.lower() in text.lower():
            return f"Harmful keyword detected: '{w}'"
    return None

def detect_pii(text: str):
    for pattern, kind in PII_PATTERNS:
        if re.search(pattern, text):
            return f"PII detected ({kind})"
    return None

def clean_prompt_injection(text: str):
    for pattern in PROMPT_INJECTION_PATTERNS:
        text = re.sub(pattern, "[REMOVED]", text, flags=re.IGNORECASE)
    return text

def detect_hallucination_risk(text: str):
    for key in HALLUCINATION_TRIGGERS:
        if key.lower() in text.lower():
            return True
    return False

def estimate_confidence(text: str):
    """
    Simple heuristic for confidence:
    - Presence of uncertain words lowers confidence
    """
    uncertain_words = ["maybe", "possibly", "might", "I think", "uncertain", "could"]
    score = 1.0
    for uw in uncertain_words:
        if uw.lower() in text.lower():
            score -= 0.2
    return max(0.0, min(1.0, score))  # normalize between 0.0 and 1.0

# ===============================
# MODEL
# ===============================
model = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

llm = ChatHuggingFace(
    llm=model,
    temperature=0.5,
    max_new_tokens=512
)

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(layout="wide")
st.title("🛡 Guarded AI Chatbot with Confidence Score")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show message history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if "confidence" in m:
            st.caption(f"Confidence: {m['confidence']*100:.1f}%")

user_input = st.chat_input("Say something safe...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 1️⃣ Pre-checks
    block = check_harmful(user_input)
    pii = detect_pii(user_input)

    if block or pii:
        safe_error = f"🚫 Safety Block: {block or pii}"
        st.session_state["messages"].append({"role": "assistant", "content": safe_error, "confidence": 0.0})
        with st.chat_message("assistant"):
            st.error(safe_error)
            st.caption("Confidence: 0%")
    else:
        sanitized = clean_prompt_injection(user_input)
        response = llm.invoke([HumanMessage(content=sanitized)])
        ai_text = response.content

        # 2️⃣ Post-check: hallucination warning
        if detect_hallucination_risk(ai_text):
            ai_text += "\n\n⚠ This contains claims that may require verification."

        # 3️⃣ Estimate confidence
        conf_score = estimate_confidence(ai_text)

        st.session_state["messages"].append({"role": "assistant", "content": ai_text, "confidence": conf_score})
        with st.chat_message("assistant"):
            st.write(ai_text)
            st.caption(f"Confidence: {conf_score*100:.1f}%")
