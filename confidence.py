import json
import streamlit as st
from typing import Dict, Any
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()


# ---------------- CONFIG ----------------
REPO_ID = "openai/gpt-oss-20b"
TASK = "text-generation"
GEN_TEMP = 1.1
VER_TEMP = 0.0
MAX_NEW_TOKENS = 500


# ---------------- HELPERS ----------------
def get_llm(temp: float):
    endpoint = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        task=TASK,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=temp
    )
    return ChatHuggingFace(llm=endpoint, temperature=temp)


def call_llm(llm: ChatHuggingFace, system: str, user: str) -> str:
    out = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user)
    ])
    return out.content.strip()


def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1:
            try:
                return json.loads(text[s:e+1])
            except:
                pass
    return {"confidence": 0.0, "reason": "Could not parse verifier JSON", "corrected": ""}


# ---------------- PROMPTS ----------------
VERIFY_PROMPT = """
Return ONLY JSON:
{{
  "confidence": <0.0-1.0>,
  "reason": "<very short reason>",
  "corrected": "<improved answer if needed, else repeat original>"
}}

User prompt:
{user_prompt}

Answer:
{assistant_answer}
"""


# ---------------- CORE FLOW ----------------
def generate_and_verify(user_prompt: str):
    # 1) Generate
    gen_llm = get_llm(GEN_TEMP)
    answer = call_llm(gen_llm, "You are a helpful assistant.", user_prompt)

    # 2) Verify
    ver_llm = get_llm(VER_TEMP)
    verify_input = VERIFY_PROMPT.format(user_prompt=user_prompt, assistant_answer=answer)
    ver_raw = call_llm(ver_llm, "Return only JSON.", verify_input)
    ver = parse_json(ver_raw)

    final_answer = ver.get("corrected", answer)
    confidence = ver.get("confidence", 0.0)
    reason = ver.get("reason", "")

    return final_answer, confidence, reason


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Simple Llama Chat + Confidence", layout="wide")
st.title("🤖 Llama Chatbot with Confidence Score")

prompt = st.text_input("Ask something:")

if st.button("Send") and prompt.strip():

    with st.spinner("Thinking..."):
        final_answer, confidence, reason = generate_and_verify(prompt)

    st.subheader("Response")
    st.write(final_answer)

    st.markdown("### Confidence")
    st.markdown(f"**{confidence:.2f}**")

    st.markdown("### Reason (short)")
    st.markdown(f"**{reason}**")
