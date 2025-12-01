# Front_END.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3, uuid
import streamlit as st

load_dotenv()

st.set_page_config(layout="wide")

model = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
llm = ChatHuggingFace(llm=model, temperature=1.100)

# SQLite Connection
sqllite_conn = sqlite3.connect(database='chatbot_history_database.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=sqllite_conn)

# Create metadata table
def _init_threads_table():
    cur = sqllite_conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id TEXT PRIMARY KEY,
            name      TEXT
        )
        """
    )
    sqllite_conn.commit()

_init_threads_table()

# ---------------- Thread DB Helpers ----------------
def ensure_thread(thread_id: str, default_name: Optional[str] = None):
    cur = sqllite_conn.cursor()
    cur.execute("INSERT OR IGNORE INTO chat_threads(thread_id, name) VALUES (?, ?)", (thread_id, default_name))
    sqllite_conn.commit()

def set_thread_name(thread_id: str, name: str):
    cur = sqllite_conn.cursor()
    cur.execute("UPDATE chat_threads SET name = ? WHERE thread_id = ?", (name, thread_id))
    sqllite_conn.commit()

def get_thread_name(thread_id: str):
    cur = sqllite_conn.cursor()
    cur.execute("SELECT name FROM chat_threads WHERE thread_id = ?", (thread_id,))
    row = cur.fetchone()
    return row[0] if row else None

def retrieve_all_threads():
    for c in checkpointer.list(None):
        ensure_thread(c.config["configurable"]["thread_id"])
    cur = sqllite_conn.cursor()
    cur.execute("SELECT thread_id FROM chat_threads ORDER BY rowid")
    return [r[0] for r in cur.fetchall()]

def delete_chat_thread(thread_id: str):
    checkpointer.delete_thread(thread_id)
    cur = sqllite_conn.cursor()
    cur.execute("DELETE FROM chat_threads WHERE thread_id=?", (thread_id,))
    sqllite_conn.commit()

def delete_all_threads():
    for thread_id in retrieve_all_threads():
        delete_chat_thread(thread_id)

# ---------------- Chatbot Setup ----------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

# ---------------- UI Helpers ----------------
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    new_thread = generate_thread_id()
    ensure_thread(new_thread, f"Chat {len(st.session_state['chat_threads']) + 1}")
    st.session_state["thread_id"] = new_thread
    st.session_state["chat_threads"].append(new_thread)
    st.session_state["message_history"] = []

def clear_all_chats():
    delete_all_threads()
    st.session_state["chat_threads"] = []
    st.session_state.pop("thread_id", None)
    st.session_state["message_history"] = []
    st.rerun()

def load_conversation(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def format_chat(messages):
    return "\n\n".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in messages)

# ---------------- Session boot ----------------
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# ---------------- Sidebar ----------------
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat ➕"):
    reset_chat()

st.sidebar.header("My Conversations")

for tid in st.session_state["chat_threads"][::-1]:
    name = get_thread_name(tid) or tid

    # Create two columns for inline layout
    col1, col2 = st.sidebar.columns([4, 1])

    with col1:
        if st.button(name, key=f"open_{tid}"):
            st.session_state["thread_id"] = tid
            msgs = load_conversation(tid)
            conv = []
            for m in msgs:
                if isinstance(m, HumanMessage):
                    conv.append({"role": "user", "content": m.content})
                if isinstance(m, AIMessage):
                    conv.append({"role": "assistant", "content": m.content})
            st.session_state["message_history"] = conv

    with col2:
        if st.button("🗑", key=f"del_{tid}"):
            delete_chat_thread(tid)
            st.session_state["chat_threads"].remove(tid)
            st.rerun()

# ---------------- Main UI ----------------
if "thread_id" not in st.session_state:
    st.markdown("### Welcome! Start a chat from the sidebar.")
    st.stop()

tid = st.session_state["thread_id"]
title = get_thread_name(tid) or "Untitled"
new_title = st.text_input("", value=title, key=f"title_{tid}", label_visibility="collapsed")
if new_title != title: 
    set_thread_name(tid, new_title)

for m in st.session_state["message_history"]:
    with st.chat_message(m["role"]): 
        st.markdown(m["content"])

if st.session_state["message_history"]:
    st.download_button(
        "Download Chat ⬇️", 
        format_chat(st.session_state["message_history"]),
        file_name=f"{new_title}.txt", 
        mime="text/plain"
    )

# Input handling
user_input = st.chat_input("Type here")
if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"): 
        st.markdown(user_input)

    CONFIG = {"configurable": {"thread_id": tid}}

    response_chunks = []

    def stream():
        for chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG, 
            stream_mode="messages"
        ):
            text = chunk.content or ""
            response_chunks.append(text)
            yield text

    with st.chat_message("assistant"):
        st.write_stream(stream())

    ai_msg = "".join(response_chunks)
    st.session_state["message_history"].append({"role": "assistant", "content": ai_msg})
