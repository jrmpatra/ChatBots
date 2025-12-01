# backend_database.py
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional, List
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
import sqlite3, uuid
import streamlit as st
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool


DB_PATH = "chat_history.db"

# ----------------- SQLite connection -----------------

conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# Simple table for thread metadata (id + display name)
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS threads (
        thread_id TEXT PRIMARY KEY,
        name      TEXT
    )
    """
)
conn.commit()

# ----------------- Thread helper functions -----------------


def retrieve_all_threads() -> list[str]:
    """Return all thread_ids in creation order."""
    cur = conn.execute("SELECT thread_id FROM threads ORDER BY rowid ASC")
    return [row[0] for row in cur.fetchall()]


def ensure_thread(thread_id: str, default_name: str) -> None:
    """Insert a thread row if it doesn't already exist."""
    conn.execute(
        "INSERT OR IGNORE INTO threads (thread_id, name) VALUES (?, ?)",
        (thread_id, default_name),
    )
    conn.commit()


def get_thread_name(thread_id: str) -> Optional[str]:
    cur = conn.execute("SELECT name FROM threads WHERE thread_id = ?", (thread_id,))
    row = cur.fetchone()
    return row[0] if row else None


def set_thread_name(thread_id: str, name: str) -> None:
    conn.execute(
        "UPDATE threads SET name = ? WHERE thread_id = ?",
        (name, thread_id),
    )
    conn.commit()


def delete_chat_thread(thread_id: str):
    """
    Permanently delete a chat thread:
    - Remove all LangGraph checkpoints for this thread
    - Remove the thread row from chat_threads table
    """
    # Delete checkpoints
    checkpointer.delete_thread(thread_id)

    # Delete metadata
    cur = sqllite_conn.cursor()
    cur.execute("DELETE FROM chat_threads WHERE thread_id=?", (thread_id,))
    sqllite_conn.commit()


# ----------------- LangGraph checkpointer -----------------

checkpointer = SqliteSaver(conn)


# ----------------- Tools -----------------


###Tools
search_tool = Tool(
    name="Search",
    func=DuckDuckGoSearchRun().run,
    description="Useful for when you need to answer questions about current events or the world."
)

@tool
def calculator(first_number: float, second_number: float, operation: str) -> str:
    """Performs basic arithmetic operations on two numbers.
    Args:
        first_number (float): The first number.
        second_number (float): The second number.
        operation (str): The operation to perform: add, subtract, multiply, divide.
        """
    if operation == "add":
        return str(first_number + second_number)
    elif operation == "subtract":
        return str(first_number - second_number)
    elif operation == "multiply":
        return str(first_number * second_number)
    elif operation == "divide":
        if second_number == 0:
            return "Error: Division by zero."
        return str(first_number / second_number)
    else:
        return "Error: Unsupported operation. Please use add, subtract, multiply, or divide."


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetches the current stock price for a given symbol.
    Args:
        symbol (str): The stock symbol to look up.  """
    API_KEY = "PH3EU2KJ7EKTPBW4"

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
    response = requests.get(url)
    return response.json()
    



@tool
def echo_tool(text: str) -> str:
    """Echo back the given text. Useful for testing tools."""
    return f"Echo: {text}"


@tool
def download_conversation_tool(thread_id: str) -> str:
    """
    Prepare a downloadable transcript for the given thread_id.

    Returns a JSON string:
    {
      "file_name": "<name>.txt",
      "content": "<plain text transcript>"
    }

    UI layer will parse this JSON and show a download button.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    # Use LangGraph state for this thread
    from backend_database import chatbot  # safe global import reference

    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    lines: List[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            # Skip system/tool messages in the transcript
            continue
        lines.append(f"{role}: {msg.content}")

    transcript = "\n\n".join(lines)
    title = get_thread_name(thread_id) or "chat"
    file_name = f"{title}.txt"

    payload = {"file_name": file_name, "content": transcript}
    # Return as JSON string so UI can parse from ToolMessage.content
    return json.dumps(payload)


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]




# ----------------- LLM + ReAct Agent -----------------

model = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
llm = ChatHuggingFace(llm=model, temperature=1.100)


prompt_text = (
    "You are a helpful assistant running inside a multi-chat UI.\n"
    "- Each conversation has a unique `thread_id` provided in a system message.\n"
    "- If the user says they want to download the chat, export, save the conversation, "
    "or similar, you MUST call the tool `download_conversation_tool` with that "
    "exact thread_id string.\n"
    "- For normal questions, answer directly without using tools.\n"
    "Be concise and helpful."
)

chatbot = create_react_agent(
    model=llm,
    tools=[echo_tool, download_conversation_tool,search_tool, calculator, get_stock_price],
    prompt=prompt_text,
    checkpointer=checkpointer,
)


# ----------------- End of backend_database.py -----------------
