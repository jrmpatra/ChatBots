#pip install opentelemetry-api==1.39.0 opentelemetry-sdk==1.39.0 opentelemetry-exporter-otlp==1.39.0 opentelemetry-exporter-otlp-proto-grpc==1.39.0 opentelemetry-exporter-otlp-proto-http==1.39.0
## To Start  .\myenv\Scripts\python.exe -m phoenix.server.main serve
##pip install arize-phoenix

from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace

from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()


from dotenv import load_dotenv
load_dotenv()


# ------------------------------
# 1. Configure Phoenix Tracer
# ------------------------------
tracer_provider = register(
  project_name="your-next-llm-project",
  endpoint="http://localhost:6006/v1/traces",
  auto_instrument=True
)

# Instrument LangChain to automatically trace LLM calls
LangChainInstrumentor().instrument()

tracer = trace.get_tracer(__name__)

# ------------------------------
# 2. HuggingFace Llama model
# ------------------------------
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
model = ChatHuggingFace(llm=llm, temperature=1.100)

# ------------------------------
# 3. Single traced LLM call
# ------------------------------
def main():
    with tracer.start_as_current_span("hf_test_call"):
        message = HumanMessage(content="Explain what Phoenix tracing is.")
        result = model([message]).content
        print("\nLLM RESPONSE:\n", result)


if __name__ == "__main__":
    main()

