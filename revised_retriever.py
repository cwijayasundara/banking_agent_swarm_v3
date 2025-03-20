from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
import logging

_ = load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

VECTOR_STORE_PATH= "./vector_store"
VECTOR_STORE_NAME = "bank_accounts"

google_api_key = os.getenv("GOOGLE_API_KEY")

llm = init_chat_model("gemini-1.5-pro", 
                      model_provider="google_genai", 
                      api_key=google_api_key)

vector_store = Chroma(
    collection_name=VECTOR_STORE_NAME,
    embedding_function=embeddings,
    persist_directory=VECTOR_STORE_PATH,
)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    logging.info(f"Inside retrieve tool: {query}")
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

memory = MemorySaver()
  
# agents
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
config = {"configurable": {"thread_id": "def234"}}

def execute_rag_agent(input_message):
    """Execute the RAG agent with the input message"""
    logging.info(f"Inside execute_rag_agent: {input_message}")
    response = agent_executor.invoke(
        {"messages": [{"role": "user", "content": input_message}]},
        config=config,
    )
    logging.info(f"Response from agent_executor: {response}")
    # Access the last message in the response
    ai_message = None
    if "messages" in response:
        for message in response["messages"]:
            # Check if it's an AIMessage instance or has a type/role attribute
            if isinstance(message, AIMessage):
                ai_message = message
            elif hasattr(message, "type") and message.type == "ai":
                ai_message = message
            elif hasattr(message, "role") and message.role == "assistant":
                ai_message = message
    else:
        logging.error(f"Invalid response from agent_executor: {response}")
    
    return ai_message

# input_message = (
#     "Whats the Annual Gross/AER for Online Fixed Bond - 1 Year accounts?"
# )

# response = execute_rag_agent(input_message)
# print(response.content)




