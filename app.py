from dotenv import load_dotenv
from revised_retriever import execute_rag_agent
from pending_tx_pandas_agent import get_pending_tx_details
from customer_details_sql_agent import get_customer_details
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from datetime import datetime
from langchain.chat_models import init_chat_model
import streamlit as st
import os
import logging
from prompt_txt import supervisor_prompt

_ = load_dotenv()

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
os.environ["USER_AGENT"] = "banking_agents_v2/1.0"

today = datetime.now().strftime("%d/%m/%Y")

google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = init_chat_model("gemini-1.5-pro", 
                      model_provider="google_genai", 
                      api_key=google_api_key)

llm_thinking = init_chat_model("o3-mini", 
                      model_provider="openai", 
                      api_key=openai_api_key,
                      temperature=1.0)

# tools
def get_interest_rates_from_vector_store(query:str) -> list[str]:
    """Retrieve interest rates from the vector store."""
    logging.info(f"Inside get_interest_rates_from_vector_store: {query}")
    documents = execute_rag_agent(query)
    return documents

def get_pending_tx_details_from_pandas_agent(query:str) -> str:
    """Retrieve pending transactions details from the pandas agent."""
    logging.info(f"Inside get_pending_tx_details_from_pandas_agent: {query}")
    response = get_pending_tx_details(query)
    return response

def get_customer_details_from_sql_agent(query:str) -> str:
    """Retrieve customer details from the sql agent."""
    logging.info(f"Inside get_customer_details_from_sql_agent: {query}")
    response = get_customer_details(query)
    return response

# agents
interest_rate_agent = create_react_agent(
    model=llm,
    tools=[get_interest_rates_from_vector_store],
    name="interest_rate_agent",
    prompt="You are an interest rate agent. Use the tools provided to retrieve the interest rates from the vector store."
)

pending_tx_agent = create_react_agent(
    model=llm,
    tools=[get_pending_tx_details_from_pandas_agent],
    name="pending_tx_agent",
    prompt="You are a pending transactions agent. Use the tools provided to retrieve the pending transactions details from the pandas agent."
)

customer_details_agent = create_react_agent(
    model=llm,
    tools=[get_customer_details_from_sql_agent],
    name="customer_details_agent",
    prompt="You are a customer details agent. Use the tools provided to retrieve the customer details from the sql agent."
)

# supervisor_prompt = """
# You are a bank supervisor managing an interest_rate_agent, customer_details_agent and pending_tx_agent.
# - For interest rate related queries, use interest_rate_agent.
# - For pending transactions related queries, use pending_tx_agent. Pending_tx_agent's Pandas schema contains pending_tx_id,customer_id,pending_date,pending_amount.
# - For customer details related queries, use customer_details_agent. Customer_details_agent's schema contains id,first_name,last_name,address,account_balance,income,gender,date_of_birth.
# - For more complex queries, do use more than one agent to retrieve required information and pass the information to the next agent.
# Some examples below given as query, approach syntax.
# - Query: What is the sum of pending transactions of John Doe?
# - Approach: first get the customer_id from the customer_details_agent by using the customer first_name & last_name and then pass the customer_id to the pending_tx_agent as here customer first_name & last_name are not provided. 
# """

# supervisor - Process each query individually instead of all at once
workflow = create_supervisor(
    [interest_rate_agent, pending_tx_agent, customer_details_agent],
    model=llm_thinking,
    prompt=supervisor_prompt
)

app = workflow.compile()

st.title("Banking Assistant")

with st.sidebar:
    st.image("image/image.jpg", width=600)

query = st.text_input("How can I help you today?")

st.write("sample questions:")
st.write("RAG agent : Whats the Annual Gross/AER for Online Fixed Bond - 1 Year accounts?")
st.write("Pandas agent: What is the total amount of pending transactions for customer c004?")
st.write("SQL agent: List all the details of Olivia Stephenson?")
st.write("Supervisor: Whats the sum of all the pending transactions for Kerry Lewis?")

if st.button("Chat"):
    if query:
        with st.spinner("Thinking..."):
            response = app.invoke({
                "messages": [
                    {"role": "user", "content": query}
                ]
            })

            # Use the helper function to run the async function
            st.write(response["messages"][-1].content)
    else:
        st.warning("Please enter a query first.")


