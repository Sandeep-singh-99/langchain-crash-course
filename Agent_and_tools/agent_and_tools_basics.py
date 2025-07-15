import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain import hub
from langchain.agents import ( AgentExecutor, create_react_agent )
from langchain_core.tools import Tool

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Agent and Tools Basics", page_icon=":books:", layout="wide")
st.title("Agent and Tools Basics")


def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

# List of tools available to the agent
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time."
    )
]

# Pull the prompt template from the hub
# React = Reason and Action
prompt = hub.pull("hwchase17/react")


# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# create the React agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# create the agent executor from the agent and tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

response = agent_executor.invoke({
    "input": "What time is it?",
})

st.success("Agent executed successfully!")
st.write("Response from agent:", response)
