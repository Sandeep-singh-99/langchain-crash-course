# import streamlit as st
# from dotenv import load_dotenv
# import os
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain import hub
# from langchain.memory import ConversationBufferMemory
# from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
# from langchain.agents import ( AgentExecutor, create_structured_chat_agent )
# from langchain_core.tools import Tool

# load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# st.set_page_config(page_title="Agent React Chat", page_icon=":books:", layout="wide")
# st.title("Agent React Chat")


# def get_current_time(*args, **kwargs):
#     """Returns the current time in H:MM AM/PM format."""
#     import datetime

#     now = datetime.datetime.now()
#     return now.strftime("%I:%M %p")


# def search_wikipedia(query):
#     """Searches Wikipedia and return the summary of the first result."""
#     from wikipedia import summary

#     try:
#         return summary(query, sentences=2)
#     except:
#         return "I Couldn't find any information on that."
    

# # List the tools that the agent can use
# tools = [
#     Tool(
#         name="Current Time",
#         description="Useful for when you need to know the current time.",
#         func=get_current_time
#     ),
#     Tool(
#         name="Search Wikipedia",
#         description="Useful for when you need to know information about a topic.",
#         func=search_wikipedia
#     )
# ]

# prompt = hub.pull("hwchase17/react")

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     memory=memory,
#     verbose=True,
#     handle_parsing_errors=True,
# )

# initial_message = "You are an AI assistant that can provide helpful answers using available tools. you are unable to answer questions that require personal opinions or subjective judgments. Use the tools provided to answer questions when necessary."
# memory.chat_memory.add_message(SystemMessage(content=initial_message))

# while True:
#     user_input = st.text_input("You: ")
#     if user_input.lower() == "exit":
#         break

#     memory.chat_memory.add_message(HumanMessage(content=user_input))

#     response = agent_executor.invoke({ "input": user_input })
#     st.write("Bot: ", response["output"])

#     memory.chat_memory.add_messages(AIMessage(content=response["output"]))



import streamlit as st
from dotenv import load_dotenv
import os
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.tools import Tool
from wikipedia import summary

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Page setup
st.set_page_config(page_title="Agent React Chat", page_icon=":books:", layout="wide")
st.title("🤖 Agent React Chat with Tools")

# Functions as tools
def get_current_time(*args, **kwargs):
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def search_wikipedia(query):
    try:
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."

# Tool definitions
tools = [
    Tool(
        name="Current Time",
        description="Useful for when you need to know the current time.",
        func=get_current_time
    ),
    Tool(
        name="Search Wikipedia",
        description="Useful for when you need to know information about a topic.",
        func=search_wikipedia
    )
]

# Pull a valid prompt
prompt = hub.pull("hwchase17/react")  # <- you had this correct

# Setup LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Setup memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    initial_message = "You are an AI assistant that can provide helpful answers using available tools. You cannot give opinions or make subjective judgments."
    st.session_state.memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Setup agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=st.session_state.memory,
    verbose=True,
    handle_parsing_errors=True,
)

# Input box
user_input = st.text_input("You: ", key="input")

if user_input:
    st.session_state.memory.chat_memory.add_message(HumanMessage(content=user_input))
    response = agent_executor.invoke({ "input": user_input })
    bot_response = response["output"]
    st.session_state.memory.chat_memory.add_message(AIMessage(content=bot_response))
    st.write(f"**Bot:** {bot_response}")

# Optional: Display chat history
with st.expander("🔁 Chat History"):
    for msg in st.session_state.memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"**Bot:** {msg.content}")
        elif isinstance(msg, SystemMessage):
            st.markdown(f"**System:** {msg.content}")
