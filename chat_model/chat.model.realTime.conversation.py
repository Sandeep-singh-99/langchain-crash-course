from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st
import os

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit setup
st.set_page_config(page_title="Chat Model Basic Conversation", page_icon=":robot:")
st.title("Chat Model Basic Conversation with Google")

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Start conversation history
chat_history = [
    SystemMessage(content="You are a friendly assistant.")
]

query = st.text_input("Ask a question:")

if query:
    chat_history.append(HumanMessage(content=query))

  
    response_message = model.predict_messages(messages=chat_history)

    chat_history.append(AIMessage(content=response_message.content))

    st.success(response_message.content)

    st.write("------- Chat History -------")
    for message in chat_history:
        st.write(f"{message.type.capitalize()}: {message.content}")
