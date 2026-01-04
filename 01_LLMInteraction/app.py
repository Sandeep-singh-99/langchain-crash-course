import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LangChain Streaming", layout="centered")
st.title("🤖 LangChain Streaming (Gemini)")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    streaming=True
)

prompt = st.text_input(
    "Ask something",
    "What is the capital of France?"
)

if st.button("Ask", use_container_width=True):
    placeholder = st.empty()
    full_response = ""

    with st.spinner("🤔 AI is thinking..."):
        for chunk in model.stream(prompt):
            if chunk.content:  # IMPORTANT: avoid None
                full_response += chunk.content
                placeholder.markdown(full_response)
