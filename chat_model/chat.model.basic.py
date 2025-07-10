from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Chat Model Basic", page_icon=":robot:")
st.title("Chat Model Basic with Google")

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

result = model.invoke("What is the capital of France?")

st.write("Full Result")
st.write(result)
st.write("Result Content")
st.write(result.content)