from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Text Summarizer", layout="centered")
st.title("Text Summarizer")

input_text = st.text_area("Enter text to summarize:", height=300)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes text clearly and concisely."),
    ("user", "Summarize the following text: {input_text}"),
])

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
output_parser = StrOutputParser()


# combine into chain
chain = prompt | llm | output_parser


if st.button("Summarize") and input_text.strip():
    with st.spinner("Summarizing..."):
        summary = chain.invoke({"input_text": input_text})
    st.success("Summary generated!")
    st.write(summary)