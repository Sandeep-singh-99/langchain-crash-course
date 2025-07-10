from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Simple Add Math Operation", page_icon=":robot:")
st.title("Simple Add Math Operation with Google Generative AI")

input_text = st.text_input("Enter a simple math operation (e.g., 2 + 2)")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that performs only addition math operations. if the input is not a addition operation, respond with 'I can only perform addition operations.'"),
    ("user", "{input_text}"),
]).partial(input_text=input_text)


llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if st.button("Calculate") and input_text.strip():
    with st.spinner("Calculating..."):
        result = chain.invoke({"input_text": input_text})
    st.success(f"Result: {result}")
    st.write(result)