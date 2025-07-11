import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="RAG Basics", page_icon=":books:", layout="wide")
st.title("RAG Basics Metadata")

# Define the persistent directory 

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# load the existing Chroma vector store with the embeddings
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

txt_input = st.text_input("Enter a query to search the vector store:")

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
revlevant_docs = retriever.invoke(txt_input)

# Display the retrieved documents with metadata

st.success("--- Retrieved Documents ---")
for i, doc in enumerate(revlevant_docs):
    st.write(f"Document {i}: {doc.page_content}")
    st.write(f"source: {doc.metadata['source']}")
