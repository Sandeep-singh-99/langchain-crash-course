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
st.title("RAG Basics with LangChain and Google GenAI")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "ben.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Please create it first.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the document from the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display the number of chunks created
    st.write("--- Document Chunks Information ---")
    st.write(f"Total number of chunks: {len(docs)}")
    st.write(f"First chunk: {docs[0].page_content}...")

    # Create Embeddings
    st.write("--- Creating Embeddings ---")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.write("Embeddings created successfully.")

    # Create a Vector Store
    st.write("--- Creating Vector Store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    st.write("Vector store created successfully.")
else:
    st.write("Persistent directory already exists. Loading existing vector store.")
    db = Chroma(persist_directory=persistent_directory, embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-1.5-flash"))