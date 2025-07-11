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

# Define the directory for the text files and the persistent storage
current_dir = os.path.dirname(os.path.abspath(__file__))
text_files_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_dir = os.path.join(db_dir, "chroma_db_with_metadata")

st.write("Books Directory:", text_files_dir)
st.write("Persistent Directory:", persistent_dir)

# check id the chroma vector store already exists
if not os.path.exists(persistent_dir):
    st.write("Chroma vector store does not exist. Creating a new one...")

    # Ensure the books directory exists
    if not os.path.exists(text_files_dir):
        raise FileNotFoundError(f"Books directory '{text_files_dir}' does not exist. Please check the path.")
    
    # List all text files in the directory
    books_files = [f for f in os.listdir(text_files_dir) if f.endswith('.txt')]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in books_files:
        file_path = os.path.join(text_files_dir, book_file)
        loader = TextLoader(file_path)
        book_doc = loader.load()
        for doc in book_doc:
            # Add metadata to each document including the file name
            doc.metadata = {"source": book_file}
            documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    st.write("--- Document Chunks Information ---")
    st.write(f"Total number of documents: {len(docs)}")

    # Create embeddings 
    st.write("Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.write("Embeddings created successfully.")

    # Create a Chroma vector store with metadata
    st.write("Creating Chroma vector store...")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_dir)
    st.write("Chroma vector store created successfully.")
else:
    st.write("Vector store already exists. Loading the existing vector store...")