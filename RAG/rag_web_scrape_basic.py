import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import ( RecursiveCharacterTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter, TextSplitter, TokenTextSplitter )
from langchain.vectorstores import Chroma


load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="RAG Web Scrape Basic", page_icon=":books:", layout="wide")
st.title("RAG Web Scrape Basic")


current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "rag_web_scrape_basic")

# Scrape the content from example.com using webBaseLoader
# WebBaseLoader loads web pages and extracts text content.

url = ["https://www.apple.com/"]

# Create a loader for the web content
loader = WebBaseLoader(url)
documents = loader.load()


# split the scraped documents into chunks
# Character Text Splitter splits text into smaller chunks

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Display information about the documents
st.write("---- Scraped Documents ----")
st.write(f"Number of documents: {len(docs)}")
st.write(f"Sample chunk: {docs[0].page_content}")

# create embeddings for the documents
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# create and persist the vector store with the embeddings
# chroma stores the embeddings for efficient search and retrieval

if  not os.path.exists(persistent_directory):
    st.write("--- Creating vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    st.write(f"Vector store created at: {persistent_directory}")
else:
    st.write(f"Vector store {persistent_directory} already exists. Loading existing vector store.")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)


# query the vector store
# create a retriever for querying the vector store
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})


input_query = st.text_input("Enter a query to search the vector store:")
relevant_docs = retriever.invoke(input_query)


st.write("--- Retrieved Documents ---")
for i, doc in enumerate(relevant_docs):
    st.write(f"Document {i}: {doc.page_content}")
    if doc.metadata:
        st.write(f"Source: {doc.metadata.get('source', 'No source metadata available')}")

