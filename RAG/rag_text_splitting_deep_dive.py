import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import ( RecursiveCharacterTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter, TextSplitter, TokenTextSplitter )
from langchain.vectorstores import Chroma

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="RAG Text Splitting Deep Dive", page_icon=":books:", layout="wide")
st.title("RAG Text Splitting Deep Dive")

# Define the directory for the text files and the persistent storage
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "ben.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(file_path):
    st.error(f"File not found: {file_path}")

# Read the text content from the file
loader = TextLoader(file_path)
documents = loader.load()


# Define the embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# function to create a vector store 
def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        st.write("--- Creating vector store ---")
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        st.write(f"Vector store created at: {store_name}")
    else: 
        st.write(f"Vector store {store_name} already exists. Loading existing vector store.")


# Character Text Splitter
# Splits text into chunks based on a specified number of characters.
# Useful for consistent chunk sizes regardless of content structure.
st.write("### Character Text Splitter")
char_splitting = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitting.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char_split")

# Sentence-based Splitting
# Splits text into chunks based on sentence, ensuring that chunks end at sentence boundaries.
# Ideal for maintaining semantic coherence within chunks.
st.write("### Sentence Text Splitter")
sentence_splitting = SentenceTransformersTokenTextSplitter(chunk_size=1000, chunk_overlap=100)
sentence_docs = sentence_splitting.split_documents(documents)
create_vector_store(sentence_docs, "chroma_db_sentence_split")

# Token-based Splitting
# Splits text into chunks based on tokens, using tokenization like GPT-2.
# Useful for transformers models with strict token limits.
st.write("### Token Text Splitter")
token_splitting = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
token_docs = token_splitting.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token_split")

# Recursive Character Text Splitter
# Attempts to split text at natural boundaries (like paragraphs) while respecting character limits.
# Balance between chunk size and content structure.
st.write("### Recursive Character Text Splitter")
recursive_splitting = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
recursive_docs = recursive_splitting.split_documents(documents)
create_vector_store(recursive_docs, "chroma_db_recursive_split")


# Custom Text Splitter
# Allows creating custom splitting logic based on specific requirements.
# Useful for documents with unique structures that standard splitters can't handle.
st.write("### Custom Text Splitter")

class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        # Implement custom splitting logic here
        return text.split("\n\n")  # Example: split by double newlines

CustomTextSplitter = CustomTextSplitter()
custom_docs = CustomTextSplitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom_split")

# Function to query a vector store
def query_vector_store(store_name, query):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        st.write(f"--- Querying vector store: {store_name} ---")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)

        st.success(f"--- Retrieved Documents {store_name} ---")
        for i,doc in enumerate(relevant_docs):
            st.write(f"Document {i}: {doc.page_content}")
            if doc.metadata:
                st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
    else:
        st.error(f"Vector store {store_name} does not exist.")


input = st.text_input("Enter your query:")

query_vector_store("chroma_db_char_split", input)
query_vector_store("chroma_db_sentence_split", input)
query_vector_store("chroma_db_token_split", input)
query_vector_store("chroma_db_recursive_split", input)
query_vector_store("chroma_db_custom_split", input)
