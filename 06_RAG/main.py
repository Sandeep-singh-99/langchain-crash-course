import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Chat (HF Embeddings)", layout="wide")
st.title("📄 Chat with PDF using Hugging Face Embeddings")

# ---------------------------
# Load PDF
# ---------------------------
pdf_path = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_path:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_path.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    st.success(f"Loaded {len(documents)} pages")

    # ---------------------------
    # Split text
    # ---------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    st.info(f"Split into {len(chunks)} chunks")

    # ---------------------------
    # Hugging Face Embeddings
    # ---------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------------------
    # Vector Store
    # ---------------------------
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ---------------------------
    # LLM (Gemini for answering)
    # ---------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        streaming=True
    )

    # ---------------------------
    # Ask Question
    # ---------------------------
    query = st.text_input("Ask a question from the PDF")

    if st.button("Ask"):
        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        Use the following context to answer the question.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        placeholder = st.empty()
        answer = ""

        with st.spinner("🤔 AI is thinking..."):
            for chunk in llm.stream(prompt):
                if chunk.content:
                    answer += chunk.content
                    placeholder.markdown(answer)
