import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------------------------
# Config
# -------------------------------------------------
load_dotenv()

st.set_page_config(page_title="HITL PDF RAG", layout="wide")
st.title("📄 Human-in-the-Loop PDF RAG")

VECTOR_DIR = "vectorstore"

# -------------------------------------------------
# Upload PDF
# -------------------------------------------------
uploaded_pdf = st.file_uploader("📤 Upload a PDF", type=["pdf"])

if not uploaded_pdf:
    st.info("Please upload a PDF to continue")
    st.stop()

# -------------------------------------------------
# Save PDF temporarily
# -------------------------------------------------
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_pdf.read())
    pdf_path = tmp.name

# -------------------------------------------------
# Load PDF
# -------------------------------------------------
loader = PyPDFLoader(pdf_path)
documents = loader.load()
st.success(f"Loaded {len(documents)} pages")

# -------------------------------------------------
# Split text
# -------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
st.info(f"Created {len(chunks)} text chunks")

# -------------------------------------------------
# Vector Store (cached per session)
# -------------------------------------------------
@st.cache_resource
def build_vectorstore(_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DIR
    )

vectorstore = build_vectorstore(chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -------------------------------------------------
# LLM
# -------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    streaming=True
)

# -------------------------------------------------
# Human-in-the-Loop State
# -------------------------------------------------
if "docs" not in st.session_state:
    st.session_state.docs = None
    st.session_state.approved = False
    st.session_state.final_context = ""

# -------------------------------------------------
# Ask Question
# -------------------------------------------------
query = st.text_input("❓ Ask a question from the PDF")

if st.button("🔍 Retrieve Context"):
    st.session_state.docs = retriever.invoke(query)
    st.session_state.approved = False

# -------------------------------------------------
# Human Review Step
# -------------------------------------------------
if st.session_state.docs and not st.session_state.approved:

    st.subheader("🔍 Retrieved Context (Human Review Required)")

    combined_context = ""
    for i, d in enumerate(st.session_state.docs, 1):
        st.markdown(f"**Chunk {i} (Page {d.metadata.get('page', 'N/A')}):**")
        st.write(d.page_content)
        combined_context += d.page_content + "\n\n"

    edited_context = st.text_area(
        "✏️ Edit / clean the context before approval",
        value=combined_context,
        height=300
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve & Generate Answer"):
            st.session_state.approved = True
            st.session_state.final_context = edited_context

    with col2:
        if st.button("❌ Reject Context"):
            st.session_state.docs = None
            st.warning("Context rejected. Please retrieve again.")

# -------------------------------------------------
# Final Answer (After Human Approval)
# -------------------------------------------------
if st.session_state.approved:

    st.subheader("🤖 Final Answer")

    prompt = f"""
    Use ONLY the approved context to answer the question.

    Context:
    {st.session_state.final_context}

    Question:
    {query}

    Answer clearly and accurately.
    """

    placeholder = st.empty()
    answer = ""

    with st.spinner("AI is answering..."):
        for chunk in llm.stream(prompt):
            if chunk.content:
                answer += chunk.content
                placeholder.markdown(answer)
