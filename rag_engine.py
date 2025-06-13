import os
import tempfile
from pathlib import Path
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

# Constants
TMP_DIR = Path("data/tmp")
LOCAL_VECTOR_STORE_DIR = Path("data/vector_store")
LLAMA_DEFAULT_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(LOCAL_VECTOR_STORE_DIR, exist_ok=True)

st.set_page_config(page_title="RAG with LLaMA")
st.title("🔍 Retrieval-Augmented Generation (RAG) with LLaMA")

# Session init
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llama_path" not in st.session_state:
    st.session_state.llama_path = LLAMA_DEFAULT_PATH


def input_fields():
    with st.sidebar:
        st.session_state.source_docs = st.file_uploader("📄 Upload PDF Documents", type="pdf", accept_multiple_files=True)
        st.session_state.llama_path = st.text_input("📍 Path to LLaMA GGUF model", value=st.session_state.llama_path)


def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob="**/*.pdf")
    return loader.load()


def split_documents(documents):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)


def create_local_vectordb(texts):
    st.write("📦 Creating vector store...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        texts,
        embedding=embedding,
        persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
    )
    vectordb.persist()
    return vectordb.as_retriever(search_kwargs={"k": 2})


def query_llm(retriever, query):
    model_path = st.session_state.llama_path
    if not Path(model_path).exists():
        st.error(f"❌ LLaMA model not found at: {model_path}")
        return "Model path invalid."

    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.5,
        max_tokens=512,
        top_p=0.95,
        verbose=False
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({
        "question": query,
        "chat_history": []
    })
    answer = result["answer"]
    st.session_state.messages.append((query, answer))
    return answer


def process_documents():
    if not st.session_state.source_docs:
        st.warning("⚠️ Please upload at least one document.")
        return

    for doc in st.session_state.source_docs:
        with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR, suffix=".pdf") as tmp_file:
            tmp_file.write(doc.read())

    documents = load_documents()
    texts = split_documents(documents)

    st.session_state.retriever = create_local_vectordb(texts)
    st.success("✅ Documents processed and retriever is ready.")

    # Cleanup
    for f in TMP_DIR.iterdir():
        f.unlink()


def boot():
    input_fields()
    st.button("🚀 Submit & Process", on_click=process_documents)

    for message in st.session_state.messages:
        st.chat_message("user").write(message[0])
        st.chat_message("ai").write(message[1])

    if query := st.chat_input("💬 Ask a question"):
        st.chat_message("user").write(query)
        if st.session_state.retriever:
            response = query_llm(st.session_state.retriever, query)
            st.chat_message("ai").write(response)
        else:
            st.warning("⚠️ Submit and process documents first.")

boot()
