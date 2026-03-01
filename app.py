import streamlit as st
import tempfile
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page config
st.set_page_config(page_title="My Personal RAG", page_icon="🤖")
st.title("🤖 My Personal RAG Chatbot")

# Load models (only once)
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatOllama(model="llama3.2")
    return embeddings, llm

embeddings, llm = load_models()

# Load vectorstore
@st.cache_resource
def load_vectorstore(_embeddings):
    return Chroma(persist_directory="./chroma_db", embedding_function=_embeddings)

vectorstore = load_vectorstore(embeddings)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # PDF Upload
    st.subheader("📄 Upload a Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.button("➕ Add to Knowledge Base"):
            with st.spinner("Processing document..."):
                # Save to a temp file
                suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Load and chunk
                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path, encoding="utf-8")

                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
                chunks = splitter.split_documents(docs)

                # Add to existing vectorstore
                vectorstore.add_documents(chunks)
                os.unlink(tmp_path)  # clean up temp file

            st.success(f"✅ Added {len(chunks)} chunks from '{uploaded_file.name}'")

    st.divider()

    # Show sources toggle
    show_sources = st.toggle("🔍 Show source chunks", value=False)

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Your RAG chatbot remembers conversation context within a session.")

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if show_sources and msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Source chunks used"):
                for i, source in enumerate(msg["sources"]):
                    st.caption(f"Chunk {i+1}: {source}")

# Chat input
if question := st.chat_input("Ask me anything..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Build conversation history
    history = ""
    for msg in st.session_state.messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    # Retrieve relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    source_chunks = [doc.page_content for doc in docs]

    # Build prompt
    prompt = f"""You are a helpful personal assistant. Use the context and conversation history below to answer the question.
If the answer isn't in the context, say you don't know.

Context from documents:
{context}

Conversation history:
{history}
User: {question}
Answer:"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)
            answer = response.content
        st.write(answer)

        # Show sources if toggle is on
        if show_sources:
            with st.expander("📚 Source chunks used"):
                for i, chunk in enumerate(source_chunks):
                    st.caption(f"Chunk {i+1}: {chunk}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_chunks
    })