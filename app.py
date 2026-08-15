import os
os.environ["HF_HOME"] = "E:\\hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "E:\\hf_cache"
import streamlit as st
import tempfile
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import rag_pipeline
import gemini_agent

# Page config
st.set_page_config(page_title="My Personal RAG", page_icon="🤖")
st.title("🤖 My Personal RAG Chatbot")


# Load embeddings (only once)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Load vectorstore
@st.cache_resource
def load_vectorstore(_embeddings):
    return Chroma(persist_directory="./chroma_db", embedding_function=_embeddings)

vectorstore = load_vectorstore(embeddings)

# LLM Loading
@st.cache_resource
def load_llm(provider, api_key):
    if provider == "Google Gemini (Cloud)":
        if not api_key:
            return None
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    elif provider == "Groq (Cloud)":
        if not api_key:
            return None
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)
    else:
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model="llama3.2")
        except ImportError:
            st.error("Ollama package is not installed.")
            return None

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # LLM Settings
    st.subheader("🤖 LLM Settings")
    provider = st.selectbox(
        "LLM Provider",
        ["Google Gemini (Cloud)", "Groq (Cloud)", "Ollama (Local)"],
        index=0
    )
    
    api_key = None
    if provider == "Google Gemini (Cloud)":
        env_key = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
        if not env_key:
            api_key = st.text_input("Enter Google API Key", type="password")
            if not api_key:
                st.warning("⚠️ Please enter a Google API Key to use Gemini.")
        else:
            api_key = env_key
            st.info("Using Google API Key from environment/secrets.")
            
    elif provider == "Groq (Cloud)":
        env_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
        if not env_key:
            api_key = st.text_input("Enter Groq API Key", type="password")
            if not api_key:
                st.warning("⚠️ Please enter a Groq API Key to use Groq.")
        else:
            api_key = env_key
            st.info("Using Groq API Key from environment/secrets.")

    st.divider()

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

    # Search Mode settings
    st.subheader("🔍 Retrieval & Ranking Settings")
    search_mode = st.selectbox(
        "Search Mode",
        ["Pure Semantic (No Rerank)", "Hybrid + Rerank", "Agentic (Gemini / Groq tool-use)"],
        index=0,
        help="Compare Pure Semantic vs Hybrid + Rerank vs Agentic tool-use mode (where Gemini or Groq dynamically decides when to retrieve context)."
    )

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Your RAG chatbot remembers conversation context within a session.")

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("used_rag"):
            st.info("🔎 Using RAG pipeline to retrieve context...")
        st.write(msg["content"])
        if show_sources and msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Source chunks used"):
                for i, source in enumerate(msg["sources"]):
                    st.caption(f"Chunk {i+1}: {source}")

llm = load_llm(provider, api_key)

# Chat input
disabled = (provider in ["Google Gemini (Cloud)", "Groq (Cloud)"] and not api_key)
placeholder = "Ask me anything..." if not disabled else f"Please enter your {provider.split()[0]} API Key in the sidebar to chat."

if question := st.chat_input(placeholder, disabled=disabled):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    if search_mode == "Agentic (Gemini / Groq tool-use)":
        with st.chat_message("assistant"):
            with st.spinner("Thinking (Agent evaluating tool call)..."):
                try:
                    agent_res = gemini_agent.run_agent(question, provider=provider, api_key=api_key)
                    answer = agent_res["answer"]
                    used_rag = agent_res["used_rag"]
                except Exception as e:
                    answer = f"Error running agent: {e}"
                    used_rag = False

            
            if used_rag:
                st.info("🔎 Using RAG pipeline to retrieve context...")
            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "used_rag": used_rag
        })
    else:
        # Build conversation history
        history = ""
        for msg in st.session_state.messages[:-1]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history += f"{role}: {msg['content']}\n"

        # Retrieve relevant chunks based on search mode
        if search_mode == "Pure Semantic (No Rerank)":
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(question)
        else:
            # Hybrid + Rerank configuration
            # Retrieve candidate pool of size 15 from dense + sparse sources
            candidates = rag_pipeline.hybrid_retrieve(vectorstore, question, k_pool=15)
            
            # Load and cache CrossEncoder using st.cache_resource
            @st.cache_resource
            def get_cached_cross_encoder():
                return rag_pipeline.get_cross_encoder()
                
            cross_encoder = get_cached_cross_encoder()
            # Rerank down to top 3 (apples-to-apples comparison depth)
            docs = rag_pipeline.rerank_documents(question, candidates, cross_encoder, k_final=3)

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