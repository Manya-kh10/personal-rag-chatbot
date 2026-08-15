import os
os.environ["HF_HOME"] = "E:\\hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "E:\\hf_cache"
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

_cross_encoders = {}

def get_cross_encoder(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2"):
    """
    Load and cache the CrossEncoder model globally for script/CLI execution.
    For Streamlit, a wrapper with @st.cache_resource should be used.
    """
    global _cross_encoders
    if model_name not in _cross_encoders:
        print(f"Loading CrossEncoder model '{model_name}' (this may take a minute on first run)...")
        _cross_encoders[model_name] = CrossEncoder(model_name)
    return _cross_encoders[model_name]

def get_all_documents_from_vectorstore(vectorstore):
    """
    Extract all unique documents stored in ChromaDB to build the sparse retriever index.
    """
    try:
        res = vectorstore.get()
        documents = []
        if res and "documents" in res and res["documents"]:
            for content, metadata in zip(res["documents"], res["metadatas"]):
                documents.append(Document(page_content=content, metadata=metadata or {}))
        return documents
    except Exception as e:
        print(f"Error fetching documents from ChromaDB: {e}")
        return []

def get_bm25_retriever(vectorstore, k=15):
    """
    Create a BM25Retriever dynamically from documents present in the Chroma vector store.
    """
    documents = get_all_documents_from_vectorstore(vectorstore)
    if not documents:
        return None
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    return bm25_retriever

def reciprocal_rank_fusion(dense_docs, sparse_docs, k_rrf=60):
    """
    Perform Reciprocal Rank Fusion on dense and sparse retrieval results.
    Eliminates duplicates and ranks documents according to the RRF score.
    """
    rrf_scores = {}
    doc_map = {}
    
    # Process dense (vector) retrieval results
    for rank, doc in enumerate(dense_docs):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (k_rrf + rank + 1)
        
    # Process sparse (keyword) retrieval results
    for rank, doc in enumerate(sparse_docs):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (k_rrf + rank + 1)
        
    # Sort documents by their RRF scores descending
    sorted_contents = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[content] for content in sorted_contents]

def hybrid_retrieve(vectorstore, query, k_pool=15, k_rrf=60):
    """
    Retrieves candidate pool using hybrid search (Chroma vector search + BM25 keyword search)
    fused via Reciprocal Rank Fusion (RRF).
    """
    # 1. Get dense results
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k_pool})
    dense_docs = dense_retriever.invoke(query)
    
    # 2. Get sparse results
    bm25_retriever = get_bm25_retriever(vectorstore, k=k_pool)
    if bm25_retriever:
        sparse_docs = bm25_retriever.invoke(query)
    else:
        sparse_docs = []
        
    # 3. Fuse lists
    fused_docs = reciprocal_rank_fusion(dense_docs, sparse_docs, k_rrf=k_rrf)
    
    # Return top k_pool candidates
    return fused_docs[:k_pool]

def rerank_documents(query, documents, cross_encoder_model, k_final=3):
    """
    Reranks a list of candidate documents using a Cross-Encoder and returns the top k_final items.
    """
    if not documents:
        return []
        
    pairs = [(query, doc.page_content) for doc in documents]
    scores = cross_encoder_model.predict(pairs)
    
    # Pair documents with scores
    scored_docs = list(zip(documents, scores))
    # Sort descending by Cross-Encoder relevance score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k_final
    return [doc for doc, score in scored_docs[:k_final]]

_vectorstore = None
_embeddings = None

def get_default_vectorstore():
    global _vectorstore, _embeddings
    if _vectorstore is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=_embeddings)
    return _vectorstore

def retrieve_context(query: str) -> str:
    """
    Calls the existing hybrid search + cross-encoder reranking pipeline
    and returns top 3 reranked chunks as a single formatted string with source labels.
    """
    vectorstore = get_default_vectorstore()
    candidates = hybrid_retrieve(vectorstore, query, k_pool=15)
    cross_encoder = get_cross_encoder()
    top_docs = rerank_documents(query, candidates, cross_encoder, k_final=3)
    
    if not top_docs:
        return "No relevant documents found."
        
    formatted_chunks = []
    for idx, doc in enumerate(top_docs):
        source = doc.metadata.get("source", f"Chunk {idx+1}")
        formatted_chunks.append(f"[Source: {source}]\n{doc.page_content}")
        
    return "\n\n".join(formatted_chunks)

