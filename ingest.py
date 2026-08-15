import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def split_document_preserving_tables(doc: Document, chunk_size=200, chunk_overlap=50) -> list[Document]:
    """
    Splits a Document into chunks while keeping Markdown tables atomic (un-split).
    Prose paragraphs outside tables are split using RecursiveCharacterTextSplitter(chunk_size, chunk_overlap).
    """
    text = doc.page_content
    metadata = doc.metadata or {}
    lines = text.split("\n")
    
    sections = []
    current_block = []
    is_in_table = False
    
    for line in lines:
        stripped = line.strip()
        # Table lines start with '|' and contain column delimiters
        line_is_table = stripped.startswith("|") and "|" in stripped[1:]
        
        if line_is_table:
            if not is_in_table:
                if current_block:
                    sections.append(("text", "\n".join(current_block)))
                    current_block = []
                is_in_table = True
            current_block.append(line)
        else:
            if is_in_table:
                if current_block:
                    sections.append(("table", "\n".join(current_block)))
                    current_block = []
                is_in_table = False
            current_block.append(line)
            
    if current_block:
        sections.append(("table" if is_in_table else "text", "\n".join(current_block)))
        
    final_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    for sec_type, sec_content in sections:
        sec_content_stripped = sec_content.strip()
        if not sec_content_stripped:
            continue
            
        if sec_type == "table":
            final_chunks.append(Document(page_content=sec_content_stripped, metadata=metadata.copy()))
        else:
            sub_docs = text_splitter.create_documents([sec_content_stripped], metadatas=[metadata.copy()])
            final_chunks.extend(sub_docs)
            
    return final_chunks

def split_documents_preserving_tables(documents: list[Document], chunk_size=200, chunk_overlap=50) -> list[Document]:
    all_chunks = []
    for doc in documents:
        chunks = split_document_preserving_tables(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks

def ingest_all():
    print("Loading documents...")
    files_to_ingest = []
    for f in ["sample.txt", "README.md"]:
        if os.path.exists(f):
            files_to_ingest.append(f)
            
    documents = []
    for filepath in files_to_ingest:
        loader = TextLoader(filepath, encoding="utf-8")
        documents.extend(loader.load())
        
    print(f"Loaded {len(documents)} document(s): {files_to_ingest}")
    
    print("Splitting into chunks with table-preservation...")
    chunks = split_documents_preserving_tables(documents, chunk_size=200, chunk_overlap=50)
    
    print(f"Created {len(chunks)} chunks.")
    
    # Remove existing vector database to ensure clean re-indexing
    if os.path.exists("./chroma_db"):
        print("Clearing existing ChromaDB directory...")
        shutil.rmtree("./chroma_db")
        
    print("\nCreating embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    print("\nDone! Vector store rebuilt cleanly in ChromaDB.")
    return vectorstore

if __name__ == "__main__":
    ingest_all()