from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1: Load your document
print("Loading document...")
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

# Step 2: Split into chunks
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.page_content}")

# Step 3: Create embeddings and store in ChromaDB
print("\nCreating embeddings and storing in ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

print("\nDone! Your data is stored in ChromaDB.")