from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama

# Step 1: Load the vectorstore
print("Loading vectorstore...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 2: Load the LLM
print("Loading LLM...")
llm = ChatOllama(model="llama3.2")

# Step 3: Ask questions in a loop
print("\nRAG Chatbot ready! Type 'exit' to quit.\n")
while True:
    question = input("You: ")
    if question.lower() == "exit":
        break

    # fetch relevant chunks from chromadb
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # build the prompt manually
    prompt = f"""Use the following context to answer the question. 
If you don't know the answer from the context, just say you don't know.

Context:
{context}

Question: {question}
Answer:"""

    response = llm.invoke(prompt)
    print(f"\nBot: {response.content}")
    print("\n" + "-"*50 + "\n")