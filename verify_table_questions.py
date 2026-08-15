import os
from eval import load_env, load_llm
load_env()

from rag_pipeline import retrieve_context

llm = load_llm()

test_questions = [
    "What is the MRR@3 score for the hybrid plus TinyBERT configuration?",
    "What is the Faithfulness score for the hybrid plus MiniLM configuration?",
    "What is the Recall@3 score across all three configurations?"
]

print("=========================================================")
print("RAG PIPELINE TABLE RETRIEVAL AND ANSWER VERIFICATION")
print("=========================================================\n")

for i, q in enumerate(test_questions):
    print(f"=== [Question {i+1}] {q} ===")
    context = retrieve_context(q)
    print("\n[Retrieved Source Chunks]")
    print(context)
    
    prompt = f"""You are a helpful personal assistant. Use the context below from the document to answer the user's question accurately.

Context:
{context}

Question: {q}
Answer:"""

    response = llm.invoke(prompt)
    print("\n[Generated RAG Answer]")
    print(response.content)
    print("=" * 65 + "\n")
