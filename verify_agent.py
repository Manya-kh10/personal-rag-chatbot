import os
import sys
import json
from eval import load_env

load_env()

from gemini_agent import run_agent

groq_key = os.environ.get("GROQ_API_KEY")
google_key = os.environ.get("GOOGLE_API_KEY")

test_queries = [
    "hi",
    "what's 2+2?",
    "Who was the first president of the United States?",
    "What is the main topic of the uploaded sample document?",
    "Can you summarize key details from the document?"
]

results = []

for q in test_queries:
    row = {"query": q}
    
    # Test Groq
    if groq_key:
        try:
            res_groq = run_agent(q, provider="Groq (Cloud)", api_key=groq_key)
            row["groq_used_rag"] = res_groq["used_rag"]
            row["groq_answer"] = res_groq["answer"].strip().replace("\n", " ")[:120]
        except Exception as e:
            row["groq_used_rag"] = "Error"
            row["groq_answer"] = str(e)[:100]
    else:
        row["groq_used_rag"] = "N/A"
        row["groq_answer"] = "GROQ_API_KEY missing"

    # Test Gemini
    if google_key:
        try:
            res_gemini = run_agent(q, provider="Google Gemini (Cloud)", api_key=google_key)
            row["gemini_used_rag"] = res_gemini["used_rag"]
            row["gemini_answer"] = res_gemini["answer"].strip().replace("\n", " ")[:120]
        except Exception as e:
            row["gemini_used_rag"] = "Error"
            row["gemini_answer"] = str(e)[:100]
    else:
        row["gemini_used_rag"] = "N/A"
        row["gemini_answer"] = "GOOGLE_API_KEY missing"

    results.append(row)

print("\n=== VERIFICATION RESULTS JSON ===")
print(json.dumps(results, indent=2))
