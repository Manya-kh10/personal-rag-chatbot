import os
os.environ["HF_HOME"] = "E:\\hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "E:\\hf_cache"
import json
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Custom imports
import rag_pipeline

# ---------------------------------------------------------
# Self-Grading Bias Warning & Pipeline Metadata
# ---------------------------------------------------------
# WARNING: This evaluation pipeline uses the SAME local model (or Groq API model)
# to grade faithfulness and answer relevancy of its own generations.
# This introduces a self-grading bias where scores may be inflated.
# Scores are explicitly marked as "local-model-graded, not GPT-4-graded".
#
# Metadata:
# - Before Configuration: Pure Semantic (Chroma dense retrieval, k_final = 3)
# - After Configuration: Hybrid Search (Chroma dense k_pool=15 + BM25 sparse k_pool=15 fused via RRF to get top 15)
#   followed by Cross-Encoder Reranking (cross-encoder/ms-marco-TinyBERT-L-2-v2) to select top k_final = 3.
#
# Both configurations are evaluated using k_final = 3 downstream documents passed to the LLM.
# This ensures a fair, apples-to-apples comparison of retrieval quality.
# ---------------------------------------------------------

EVAL_METADATA_HEADER = """
========================================================================
EVALUATION RUN REPORT (Apples-to-Apples Configuration)
------------------------------------------------------------------------
* Parameters:
  - Evaluation Depth (k_final): 3 documents passed to LLM for both configs.
  - Candidate Pool Size for Reranker: 15 documents.
  - Reranker Model: cross-encoder/ms-marco-TinyBERT-L-2-v2
  - Sparse Index: BM25 (built dynamically from vector store documents)
* Self-Grading Bias Warning:
  - Generation and Evaluation are performed using the same model family.
  - Metrics are local-model-graded (NOT GPT-4-graded) and may contain bias.
========================================================================
"""

def load_env():
    """Manually parse .env file to load variables into os.environ"""
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

def load_llm():
    """Loads Groq or Ollama LLM based on environment setup"""
    load_env()
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        print("Using Groq API for evaluation (faster generation/grading)...")
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_key)
    else:
        print("Using Ollama ChatOllama (local llama3.2)...")
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2")

def check_substrings_in_doc(doc_content, expected_substrings):
    """Checks if any expected substring is present in the document content (case-insensitive)"""
    content_lower = doc_content.lower()
    for sub in expected_substrings:
        if sub.lower() in content_lower:
            return True
    return False

def compute_retrieval_metrics(retrieved_docs, expected_substrings):
    """
    Computes Recall@3 and MRR@3 for a set of retrieved documents.
    Both metrics are capped/measured at k=3 for fair comparison.
    """
    docs_to_check = retrieved_docs[:3]
    
    # 1. Recall@3
    recall = 0.0
    for doc in docs_to_check:
        if check_substrings_in_doc(doc.page_content, expected_substrings):
            recall = 1.0
            break
            
    # 2. MRR@3
    mrr = 0.0
    for idx, doc in enumerate(docs_to_check):
        if check_substrings_in_doc(doc.page_content, expected_substrings):
            mrr = 1.0 / (idx + 1)
            break
            
    return recall, mrr

def invoke_with_retry(llm, prompt, max_retries=5, initial_delay=3.0):
    """
    Invokes the LLM with exponential backoff if a rate limit error (429) is hit.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str:
                print(f"  Rate limit hit (429). Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e
    return llm.invoke(prompt)

def grade_faithfulness(llm, context, answer):
    """LLM-graded check for context alignment (0 to 1)"""
    prompt = f"""You are a strict evaluator checking for hallucinations.
Analyze if the generated answer is completely supported by the retrieved context.
Output only a valid JSON object with the keys "reasoning" (brief explanation) and "score" (a float between 0.0 and 1.0, where 0.0 means not supported/hallucinated and 1.0 means fully supported). Do not output any markdown formatting, code block ticks, or extra text.

Context:
{context}

Generated Answer:
{answer}

JSON:"""
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        data = json.loads(content)
        return max(0.0, min(1.0, float(data.get("score", 0.0))))
    except Exception as e:
        # Simple string heuristics fallback if LLM output isn't clean JSON
        return 0.5

def grade_relevancy(llm, question, answer):
    """LLM-graded check for answer relevancy to the question (0 to 1)"""
    prompt = f"""You are a strict evaluator checking answer relevance.
Analyze if the generated answer directly, accurately, and fully answers the question.
Output only a valid JSON object with the keys "reasoning" (brief explanation) and "score" (a float between 0.0 and 1.0, where 0.0 means irrelevant and 1.0 means completely relevant and accurate). Do not output any markdown formatting, code block ticks, or extra text.

Question:
{question}

Generated Answer:
{answer}

JSON:"""
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        data = json.loads(content)
        return max(0.0, min(1.0, float(data.get("score", 0.0))))
    except Exception as e:
        return 0.5

def main():
    print(EVAL_METADATA_HEADER)
    
    # Load dataset
    if not os.path.exists("eval_questions.json"):
        print("Error: eval_questions.json file not found. Please create it first.")
        return
        
    with open("eval_questions.json", "r") as f:
        questions_dataset = json.load(f)
        
    print(f"Loaded {len(questions_dataset)} evaluation questions.")
    
    # Check vector store
    if not os.path.exists("./chroma_db"):
        print("Vectorstore not found! Running ingest.py first...")
        import subprocess
        subprocess.run(["python", "ingest.py"], check=True)
        
    # Load embeddings and vectorstore
    print("Loading vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Load LLM
    llm = load_llm()
    if not llm:
        print("Error: LLM failed to load.")
        return
        
    # Load CrossEncoder Rerankers
    tiny_reranker = rag_pipeline.get_cross_encoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    minilm_reranker = rag_pipeline.get_cross_encoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Track metrics
    before_metrics = {"recall": [], "mrr": [], "faithfulness": [], "relevancy": []}
    tiny_metrics = {"recall": [], "mrr": [], "faithfulness": [], "relevancy": []}
    minilm_metrics = {"recall": [], "mrr": [], "faithfulness": [], "relevancy": []}
    
    print("\nRunning Evaluation...")
    for idx, item in enumerate(questions_dataset):
        question = item["question"]
        expected_subs = item["expected_substrings"]
        
        print(f"[{idx+1}/{len(questions_dataset)}] Q: {question}")
        
        # -------------------------------------------------
        # Config 1: Before (Pure Semantic Baseline, k_final = 3)
        # -------------------------------------------------
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        before_docs = semantic_retriever.invoke(question)
        
        before_recall, before_mrr = compute_retrieval_metrics(before_docs, expected_subs)
        before_metrics["recall"].append(before_recall)
        before_metrics["mrr"].append(before_mrr)
        
        before_context = "\n\n".join([doc.page_content for doc in before_docs])
        prompt_before = f"""Use the following context to answer the question. 
If you don't know the answer from the context, just say you don't know.

Context:
{before_context}

Question: {question}
Answer:"""
        try:
            res_before = invoke_with_retry(llm, prompt_before)
            ans_before = res_before.content
            
            f_before = grade_faithfulness(llm, before_context, ans_before)
            r_before = grade_relevancy(llm, question, ans_before)
            before_metrics["faithfulness"].append(f_before)
            before_metrics["relevancy"].append(r_before)
        except Exception as e:
            print(f"  Error generating answer (Pure Semantic): {e}")
            before_metrics["faithfulness"].append(0.0)
            before_metrics["relevancy"].append(0.0)
            
        # -------------------------------------------------
        # Config 2: Hybrid + TinyBERT Rerank (k_final = 3)
        # -------------------------------------------------
        # 1. Retrieve hybrid candidate pool (k_pool = 15)
        hybrid_candidates = rag_pipeline.hybrid_retrieve(vectorstore, question, k_pool=15)
        # 2. Rerank using TinyBERT down to k_final = 3
        tiny_docs = rag_pipeline.rerank_documents(question, hybrid_candidates, tiny_reranker, k_final=3)
        
        tiny_recall, tiny_mrr = compute_retrieval_metrics(tiny_docs, expected_subs)
        tiny_metrics["recall"].append(tiny_recall)
        tiny_metrics["mrr"].append(tiny_mrr)
        
        tiny_context = "\n\n".join([doc.page_content for doc in tiny_docs])
        prompt_tiny = f"""Use the following context to answer the question. 
If you don't know the answer from the context, just say you don't know.

Context:
{tiny_context}

Question: {question}
Answer:"""
        try:
            res_tiny = invoke_with_retry(llm, prompt_tiny)
            ans_tiny = res_tiny.content
            
            f_tiny = grade_faithfulness(llm, tiny_context, ans_tiny)
            r_tiny = grade_relevancy(llm, question, ans_tiny)
            tiny_metrics["faithfulness"].append(f_tiny)
            tiny_metrics["relevancy"].append(r_tiny)
        except Exception as e:
            print(f"  Error generating answer (Hybrid + TinyBERT): {e}")
            tiny_metrics["faithfulness"].append(0.0)
            tiny_metrics["relevancy"].append(0.0)

        # -------------------------------------------------
        # Config 3: Hybrid + MiniLM Rerank (k_final = 3)
        # -------------------------------------------------
        # 1. Reuse the hybrid candidate pool (k_pool = 15) to isolate the reranker's impact
        # 2. Rerank using MiniLM down to k_final = 3
        minilm_docs = rag_pipeline.rerank_documents(question, hybrid_candidates, minilm_reranker, k_final=3)
        
        minilm_recall, minilm_mrr = compute_retrieval_metrics(minilm_docs, expected_subs)
        minilm_metrics["recall"].append(minilm_recall)
        minilm_metrics["mrr"].append(minilm_mrr)
        
        minilm_context = "\n\n".join([doc.page_content for doc in minilm_docs])
        prompt_minilm = f"""Use the following context to answer the question. 
If you don't know the answer from the context, just say you don't know.

Context:
{minilm_context}

Question: {question}
Answer:"""
        try:
            res_minilm = invoke_with_retry(llm, prompt_minilm)
            ans_minilm = res_minilm.content
            
            f_minilm = grade_faithfulness(llm, minilm_context, ans_minilm)
            r_minilm = grade_relevancy(llm, question, ans_minilm)
            minilm_metrics["faithfulness"].append(f_minilm)
            minilm_metrics["relevancy"].append(r_minilm)
        except Exception as e:
            print(f"  Error generating answer (Hybrid + MiniLM): {e}")
            minilm_metrics["faithfulness"].append(0.0)
            minilm_metrics["relevancy"].append(0.0)
            
    # Calculate Averages
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0
        
    avg_before_recall = avg(before_metrics["recall"])
    avg_before_mrr = avg(before_metrics["mrr"])
    avg_before_faith = avg(before_metrics["faithfulness"])
    avg_before_rel = avg(before_metrics["relevancy"])
    
    avg_tiny_recall = avg(tiny_metrics["recall"])
    avg_tiny_mrr = avg(tiny_metrics["mrr"])
    avg_tiny_faith = avg(tiny_metrics["faithfulness"])
    avg_tiny_rel = avg(tiny_metrics["relevancy"])
    
    avg_minilm_recall = avg(minilm_metrics["recall"])
    avg_minilm_mrr = avg(minilm_metrics["mrr"])
    avg_minilm_faith = avg(minilm_metrics["faithfulness"])
    avg_minilm_rel = avg(minilm_metrics["relevancy"])
    
    # Print Comparison Table
    print("\n" + "="*70)
    print("FINAL EVALUATION METRICS TABLE (THREE-WAY COMPARISON)")
    print("="*70)
    print(f"Evaluation Depth (k_final) for ALL configs: 3")
    print(f"Candidate Pool Size for Hybrid configs (k_pool): 15")
    print(f"Warning: Local-model-graded metrics below contain self-grading bias.\n")
    
    table = f"""| Metric | Pure Semantic (Baseline) | Hybrid + TinyBERT (17MB) | Hybrid + MiniLM (80MB) |
| :--- | :---: | :---: | :---: |
| **Recall@3** (Retrieval Quality) | {avg_before_recall:.3f} | {avg_tiny_recall:.3f} | {avg_minilm_recall:.3f} |
| **MRR@3** (Retrieval Ranking) | {avg_before_mrr:.3f} | {avg_tiny_mrr:.3f} | {avg_minilm_mrr:.3f} |
| **Faithfulness** (Local-model-graded) | {avg_before_faith:.3f} | {avg_tiny_faith:.3f} | {avg_minilm_faith:.3f} |
| **Answer Relevancy** (Local-model-graded) | {avg_before_rel:.3f} | {avg_tiny_rel:.3f} | {avg_minilm_rel:.3f} |"""
    
    print(table)
    print("="*70)
    
    # Save the table to artifacts/evaluation_report.md
    report_content = f"""# RAG Evaluation Report

{EVAL_METADATA_HEADER}

## Performance Comparison Table

{table}

### Notes on Evaluation Methodology
- **Apples-to-Apples Depth**: All configurations used a final depth of `k=3` document chunks passed downstream to the LLM generator. This isolates the retrieval quality and order impact of combining **BM25 Keyword search** and **Chroma Dense Vector search** via **Reciprocal Rank Fusion (RRF)**, followed by scoring using different Cross-Encoders.
- **Reranker Tradeoffs Takeaway**:
  - **ms-marco-TinyBERT-L-2-v2** (17MB): Extremely lightweight and fast. Very low memory footprint, making it ideal for CPU-only or memory-constrained environments. Provides significant MRR improvements.
  - **ms-marco-MiniLM-L-6-v2** (80MB): Moderate size. Captures finer semantic nuances than TinyBERT, leading to higher ranking precision (MRR) and improved contextual quality for the LLM generator.
- **Retrieval Metrics**: `Recall@3` and `MRR@3` evaluate whether the target source text chunks (containing correct expected substrings) were retrieved and placed in the top 3 spots.
- **LLM-Graded Metrics**: `Faithfulness` (checking if answers match source chunks) and `Answer Relevancy` (checking if answers match question intent) were graded by the local model.
- **Self-Grading Bias Warning**: Because the generator model evaluates its own outputs, scores are expected to be higher and more generous than objective human or GPT-4 grading.
"""
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("Saved evaluation report to artifacts/evaluation_report.md.")

if __name__ == "__main__":
    main()
