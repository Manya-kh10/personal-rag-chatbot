# RAG Evaluation Report


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


## Performance Comparison Table

| Metric | Pure Semantic (Baseline) | Hybrid + TinyBERT (17MB) | Hybrid + MiniLM (80MB) |
| :--- | :---: | :---: | :---: |
| **Recall@3** (Retrieval Quality) | 1.000 | 1.000 | 1.000 |
| **MRR@3** (Retrieval Ranking) | 0.865 | 0.962 | 0.942 |
| **Faithfulness** (Local-model-graded) | 0.874 | 0.919 | 0.940 |
| **Answer Relevancy** (Local-model-graded) | 0.690 | 0.707 | 0.698 |

### Notes on Evaluation Methodology
- **Apples-to-Apples Depth**: All configurations used a final depth of `k=3` document chunks passed downstream to the LLM generator. This isolates the retrieval quality and order impact of combining **BM25 Keyword search** and **Chroma Dense Vector search** via **Reciprocal Rank Fusion (RRF)**, followed by scoring using different Cross-Encoders.
- **Reranker Tradeoffs Takeaway**:
  - **ms-marco-TinyBERT-L-2-v2** (17MB): Extremely lightweight and fast. Very low memory footprint, making it ideal for CPU-only or memory-constrained environments. Provides significant MRR improvements.
  - **ms-marco-MiniLM-L-6-v2** (80MB): Moderate size. Captures finer semantic nuances than TinyBERT, leading to higher ranking precision (MRR) and improved contextual quality for the LLM generator.
- **Retrieval Metrics**: `Recall@3` and `MRR@3` evaluate whether the target source text chunks (containing correct expected substrings) were retrieved and placed in the top 3 spots.
- **LLM-Graded Metrics**: `Faithfulness` (checking if answers match source chunks) and `Answer Relevancy` (checking if answers match question intent) were graded by the local model.
- **Self-Grading Bias Warning**: Because the generator model evaluates its own outputs, scores are expected to be higher and more generous than objective human or GPT-4 grading.
