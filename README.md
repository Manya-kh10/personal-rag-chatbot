# 🤖 Personal RAG Chatbot

A personal AI chatbot built using **Retrieval Augmented Generation (RAG)** that answers questions based on your own documents. Built as a first GenAI project while learning transformer architecture and PyTorch.

---

## 💡 What is RAG?

Instead of training a model from scratch, RAG works by:
1. **Chunking** your documents into small pieces and converting them into vectors (embeddings)
2. **Storing** those vectors in a vector database (ChromaDB)
3. At **query time**, finding the most relevant chunks and passing them as context to the LLM

This means the LLM answers based on *your* data — not just its general training knowledge.

---

## ✨ Features

- 📄 Upload your own PDFs or TXT files directly from the UI
- 🧠 Answers questions based on your uploaded documents
- 💬 Maintains conversation history within a session
- 🔍 Toggle to view the source chunks used to generate each answer
- 🗑️ Clear chat button to reset the conversation
- 🏠 Runs completely locally — no data leaves your machine

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Ollama](https://ollama.com) + Llama 3.2 | Local LLM inference |
| [LangChain](https://langchain.com) | RAG pipeline |
| [ChromaDB](https://www.trychroma.com) | Vector database |
| [HuggingFace sentence-transformers](https://huggingface.co/sentence-transformers) | Text embeddings (`all-MiniLM-L6-v2`) |
| [Streamlit](https://streamlit.io) | Frontend UI |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 1. Clone the repository
```bash
git clone https://github.com/Manya-kh10/personal-rag-chatbot.git
cd personal-rag-chatbot
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull the Ollama model
```bash
ollama pull llama3.2
```

### 5. Ingest your documents
Add your documents to the project folder and run:
```bash
python ingest.py
```
This creates a `chroma_db/` folder with your embedded documents.

### 6. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
personal-rag-chatbot/
├── app.py          # Streamlit frontend + RAG query pipeline
├── ingest.py       # Document loading, chunking, and embedding
├── query.py        # Terminal-based query interface
├── sample.txt      # Sample document for testing
├── requirements.txt
└── .gitignore
```

---

## 🧑‍💻 How It Works

1. `ingest.py` loads your documents, splits them into chunks, generates embeddings using `all-MiniLM-L6-v2`, and stores them in ChromaDB
2. When you ask a question in the UI, the question is also embedded and the top 3 most similar chunks are retrieved
3. Those chunks are injected into the prompt along with conversation history and sent to Llama 3.2 via Ollama
4. The model responds based only on the retrieved context

---

## 🌱 What I Learned

- How RAG pipelines work end to end
- Text embeddings and vector similarity search
- Working with ChromaDB as a local vector store
- Building LLM-powered apps with LangChain
- Creating interactive AI UIs with Streamlit
- Running LLMs locally with Ollama

---

## 📊 Evaluation & Benchmarks

A standalone evaluation pipeline (`eval.py`) was built to compare the RAG performance across different search and reranking configurations.

### Configuration Details
- **Evaluation Depth (k_final)**: `3` (across all three configurations to ensure an apples-to-apples comparison).
- **Candidate Pool Size (k_pool)**: `15` (for hybrid retrieval configurations).
- **Sparse Index**: BM25 keyword search.
- **Reranker Models**:
  - `cross-encoder/ms-marco-TinyBERT-L-2-v2` (17MB lightweight local Cross-Encoder).
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` (80MB moderate local Cross-Encoder).
- **Grading Model**: Local-model-graded (using the configured LLM).

### Performance Metrics

| Metric | Pure Semantic (Baseline) | Hybrid + TinyBERT (17MB) | Hybrid + MiniLM (80MB) |
| :--- | :---: | :---: | :---: |
| **Recall@3** (Retrieval Quality) | 1.000 | 1.000 | 1.000 |
| **MRR@3** (Retrieval Ranking) | 0.865 | 0.962 | 0.942 |
| **Faithfulness** (Local-model-graded) | 0.874 | 0.919 | 0.940 |
| **Answer Relevancy** (Local-model-graded) | 0.690 | 0.707 | 0.698 |

### 📌 Reranker Tradeoffs Takeaway
- **Pure Semantic (Baseline)**: No extra dependencies, lowest latency, but suffers from lower ranking precision (MRR: `0.865`), leading to less faithful LLM generations due to noise in the retrieved chunks.
- **Hybrid + TinyBERT (17MB)**: Extremely fast and lightweight. Very low memory footprint, making it ideal for CPU-only or resource-constrained local environments. Significantly improves ranking precision (MRR: `0.962`) and raises Faithfulness to `0.919`.
- **Hybrid + MiniLM (80MB)**: Moderate download size. Captures finer semantic nuances than TinyBERT, achieving the highest overall **Faithfulness** (`0.940`) by ensuring context chunks align closely with target factual query details, with only a minor ranking precision variation on this dataset size (MRR: `0.942`).

> [!WARNING]
> **Self-Grading Bias Warning**: Faithfulness and Answer Relevancy are graded by the local model itself. These scores are **local-model-graded, not GPT-4-graded**, which may introduce a self-grading bias resulting in more generous or slightly inconsistent scoring. They serve as a relative comparison rather than an objective benchmark.

---

## 📌 Future Improvements

- [ ] Swap Ollama for a cloud LLM API for deployment
- [ ] Add support for more file types (DOCX, CSV)
- [ ] Rebuild UI with Chainlit for a cleaner chat experience
- [ ] Fine-tune a small model on personal data using LoRA

---

*Built as a first GenAI project — part of a learning journey into Generative AI and LLMs.*

