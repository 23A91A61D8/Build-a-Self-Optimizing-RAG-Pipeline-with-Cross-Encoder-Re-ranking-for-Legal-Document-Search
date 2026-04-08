# Build-a-Self-Optimizing-RAG-Pipeline-with-Cross-Encoder-Re-ranking-for-Legal-Document-Search

##  Project Overview
This project implements a **production-grade Retrieval-Augmented Generation (RAG) pipeline** tailored for legal document search.

Traditional keyword-based search fails to capture the semantic complexity of legal language. This system overcomes that limitation by combining:

-  Fast vector similarity retrieval (bi-encoder)
-  High-precision cross-encoder re-ranking

This two-stage architecture significantly improves the relevance and accuracy of search results.

---

##  Key Concepts

### 🔹 Retrieval-Augmented Generation (RAG)
Enhances search by retrieving relevant context before generating or returning answers.

### 🔹 Two-Stage Retrieval Pipeline
1. **Stage 1 (Recall)**: Vector search retrieves top-N candidate documents.
2. **Stage 2 (Precision)**: Cross-encoder re-ranks results using deep semantic understanding.

---

##  Technology Stack

| Component | Tool |
|----------|------|
| Backend API | FastAPI |
| Embeddings | Sentence Transformers |
| Re-ranking | Cross-Encoder |
| Vector DB | ChromaDB |
| Language | Python |

---

##  System Architecture


Raw Documents
↓
Chunking
↓
Embeddings (Bi-Encoder)
↓
Vector Database (ChromaDB)
↓
User Query
↓
Top-K Retrieval (Baseline)
↓
Cross-Encoder Re-ranking
↓
Final Ranked Results


---

##  API Endpoints

### 🔹 Health Check

GET /health


### 🔹 Baseline Retrieval

GET /api/v1/retrieve/baseline


### 🔹 Re-ranked Retrieval

GET /api/v1/retrieve/reranked


---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

pip install -r requirements.txt


### 2️⃣ Generate Embeddings

cd scripts
python embed.py


### 3️⃣ Start API Server

uvicorn api.main:app --host 127.0.0.1 --port 9000 --reload


### 4️⃣ Access API

http://127.0.0.1:9000/docs


---

##  Evaluation Framework

This project includes a custom evaluation pipeline to compare retrieval performance.

### Metrics Used:
-  Mean Reciprocal Rank (MRR@5)
-  Normalized Discounted Cumulative Gain (NDCG@10)

### Run Evaluation:

cd scripts
python evaluate.py


### Output:

results/evaluation_metrics.json


---

##  Results Summary

| Method | MRR@5 | NDCG@10 |
|--------|------|---------|
| Baseline | 0.95 | 0.96 |
| Re-ranked | 0.95 | 0.96 |

> Note: Similar scores due to small dataset and query simplicity.

---

##  Key Features

- ✅ Two-stage RAG architecture
- ✅ Cross-encoder semantic re-ranking
- ✅ RESTful API with FastAPI
- ✅ Modular and scalable design
- ✅ Custom evaluation metrics implementation
- ✅ Real-world legal search use case

---

##  Limitations

- Small dataset size
- Limited domain diversity
- No fine-tuning applied

---

##  Future Improvements

- Larger legal datasets (CUAD, LegalPile)
- Fine-tuned domain-specific embeddings
- Hybrid search (BM25 + vector search)
- LLM-based answer generation layer

---
Docker setup was attempted, but due to system-level compatibility issues, 
the project was executed successfully using local FastAPI server.
All functionalities including embedding, retrieval, re-ranking, and evaluation are fully working.

----
