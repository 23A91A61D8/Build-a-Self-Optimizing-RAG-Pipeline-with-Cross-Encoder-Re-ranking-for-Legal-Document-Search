### Chunking Strategy

A fixed-size chunking strategy with logical segmentation was used to preserve the semantic integrity of legal clauses. Legal documents contain structured sections, and improper splitting can distort meaning. Therefore, chunking ensured that clauses remain intact while maintaining manageable input sizes for embedding models.

---

### Model Selection

#### 🔹 Bi-Encoder Model
**all-MiniLM-L6-v2**

- Lightweight and fast
- Suitable for large-scale retrieval
- Good balance between performance and efficiency

#### 🔹 Cross-Encoder Model
**cross-encoder/ms-marco-MiniLM-L-6-v2**

- Performs deep interaction between query and document
- Provides higher accuracy in ranking
- Ideal for second-stage precision optimization

---

### System Design Trade-offs

| Component | Trade-off |
|----------|----------|
| Bi-Encoder | Fast but less precise |
| Cross-Encoder | Accurate but slower |
| Two-stage system | Combines both advantages |

---

### Failure Mode Analysis

#### 1. Similar Results in Baseline and Re-ranked
Due to the small dataset and relatively simple queries, both models often retrieve similar top results. This reduces observable improvement in evaluation metrics.

#### 2. Query Ambiguity
Ambiguous or short queries may lead to less relevant retrieval due to lack of context.

#### 3. Limited Dataset Coverage
The system is trained on a small synthetic dataset, which limits generalization.

---

### Future Enhancements

- Increase dataset size using CUAD or LegalPile
- Apply domain-specific fine-tuning
- Improve chunking using sentence-aware splitting
- Integrate hybrid retrieval methods (BM25 + vector search)
- Add LLM-based answer generation layer