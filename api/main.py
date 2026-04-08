from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

app = FastAPI()

# ✅ Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Cross-encoder model (NEW)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ✅ Absolute DB Path
DB_PATH = "E:/partnr/Week-17(MANDRATORY)/Re-ranking-for-Legal-Document-Search/chroma_db"

# ✅ Load DB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="legal_docs")


# 🔹 Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# 🔹 BASELINE RETRIEVAL
@app.get("/api/v1/retrieve/baseline")
def retrieve_baseline(query: str, k: int = 10):
    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    output = []

    if results["ids"]:
        for i in range(len(results["ids"][0])):
            output.append({
                "doc_id": results["metadatas"][0][i]["doc_id"],
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": float(results["distances"][0][i])
            })

    return {"results": output}


# 🔥 RERANKED RETRIEVAL (STEP 6)
@app.get("/api/v1/retrieve/reranked")
def retrieve_reranked(query: str, k: int = 10):
    # Step 1: Get more candidates
    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=50  # get more results for reranking
    )

    candidates = []

    if results["ids"]:
        for i in range(len(results["ids"][0])):
            candidates.append({
                "doc_id": results["metadatas"][0][i]["doc_id"],
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i]
            })

    # Step 2: Prepare query-text pairs
    pairs = [[query, item["text"]] for item in candidates]

    # Step 3: Get cross-encoder scores
    scores = cross_encoder.predict(pairs)

    # Step 4: Attach scores
    for i in range(len(candidates)):
        candidates[i]["score"] = float(scores[i])

    # Step 5: Sort (HIGHER = better)
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    # Step 6: Return top k
    return {"results": candidates[:k]}