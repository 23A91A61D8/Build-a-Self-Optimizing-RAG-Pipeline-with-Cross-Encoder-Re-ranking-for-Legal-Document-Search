import json
from sentence_transformers import SentenceTransformer
import chromadb

CHUNKS_FILE = "../data/processed/chunks.jsonl"

DB_PATH = "E:/partnr/Week-17(MANDRATORY)/Re-ranking-for-Legal-Document-Search/chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_chunks():
    data = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_embeddings():
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection("legal_docs")
    except:
        pass

    collection = client.get_or_create_collection(name="legal_docs")

    chunks = load_chunks()

    for item in chunks:
        emb = model.encode(item["text"])

        collection.add(
            documents=[item["text"]],
            embeddings=[emb],
            ids=[item["chunk_id"]],
            metadatas=[{"doc_id": item["doc_id"]}]
        )

    print("Embeddings stored successfully ✅")

if __name__ == "__main__":
    create_embeddings()