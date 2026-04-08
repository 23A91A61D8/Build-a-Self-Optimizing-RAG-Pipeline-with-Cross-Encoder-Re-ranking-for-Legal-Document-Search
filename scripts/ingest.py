import os
import json

RAW_DIR = "../data/raw"
OUTPUT_FILE = "../data/processed/chunks.jsonl"

def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks

def process_documents():
    os.makedirs("../data/processed", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        for filename in os.listdir(RAW_DIR):
            if filename.endswith(".txt"):
                doc_id = filename.replace(".txt", "")
                
                with open(os.path.join(RAW_DIR, filename), "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = chunk_text(text)

                for i, chunk in enumerate(chunks):
                    data = {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}-{i+1}",
                        "text": chunk
                    }
                    out_file.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    process_documents()