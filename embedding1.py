import json
import faiss
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

CHUNK_FILE = "chunks/all_chunks.jsonl"
INDEX_FILE = "faiss_index/index.faiss"
METADATA_FILE = "faiss_index/metadata.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_metadata(metadata, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in metadata:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def build_index(chunks, model):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # inner product with normalized vectors = cosine similarity
    index.add(np.array(embeddings, dtype="float32"))

    return index, embeddings

def main():
    print("📦 Loading chunks...")
    chunks = load_chunks(CHUNK_FILE)

    print(f"🧠 Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("🔧 Building FAISS index...")
    index, _ = build_index(chunks, model)

    print(f"💾 Saving index to: {INDEX_FILE}")
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    print(f"💾 Saving metadata to: {METADATA_FILE}")
    save_metadata(chunks, METADATA_FILE)

    print("✅ Index and metadata saved.")

if __name__ == "__main__":
    main()