from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional
import base64
from PIL import Image
import io
import pytesseract
import os

app = FastAPI()

INDEX_PATH = "./faiss_index/index.faiss"
METADATA_PATH = "./faiss_index/metadata.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
print("🔁 Loading model, index, and metadata...")
model = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(INDEX_PATH)
metadata = [json.loads(line) for line in open(METADATA_PATH, "r", encoding="utf-8")]

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/")
def ask(data: QueryRequest):
    question = data.question.strip()

    if data.image:
        try:
            image_data = base64.b64decode(data.image)
            image = Image.open(io.BytesIO(image_data))
            extracted_text = pytesseract.image_to_string(image)
            question += " " + extracted_text.strip()
        except Exception:
            return {"answer": "❌ Could not decode image.", "links": []}

    embedding = model.encode([question], normalize_embeddings=True)
    D, I = index.search(np.array(embedding, dtype="float32"), TOP_K)

    relevant_chunks = [metadata[i]["text"] for i in I[0]]
    context = "\n---\n".join(relevant_chunks)

    dummy_links = []
    unique_urls = set()
    for i in I[0]:
        if "url" in metadata[i]:
            url = metadata[i]["url"]
            if url not in unique_urls:
                dummy_links.append({
                    "url": url,
                    "text": metadata[i].get("title", "Related Discourse Thread")
                })
                unique_urls.add(url)
        if len(dummy_links) >= 5:
            break

    return {
        "answer": context,
        "links": dummy_links
    }
