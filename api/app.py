from fastapi import FastAPI, Request
from pydantic import BaseModel
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import base64
from PIL import Image
import io
import pytesseract

import os

INDEX_PATH = "./faiss_index/index.faiss"
METADATA_PATH = "./faiss_index/metadata.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
print("🔁 Loading model, index, and metadata...")
model = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(INDEX_PATH)
metadata = [json.loads(line) for line in open(METADATA_PATH, "r", encoding="utf-8")]

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  

def answer_question(data: QueryRequest):
    question = data.question.strip()

    if data.image:
        try:
            image_data = base64.b64decode(data.image)
            image = Image.open(io.BytesIO(image_data))
            extracted_text = pytesseract.image_to_string(image)
            question += " " + extracted_text.strip()
        except Exception as e:
            return {"answer": "❌ Could not decode image.", "links": []}

    embedding = model.encode([question], normalize_embeddings=True)
    D, I = index.search(np.array(embedding, dtype="float32"), TOP_K)

    relevant_chunks = [metadata[i]["text"] for i in I[0]]
    context = "\n---\n".join(relevant_chunks)

    dummy_links = []
    for i in I[0]:
        if "url" in metadata[i]:
            dummy_links.append({
                "url": metadata[i]["url"],
                "text": metadata[i].get("title", "Related Discourse Thread")
            })

    answer = f"📚 **Context:**\n{context}\n\n🤖 **Answer:**\nBased on the above, your question was: '{data.question.strip()}'."

    return {"answer": answer, "links": dummy_links}

@app.post("/api/")
def ask(data: QueryRequest):
    return answer_question(data)