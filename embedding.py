import os
import json
import re
import glob
import nltk
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

CHUNK_SIZE = 500  # ~words
CHUNK_OVERLAP = 100  # overlap between chunks

def clean_text(text):
    # Remove HTML and markdown leftovers
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    total_words = 0
    for sentence in sentences:
        words = sentence.split()
        total_words += len(words)
        current_chunk.append(sentence)

        if total_words >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # overlap part
            current_chunk = current_chunk[-(overlap//10):]  # approx overlap
            total_words = sum(len(s.split()) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def process_discourse(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    chunked = []
    for post in posts:
        content = clean_text(post.get("content", ""))
        if not content.strip():
            continue

        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            chunked.append({
                "source": "discourse",
                "topic_title": post.get("topic_title"),
                "url": post.get("url"),
                "chunk_id": f"{post['post_id']}_{i}",
                "text": chunk
            })
    return chunked

def process_markdown_folder(folder_path):
    chunked = []
    md_files = glob.glob(os.path.join(folder_path, "*.md"))
    
    for path in md_files:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        meta_match = re.search(r'^---(.*?)---', raw, re.DOTALL)
        content = raw
        metadata = {}

        if meta_match:
            meta_text = meta_match.group(1)
            content = raw[meta_match.end():]
            for line in meta_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    metadata[k.strip()] = v.strip().strip('"')

        title = metadata.get("title", os.path.basename(path).replace(".md", ""))
        url = metadata.get("original_url", "")
        clean = clean_text(content)
        if not clean.strip():
            continue

        chunks = chunk_text(clean)
        for i, chunk in enumerate(chunks):
            chunked.append({
                "source": "tds",
                "title": title,
                "url": url,
                "chunk_id": f"{title}_{i}",
                "text": chunk
            })

    return chunked

def save_chunks(chunks, output_file="chunks/all_chunks.jsonl"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

def main():
    discourse_chunks = process_discourse("discourse_posts.json")
    tds_chunks = process_markdown_folder("tds_pages_md")
    all_chunks = discourse_chunks + tds_chunks

    print(f"✅ Total chunks: {len(all_chunks)} (Discourse: {len(discourse_chunks)}, TDS: {len(tds_chunks)})")
    save_chunks(all_chunks)

if __name__ == "__main__":
    main()
