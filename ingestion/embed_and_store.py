"""
ingestion/embed_and_store.py

Embeds each chunk with BAAI/bge-small-en-v1.5 and writes it, along with its
metadata, into a persistent local ChromaDB collection.
"""

import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

INPUT_PATH = Path(__file__).parent / "data" / "processed" / "chunks.json"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME = "graphqa_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found — did you run chunk.py first?")
        return

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Fresh start each run: drop any existing collection with this name so
    # re-running ingestion doesn't create duplicate entries.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    print(f"Embedding and storing {len(chunks)} chunks ...")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # bge models use a "query: " instruction prefix convention for
        # asymmetric search; passages are embedded as-is (no prefix). The
        # matching query-side prefix is applied later, in the retriever node.
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"url": c["url"], "title": c["title"], "section": c["section"]}
                for c in batch
            ],
        )
        print(f"  stored {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")
    print(f"Persisted at: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()