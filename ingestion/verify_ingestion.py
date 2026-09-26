"""
ingestion/verify_ingestion.py

Quick manual sanity check for M1: embeds a test query and prints the top-k
retrieved chunks so you can eyeball whether retrieval is on-topic.
"""

import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME = "graphqa_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge models recommend prefixing the QUERY (not the passage) with this
# instruction string for retrieval tasks.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

TEST_QUERY = "how do I add a dependency in FastAPI?"
TOP_K = 3


def main():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode(
        [QUERY_INSTRUCTION + TEST_QUERY], normalize_embeddings=True
    ).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)

    print(f"Query: {TEST_QUERY}\n")
    for i in range(len(results["ids"][0])):
        print(f"--- Result {i + 1} ---")
        print(f"URL: {results['metadatas'][0][i]['url']}")
        print(f"Section: {results['metadatas'][0][i]['section']}")
        print(f"Distance: {results['distances'][0][i]:.4f}")
        print(results["documents"][0][i][:300].replace("\n", " ") + " ...")
        print()


if __name__ == "__main__":
    main()