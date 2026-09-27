"""
api/rag_pipeline.py

Baseline (non-agentic) RAG pipeline for M2: retrieve top-k chunks from
Chroma, then generate an answer with Groq, citing sources inline.

This module intentionally has no retry/grading/self-check logic yet —
that's the M5 agentic upgrade. Keeping this simple version around also
gives us the "before" pipeline for the M4 baseline eval numbers.
"""

import os

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME = "graphqa_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TOP_K = 5

SYSTEM_PROMPT = """You are a documentation assistant for FastAPI. Answer the \
user's question using ONLY the provided context chunks below. Each chunk is \
numbered and includes its source URL.

Rules:
- Cite the chunk number(s) you used inline using ONLY plain ASCII square \
brackets, like [1] or [1][3]. Do not use any other bracket style, special \
Unicode characters, or citation formatting.
- If the context does not contain enough information to answer, say so \
explicitly instead of guessing or using outside knowledge.
- Be concise and technically precise. Include code examples from the \
context when they directly answer the question.
- Use plain ASCII punctuation only (regular hyphens, straight quotes) — \
avoid em-dashes, curly quotes, or other special Unicode punctuation.
"""

# Lazily initialized singletons so importing this module doesn't
# immediately load the embedding model or open a Chroma connection.
_embedding_model: SentenceTransformer | None = None
_chroma_collection = None
_groq_client: Groq | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    model = get_embedding_model()
    collection = get_chroma_collection()

    query_embedding = model.encode(
        [QUERY_INSTRUCTION + query], normalize_embeddings=True
    ).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "url": results["metadatas"][0][i]["url"],
            "title": results["metadatas"][0][i]["title"],
            "section": results["metadatas"][0][i]["section"],
            "distance": results["distances"][0][i],
        })
    return chunks


def build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Source: {chunk['url']} — {chunk['title']} / {chunk['section']}\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> str:
    context_block = build_context_block(chunks)
    client = get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n\n{context_block}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
	max_tokens=800,
    )
    return response.choices[0].message.content


def answer_question(query: str) -> dict:
    chunks = retrieve(query)
    answer = generate_answer(query, chunks)
    sources = sorted({chunk["url"] for chunk in chunks})
    return {"answer": answer, "sources": sources}