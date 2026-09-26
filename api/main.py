"""
api/main.py

FastAPI app exposing the GraphQA-Sentinel baseline RAG pipeline.
POST /chat  - ask a question, get an answer + source URLs
GET  /health - liveness check
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.rag_pipeline import answer_question

app = FastAPI(title="GraphQA-Sentinel API")

# Dev-friendly CORS: the frontend is a static HTML file with no fixed
# origin, so we allow all origins here. Tighten this before any real
# public deployment beyond a portfolio demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = answer_question(request.question)
    return ChatResponse(**result)