"""
api/main.py

FastAPI app exposing GraphQA-Sentinel.

POST /chat          - agentic pipeline (M5): self-correcting retrieval,
                       grading, retry/web-fallback, self-check
POST /chat/baseline  - original M2 pipeline (plain retrieve-then-generate),
                       kept alongside for manual comparison/debugging
GET  /health         - liveness check
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.graph import answer_question_agentic
from api.rag_pipeline import answer_question as answer_question_baseline

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


class AgenticChatResponse(ChatResponse):
    used_web_fallback: bool = False
    retry_count: int = 0
    self_check_passed: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=AgenticChatResponse)
def chat(request: ChatRequest):
    result = answer_question_agentic(request.question)
    return AgenticChatResponse(**result)


@app.post("/chat/baseline", response_model=ChatResponse)
def chat_baseline(request: ChatRequest):
    result = answer_question_baseline(request.question)
    return ChatResponse(**result)