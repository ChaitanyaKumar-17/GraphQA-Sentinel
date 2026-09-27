"""
api/main.py

FastAPI app exposing GraphQA-Sentinel.

POST /chat          - agentic pipeline (M5): self-correcting retrieval,
                       grading, retry/web-fallback, self-check
POST /chat/baseline  - original M2 pipeline (plain retrieve-then-generate),
                       kept alongside for manual comparison/debugging
GET  /health         - liveness check
GET  /               - serves the frontend chat UI (frontend/index.html)
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.graph import answer_question_agentic
from api.rag_pipeline import answer_question as answer_question_baseline
from api.rag_pipeline import get_embedding_model

app = FastAPI(title="GraphQA-Sentinel API")

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


@app.on_event("startup")
def prewarm_embedding_model():
    # Loads the bge-small-en-v1.5 weights once at container startup, so
    # the first real user request isn't the one waiting on that download.
    get_embedding_model()


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


# Serves the chat UI for every path not already matched above. Mounted
# last so it acts as a fallback rather than shadowing the API routes.
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")