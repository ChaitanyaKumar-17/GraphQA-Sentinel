"""
eval/metrics.py

RAGAS metric wrappers + judge LLM configuration.

Judge model: qwen/qwen3.8-27b via Groq. Both qwen/qwen3.8-27b and
openai/gpt-oss-20b are documented by Groq as supporting strict-mode
structured outputs, but in practice only qwen/qwen3.8-27b reliably
produced valid JSON through this project's actual call path (via
ragas -> instructor -> Groq's OpenAI-compatible endpoint); gpt-oss-20b
consistently failed JSON validation. qwen/qwen3.8-27b's real constraint
is a tight per-model output-token-per-minute (OTPM) budget on Groq's
free tier (measured at 1000 tokens/minute), addressed here by giving
each metric its own right-sized max_tokens (Faithfulness decomposes an
entire answer into individual claims and needs real room; the other
three metrics return a much simpler verdict and need far less).

qwen/qwen3.8-27b (Alibaba) is a genuinely different model family from
the app LLM (openai/gpt-oss-120b via Groq), preserving the project's
anti-self-preference-bias design.

Embeddings for AnswerRelevancy reuse the same local bge-small-en-v1.5
model used elsewhere in the pipeline, so the eval harness itself doesn't
introduce any extra embedding-API cost.
"""

import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

load_dotenv()

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b")
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
JUDGE_RATE_LIMIT_DELAY_SECONDS = 15  # generous spacing to let the 60s OTPM window clear

_judge_client = None
_judge_embeddings = None


def get_judge_client() -> AsyncOpenAI:
    global _judge_client
    if _judge_client is None:
        _judge_client = AsyncOpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url=GROQ_OPENAI_BASE_URL,
        )
    return _judge_client


def make_judge_llm(max_tokens: int):
    """Creates a fresh judge LLM wrapper pinned to a specific max_tokens
    budget. Each metric gets its own instance so lightweight metrics
    don't reserve as much OTPM headroom as Faithfulness needs."""
    client = get_judge_client()
    return llm_factory(JUDGE_MODEL, provider="openai", client=client, max_tokens=max_tokens)


def get_judge_embeddings():
    global _judge_embeddings
    if _judge_embeddings is None:
        _judge_embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL_NAME, device="cpu")
    return _judge_embeddings


def build_metrics() -> dict:
    """Returns the four RAGAS metrics this project tracks, each wired to
    its own right-sized judge LLM instance (and local embeddings for
    AnswerRelevancy)."""
    embeddings = get_judge_embeddings()
    return {
        "faithfulness": Faithfulness(llm=make_judge_llm(max_tokens=900)),
        "answer_relevancy": AnswerRelevancy(llm=make_judge_llm(max_tokens=400), embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=make_judge_llm(max_tokens=400)),
        "context_recall": ContextRecall(llm=make_judge_llm(max_tokens=400)),
    }


def score_sample(
    metrics: dict, question: str, answer: str, contexts: list[str], reference: str
) -> dict:
    """Scores one Q/A pair across all four metrics. Returns {metric_name: float}."""
    scores = {}

    faith_result = metrics["faithfulness"].score(
        user_input=question, response=answer, retrieved_contexts=contexts
    )
    scores["faithfulness"] = faith_result.value
    time.sleep(JUDGE_RATE_LIMIT_DELAY_SECONDS)

    relevancy_result = metrics["answer_relevancy"].score(
        user_input=question, response=answer
    )
    scores["answer_relevancy"] = relevancy_result.value
    time.sleep(JUDGE_RATE_LIMIT_DELAY_SECONDS)

    precision_result = metrics["context_precision"].score(
        user_input=question, retrieved_contexts=contexts, reference=reference
    )
    scores["context_precision"] = precision_result.value
    time.sleep(JUDGE_RATE_LIMIT_DELAY_SECONDS)

    recall_result = metrics["context_recall"].score(
        user_input=question, retrieved_contexts=contexts, reference=reference
    )
    scores["context_recall"] = recall_result.value

    return scores