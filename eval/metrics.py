"""
eval/metrics.py

RAGAS metric wiring + judge LLM configuration.

Judge models run on Groq through its OpenAI-compatible endpoint, using
ragas's OpenAI adapter (its most mature provider path).

Why this file looks the way it does
-----------------------------------
1. Groq's qwen/qwen3.8-27b and openai/gpt-oss-* are *reasoning* models.
   Their hidden thinking tokens are drawn from the same max_tokens budget
   as the answer, so a small max_tokens meant the model ran out of budget
   before writing its JSON ("output incomplete due to max_tokens" for
   Qwen, empty "failed_generation" for gpt-oss). We now switch thinking
   off for Qwen (reasoning_effort="none") and to the minimum for gpt-oss
   ("low"), so max_tokens only has to cover the JSON verdict itself.

2. ragas/instructor give us no hook to pass extra request parameters, so
   we patch the OpenAI client's create() *before* handing the client to
   ragas. The patch (a) adds the reasoning setting to every request and
   (b) handles Groq rate limits: per-minute limits are waited out using
   the "try again in Xs" hint in the error, while per-day limits raise
   DailyQuotaExhausted so the eval runner can stop cleanly and resume
   later instead of burning retries.

Judge model choice: qwen/qwen3.8-27b (Alibaba) is a different model
family from the app LLM (openai/gpt-oss-120b), which limits
self-preference bias. Optionally set JUDGE_MODEL_LIGHT to move the two
cheaper metrics (answer relevancy, context precision) onto a second
model with its own daily quota - see the README for the tradeoff.

Embeddings for AnswerRelevancy reuse the local bge-small-en-v1.5 model,
so the harness adds no embedding-API cost.
"""

import asyncio
import json
import math
import os
import re

from dotenv import load_dotenv
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
)

load_dotenv()

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b")
JUDGE_MODEL_LIGHT = os.getenv("JUDGE_MODEL_LIGHT", JUDGE_MODEL)
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

MAX_RATE_LIMIT_RETRIES = 8
MAX_BACKOFF_SECONDS = 90.0
DAILY_WAIT_CEILING_SECONDS = 45 * 60  # per-day limits replenish continuously; wait it out below this


class DailyQuotaExhausted(RuntimeError):
    """Groq reported a per-day limit; waiting is not a practical option."""


def parse_retry_after(message: str) -> float | None:
    """Turns Groq's 'Please try again in 2m51.936s' / '592.5ms' into seconds."""
    match = re.search(r"try again in ([0-9hms.]+)", message)
    if not match:
        return None
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    total = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", match.group(1)):
        total += float(value) * units[unit]
    return total or None


def reasoning_params(model: str) -> dict:
    """Request parameters that stop hidden thinking from eating max_tokens."""
    if model.startswith("qwen/"):
        return {"reasoning_effort": "none"}
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    return {}


_embeddings = None

MAX_CONNECTION_RETRIES = 3
CONNECTION_RETRY_DELAY = 2.0


def get_judge_client(model: str) -> AsyncOpenAI:
    """Fresh client per call — avoids stale event-loop bugs.

    RAGAS's metric.score() calls asyncio.run() internally, which creates
    and then closes an event loop each time.  An AsyncOpenAI client cached
    across those calls holds an httpx transport bound to a dead loop,
    causing 'RuntimeError: Event loop is closed' (surfaced by openai as
    APIConnectionError).  Creating a new client each time is cheap and
    side-steps the problem entirely.
    """
    client = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_OPENAI_BASE_URL,
        max_retries=0,  # we do our own, smarter, waiting below
    )
    original_create = client.chat.completions.create
    extra = reasoning_params(model)

    async def create_with_backoff(*args, **kwargs):
        if extra:
            body = dict(kwargs.get("extra_body") or {})
            for key, value in extra.items():
                body.setdefault(key, value)
            kwargs["extra_body"] = body

        last_exc = None
        for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return await original_create(*args, **kwargs)
            except RateLimitError as exc:
                message = str(exc)
                if "per day" in message:
                    wait = parse_retry_after(message)
                    if wait is not None and wait <= DAILY_WAIT_CEILING_SECONDS:
                        print(f"  Daily limit reported, but it replenishes continuously - "
                              f"waiting {wait:.0f}s and retrying …")
                        await asyncio.sleep(wait + 1.0)
                        continue
                    raise DailyQuotaExhausted(f"DAILY_QUOTA_EXHAUSTED: {message}") from exc
                if "Request too large" in message:
                    # OTPM (output tokens per minute) limits are retriable —
                    # wait for the minute window to roll over.
                    if "per minute" in message:
                        wait = parse_retry_after(message) or 62.0
                        if attempt < MAX_RATE_LIMIT_RETRIES:
                            print(f"  OTPM limit hit, waiting {wait:.0f}s for the minute window to reset …")
                            await asyncio.sleep(wait)
                            continue
                    # Truly non-retriable (e.g. input exceeds context window)
                    raise
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                wait = parse_retry_after(message) or 10.0
                await asyncio.sleep(min(wait, MAX_BACKOFF_SECONDS) + 1.0)
            except APIConnectionError as exc:
                last_exc = exc
                if attempt >= MAX_CONNECTION_RETRIES:
                    raise
                print(f"  API connection error (attempt {attempt}/{MAX_CONNECTION_RETRIES}), "
                      f"retrying in {CONNECTION_RETRY_DELAY}s …")
                await asyncio.sleep(CONNECTION_RETRY_DELAY)

    # instructor (inside ragas) wraps client.chat.completions.create when
    # llm_factory() is called, so this must be in place before that.
    client.chat.completions.create = create_with_backoff
    return client


def make_judge_llm(model: str, max_tokens: int):
    if model.startswith("openai/gpt-oss"):
        max_tokens += 600  # gpt-oss can't fully disable thinking; leave room for it
    return llm_factory(model, provider="openai", client=get_judge_client(model), max_tokens=max_tokens)


def get_judge_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL_NAME, device="cpu")
    return _embeddings


def build_metrics() -> dict:
    """The RAGAS metrics plus a lightweight faithfulness scorer.

    RAGAS's built-in Faithfulness uses a verbose NLI schema that repeats
    every claim + adds a reason, easily needing 800+ output tokens for
    8 claims.  Groq free tier caps qwen/qwen3.8-27b at 1000 OTPM, so
    max_tokens can't exceed ~950 and RAGAS's schema doesn't fit.

    We replace it with a compact 2-call scorer (_score_faithfulness_compact)
    that produces the same metric — ratio of context-supported claims —
    with ~30 output tokens for the verdict step.
    """
    embeddings = get_judge_embeddings()
    return {
        "faithfulness": {
            "model": JUDGE_MODEL,
            "max_tokens": 950,
        },
        "answer_relevancy": AnswerRelevancy(llm=make_judge_llm(JUDGE_MODEL_LIGHT, 800), embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=make_judge_llm(JUDGE_MODEL_LIGHT, 800)),
        "context_recall": ContextRecall(llm=make_judge_llm(JUDGE_MODEL, 950)),
    }


# ---------------------------------------------------------------------------
# Compact faithfulness scorer (fits within 1000 OTPM)
# ---------------------------------------------------------------------------
#
# RAGAS Faithfulness internally uses two LLM calls:
#   1. StatementGenerator — extracts atomic claims from the answer
#   2. NLIStatement       — for each claim outputs {statement, reason, verdict}
#
# The NLI output repeats every claim verbatim + writes a reason (~80-100
# tokens per claim).  For 8+ claims this blows past the 950 max_tokens
# ceiling imposed by the 1000 OTPM hard cap on qwen free tier.
#
# Our replacement keeps the same metric semantics (ratio of context-supported
# claims) but uses a compact output format:
#   Call 1: claims as JSON array of strings  (~200-400 output tokens)
#   Call 2: verdicts as JSON array of 0/1    (~30 output tokens)

_EXTRACT_CLAIMS_PROMPT = """\
Given a question and an answer, break the answer into simple, atomic factual \
claims.  Each claim must be a standalone sentence (no pronouns).  Return ONLY \
a JSON object: {{"claims": ["claim1", "claim2", ...]}}

Question: {question}
Answer: {answer}"""

_VERIFY_CLAIMS_PROMPT = """\
Context:
{context}

For each numbered claim below, output 1 if it is directly supported by the \
context, or 0 if not.  Return ONLY a JSON object: {{"verdicts": [1, 0, ...]}}

{claims_numbered}"""


async def _faithfulness_async(
    model: str, max_tokens: int, question: str, answer: str, contexts: list[str]
) -> float | None:
    """Two-call faithfulness scorer with compact output."""
    client = get_judge_client(model)

    # --- Call 1: extract claims -------------------------------------------
    resp1 = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": _EXTRACT_CLAIMS_PROMPT.format(question=question, answer=answer),
        }],
        max_tokens=max_tokens,
        temperature=0,
        response_format={"type": "json_object"},
    )
    text1 = resp1.choices[0].message.content or ""
    try:
        claims = json.loads(text1).get("claims", [])
    except json.JSONDecodeError:
        return None

    if not claims:
        return None  # no claims to verify — same as RAGAS returning NaN

    # --- Call 2: verify claims against context ----------------------------
    context_str = "\n".join(contexts)
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    resp2 = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": _VERIFY_CLAIMS_PROMPT.format(
                context=context_str, claims_numbered=numbered
            ),
        }],
        max_tokens=200,  # verdicts array is tiny
        temperature=0,
        response_format={"type": "json_object"},
    )
    text2 = resp2.choices[0].message.content or ""
    try:
        verdicts = json.loads(text2).get("verdicts", [])
    except json.JSONDecodeError:
        return None

    if not verdicts:
        return None

    # Use the shorter list if lengths mismatch
    n = min(len(claims), len(verdicts))
    return sum(1 for v in verdicts[:n] if v) / n


def _score_faithfulness_compact(
    model: str, max_tokens: int,
    question: str, answer: str, contexts: list[str],
) -> float | None:
    """Synchronous wrapper around the async faithfulness scorer."""
    return asyncio.run(
        _faithfulness_async(model, max_tokens, question, answer, contexts)
    )


def score_metric(
    metrics: dict, name: str, question: str, answer: str, contexts: list[str], reference: str
) -> float | None:
    """Scores one metric for one answer. Returns None when the score
    is undefined (NaN), which is a legitimate outcome."""
    if name == "faithfulness":
        cfg = metrics["faithfulness"]
        return _score_faithfulness_compact(
            cfg["model"], cfg["max_tokens"], question, answer, contexts
        )
    metric = metrics[name]
    if name == "answer_relevancy":
        result = metric.score(user_input=question, response=answer)
    elif name in ("context_precision", "context_recall"):
        result = metric.score(user_input=question, retrieved_contexts=contexts, reference=reference)
    else:
        raise ValueError(f"Unknown metric: {name}")

    value = float(result.value)
    return None if math.isnan(value) else value