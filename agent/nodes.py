"""
agent/nodes.py

LangGraph node functions for the agentic RAG pipeline (M5):
query_analyzer, retriever, grader, generator, self_check - plus the
two conditional-edge routing functions the graph uses.

Each node takes the current graph state and returns a dict of only the
keys it updates; LangGraph merges these into the running state.
"""

import json

from agent.tools import web_search
from api.rag_pipeline import (
    GROQ_MODEL,
    SYSTEM_PROMPT as GENERATOR_SYSTEM_PROMPT,
    build_context_block,
    get_groq_client,
    retrieve,
)

import time

from groq import RateLimitError

MAX_RETRIES = 2
MIN_RELEVANT_CHUNKS = 2

QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite user questions into effective \
search queries for a documentation retrieval system covering FastAPI. \
Given the original question (and, if this is a retry, the previous search \
query that failed to find enough relevant documentation), produce ONE \
improved search query. Make it specific, use terminology likely to appear \
in official docs, and if this is a retry, try a genuinely different angle \
or phrasing than the previous attempt - do not just lightly reword it. \
Respond with ONLY the rewritten query text, nothing else - no quotes, no \
explanation.
"""

GRADER_SYSTEM_PROMPT = """You grade the relevance of retrieved documentation \
chunks against a search query. For EACH chunk, decide if it is genuinely \
relevant and useful for answering the query (1) or not (0). Respond with \
ONLY a JSON array of 0/1 integers, one per chunk, in the same order as the \
chunks were given. Example for 3 chunks: [1, 0, 1]. No other text.
"""

SELF_CHECK_SYSTEM_PROMPT = """You fact-check a draft answer against the \
context it was supposed to be based on. Check whether EVERY factual claim \
in the draft answer is actually supported by the provided context. Respond \
with ONLY a JSON object: {"faithful": true or false, "unsupported_claims": \
[list of specific claims not supported by the context, empty list if none]}. \
No other text.
"""

STRICT_REGENERATE_SYSTEM_PROMPT = """You are a documentation assistant for \
FastAPI. The previous answer included claims not supported by the provided \
context. Regenerate the answer using ONLY information explicitly stated in \
the context below. If the context does not fully support an answer, say so \
explicitly rather than filling gaps with outside knowledge. Cite chunk \
numbers inline like [1] or [1][3]. Use plain ASCII punctuation only (no \
em-dashes or curly quotes).
"""


LLM_CALL_DELAY_SECONDS = 6  # keep comfortably under gpt-oss-120b's 8000 TPM free-tier ceiling
MAX_LLM_RETRIES = 3


def _llm_call(system_prompt: str, user_content: str, temperature: float = 0.1) -> str:
    client = get_groq_client()

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=800,
            )
            time.sleep(LLM_CALL_DELAY_SECONDS)
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            if attempt == MAX_LLM_RETRIES:
                raise
            wait = 10 * attempt
            print(f"  [agent LLM call rate-limited, retrying in {wait}s] {e}")
            time.sleep(wait)


def query_analyzer_node(state: dict) -> dict:
    """Rewrites the question into a search-friendly query. On a retry
    (rewritten_query already set from a previous pass), tries a
    genuinely different angle rather than a light reword."""
    is_retry = bool(state.get("rewritten_query"))

    if is_retry:
        user_content = (
            f"Original question: {state['question']}\n"
            f"Previous search query (found too little relevant documentation): "
            f"{state['rewritten_query']}\n"
            "Produce a different search query."
        )
        retry_count = state.get("retry_count", 0) + 1
    else:
        user_content = f"Original question: {state['question']}"
        retry_count = 0

    rewritten = _llm_call(QUERY_REWRITE_SYSTEM_PROMPT, user_content)
    return {"rewritten_query": rewritten, "retry_count": retry_count}


def retriever_node(state: dict) -> dict:
    """Retrieves top-k chunks from Chroma using the current rewritten query."""
    chunks = retrieve(state["rewritten_query"])
    return {"retrieved_chunks": chunks}


def grader_node(state: dict) -> dict:
    """Grades each retrieved chunk 0/1 relevant against the rewritten
    query, in a single batched LLM call (cheaper and faster than one
    call per chunk, and just as effective for this workload)."""
    chunks = state["retrieved_chunks"]
    if not chunks:
        return {"graded_chunks": []}

    numbered = "\n\n".join(
        f"[{i}] {c['text'][:600]}" for i, c in enumerate(chunks, start=1)
    )
    user_content = f"Query: {state['rewritten_query']}\n\nChunks:\n\n{numbered}"

    raw = _llm_call(GRADER_SYSTEM_PROMPT, user_content, temperature=0.0)

    try:
        verdicts = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        verdicts = [0] * len(chunks)

    graded = [c for c, verdict in zip(chunks, verdicts) if verdict == 1]
    return {"graded_chunks": graded}


def route_after_grading(state: dict) -> str:
    """Conditional edge: enough relevant chunks -> generate. Otherwise
    retry (rewrite + retrieve again) up to MAX_RETRIES times total,
    then fall back to web search."""
    if len(state.get("graded_chunks", [])) >= MIN_RELEVANT_CHUNKS:
        return "generate"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "web_search"


def web_search_node(state: dict) -> dict:
    """Falls back to a scoped web search when local retrieval, even
    after retries, didn't find enough relevant documentation."""
    results = web_search(state["rewritten_query"])
    return {"web_results": results, "used_web_fallback": True}


def generator_node(state: dict) -> dict:
    """Generates an answer with inline citations from whichever context
    is available (graded local chunks, or web fallback results). Uses a
    stricter prompt on the one-time regeneration pass after a failed
    self-check."""
    context_chunks = state.get("web_results") or state["graded_chunks"]
    context_block = build_context_block(context_chunks)

    is_regeneration = state.get("self_check_passed") is False
    system_prompt = STRICT_REGENERATE_SYSTEM_PROMPT if is_regeneration else GENERATOR_SYSTEM_PROMPT
    user_content = f"Context:\n\n{context_block}\n\nQuestion: {state['question']}"

    answer = _llm_call(system_prompt, user_content)
    sources = sorted({c["url"] for c in context_chunks})

    update = {"draft_answer": answer, "final_answer": answer, "sources": sources}
    if is_regeneration:
        update["regenerated"] = True
    return update


def self_check_node(state: dict) -> dict:
    """Verifies every claim in the draft answer traces back to the
    context; the router decides whether to regenerate based on this."""
    context_chunks = state.get("web_results") or state["graded_chunks"]
    context_block = build_context_block(context_chunks)

    user_content = f"Context:\n\n{context_block}\n\nDraft answer:\n\n{state['draft_answer']}"
    raw = _llm_call(SELF_CHECK_SYSTEM_PROMPT, user_content, temperature=0.0)

    try:
        result = json.loads(raw)
        faithful = bool(result.get("faithful", True))
    except (json.JSONDecodeError, TypeError):
        faithful = True

    return {"self_check_passed": faithful}


def route_after_self_check(state: dict) -> str:
    """Conditional edge: regenerate once (strictly) if the self-check
    failed and we haven't already regenerated; otherwise finish."""
    if not state.get("self_check_passed", True) and not state.get("regenerated"):
        return "regenerate"
    return "end"