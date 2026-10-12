# Eval Pipeline Bug Analysis

Reference document for all bugs found and fixed in the eval scoring pipeline
and the agentic RAG system.

---

## Bug 1: "Connection error" — Stale AsyncOpenAI client across event loops

**File:** `eval/metrics.py`
**Status:** FIXED ✅

### Problem

The global `_clients: dict[str, AsyncOpenAI]` cached `AsyncOpenAI` instances.
RAGAS's `metric.score()` calls `asyncio.run(self.ascore(...))` internally, which:
1. Creates a **new event loop**
2. Runs the async scoring
3. **Closes and destroys** the event loop

After the first `metric.score()` call, the cached client's `httpx.AsyncClient`
transport is bound to the now-dead event loop. The next call creates a new loop,
but the cached client still references the old closed one, leading to
`APIConnectionError("Connection error.")`.

### Fix applied

Removed global client cache. `get_judge_client()` now creates a fresh
`AsyncOpenAI` client per call. Also added `APIConnectionError` to the retry
wrapper (3 retries, 2s delay).

---

## Bug 2: "max_tokens length limit" — RAGAS NLI schema too verbose for OTPM cap

**File:** `eval/metrics.py`
**Status:** FIXED ✅

### Problem (evolved through investigation)

**Initial diagnosis:** Token budgets in `build_metrics()` were too small
(faithfulness=700). Increasing to 950 was the max possible because Groq's free
tier caps `qwen/qwen3.8-27b` at **1000 OTPM** (output tokens per minute) — a
single request can't have `max_tokens > ~989`.

**Root cause:** Even 950 tokens wasn't enough. RAGAS Faithfulness internally
uses an `NLIStatementOutput` schema that repeats every claim verbatim + adds a
reason per verdict:

```json
{"statement": "full claim repeated", "reason": "explanation...", "verdict": 0}
```

That's ~80-100 tokens per claim. For 8+ claims this blows past 950, regardless
of answer length. The answers were only 116-193 words — truncation was never the
real issue.

### Fix applied

Replaced RAGAS's built-in `Faithfulness` metric with a custom 2-call scorer
(`_score_faithfulness_compact`) that keeps the same metric semantics (ratio of
context-supported claims) but uses compact output:

| Call | RAGAS output | Custom output |
|---|---|---|
| 1: Extract claims | `{"statements": [...]}` | `{"claims": [...]}` |
| 2: Verify claims | `[{statement, reason, verdict}, ...]` ~800-1200 tok | `{"verdicts": [1,0,1,...]}` ~30 tok |

Same metric definition, fraction of the output tokens.

---

## Bug 3: Self-check "fails open" — strict regeneration never fires

**File:** `agent/nodes.py`
**Status:** FIXED ✅

### Problem

```python
# Old code
faithful = bool(result.get("faithful", True))   # missing key → True
except ...:
    faithful = True                               # parse error → True
```

The self-check node defaulted to `faithful = True` on both JSON parse errors and
missing keys. This meant `STRICT_REGENERATE_SYSTEM_PROMPT` was practically never
used — the draft answer passed through even when it contained unsupported claims.
The README's own "Known limitations" section documented this as a known issue.

### Fix applied

Changed both defaults to `False` (fail closed). If we can't verify faithfulness,
assume it's not faithful and regenerate with the stricter prompt.

---

## Bug 4: Generator prompt encourages hallucination

**File:** `agent/nodes.py`
**Status:** FIXED ✅

### Problem

The agentic pipeline's first-pass generator used the same `SYSTEM_PROMPT` as the
baseline pipeline, which contained:

- *"the way a knowledgeable colleague would explain it"*
- *"Explain the 'why' briefly where it helps"*

These phrases invited the LLM to add background knowledge from its training data.
Combined with the grader narrowing context from 5 chunks to 2-3, the generator
had fewer chunks to work with and was more likely to pad with parametric
knowledge. RAGAS then checked faithfulness against only the graded chunks,
finding the added claims unsupported.

### Fix applied

Defined a new `AGENTIC_GENERATOR_SYSTEM_PROMPT` specifically for the agentic
pipeline, replacing "explain the why like a colleague" with **"every factual
claim in your answer must be directly supported by the provided context"**. The
baseline pipeline's prompt is untouched.

---

## Results

### Before fixes

Faithfulness scoring failed entirely — 0/8 factual+multi_hop questions scored
due to `IncompleteOutputException` (Bug 2) and `APIConnectionError` (Bug 1).

### After all fixes

| Metric | Baseline | Agentic (before Bug 3+4 fix) | Agentic (after all fixes) |
|---|---|---|---|
| Faithfulness | 0.798 | 0.740 | **0.862** |
| Answer Relevancy | 0.625 | 0.663 | **0.765** |
| Context Precision | 0.574 | 0.571 | 0.540 |
| Context Recall | 0.597 | 0.542 | 0.542 |

Factual faithfulness: 0.727 → **0.916** (now beats baseline's 0.866).
Multi-hop faithfulness: 0.802 → **0.880** (beats baseline's 0.610).

---

## Remaining issue: Out-of-scope leakage

**Status:** NOT FIXED — documented for future work

The agentic pipeline's web fallback answers questions it should decline
(`answer_relevancy` 0.694 vs baseline's 0.000 for out-of-scope). Per the project
spec (`01_agentic_rag_eval_harness.md` line 132), out-of-scope questions should
be **refused**. The web fallback is scoped to FastAPI docs/GitHub and is meant
for retrieval gaps in the local corpus, not for answering off-topic questions.

**Root cause:** There is no scope check before answering. When local retrieval
finds < 2 relevant chunks for an off-topic question, the system retries and
eventually falls back to web search. The web search may find tangentially
related content, and the generator answers from whatever it gets instead of
recognizing the question is out of scope.

---

## Milestone status (updated)

| Milestone | Status |
|---|---|
| M1–M3 Ingestion, baseline RAG, golden set | ✅ Done |
| M4 Eval harness with real baseline numbers | ✅ Done |
| M5 Agentic pipeline scored | ✅ Done |
| M6 Regression log + dashboard | ✅ Has data (3 logged runs). Dashboard needs verification |
| M7 CI gate deliberate-failure test | ⬜ Not run yet |
| M8 Docker | ✅ Done (local only) |
| M9 README | ⬜ Needs eval numbers table + known limitations update |

## Next steps

1. **Update README** — fill in the eval numbers table (currently "pending")
   and update the known limitations section (self-check is no longer fail-open)
2. **Verify dashboard** — run `streamlit run dashboard/app.py` and confirm
   the 3 logged runs plot correctly (definition of done requires 3+ runs)
3. **CI gate test** — intentionally break something (e.g. reduce k to 1),
   push, confirm CI fails, then revert
4. **Fix out-of-scope leakage** (optional) — add a scope classifier before
   the web fallback that refuses questions unrelated to the docs corpus
