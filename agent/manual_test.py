"""
agent/manual_test.py

Quick manual sanity check for M5: runs a few questions through the
agentic graph and prints the diagnostic fields (retries used, web
fallback triggered, self-check outcome) alongside the answer.
"""

from agent.graph import answer_question_agentic

TEST_QUESTIONS = [
    "How do I add a dependency in FastAPI?",
    "How do I connect FastAPI to a Kafka message queue?",
    "How do dependencies with yield interact with background tasks?",
]

for q in TEST_QUESTIONS:
    print(f"\n{'=' * 70}\nQ: {q}\n{'=' * 70}")
    result = answer_question_agentic(q)
    print(f"Retries used: {result['retry_count']}")
    print(f"Web fallback used: {result['used_web_fallback']}")
    print(f"Self-check passed: {result['self_check_passed']}")
    print(f"\nA: {result['answer']}")
    print(f"\nSources: {result['sources']}")