"""
agent/graph.py

LangGraph state machine for GraphQA-Sentinel's agentic RAG pipeline (M5):

    query_analyzer -> retriever -> grader
        --(>=2 relevant chunks)--------> generator -> self_check -> [regenerate once, or end]
        --(not enough, retries left)---> query_analyzer (retry, different angle)
        --(not enough, retries used)---> web_search -> generator -> self_check -> ...
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    generator_node,
    grader_node,
    query_analyzer_node,
    retriever_node,
    route_after_grading,
    route_after_self_check,
    self_check_node,
    web_search_node,
)


class GraphState(TypedDict, total=False):
    question: str
    rewritten_query: str
    retry_count: int
    retrieved_chunks: list[dict]
    graded_chunks: list[dict]
    used_web_fallback: bool
    web_results: list[dict]
    draft_answer: str
    final_answer: str
    self_check_passed: bool
    regenerated: bool
    sources: list[str]


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("query_analyzer", query_analyzer_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generator", generator_node)
    graph.add_node("self_check", self_check_node)

    graph.add_edge(START, "query_analyzer")
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever", "grader")

    graph.add_conditional_edges(
        "grader",
        route_after_grading,
        {
            "generate": "generator",
            "retry": "query_analyzer",
            "web_search": "web_search",
        },
    )

    graph.add_edge("web_search", "generator")
    graph.add_edge("generator", "self_check")

    graph.add_conditional_edges(
        "self_check",
        route_after_self_check,
        {
            "regenerate": "generator",
            "end": END,
        },
    )

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def answer_question_agentic(question: str) -> dict:
    """Runs the full agentic pipeline for one question. Shaped like
    api.rag_pipeline.answer_question()'s output (answer, sources), plus
    diagnostic fields, plus the context passages the answer was actually
    generated from (needed by the eval harness; the API ignores it)."""
    app = get_graph()
    result = app.invoke({"question": question})

    context_chunks = result.get("web_results") or result.get("graded_chunks") or []

    return {
        "answer": result.get("final_answer", ""),
        "sources": result.get("sources", []),
        "contexts": [c["text"] for c in context_chunks],
        "used_web_fallback": result.get("used_web_fallback", False),
        "retry_count": result.get("retry_count", 0),
        "self_check_passed": result.get("self_check_passed", True),
    }