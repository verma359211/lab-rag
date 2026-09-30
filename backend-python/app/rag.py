"""Coordinate retrieval and generation for one RAG question.

This file is intentionally short. It reads like a map of the answering flow,
while the detailed retrieval and generation work lives in focused modules.
"""

from dataclasses import dataclass, field
from time import perf_counter

from app.generation import generate_answer
from app.retrieval import (
    RetrievalTrace,
    build_context,
    build_retrieval_details,
    retrieve_with_diagnostics,
)
from app.search.models import SearchMode


NO_CONTEXT_ANSWER = "I could not find relevant information in the uploaded documents."


@dataclass
class RagExecution:
    """Internal result shared by production chat and offline evaluation."""

    question: str
    search_mode: SearchMode
    answer: str
    context: str
    contexts: list[str]
    sources: list[str]
    retrieval: RetrievalTrace
    latency_ms: dict[str, float] = field(default_factory=dict)


def _milliseconds(started_at: float) -> float:
    """Return elapsed milliseconds for one measured stage."""

    return round((perf_counter() - started_at) * 1000, 3)


def execute_rag(
    question: str,
    search_mode: SearchMode = "hybrid",
) -> RagExecution:
    """Run the application's actual retrieval, context, and generation flow.

    The returned diagnostics are used only by trusted internal callers such as
    the offline evaluator. ``answer_question`` below deliberately converts
    this object into the existing, smaller public response.
    """

    total_started_at = perf_counter()
    retrieval = retrieve_with_diagnostics(question, search_mode)

    if not retrieval.final_results:
        return RagExecution(
            question=question,
            search_mode=search_mode,
            answer=NO_CONTEXT_ANSWER,
            context="",
            contexts=[],
            sources=[],
            retrieval=retrieval,
            latency_ms={"total": _milliseconds(total_started_at)},
        )

    context_started_at = perf_counter()
    context, sources = build_context(retrieval.final_results)
    context_latency = _milliseconds(context_started_at)

    generation_started_at = perf_counter()
    answer = generate_answer(context, question)
    generation_latency = _milliseconds(generation_started_at)

    return RagExecution(
        question=question,
        search_mode=search_mode,
        answer=answer,
        context=context,
        contexts=[
            result.document.page_content
            for result in retrieval.final_results
        ],
        sources=sources,
        retrieval=retrieval,
        latency_ms={
            "context_building": context_latency,
            "generation": generation_latency,
            "total": _milliseconds(total_started_at),
        },
    )


def answer_question(question: str, search_mode: SearchMode = "hybrid") -> dict:
    """Retrieve relevant chunks, generate an answer, and format the response."""

    execution = execute_rag(question, search_mode)
    results = execution.retrieval.final_results

    if not results:
        return {
            "answer": execution.answer,
            "sources": [],
            "topSimilarity": None,
            "retrievalMethod": search_mode,
            "retrievalDetails": [],
        }

    # Preserve the existing frontend field. With hybrid retrieval, the first
    # result may have come from keyword search only, so we use the first vector
    # score that is available among the chunks included in the prompt.
    top_similarity = next(
        (
            result.vector_score
            for result in results
            if result.vector_score is not None
        ),
        None,
    )

    return {
        "answer": execution.answer,
        "sources": execution.sources,
        "topSimilarity": top_similarity,
        "retrievalMethod": search_mode,
        "retrievalDetails": build_retrieval_details(results),
    }
