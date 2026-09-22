"""Coordinate retrieval and generation for one RAG question.

This file is intentionally short. It reads like a map of the answering flow,
while the detailed retrieval and generation work lives in focused modules.
"""

from app.generation import generate_answer
from app.retrieval import build_context, retrieve_documents


def answer_question(question: str) -> dict:
    """Retrieve relevant chunks, generate an answer, and format the response."""

    results = retrieve_documents(question)

    if not results:
        return {
            "answer": "I could not find relevant information in the uploaded documents.",
            "sources": [],
            "topSimilarity": None,
        }

    context, sources = build_context(results)
    answer = generate_answer(context, question)

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
        "answer": answer,
        "sources": sources,
        "topSimilarity": top_similarity,
        "retrievalMethod": "hybrid",
    }
