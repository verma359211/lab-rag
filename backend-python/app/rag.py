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

    # Results are already ordered from most relevant to least relevant, so the
    # first score gives the frontend a simple view of the strongest match.
    top_similarity = results[0][1]

    return {
        "answer": answer,
        "sources": sources,
        "topSimilarity": top_similarity,
    }
