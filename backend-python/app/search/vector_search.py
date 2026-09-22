"""Retrieve semantically similar chunks with Gemini and PGVector."""

from app.config import SEARCH_CANDIDATE_LIMIT
from app.search.models import SearchResult
from app.vector_store import get_vector_store


def search_by_vector(question: str) -> list[SearchResult]:
    """Return chunks whose meaning is closest to the question.

    LangChain embeds the question and asks PGVector for nearby chunk vectors.
    We request more results than will reach the prompt because rank fusion needs
    a useful candidate pool from both search methods.
    """

    matches = get_vector_store().similarity_search_with_relevance_scores(
        query=question,
        k=SEARCH_CANDIDATE_LIMIT,
    )

    return [
        SearchResult(document=document, vector_score=float(score))
        for document, score in matches
    ]
