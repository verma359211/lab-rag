"""Rerank first-stage search candidates with a local cross-encoder."""

from typing import Any

from app.config import get_reranker_model_name
from app.search.models import SearchResult


# Loading a transformer is expensive, so the same model instance is reused for
# every request after its first use.
_reranker_model: Any = None


def get_reranker_model() -> Any:
    """Load the configured cross-encoder once and return the shared instance."""

    global _reranker_model

    if _reranker_model is None:
        # The import stays here so vector, keyword, and normal hybrid modes do
        # not load PyTorch or the reranker model when they do not need it.
        from sentence_transformers import CrossEncoder

        _reranker_model = CrossEncoder(get_reranker_model_name())

    return _reranker_model


def rerank_results(
    question: str,
    candidates: list[SearchResult],
    limit: int | None = None,
    model: Any = None,
) -> list[SearchResult]:
    """Score each question-and-chunk pair and return the strongest chunks.

    ``model`` is optional and exists mainly so unit tests can use a tiny fake
    model instead of downloading a transformer.
    """

    if not candidates:
        return []

    active_model = model if model is not None else get_reranker_model()
    pairs = [
        (question, candidate.document.page_content)
        for candidate in candidates
    ]
    scores = active_model.predict(pairs)

    for candidate, score in zip(candidates, scores, strict=True):
        candidate.reranker_score = float(score)

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.reranker_score
            if candidate.reranker_score is not None
            else float("-inf")
        ),
        reverse=True,
    )

    for rank, candidate in enumerate(ranked, start=1):
        candidate.reranker_rank = rank

    return ranked if limit is None else ranked[:limit]
