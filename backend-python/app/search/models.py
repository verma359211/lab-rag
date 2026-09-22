"""Shared result shape used by vector, keyword, and fused search."""

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class SearchResult:
    """Hold one chunk and the scores or ranks it received.

    A value is ``None`` when that search method did not return the chunk. The
    final fusion score is calculated from ranks, not from the raw scores.
    """

    document: Document
    vector_score: float | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fusion_score: float = 0.0
