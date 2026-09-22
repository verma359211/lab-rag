"""Combine vector and keyword rankings with Reciprocal Rank Fusion."""

from langchain_core.documents import Document

from app.config import RETRIEVAL_LIMIT, RRF_K
from app.search.models import SearchResult


def get_document_key(document: Document) -> str:
    """Return a stable identity used to detect the same chunk in both lists."""

    # New chunks contain chunk_id metadata. LangChain also gives every stored
    # row an ID, so documents ingested before this feature still work.
    chunk_id = document.metadata.get("chunk_id")

    if chunk_id:
        return str(chunk_id)

    if document.id:
        return str(document.id)

    # This final fallback mainly helps small unit tests or manually constructed
    # documents. Database results should always have one of the IDs above.
    source = document.metadata.get("source", "")
    page = document.metadata.get("page", "")
    return f"{source}|{page}|{document.page_content}"


def fuse_results(
    vector_results: list[SearchResult],
    keyword_results: list[SearchResult],
) -> list[SearchResult]:
    """Merge two ordered result lists and return the strongest chunks.

    Raw vector and keyword scores cannot be compared because they use different
    scales. Reciprocal Rank Fusion uses positions instead. A chunk receives one
    contribution for its vector rank and another for its keyword rank.
    """

    combined: dict[str, SearchResult] = {}

    for rank, result in enumerate(vector_results, start=1):
        key = get_document_key(result.document)
        fused = combined.setdefault(key, SearchResult(document=result.document))

        fused.vector_score = result.vector_score
        fused.vector_rank = rank
        fused.fusion_score += 1 / (RRF_K + rank)

    for rank, result in enumerate(keyword_results, start=1):
        key = get_document_key(result.document)
        fused = combined.setdefault(key, SearchResult(document=result.document))

        fused.keyword_score = result.keyword_score
        fused.keyword_rank = rank
        fused.fusion_score += 1 / (RRF_K + rank)

    ranked_results = sorted(
        combined.values(),
        key=lambda result: result.fusion_score,
        reverse=True,
    )

    return ranked_results[:RETRIEVAL_LIMIT]
