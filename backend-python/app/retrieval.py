"""Select a retrieval method and prepare its chunks for the language model.

Retrieval is the first half of answering a question. Vector search finds
similar meanings, keyword search finds exact terms, and Reciprocal Rank Fusion
combines their ordered result lists.
"""

from dataclasses import dataclass, field
from time import perf_counter

from app.config import RERANK_CANDIDATE_LIMIT, RETRIEVAL_LIMIT
from app.search.fusion import fuse_results
from app.search.keyword_search import search_by_keyword
from app.search.models import SearchMode, SearchResult
from app.search.reranker import rerank_results
from app.search.vector_search import search_by_vector


@dataclass
class RetrievalTrace:
    """Keep the detailed retrieval work needed by offline evaluation.

    This object is internal. The normal chat endpoint never serializes it, so
    candidate lists and timing details do not become part of the public API.
    """

    vector_candidates: list[SearchResult] = field(default_factory=list)
    keyword_candidates: list[SearchResult] = field(default_factory=list)
    fused_candidates: list[SearchResult] = field(default_factory=list)
    reranked_candidates: list[SearchResult] = field(default_factory=list)
    final_results: list[SearchResult] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    reranker_applied: bool = False
    reranker_error: str | None = None


def _milliseconds(started_at: float) -> float:
    """Return elapsed milliseconds for one measured stage."""

    return round((perf_counter() - started_at) * 1000, 3)


def retrieve_with_diagnostics(
    question: str,
    search_mode: SearchMode = "hybrid",
) -> RetrievalTrace:
    """Run the real retrievers and retain their intermediate rankings.

    Production chat and offline evaluation both use this function. That keeps
    evaluation honest: changing application retrieval also changes evaluation
    automatically instead of leaving a second pipeline out of date.
    """

    total_started_at = perf_counter()
    trace = RetrievalTrace()

    if search_mode in ("vector", "hybrid", "hybrid_rerank"):
        stage_started_at = perf_counter()
        trace.vector_candidates = search_by_vector(question)
        trace.latency_ms["vector_search"] = _milliseconds(stage_started_at)

    if search_mode in ("keyword", "hybrid", "hybrid_rerank"):
        stage_started_at = perf_counter()
        trace.keyword_candidates = search_by_keyword(question)
        trace.latency_ms["keyword_search"] = _milliseconds(stage_started_at)

    if search_mode == "vector":
        trace.final_results = trace.vector_candidates[:RETRIEVAL_LIMIT]
    elif search_mode == "keyword":
        trace.final_results = trace.keyword_candidates[:RETRIEVAL_LIMIT]
    else:
        stage_started_at = perf_counter()
        trace.fused_candidates = fuse_results(
            trace.vector_candidates,
            trace.keyword_candidates,
            limit=None,
        )
        trace.latency_ms["fusion"] = _milliseconds(stage_started_at)

        if search_mode == "hybrid_rerank":
            stage_started_at = perf_counter()
            candidates = trace.fused_candidates[:RERANK_CANDIDATE_LIMIT]

            try:
                trace.reranked_candidates = rerank_results(
                    question,
                    candidates,
                    limit=None,
                )
                trace.reranker_applied = True
            except Exception as error:
                # RRF remains a safe fallback if the optional local model is
                # unavailable. Diagnostics keep the failure visible.
                trace.reranked_candidates = candidates
                trace.reranker_error = str(error)

            trace.latency_ms["reranking"] = _milliseconds(stage_started_at)
            trace.final_results = trace.reranked_candidates[:RETRIEVAL_LIMIT]
        else:
            trace.final_results = trace.fused_candidates[:RETRIEVAL_LIMIT]

    trace.latency_ms["retrieval_total"] = _milliseconds(total_started_at)
    return trace


def retrieve_documents(
    question: str,
    search_mode: SearchMode = "hybrid",
) -> list[SearchResult]:
    """Return the best chunks using the retrieval method selected by the user.

    Vector-only and keyword-only modes simply keep their five highest-ranked
    chunks. Hybrid mode combines both lists with Reciprocal Rank Fusion. The
    reranked hybrid mode then uses a cross-encoder before keeping five chunks.
    """

    return retrieve_with_diagnostics(question, search_mode).final_results


def build_retrieval_details(results: list[SearchResult]) -> list[dict]:
    """Create frontend-friendly score details for every retrieved chunk.

    Scores remain separate because vector relevance, PostgreSQL keyword rank,
    and RRF fusion score use different scales and must not be compared as if
    they were the same measurement.
    """

    details: list[dict] = []

    for result in results:
        document = result.document
        page = document.metadata.get("page")

        details.append(
            {
                "chunkId": document.metadata.get("chunk_id") or document.id,
                "source": document.metadata.get("source", "Unknown source"),
                "page": page + 1 if isinstance(page, int) else None,
                "vectorScore": result.vector_score,
                "keywordScore": result.keyword_score,
                "hybridScore": result.fusion_score or None,
                "hybridRank": result.fusion_rank,
                "rerankerScore": result.reranker_score,
                "rerankerRank": result.reranker_rank,
                "vectorRank": result.vector_rank,
                "keywordRank": result.keyword_rank,
            }
        )

    return details


def build_context(results: list[SearchResult]) -> tuple[str, list[str]]:
    """Convert retrieved documents into prompt context and source labels.

    The language model receives the combined text. The API separately returns
    unique source labels so the frontend can show which PDF pages contributed
    to the answer.
    """

    context_parts: list[str] = []
    sources: list[str] = []

    for result in results:
        document = result.document
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page")

        # PyPDF counts pages from zero, while people normally count from one.
        page_label = f"page {page + 1}" if isinstance(page, int) else "page unknown"
        source_label = f"{source} ({page_label})"

        context_parts.append(f"Source: {source_label}\n{document.page_content}")

        if source_label not in sources:
            sources.append(source_label)

    # Separators help the model see where one retrieved chunk ends and the next
    # one begins. They are prompt formatting, not text from the PDF.
    context = "\n\n---\n\n".join(context_parts)

    return context, sources
