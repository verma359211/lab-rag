"""Run hybrid search and prepare the winning chunks for the language model.

Retrieval is the first half of answering a question. Vector search finds
similar meanings, keyword search finds exact terms, and Reciprocal Rank Fusion
combines their ordered result lists.
"""

from app.search.fusion import fuse_results
from app.search.keyword_search import search_by_keyword
from app.search.models import SearchResult
from app.search.vector_search import search_by_vector


def retrieve_documents(question: str) -> list[SearchResult]:
    """Return the best chunks from vector and keyword retrieval together.

    The two searches intentionally stay separate until the fusion step. This
    makes their behavior easier to inspect and lets us add a reranker later
    without rewriting the database search functions.
    """

    vector_results = search_by_vector(question)
    keyword_results = search_by_keyword(question)

    return fuse_results(vector_results, keyword_results)


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
