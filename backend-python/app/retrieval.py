"""Find relevant PDF chunks and prepare them for the language model.

Retrieval is the first half of answering a question. It searches the stored
vectors but does not generate an answer. Keeping retrieval separate makes it
easy to add keyword search and reranking later.
"""

from langchain_core.documents import Document

from app.config import RETRIEVAL_LIMIT
from app.vector_store import get_vector_store


SearchResult = tuple[Document, float]


def retrieve_documents(question: str) -> list[SearchResult]:
    """Return the PDF chunks whose vectors are closest to the question.

    LangChain first embeds the question with the same Gemini model used during
    ingestion. PGVector then compares that question vector with stored chunk
    vectors and returns the best matches together with relevance scores.
    """

    return get_vector_store().similarity_search_with_relevance_scores(
        query=question,
        k=RETRIEVAL_LIMIT,
    )


def build_context(results: list[SearchResult]) -> tuple[str, list[str]]:
    """Convert retrieved documents into prompt context and source labels.

    The language model receives the combined text. The API separately returns
    unique source labels so the frontend can show which PDF pages contributed
    to the answer.
    """

    context_parts: list[str] = []
    sources: list[str] = []

    for document, _score in results:
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
