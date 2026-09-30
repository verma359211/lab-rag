"""Prepare retrieved chunks for reliable evaluator judgments.

The application still generates its answer from the normal retrieval ranking.
This module only gives the evaluation judge a readable view of the same chunks.
Adjacent chunks are restored to document order so a sentence split across a
chunk boundary is not accidentally shown backwards.
"""

from typing import Any


def _sort_value(item: dict[str, Any]) -> tuple[str, str, int, int, int]:
    """Return a stable document-order key for one serialized chunk."""

    chunk_number = item.get("chunk_number")
    page = item.get("page")
    rank = item.get("rank")
    return (
        str(item.get("document_id") or ""),
        str(item.get("source") or ""),
        chunk_number if isinstance(chunk_number, int) else 1_000_000,
        page if isinstance(page, int) else 1_000_000,
        rank if isinstance(rank, int) else 1_000_000,
    )


def _is_adjacent(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Return whether two chunks are consecutive pieces of one document."""

    previous_number = previous.get("chunk_number")
    current_number = current.get("chunk_number")
    return (
        previous.get("document_id") == current.get("document_id")
        and isinstance(previous_number, int)
        and isinstance(current_number, int)
        and current_number == previous_number + 1
    )


def _merge_overlapping_text(left: str, right: str) -> str:
    """Join adjacent chunks without repeating their exact shared text.

    The ingestion splitter intentionally overlaps nearby chunks. Sending that
    overlap twice wastes judge tokens and can make repeated facts look more
    important. Only an exact overlap of at least 40 characters is removed, so
    similar but genuinely separate sentences are kept.
    """

    maximum = min(len(left), len(right))
    for size in range(maximum, 39, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return f"{left}\n{right}"


def _merge_group_content(group: list[dict[str, Any]]) -> str:
    """Merge the text of one adjacent chunk group in document order."""

    content = str(group[0].get("content") or "")
    for item in group[1:]:
        content = _merge_overlapping_text(
            content,
            str(item.get("content") or ""),
        )
    return content


def prepare_judge_contexts(
    ranked_chunks: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return exact judge inputs and a readable description of their order.

    Retrieval ranks are preserved in every trace entry. Sorting here does not
    change Hit@K, RRF, the generated answer, or any application search result.
    It only prevents the evaluation model from seeing adjacent text backwards.
    """

    ordered = sorted(ranked_chunks, key=_sort_value)
    groups: list[list[dict[str, Any]]] = []

    for chunk in ordered:
        if groups and _is_adjacent(groups[-1][-1], chunk):
            groups[-1].append(chunk)
        else:
            groups.append([chunk])

    judge_contexts: list[str] = []
    trace: list[dict[str, Any]] = []

    for group in groups:
        source = group[0].get("source") or "Unknown source"
        pages = list(dict.fromkeys(item.get("page") for item in group))
        chunk_ids = [str(item.get("chunk_id") or "") for item in group]
        chunk_numbers = [item.get("chunk_number") for item in group]
        original_ranks = [item.get("rank") for item in group]
        content = _merge_group_content(group)
        header = (
            f"[Source: {source}; pages: {pages}; "
            f"chunk numbers: {chunk_numbers}]"
        )
        prepared_text = f"{header}\n{content}"

        judge_contexts.append(prepared_text)
        trace.append(
            {
                "source": source,
                "pages": pages,
                "chunk_ids": chunk_ids,
                "chunk_numbers": chunk_numbers,
                "original_retrieval_ranks": original_ranks,
                "text": prepared_text,
            }
        )

    return judge_contexts, trace
