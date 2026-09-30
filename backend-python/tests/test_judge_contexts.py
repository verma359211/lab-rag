"""Verify evaluator-only context preparation without touching retrieval ranks."""

from evals.judge_contexts import prepare_judge_contexts


def test_adjacent_chunks_are_restored_to_document_order() -> None:
    ranked_chunks = [
        {
            "rank": 1,
            "chunk_id": "second",
            "document_id": "doc-1",
            "source": "policy.pdf",
            "page": 4,
            "chunk_number": 11,
            "content": "each month are excluded.",
        },
        {
            "rank": 2,
            "chunk_id": "first",
            "document_id": "doc-1",
            "source": "policy.pdf",
            "page": 4,
            "chunk_number": 10,
            "content": "The first 50,000 calls",
        },
    ]

    contexts, trace = prepare_judge_contexts(ranked_chunks)

    assert len(contexts) == 1
    assert contexts[0].index("The first 50,000 calls") < contexts[0].index(
        "each month are excluded."
    )
    assert trace[0]["chunk_ids"] == ["first", "second"]
    assert trace[0]["original_retrieval_ranks"] == [2, 1]


def test_non_adjacent_chunks_remain_separate() -> None:
    ranked_chunks = [
        {
            "rank": 1,
            "chunk_id": "one",
            "document_id": "doc-1",
            "source": "policy.pdf",
            "page": 1,
            "chunk_number": 1,
            "content": "First",
        },
        {
            "rank": 2,
            "chunk_id": "three",
            "document_id": "doc-1",
            "source": "policy.pdf",
            "page": 3,
            "chunk_number": 3,
            "content": "Third",
        },
    ]

    contexts, _trace = prepare_judge_contexts(ranked_chunks)

    assert len(contexts) == 2


def test_exact_chunk_overlap_is_sent_to_the_judge_once() -> None:
    """Splitter overlap should not consume judge tokens twice."""

    overlap = "This exact policy sentence is deliberately longer than forty characters."
    ranked_chunks = [
        {
            "rank": 1,
            "chunk_id": "first",
            "document_id": "doc-1",
            "source": "policy.pdf",
            "page": 1,
            "chunk_number": 1,
            "content": f"Opening text. {overlap}",
        },
        {
            "rank": 2,
            "chunk_id": "second",
            "document_id": "doc-1",
            "source": "policy.pdf",
            "page": 2,
            "chunk_number": 2,
            "content": f"{overlap} Closing text.",
        },
    ]

    contexts, _trace = prepare_judge_contexts(ranked_chunks)

    assert len(contexts) == 1
    assert contexts[0].count(overlap) == 1
    assert "Opening text." in contexts[0]
    assert "Closing text." in contexts[0]
