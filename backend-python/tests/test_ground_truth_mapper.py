"""Test deterministic ground-truth mapping without PostgreSQL or retrieval."""

from evals.map_ground_truth import map_record


def chunk(chunk_id: str, content: str, page: int = 1) -> dict:
    """Create a tiny stored-chunk shape using zero-based metadata pages."""

    return {
        "id": chunk_id,
        "content": content,
        "metadata": {
            "chunk_id": chunk_id,
            "document_id": "document-1",
            "source": "Northstar_RAG_Torture_Corpus_v2.pdf",
            "page": page,
            "chunk_number": 2,
        },
    }


def record(record_id: str, answerable: bool = True) -> dict:
    """Create the minimum raw record used by ``map_record``."""

    return {
        "id": record_id,
        "question": "Test question",
        "answerable": answerable,
        "expected_chunk_ids": [],
        "reference_sources": [
            {
                "source": "Northstar_RAG_Torture_Corpus_v2.pdf",
                "page": 2,
                "section": "Test section",
            }
        ],
    }


def test_maps_only_the_chunk_containing_explicit_evidence() -> None:
    dataset_record = record("Q01")
    evidence = (
        "Nova Edge is not an upgrade of Nova Plus; it is a deployment "
        "profile for intermittently connected sites"
    )

    report = map_record(
        dataset_record,
        [
            chunk("relevant", evidence),
            chunk("same-page-distractor", "Nova Edge is mentioned here."),
        ],
    )

    assert report["status"] == "mapped"
    assert dataset_record["expected_chunk_ids"] == ["relevant"]


def test_repeated_evidence_is_ambiguous_instead_of_guessed() -> None:
    dataset_record = record("Q01")
    evidence = (
        "Nova Edge is not an upgrade of Nova Plus; it is a deployment "
        "profile for intermittently connected sites"
    )

    report = map_record(
        dataset_record,
        [chunk("first", evidence), chunk("second", evidence)],
    )

    assert report["status"] == "ambiguous"
    assert dataset_record["expected_chunk_ids"] == []


def test_unanswerable_record_always_has_no_supporting_ids() -> None:
    dataset_record = record("Q14", answerable=False)
    dataset_record["expected_chunk_ids"] = ["old-value"]

    report = map_record(dataset_record, [])

    assert report["status"] == "intentionally_unanswerable"
    assert dataset_record["expected_chunk_ids"] == []


def test_mapper_keeps_alternative_evidence_paths_separate() -> None:
    """The mapper must not merge duplicate valid routes into one requirement."""

    dataset_record = record("Q32")
    dataset_record["reference_sources"][0]["page"] = 20
    primary = (
        "A credit issued on 20 August 2026 therefore uses the 180-day rule "
        "unless its memo explicitly sets an earlier expiration"
    )
    alternative = (
        "Credits issued on or after 1 July 2026 expire 180 days after "
        "issuance unless the individual credit memo specifies an earlier date"
    )

    report = map_record(
        dataset_record,
        [
            chunk("scenario", primary, page=19),
            chunk("general-rule", alternative, page=19),
        ],
    )

    assert report["status"] == "mapped"
    assert dataset_record["expected_evidence_sets"] == [
        ["scenario"],
        ["general-rule"],
    ]
