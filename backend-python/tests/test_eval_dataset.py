"""Verify JSONL parsing without connecting to PostgreSQL or an LLM."""

import json

import pytest

from evals.dataset import DatasetError, load_dataset


def valid_sample(sample_id: str) -> dict:
    """Return one minimal human-label-shaped test record."""

    return {
        "id": sample_id,
        "question": "What is the limit?",
        "reference_answer": "The limit is ten.",
        "reference_sources": [{"source": "guide.pdf", "page": 2}],
        "expected_chunk_ids": ["chunk-1"],
        "category": "fact_lookup",
        "answerable": True,
        "notes": None,
    }


def test_loads_jsonl_and_optional_chunk_ids(tmp_path) -> None:
    dataset_path = tmp_path / "golden.jsonl"
    first = valid_sample("one")
    second = valid_sample("two")
    second.pop("expected_chunk_ids")
    dataset_path.write_text(
        json.dumps(first) + "\n\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    samples = load_dataset(dataset_path)

    assert [sample.id for sample in samples] == ["one", "two"]
    assert samples[1].expected_chunk_ids is None


def test_loads_alternative_evidence_sets(tmp_path) -> None:
    """A sample may define multiple independently complete evidence paths."""

    dataset_path = tmp_path / "golden.jsonl"
    sample = valid_sample("alternatives")
    sample["expected_evidence_sets"] = [["chunk-1"], ["chunk-2", "chunk-3"]]
    dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    loaded = load_dataset(dataset_path)

    assert loaded[0].expected_evidence_sets == [
        ["chunk-1"],
        ["chunk-2", "chunk-3"],
    ]


def test_rejects_duplicate_ids(tmp_path) -> None:
    dataset_path = tmp_path / "golden.jsonl"
    record = json.dumps(valid_sample("duplicate"))
    dataset_path.write_text(record + "\n" + record + "\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="Duplicate sample id"):
        load_dataset(dataset_path)


def test_reports_the_line_containing_invalid_json(tmp_path) -> None:
    dataset_path = tmp_path / "golden.jsonl"
    dataset_path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(DatasetError, match=":1"):
        load_dataset(dataset_path)
