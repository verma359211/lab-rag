"""Ensure the fast benchmark remains an exact subset of the full dataset."""

import json
from pathlib import Path


DATASETS_DIRECTORY = Path(__file__).resolve().parents[1] / "evals" / "datasets" / "v1"
CORE_DATASET = DATASETS_DIRECTORY / "golden.core15.jsonl"
FULL_DATASET = DATASETS_DIRECTORY / "golden.mapped.jsonl"

EXPECTED_CORE_IDS = [
    "Q03",
    "Q06",
    "Q10",
    "Q11",
    "Q16",
    "Q18",
    "Q23",
    "Q25",
    "Q27",
    "Q32",
    "Q34",
    "Q40",
    "Q43",
    "Q48",
    "Q51",
]


def non_empty_lines(path: Path) -> list[str]:
    """Return JSONL records exactly as they are stored in the file."""

    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_core_dataset_is_the_exact_requested_subset() -> None:
    core_lines = non_empty_lines(CORE_DATASET)
    full_lines = non_empty_lines(FULL_DATASET)
    core_records = [json.loads(line) for line in core_lines]
    full_records = [json.loads(line) for line in full_lines]
    full_lines_by_id = {
        record["id"]: line
        for record, line in zip(full_records, full_lines, strict=True)
    }

    assert len(core_records) == 15
    assert [record["id"] for record in core_records] == EXPECTED_CORE_IDS

    # Comparing the original JSONL lines guarantees that no field, value, or
    # existing ground-truth chunk ID was regenerated or changed.
    assert core_lines == [full_lines_by_id[record_id] for record_id in EXPECTED_CORE_IDS]
