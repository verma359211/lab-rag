"""Load and validate the version-controlled JSONL evaluation dataset."""

import json
from pathlib import Path

from pydantic import ValidationError

from evals.models import GoldenSample


class DatasetError(ValueError):
    """Explain a malformed dataset with its file and line number."""


def load_dataset(path: str | Path) -> list[GoldenSample]:
    """Read one JSON object per line and reject duplicate sample IDs."""

    dataset_path = Path(path)
    samples: list[GoldenSample] = []
    seen_ids: set[str] = set()

    if not dataset_path.is_file():
        raise DatasetError(f"Dataset does not exist: {dataset_path}")

    with dataset_path.open(encoding="utf-8") as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            line = raw_line.strip()

            # Blank lines are allowed so people can visually group examples.
            if not line:
                continue

            try:
                raw_sample = json.loads(line)
                sample = GoldenSample.model_validate(raw_sample)
            except (json.JSONDecodeError, ValidationError) as error:
                raise DatasetError(
                    f"Invalid sample at {dataset_path}:{line_number}: {error}"
                ) from error

            if sample.id in seen_ids:
                raise DatasetError(
                    f"Duplicate sample id '{sample.id}' at "
                    f"{dataset_path}:{line_number}"
                )

            seen_ids.add(sample.id)
            samples.append(sample)

    if not samples:
        raise DatasetError(f"Dataset contains no samples: {dataset_path}")

    return samples
