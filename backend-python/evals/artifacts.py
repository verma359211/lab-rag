"""Write complete evaluation results without replacing an earlier run."""

import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"


def _safe_name(value: str) -> str:
    """Make a short experiment name safe to use inside a filename."""

    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-") or "evaluation"


def _flatten(prefix: str, value: Any, row: dict[str, Any]) -> None:
    """Flatten nested dictionaries for the human-friendly CSV artifact."""

    if isinstance(value, dict):
        for child_name, child_value in value.items():
            name = f"{prefix}.{child_name}" if prefix else child_name
            _flatten(name, child_value, row)
        return

    if isinstance(value, list):
        row[prefix] = json.dumps(value, ensure_ascii=False)
        return

    row[prefix] = value


def write_artifacts(
    report: dict[str, Any],
    experiment: str,
    results_directory: Path = RESULTS_DIRECTORY,
) -> tuple[Path, Path]:
    """Save a lossless JSON report and a spreadsheet-friendly CSV report."""

    results_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"{timestamp}_{_safe_name(experiment)}"
    json_path = results_directory / f"{filename}.json"
    csv_path = results_directory / f"{filename}.csv"

    # Exclusive creation is a final guard against accidentally replacing a
    # result, even in the unlikely event of identical microsecond timestamps.
    with json_path.open("x", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2, ensure_ascii=False)

    flat_rows: list[dict[str, Any]] = []
    for result in report["results"]:
        flat_row: dict[str, Any] = {}
        _flatten("", result, flat_row)
        flat_rows.append(flat_row)

    fieldnames = sorted({key for row in flat_rows for key in row})
    with csv_path.open("x", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    return json_path, csv_path
