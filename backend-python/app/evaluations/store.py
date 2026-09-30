"""Tiny JSON-file store for resumable local evaluation jobs."""

import json
import threading
from pathlib import Path
from typing import Any


RUNS_DIRECTORY = Path(__file__).resolve().parents[2] / "evals" / "results" / "runs"


class EvaluationStore:
    """Save each job in one readable JSON file using atomic replacement."""

    def __init__(self, directory: Path = RUNS_DIRECTORY):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._mark_interrupted_jobs()

    def _path(self, run_id: str) -> Path:
        """Resolve one server-generated ID inside the runs directory."""

        return self.directory / f"{run_id}.json"

    def _mark_interrupted_jobs(self) -> None:
        """Make jobs resumable after FastAPI or the computer restarts."""

        for path in self.directory.glob("*.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if state.get("status") in {"queued", "running", "waiting"}:
                state["status"] = "paused"
                state["status_message"] = "Server restarted; the saved run can be resumed."
                self._write(path, state)

    def _write(self, path: Path, state: dict[str, Any]) -> None:
        """Write completely before replacing the previous checkpoint."""

        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def save(self, state: dict[str, Any]) -> None:
        """Persist a complete job snapshot."""

        with self.lock:
            self._write(self._path(state["id"]), state)

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Load one job or return None when it does not exist."""

        path = self._path(run_id)
        if not path.is_file():
            return None
        with self.lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        """Return newest jobs first without their large per-question evidence."""

        jobs: list[dict[str, Any]] = []
        for path in self.directory.glob("*.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            jobs.append(
                {
                    key: value
                    for key, value in state.items()
                    if key not in {"results", "events"}
                }
            )
        return sorted(jobs, key=lambda job: job.get("updated_at", ""), reverse=True)
