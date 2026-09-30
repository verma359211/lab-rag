"""Run one evaluation at a time and expose durable progress snapshots."""

import threading
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from evals.job_runner import (
    EvaluationCancelled,
    EvaluationPaused,
    build_report,
    run_evaluation_job,
)

from app.evaluations.store import EvaluationStore


ACTIVE_STATUSES = {"queued", "running", "waiting", "cancelling"}


def _now() -> str:
    """Return an ISO timestamp that JavaScript can parse directly."""

    return datetime.now(UTC).isoformat()


class EvaluationService:
    """Coordinate background threads while the store owns durable state."""

    def __init__(self, store: EvaluationStore | None = None):
        self.store = store or EvaluationStore()
        self.cancel_events: dict[str, threading.Event] = {}
        self.lock = threading.Lock()

    def _active_job(self) -> dict[str, Any] | None:
        """Find the single job currently allowed to use provider quotas."""

        return next(
            (job for job in self.store.list() if job.get("status") in ACTIVE_STATUSES),
            None,
        )

    def create(self, config: dict[str, Any]) -> dict[str, Any]:
        """Create and immediately start one new evaluation job."""

        with self.lock:
            active = self._active_job()
            if active:
                raise RuntimeError(f"Evaluation {active['id']} is already running")

            run_id = f"eval_{uuid4().hex[:12]}"
            timestamp = _now()
            state = {
                "id": run_id,
                "status": "queued",
                "status_message": "Waiting to start",
                "created_at": timestamp,
                "updated_at": timestamp,
                "config": config,
                "progress": {"completed": 0, "total": 15 if config["dataset"] == "core15" else 60},
                "current": None,
                "current_metric": None,
                "rate_limit": None,
                "provider_quota": None,
                "aggregate": {},
                "results": [],
                "events": [],
                "artifacts": None,
            }
            self.store.save(state)
            self._start(run_id)
            return state

    def _start(self, run_id: str) -> None:
        """Start the blocking evaluation engine on one daemon thread."""

        cancel_event = threading.Event()
        self.cancel_events[run_id] = cancel_event
        thread = threading.Thread(
            target=self._run,
            args=(run_id, cancel_event),
            daemon=True,
            name=f"evaluation-{run_id}",
        )
        thread.start()

    def _save(self, state: dict[str, Any]) -> None:
        """Update timestamps, progress, partial metrics, and the JSON file."""

        state["updated_at"] = _now()
        state["progress"]["completed"] = sum(
            result.get("evaluation_status") == "complete"
            for result in state.get("results", [])
        )
        if state.get("results"):
            state["aggregate"] = build_report(
                state["config"],
                state["results"],
            )["aggregate"]
        self.store.save(state)

    def _run(self, run_id: str, cancel_event: threading.Event) -> None:
        """Translate engine callbacks into one persisted job state."""

        state = self.store.get(run_id)
        if not state:
            return
        state["status"] = "running"
        state["status_message"] = "Evaluation is running"
        self._save(state)

        def on_event(event: dict[str, Any]) -> None:
            latest = self.store.get(run_id) or state
            latest.setdefault("events", []).append(event)
            latest["events"] = latest["events"][-500:]
            latest["current"] = event.get("question_id") or latest.get("current")
            latest["current_metric"] = event.get("metric")
            if event["event"] in {
                "waiting",
                "token_budget_wait",
                "rate_limit_retry",
                "metric_error_retry",
            }:
                latest["status"] = "waiting"
            elif latest.get("status") not in {"cancelling", "paused_rate_limit"}:
                latest["status"] = "running"
            latest["status_message"] = event["message"]
            if event.get("data", {}).get("quota"):
                latest["provider_quota"] = event["data"]["quota"]
            self._save(latest)

        def on_checkpoint(values: dict[str, Any]) -> None:
            latest = self.store.get(run_id) or state
            latest.update(values)
            self._save(latest)

        try:
            report, json_path, csv_path = run_evaluation_job(
                state["config"],
                state.get("results", []),
                on_event,
                on_checkpoint,
                cancel_event.is_set,
            )
            state = self.store.get(run_id) or state
            has_errors = report["aggregate"]["failed_count"] > 0
            state["status"] = "completed_with_errors" if has_errors else "completed"
            state["status_message"] = (
                "Evaluation completed with missing metrics; retry is available"
                if has_errors
                else "Evaluation completed"
            )
            state["aggregate"] = report["aggregate"]
            state["results"] = report["results"]
            state["artifacts"] = {"json": str(json_path), "csv": str(csv_path)}
            state["rate_limit"] = None
            self._save(state)
        except EvaluationPaused as error:
            state = self.store.get(run_id) or state
            state["status"] = "paused_rate_limit"
            state["status_message"] = "Paused because the Groq token limit was reached"
            state["rate_limit"] = error.details
            self._save(state)
        except EvaluationCancelled:
            state = self.store.get(run_id) or state
            state["status"] = "cancelled"
            state["status_message"] = "Evaluation cancelled; saved progress can be resumed"
            self._save(state)
        except Exception as error:
            state = self.store.get(run_id) or state
            state["status"] = "failed"
            state["status_message"] = str(error)
            self._save(state)
        finally:
            self.cancel_events.pop(run_id, None)

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Return one complete dashboard snapshot."""

        return self.store.get(run_id)

    def list(self) -> list[dict[str, Any]]:
        """Return compact history entries."""

        return self.store.list()

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Request cooperative cancellation after the current API call."""

        state = self.store.get(run_id)
        if not state:
            raise KeyError(run_id)
        cancel_event = self.cancel_events.get(run_id)
        if cancel_event:
            cancel_event.set()
            state["status"] = "cancelling"
            state["status_message"] = "Cancellation requested"
            self._save(state)
        return state

    def resume(self, run_id: str) -> dict[str, Any]:
        """Resume saved work and skip completed questions and metrics."""

        with self.lock:
            active = self._active_job()
            if active:
                raise RuntimeError(f"Evaluation {active['id']} is already running")
            state = self.store.get(run_id)
            if not state:
                raise KeyError(run_id)
            if state["status"] == "completed":
                has_missing_metrics = any(
                    result.get("evaluation_status") != "complete"
                    or bool(result.get("errors"))
                    for result in state.get("results", [])
                )
                if not has_missing_metrics:
                    raise RuntimeError("A completed evaluation does not need to resume")
            state["status"] = "queued"
            state["status_message"] = "Waiting to resume"
            state["rate_limit"] = None
            self._save(state)
            self._start(run_id)
            return state


evaluation_service = EvaluationService()
