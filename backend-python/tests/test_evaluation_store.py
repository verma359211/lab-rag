"""Verify evaluation checkpoints survive process restarts."""

from evals.events import evaluation_event
from app.evaluations.service import EvaluationService
from app.evaluations.store import EvaluationStore


def test_store_round_trip(tmp_path) -> None:
    store = EvaluationStore(tmp_path)
    state = {
        "id": "eval_test",
        "status": "paused",
        "events": [evaluation_event("checkpoint_saved", "Saved")],
        "results": [{"id": "Q03", "evaluation_status": "complete"}],
    }

    store.save(state)

    assert store.get("eval_test") == state
    assert store.list()[0]["id"] == "eval_test"
    assert "results" not in store.list()[0]


def test_running_job_becomes_paused_after_restart(tmp_path) -> None:
    first_store = EvaluationStore(tmp_path)
    first_store.save({"id": "eval_test", "status": "running"})

    restarted_store = EvaluationStore(tmp_path)

    assert restarted_store.get("eval_test")["status"] == "paused"


def test_completed_job_with_missing_metric_can_resume(tmp_path) -> None:
    store = EvaluationStore(tmp_path)
    store.save(
        {
            "id": "eval_test",
            "status": "completed",
            "status_message": "Completed with errors",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "progress": {"completed": 1, "total": 1},
            "rate_limit": None,
            "results": [
                {
                    "id": "Q03",
                    "evaluation_status": "complete",
                    "errors": ["RAGAS faithfulness: incomplete JSON"],
                }
            ],
        }
    )
    service = EvaluationService(store)
    service._start = lambda _run_id: None
    service._save = store.save

    resumed = service.resume("eval_test")

    assert resumed["status"] == "queued"
    assert resumed["rate_limit"] is None
