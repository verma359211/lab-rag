"""Structured progress events shared by the CLI, API, and frontend."""

from datetime import UTC, datetime
from typing import Any


def evaluation_event(
    event: str,
    message: str,
    *,
    level: str = "info",
    question_id: str | None = None,
    stage: str | None = None,
    metric: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one JSON-safe event with consistent fields."""

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "event": event,
        "question_id": question_id,
        "stage": stage,
        "metric": metric,
        "message": message,
        "data": data or {},
    }
