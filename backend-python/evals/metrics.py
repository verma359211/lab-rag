"""Small deterministic retrieval metrics with no external dependencies."""

from collections.abc import Iterable


DEFAULT_K_VALUES = (1, 3, 5, 10)


def _evidence_sets(
    expected_ids: list[str] | None,
    expected_evidence_sets: list[list[str]] | None,
) -> list[set[str]]:
    """Return valid evidence combinations with legacy-label fallback."""

    alternatives = [
        set(evidence_set)
        for evidence_set in expected_evidence_sets or []
        if evidence_set
    ]
    if alternatives:
        return alternatives
    return [set(expected_ids)] if expected_ids else []


def hit_at_k(ranked_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Return 1 when any expected chunk appears in the first ``k`` results."""

    return float(bool(expected_ids.intersection(ranked_ids[:k])))


def recall_at_k(ranked_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Return the fraction of expected chunks found in the first ``k``."""

    if not expected_ids:
        return 0.0

    found_ids = expected_ids.intersection(ranked_ids[:k])
    return len(found_ids) / len(expected_ids)


def reciprocal_rank(ranked_ids: list[str], expected_ids: set[str]) -> float:
    """Return the reciprocal position of the first relevant chunk."""

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in expected_ids:
            return 1 / rank

    return 0.0


def calculate_retrieval_metrics(
    ranked_ids: list[str],
    expected_ids: list[str] | None,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    expected_evidence_sets: list[list[str]] | None = None,
) -> dict[str, float] | None:
    """Score retrieval against the best complete human-approved evidence path."""

    evidence_sets = _evidence_sets(expected_ids, expected_evidence_sets)
    if not evidence_sets:
        return None

    relevant_ids = set().union(*evidence_sets)
    metrics: dict[str, float] = {}

    for k in k_values:
        metrics[f"hit_at_{k}"] = hit_at_k(ranked_ids, relevant_ids, k)
        metrics[f"recall_at_{k}"] = max(
            recall_at_k(ranked_ids, evidence_set, k)
            for evidence_set in evidence_sets
        )

    metrics["mrr"] = reciprocal_rank(ranked_ids, relevant_ids)
    return metrics


def calculate_context_metrics(
    retrieved_ids: list[str],
    expected_ids: list[str] | None,
    expected_evidence_sets: list[list[str]] | None = None,
) -> dict[str, float] | None:
    """Score final contexts directly against human-reviewed chunk IDs.

    These deterministic metrics replace two LLM judgments when chunk labels
    exist. Duplicate retrieved IDs count only once because retrieving the same
    evidence twice should not improve precision or recall.

    Empty expected IDs represent intentionally unanswerable questions. They do
    not have positive evidence chunks, so context precision/recall are reported
    as unavailable instead of assigning a misleading perfect recall score.
    """

    evidence_sets = _evidence_sets(expected_ids, expected_evidence_sets)
    if not evidence_sets:
        return None

    unique_retrieved = list(dict.fromkeys(retrieved_ids))
    relevant_ids = set().union(*evidence_sets)
    matched = relevant_ids.intersection(unique_retrieved)

    precision = (
        len(matched) / len(unique_retrieved)
        if unique_retrieved
        else 0.0
    )
    recall = max(
        len(evidence_set.intersection(unique_retrieved)) / len(evidence_set)
        for evidence_set in evidence_sets
    )
    return {
        "context_precision": precision,
        "context_recall": recall,
    }


def average_available_metrics(rows: list[dict[str, float] | None]) -> dict[str, float]:
    """Average each metric while ignoring samples without chunk ground truth."""

    available_rows = [row for row in rows if row is not None]

    if not available_rows:
        return {}

    metric_names = available_rows[0].keys()
    return {
        name: sum(row[name] for row in available_rows) / len(available_rows)
        for name in metric_names
    }
