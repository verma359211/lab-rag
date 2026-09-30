"""Verify retrieval metrics with small, completely artificial rankings."""

import pytest

from evals.metrics import calculate_context_metrics, calculate_retrieval_metrics


def test_metrics_use_the_first_relevant_rank_and_expected_set() -> None:
    metrics = calculate_retrieval_metrics(
        ranked_ids=["noise", "relevant-b", "relevant-a", "other"],
        expected_ids=["relevant-a", "relevant-b"],
    )

    assert metrics is not None
    assert metrics["hit_at_1"] == 0
    assert metrics["hit_at_3"] == 1
    assert metrics["recall_at_1"] == 0
    assert metrics["recall_at_3"] == 1
    assert metrics["mrr"] == pytest.approx(0.5)


def test_metrics_are_unavailable_without_human_chunk_labels() -> None:
    assert calculate_retrieval_metrics(["chunk-a"], None) is None
    assert calculate_retrieval_metrics(["chunk-a"], []) is None


def test_missing_relevant_chunks_receive_zero() -> None:
    metrics = calculate_retrieval_metrics(["noise"], ["relevant"])

    assert metrics is not None
    assert metrics["hit_at_10"] == 0
    assert metrics["recall_at_10"] == 0
    assert metrics["mrr"] == 0


def test_context_metrics_use_human_chunk_labels() -> None:
    metrics = calculate_context_metrics(
        retrieved_ids=["relevant-a", "noise", "relevant-b", "noise"],
        expected_ids=["relevant-a", "relevant-b"],
    )

    assert metrics == {
        "context_precision": pytest.approx(2 / 3),
        "context_recall": 1.0,
    }


def test_context_metrics_are_unavailable_without_positive_evidence() -> None:
    assert calculate_context_metrics(["noise"], None) is None
    assert calculate_context_metrics(["noise"], []) is None


def test_retrieval_uses_the_best_complete_alternative_evidence_set() -> None:
    """Finding one complete alternative must count as full recall."""

    metrics = calculate_retrieval_metrics(
        ranked_ids=["summary", "noise"],
        expected_ids=["policy-a", "policy-b"],
        expected_evidence_sets=[
            ["policy-a", "policy-b"],
            ["summary"],
        ],
    )

    assert metrics is not None
    assert metrics["hit_at_1"] == 1
    assert metrics["recall_at_1"] == 1
    assert metrics["mrr"] == 1


def test_context_precision_accepts_chunks_from_any_valid_evidence_set() -> None:
    """Valid alternative evidence should not be counted as irrelevant noise."""

    metrics = calculate_context_metrics(
        retrieved_ids=["summary", "noise"],
        expected_ids=["policy-a", "policy-b"],
        expected_evidence_sets=[
            ["policy-a", "policy-b"],
            ["summary"],
        ],
    )

    assert metrics == {
        "context_precision": 0.5,
        "context_recall": 1.0,
    }
