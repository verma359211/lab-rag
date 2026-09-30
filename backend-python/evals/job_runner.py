"""Resumable evaluation engine used by the web dashboard."""

import statistics
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    RERANK_CANDIDATE_LIMIT,
    RETRIEVAL_LIMIT,
    RRF_K,
    SEARCH_CANDIDATE_LIMIT,
    get_chat_model_name,
    get_embedding_model_name,
    get_reranker_model_name,
)
from app.rag import execute_rag
from app.retrieval import retrieve_with_diagnostics
from evals.artifacts import write_artifacts
from evals.dataset import load_dataset
from evals.events import evaluation_event
from evals.evaluator_config import load_evaluator_config
from evals.judge_contexts import prepare_judge_contexts
from evals.metrics import average_available_metrics
from evals.ragas_metrics import RagasEvaluator
from evals.rate_limit import (
    RetrySettings,
    is_rate_limit_error,
    rate_limit_details,
    run_with_rate_limit_retry,
)
from evals.runner import _add_trace, _base_result, _percentile
from evals.token_budget import TokenBudget, estimate_tokens


DATASETS = {
    "core15": Path("evals/datasets/v1/golden.core15.jsonl"),
    "full60": Path("evals/datasets/v1/golden.mapped.jsonl"),
}

EventCallback = Callable[[dict[str, Any]], None]
CheckpointCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]


class EvaluationPaused(RuntimeError):
    """Signal that saved work should resume after a provider limit resets."""

    def __init__(self, details: dict[str, Any]):
        super().__init__("Evaluation paused because the provider limit was reached")
        self.details = details


class EvaluationCancelled(RuntimeError):
    """Signal a cooperative cancellation between external operations."""


def _sleep_with_cancel(seconds: float, is_cancelled: CancelCallback) -> None:
    """Sleep in short pieces so a dashboard cancellation stays responsive."""

    deadline = time.monotonic() + seconds
    while True:
        if is_cancelled():
            raise EvaluationCancelled("Evaluation cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.5, remaining))


def _emit(callback: EventCallback, name: str, message: str, **values: Any) -> None:
    """Build and publish one structured event."""

    callback(evaluation_event(name, message, **values))


def _remove_metric_error(result: dict[str, Any], metric_name: str) -> None:
    """Remove an older error when one missing metric is being retried."""

    prefix = f"RAGAS {metric_name}:"
    result["errors"] = [
        error for error in result.get("errors", []) if not error.startswith(prefix)
    ]
    result.get("ragas", {}).get("errors", {}).pop(metric_name, None)


def _ragas_average(results: list[dict[str, Any]]) -> dict[str, float]:
    """Average every metric that completed successfully."""

    names = {
        name
        for result in results
        for name in result.get("ragas", {}).get("scores", {})
    }
    return {
        name: statistics.fmean(
            result["ragas"]["scores"][name]
            for result in results
            if name in result.get("ragas", {}).get("scores", {})
        )
        for name in names
    }


def build_report(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Create the same final report shape used by the command-line runner."""

    retrieval_rows = [result.get("retrieval_metrics") for result in results]
    latencies = [
        result["latency_ms"].get(
            "total",
            result["latency_ms"].get("retrieval_total"),
        )
        for result in results
        if result.get("latency_ms")
    ]
    latencies = [value for value in latencies if value is not None]
    failures = Counter(
        result["category"] for result in results if result.get("errors")
    )

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": config["experiment"],
        "mode": config["mode"],
        "dataset": str(DATASETS[config["dataset"]].resolve()),
        "configuration": {
            "search_mode": config["search_mode"],
            "application_chat_model": get_chat_model_name(),
            "application_embedding_model": get_embedding_model_name(),
            "collection_name": COLLECTION_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "search_candidate_limit": SEARCH_CANDIDATE_LIMIT,
            "retrieval_limit": RETRIEVAL_LIMIT,
            "rrf_k": RRF_K,
            "reranker_model": (
                get_reranker_model_name()
                if config["search_mode"] == "hybrid_rerank"
                else None
            ),
            "rerank_candidate_limit": RERANK_CANDIDATE_LIMIT,
            "question_delay_seconds": config["question_delay"],
            "metric_delay_seconds": config["metric_delay"],
            "groq_tokens_per_minute": config["groq_tpm_limit"],
            "max_rate_limit_retries": config["max_retries"],
            "initial_retry_delay_seconds": config["initial_retry_delay"],
            "context_metrics_source": "human_expected_chunk_ids",
            "persistent_judge_cache": True,
            "evaluation_implementation": (
                RagasEvaluator.IMPLEMENTATION_VERSION
                if config["mode"] == "full"
                else None
            ),
        },
        "aggregate": {
            "sample_count": len(results),
            "successful_count": sum(
                result.get("evaluation_status") == "complete"
                and not result.get("errors")
                for result in results
            ),
            "failed_count": sum(
                result.get("evaluation_status") == "failed"
                or bool(result.get("errors"))
                for result in results
            ),
            "chunk_labeled_count": sum(row is not None for row in retrieval_rows),
            "retrieval_metrics": average_available_metrics(retrieval_rows),
            "ragas_metrics": _ragas_average(results),
            "judge_cache_hit_count": sum(
                bool(cache_hit)
                for result in results
                for metric_name, cache_hit in result.get("ragas", {})
                .get("cache_hits", {})
                .items()
                if metric_name not in {"context_precision", "context_recall"}
            ),
            "latency_ms": {
                "average": statistics.fmean(latencies) if latencies else None,
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
            "failures_by_category": dict(failures),
        },
        "results": results,
    }


def run_evaluation_job(
    config: dict[str, Any],
    saved_results: list[dict[str, Any]],
    on_event: EventCallback,
    on_checkpoint: CheckpointCallback,
    is_cancelled: CancelCallback,
) -> tuple[dict[str, Any], Path, Path]:
    """Run or resume one evaluation and checkpoint every completed stage."""

    samples = load_dataset(DATASETS[config["dataset"]])
    result_by_id = {result["id"]: result for result in saved_results}
    retry_settings = RetrySettings(
        max_retries=config["max_retries"],
        initial_delay_seconds=config["initial_retry_delay"],
    )
    evaluator = None
    token_budget = None

    if config["mode"] == "full":
        evaluator = RagasEvaluator(
            load_evaluator_config(),
            retry_settings=retry_settings,
        )
        token_budget = TokenBudget(
            config["groq_tpm_limit"],
            sleep=lambda seconds: _sleep_with_cancel(seconds, is_cancelled),
        )

    _emit(on_event, "run_started", "Evaluation started", stage="run")

    for index, sample in enumerate(samples, start=1):
        if is_cancelled():
            raise EvaluationCancelled("Evaluation cancelled")

        existing = result_by_id.get(sample.id)
        if existing and existing.get("evaluation_status") == "complete":
            if config["mode"] != "full" or not evaluator:
                continue
            completed_scores = existing.get("ragas", {}).get("scores", {})
            required_metrics = evaluator.metric_names_for(sample.answerable)
            same_evaluator = (
                existing.get("ragas", {}).get("evaluation_version")
                == evaluator.IMPLEMENTATION_VERSION
            )
            if same_evaluator and all(
                name in completed_scores for name in required_metrics
            ):
                continue

        if index > 1 and config["question_delay"] > 0:
            _emit(
                on_event,
                "waiting",
                f"Waiting {config['question_delay']:g}s before the next question",
                question_id=sample.id,
                stage="pacing",
                data={"seconds": config["question_delay"]},
            )
            _sleep_with_cancel(config["question_delay"], is_cancelled)

        _emit(
            on_event,
            "question_started",
            f"Evaluating {sample.id}",
            question_id=sample.id,
            stage="question",
            data={"number": index, "total": len(samples)},
        )

        result = existing or _base_result(sample)
        result.setdefault("evaluation_status", "running")

        if not result.get("final_contexts"):
            try:
                if config["mode"] == "retrieval":
                    trace = run_with_rate_limit_retry(
                        lambda: retrieve_with_diagnostics(
                            sample.question,
                            config["search_mode"],
                        ),
                        retry_settings,
                        sleep=lambda seconds: _sleep_with_cancel(seconds, is_cancelled),
                    )
                    _add_trace(result, trace, config["search_mode"])
                else:
                    execution = run_with_rate_limit_retry(
                        lambda: execute_rag(
                            sample.question,
                            config["search_mode"],
                        ),
                        retry_settings,
                        sleep=lambda seconds: _sleep_with_cancel(seconds, is_cancelled),
                    )
                    _add_trace(result, execution.retrieval, config["search_mode"])
                    result["answer"] = execution.answer
                    result["latency_ms"].update(execution.latency_ms)

                _emit(
                    on_event,
                    "vector_search_completed",
                    "Question embedding and vector search completed",
                    question_id=sample.id,
                    stage="retrieval",
                    data={
                        "vector_candidates": len(result.get("vector_candidates", [])),
                        "latency_ms": result.get("latency_ms", {}).get("vector_search"),
                    },
                )
                _emit(
                    on_event,
                    "keyword_search_completed",
                    "PostgreSQL keyword search completed",
                    question_id=sample.id,
                    stage="retrieval",
                    data={
                        "keyword_candidates": len(result.get("keyword_candidates", [])),
                        "latency_ms": result.get("latency_ms", {}).get("keyword_search"),
                    },
                )
                _emit(
                    on_event,
                    "fusion_completed",
                    "Retrieval ranking completed",
                    question_id=sample.id,
                    stage="retrieval",
                    data={
                        "final_contexts": len(result.get("final_contexts", [])),
                        "latency_ms": result.get("latency_ms", {}).get("retrieval_total"),
                    },
                )
                if config["search_mode"] == "hybrid_rerank":
                    _emit(
                        on_event,
                        "reranking_completed",
                        "Cross-encoder reranking completed",
                        question_id=sample.id,
                        stage="retrieval",
                        data={
                            "reranker_applied": result.get("reranker_applied"),
                            "reranker_error": result.get("reranker_error"),
                            "reranked_candidates": len(
                                result.get("reranked_candidates", [])
                            ),
                            "latency_ms": result.get("latency_ms", {}).get(
                                "reranking"
                            ),
                        },
                    )
                if config["mode"] == "full":
                    _emit(
                        on_event,
                        "generation_completed",
                        "Application answer generated",
                        question_id=sample.id,
                        stage="generation",
                        data={
                            "answer_preview": result["answer"][:240],
                            "latency_ms": result.get("latency_ms", {}).get("generation"),
                        },
                    )
            except Exception as error:
                result["errors"].append(str(error))
                result["evaluation_status"] = "failed"
                result_by_id[sample.id] = result
                on_checkpoint({"results": list(result_by_id.values()), "current": sample.id})
                continue

            result_by_id[sample.id] = result
            on_checkpoint({"results": list(result_by_id.values()), "current": sample.id})

        if config["mode"] == "full" and evaluator and token_budget:
            if (
                result.get("ragas", {}).get("evaluation_version")
                != evaluator.IMPLEMENTATION_VERSION
            ):
                # Keep the saved retrieval and generated answer, but never mix
                # judgments created by different evaluator implementations.
                result["ragas"] = {
                    "evaluation_version": evaluator.IMPLEMENTATION_VERSION,
                    "scores": {},
                    "reasons": {},
                    "errors": {},
                    "token_estimates": {},
                    "cache_hits": {},
                    "sources": {},
                    "traces": {},
                }
            ragas = result["ragas"]
            ragas.setdefault("cache_hits", {})
            ragas.setdefault("sources", {})
            ragas.setdefault("traces", {})
            contexts, judge_context_trace = prepare_judge_contexts(
                result["final_contexts"]
            )
            ragas["judge_contexts"] = judge_context_trace

            # The dataset already contains human-reviewed relevant chunk IDs.
            # Use those labels directly instead of spending six judge calls on
            # LLM-based context precision and context recall.
            context_metrics = result.get("ground_truth_context_metrics")
            if context_metrics:
                for metric_name, score in context_metrics.items():
                    if metric_name in ragas["scores"]:
                        continue
                    ragas["scores"][metric_name] = score
                    ragas["sources"][metric_name] = "human_chunk_ids"
                    ragas["cache_hits"][metric_name] = False
                    _emit(
                        on_event,
                        "metric_completed",
                        f"Completed {metric_name} from human chunk labels",
                        question_id=sample.id,
                        stage="ragas",
                        metric=metric_name,
                        data={"score": score, "source": "human_chunk_ids"},
                    )
                on_checkpoint(
                    {"results": list(result_by_id.values()), "current": sample.id}
                )

            metric_names = evaluator.metric_names_for(sample.answerable)
            for metric_index, metric_name in enumerate(metric_names):
                if metric_name in ragas["scores"]:
                    continue
                if is_cancelled():
                    raise EvaluationCancelled("Evaluation cancelled")

                cached_metric = evaluator.get_cached_metric(
                    metric_name,
                    sample.question,
                    result["answer"],
                    sample.reference_answer,
                    contexts,
                )
                if cached_metric:
                    ragas["scores"][metric_name] = cached_metric["score"]
                    if cached_metric.get("reason"):
                        ragas["reasons"][metric_name] = cached_metric["reason"]
                    if cached_metric.get("trace"):
                        ragas["traces"][metric_name] = cached_metric["trace"]
                    ragas["cache_hits"][metric_name] = True
                    ragas["sources"][metric_name] = "persistent_judge_cache"
                    _remove_metric_error(result, metric_name)
                    _emit(
                        on_event,
                        "metric_cache_hit",
                        f"Loaded {metric_name} from the persistent judge cache",
                        question_id=sample.id,
                        stage="ragas",
                        metric=metric_name,
                        data={"score": cached_metric["score"]},
                    )
                    on_checkpoint(
                        {"results": list(result_by_id.values()), "current": sample.id}
                    )
                    continue

                _remove_metric_error(result, metric_name)

                if metric_index > 0 and config["metric_delay"] > 0:
                    _emit(
                        on_event,
                        "waiting",
                        f"Waiting {config['metric_delay']:g}s before the next metric",
                        question_id=sample.id,
                        stage="pacing",
                        metric=metric_name,
                        data={"seconds": config["metric_delay"]},
                    )
                    _sleep_with_cancel(config["metric_delay"], is_cancelled)

                estimate = estimate_tokens(
                    sample.question,
                    result["answer"],
                    sample.reference_answer,
                    *contexts,
                )
                ragas["token_estimates"][metric_name] = estimate
                token_budget.reserve(
                    estimate,
                    on_wait=lambda seconds, tokens: _emit(
                        on_event,
                        "token_budget_wait",
                        "Waiting for the local Groq token budget",
                        question_id=sample.id,
                        stage="ragas",
                        metric=metric_name,
                        data={"seconds": round(seconds, 1), "estimated_tokens": tokens},
                    ),
                )
                _emit(
                    on_event,
                    "metric_started",
                    f"Starting {metric_name}",
                    question_id=sample.id,
                    stage="ragas",
                    metric=metric_name,
                    data={"estimated_tokens": estimate},
                )

                try:
                    metric_result = evaluator.evaluate_metric(
                        metric_name,
                        question=sample.question,
                        answer=result["answer"],
                        reference_answer=sample.reference_answer,
                        contexts=contexts,
                        on_retry=lambda attempt, delay, error: _emit(
                            on_event,
                            "rate_limit_retry",
                            f"Rate limited; retry {attempt} in {delay:g}s",
                            level="warning",
                            question_id=sample.id,
                            stage="ragas",
                            metric=metric_name,
                            data={
                                "attempt": attempt,
                                "seconds": delay,
                                "limit": rate_limit_details(error).__dict__,
                            },
                        ),
                        on_error_retry=lambda attempt, delay, error: _emit(
                            on_event,
                            "metric_error_retry",
                            f"Invalid or incomplete result; retry {attempt}/2 "
                            f"in {delay:g}s",
                            level="warning",
                            question_id=sample.id,
                            stage="ragas",
                            metric=metric_name,
                            data={
                                "attempt": attempt,
                                "seconds": delay,
                                "error": str(error)[:500],
                            },
                        ),
                        sleep=lambda seconds: _sleep_with_cancel(seconds, is_cancelled),
                    )
                except Exception as error:
                    if is_rate_limit_error(error):
                        details = rate_limit_details(error).__dict__
                        ragas["errors"][metric_name] = details["message"]
                        result_by_id[sample.id] = result
                        on_checkpoint(
                            {
                                "results": list(result_by_id.values()),
                                "current": sample.id,
                                "current_metric": metric_name,
                                "rate_limit": details,
                            }
                        )
                        _emit(
                            on_event,
                            "run_paused",
                            "Evaluation paused after the Groq token limit was reached",
                            level="warning",
                            question_id=sample.id,
                            stage="ragas",
                            metric=metric_name,
                            data=details,
                        )
                        raise EvaluationPaused(details) from error

                    ragas["errors"][metric_name] = str(error)
                    result["errors"].append(f"RAGAS {metric_name}: {error}")
                    on_checkpoint({"results": list(result_by_id.values()), "current": sample.id})
                    continue

                ragas["scores"][metric_name] = metric_result["score"]
                ragas["cache_hits"][metric_name] = metric_result.get("cached", False)
                ragas["sources"][metric_name] = (
                    "persistent_judge_cache"
                    if metric_result.get("cached")
                    else "judge_llm"
                )
                if metric_result["reason"]:
                    ragas["reasons"][metric_name] = metric_result["reason"]
                if metric_result.get("trace"):
                    ragas["traces"][metric_name] = metric_result["trace"]
                ragas["errors"].pop(metric_name, None)
                _remove_metric_error(result, metric_name)
                _emit(
                    on_event,
                    "metric_completed",
                    f"Completed {metric_name}",
                    question_id=sample.id,
                    stage="ragas",
                    metric=metric_name,
                    data={
                        "score": metric_result["score"],
                        "quota": evaluator.latest_quota,
                    },
                )
                on_checkpoint({"results": list(result_by_id.values()), "current": sample.id})

        missing_judge_metrics = (
            [
                name
                for name in evaluator.metric_names_for(sample.answerable)
                if name not in result.get("ragas", {}).get("scores", {})
            ]
            if evaluator
            else []
        )
        result["evaluation_status"] = (
            "partial" if missing_judge_metrics else "complete"
        )
        result_by_id[sample.id] = result
        on_checkpoint({"results": list(result_by_id.values()), "current": sample.id})
        _emit(
            on_event,
            "question_completed",
            f"Completed {sample.id}",
            question_id=sample.id,
            stage="question",
            data={"number": index, "total": len(samples)},
        )

    ordered_results = [result_by_id[sample.id] for sample in samples if sample.id in result_by_id]
    report = build_report(config, ordered_results)
    json_path, csv_path = write_artifacts(report, config["experiment"])
    _emit(
        on_event,
        "run_completed",
        "Evaluation completed",
        stage="run",
        data={"json": str(json_path), "csv": str(csv_path)},
    )
    return report, json_path, csv_path
