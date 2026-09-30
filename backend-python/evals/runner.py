"""Command-line runner for repeatable offline RAG experiments."""

import argparse
import statistics
import time
from collections import Counter
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
from app.retrieval import RetrievalTrace, retrieve_with_diagnostics
from app.search.fusion import get_document_key
from app.search.models import SearchMode, SearchResult
from evals.artifacts import write_artifacts
from evals.dataset import load_dataset
from evals.judge_contexts import prepare_judge_contexts
from evals.metrics import (
    average_available_metrics,
    calculate_context_metrics,
    calculate_retrieval_metrics,
)
from evals.rate_limit import RetrySettings, run_with_rate_limit_retry


# Day-to-day runs use the focused benchmark. Pass ``--dataset`` explicitly for
# the complete 60-question regression suite.
DEFAULT_DATASET = Path("evals/datasets/v1/golden.core15.jsonl")


def non_negative_float(value: str) -> float:
    """Reject negative delay values with an argparse-friendly message."""

    parsed_value = float(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed_value


def non_negative_int(value: str) -> int:
    """Reject negative retry counts with an argparse-friendly message."""

    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed_value


def _candidate(result: SearchResult, rank: int) -> dict[str, Any]:
    """Turn one internal search result into durable experiment evidence."""

    metadata = result.document.metadata
    page = metadata.get("page")
    return {
        "rank": rank,
        "chunk_id": get_document_key(result.document),
        "document_id": metadata.get("document_id"),
        "source": metadata.get("source"),
        "page": page + 1 if isinstance(page, int) else None,
        "chunk_number": metadata.get("chunk_number"),
        "content": result.document.page_content,
        "vector_score": result.vector_score,
        "keyword_score": result.keyword_score,
        "fusion_score": result.fusion_score or None,
        "fusion_rank": result.fusion_rank,
        "reranker_score": result.reranker_score,
        "reranker_rank": result.reranker_rank,
        "vector_rank": result.vector_rank,
        "keyword_rank": result.keyword_rank,
    }


def _candidates(results: list[SearchResult]) -> list[dict[str, Any]]:
    """Serialize a ranked list while preserving its position."""

    return [_candidate(result, rank) for rank, result in enumerate(results, 1)]


def _ranking_for_mode(trace: RetrievalTrace, mode: SearchMode) -> list[SearchResult]:
    """Select the complete ranking used for deterministic retrieval metrics."""

    if mode == "vector":
        return trace.vector_candidates
    if mode == "keyword":
        return trace.keyword_candidates
    if mode == "hybrid_rerank":
        return trace.reranked_candidates
    return trace.fused_candidates


def _percentile(values: list[float], percentile: float) -> float | None:
    """Calculate a linearly interpolated percentile for a small experiment."""

    if not values:
        return None
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def _average_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average every available RAGAS score and ignore failed metrics."""

    names = {
        name
        for row in rows
        for name in row.get("ragas", {}).get("scores", {})
    }
    averages: dict[str, float] = {}
    for name in names:
        values = [
            row["ragas"]["scores"][name]
            for row in rows
            if name in row.get("ragas", {}).get("scores", {})
        ]
        if values:
            averages[name] = statistics.fmean(values)
    return averages


def _base_result(sample: Any) -> dict[str, Any]:
    """Copy the human reference data into the per-question result."""

    return {
        "id": sample.id,
        "question": sample.question,
        "reference_answer": sample.reference_answer,
        "reference_sources": [
            source.model_dump() for source in sample.reference_sources
        ],
        "expected_chunk_ids": sample.expected_chunk_ids,
        "expected_evidence_sets": sample.expected_evidence_sets,
        "category": sample.category,
        "answerable": sample.answerable,
        "notes": sample.notes,
        "errors": [],
    }


def _add_trace(result: dict[str, Any], trace: RetrievalTrace, mode: SearchMode) -> None:
    """Attach all retrieval stages and deterministic metrics to one result."""

    ranking = _ranking_for_mode(trace, mode)
    ranked_ids = [get_document_key(item.document) for item in ranking]
    result.update(
        {
            "vector_candidates": _candidates(trace.vector_candidates),
            "keyword_candidates": _candidates(trace.keyword_candidates),
            "fused_candidates": _candidates(trace.fused_candidates),
            "reranked_candidates": _candidates(trace.reranked_candidates),
            "final_contexts": _candidates(trace.final_results),
            "reranker_applied": trace.reranker_applied,
            "reranker_error": trace.reranker_error,
            "retrieval_metrics": calculate_retrieval_metrics(
                ranked_ids,
                result["expected_chunk_ids"],
                expected_evidence_sets=result.get("expected_evidence_sets"),
            ),
            "ground_truth_context_metrics": calculate_context_metrics(
                [get_document_key(item.document) for item in trace.final_results],
                result["expected_chunk_ids"],
                expected_evidence_sets=result.get("expected_evidence_sets"),
            ),
            "latency_ms": dict(trace.latency_ms),
        }
    )


def _process_sample(
    sample: Any,
    args: argparse.Namespace,
    evaluator: Any,
) -> dict[str, Any]:
    """Run one sample from a fresh result object so retries stay clean."""

    result = _base_result(sample)
    if args.mode == "retrieval":
        trace = retrieve_with_diagnostics(sample.question, args.search_mode)
        _add_trace(result, trace, args.search_mode)
        return result

    execution = execute_rag(sample.question, args.search_mode)
    _add_trace(result, execution.retrieval, args.search_mode)
    result["answer"] = execution.answer
    result["latency_ms"].update(execution.latency_ms)
    judge_contexts, judge_context_trace = prepare_judge_contexts(
        result["final_contexts"]
    )
    result["ragas"] = evaluator.evaluate(
        question=sample.question,
        answer=execution.answer,
        reference_answer=sample.reference_answer,
        contexts=judge_contexts,
        answerable=sample.answerable,
    )
    result["ragas"]["judge_contexts"] = judge_context_trace
    context_metrics = result.get("ground_truth_context_metrics")
    if context_metrics:
        result["ragas"]["scores"].update(context_metrics)
        for metric_name in context_metrics:
            result["ragas"]["sources"][metric_name] = "human_chunk_ids"
            result["ragas"]["cache_hits"][metric_name] = False
    for metric_name, error in result["ragas"]["errors"].items():
        result["errors"].append(f"RAGAS {metric_name}: {error}")
    return result


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    """Run every dataset example, continue on errors, and save both artifacts."""

    samples = load_dataset(args.dataset)
    evaluator = None
    evaluator_settings = None
    retry_settings = RetrySettings(
        max_retries=args.max_retries,
        initial_delay_seconds=args.initial_retry_delay,
    )

    # Imports and evaluator clients are intentionally absent in retrieval mode.
    # This guarantees that mode cannot accidentally make RAGAS/LLM calls.
    if args.mode == "full":
        from evals.evaluator_config import load_evaluator_config
        from evals.ragas_metrics import RagasEvaluator

        evaluator_settings = load_evaluator_config()
        evaluator = RagasEvaluator(
            evaluator_settings,
            retry_settings=retry_settings,
            metric_delay_seconds=args.metric_delay,
            tokens_per_minute=args.groq_tpm_limit,
        )

    results: list[dict[str, Any]] = []
    needs_external_pacing = args.mode == "full" or args.search_mode != "keyword"
    for sample_number, sample in enumerate(samples, start=1):
        if (
            sample_number > 1
            and args.request_delay > 0
            and needs_external_pacing
        ):
            time.sleep(args.request_delay)

        print(f"[{sample_number}/{len(samples)}] Evaluating {sample.id}...")
        try:
            result = run_with_rate_limit_retry(
                operation=lambda: _process_sample(sample, args, evaluator),
                settings=retry_settings,
                on_retry=lambda attempt, delay, _error: print(
                    f"{sample.id} rate limited. Retry {attempt}/"
                    f"{retry_settings.max_retries} in {delay:g}s."
                ),
            )
        except Exception as error:
            result = _base_result(sample)
            result["errors"].append(str(error))

        results.append(result)

    retrieval_rows = [row.get("retrieval_metrics") for row in results]
    total_latencies = [
        row["latency_ms"].get("total", row["latency_ms"].get("retrieval_total"))
        for row in results
        if row.get("latency_ms")
    ]
    total_latencies = [value for value in total_latencies if value is not None]
    failures = Counter(
        row["category"] for row in results if row.get("errors")
    )
    aggregate = {
        "sample_count": len(results),
        "successful_count": sum(not row["errors"] for row in results),
        "failed_count": sum(bool(row["errors"]) for row in results),
        "chunk_labeled_count": sum(row is not None for row in retrieval_rows),
        "retrieval_metrics": average_available_metrics(retrieval_rows),
        "ragas_metrics": _average_scores(results),
        "judge_cache_hit_count": sum(
            bool(cache_hit)
            for row in results
            for metric_name, cache_hit in row.get("ragas", {})
            .get("cache_hits", {})
            .items()
            if metric_name not in {"context_precision", "context_recall"}
        ),
        "latency_ms": {
            "average": statistics.fmean(total_latencies) if total_latencies else None,
            "p50": _percentile(total_latencies, 0.50),
            "p95": _percentile(total_latencies, 0.95),
        },
        "failures_by_category": dict(failures),
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": args.experiment,
        "mode": args.mode,
        "dataset": str(Path(args.dataset).resolve()),
        "configuration": {
            "search_mode": args.search_mode,
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
                if args.search_mode == "hybrid_rerank"
                else None
            ),
            "rerank_candidate_limit": RERANK_CANDIDATE_LIMIT,
            "request_delay_seconds": args.request_delay,
            "max_rate_limit_retries": args.max_retries,
            "initial_retry_delay_seconds": args.initial_retry_delay,
            "metric_delay_seconds": args.metric_delay,
            "groq_tokens_per_minute": args.groq_tpm_limit,
            "context_metrics_source": "human_expected_chunk_ids",
            "persistent_judge_cache": True,
            "evaluation_implementation": (
                evaluator.IMPLEMENTATION_VERSION if evaluator else None
            ),
            "evaluator": (
                evaluator_settings.public_values()
                if evaluator_settings
                else None
            ),
        },
        "aggregate": aggregate,
        "results": results,
    }
    json_path, csv_path = write_artifacts(report, args.experiment)
    return report, json_path, csv_path


def _print_summary(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    """Print the short scorecard needed after an experiment finishes."""

    aggregate = report["aggregate"]
    print(
        f"Completed {aggregate['sample_count']} samples: "
        f"{aggregate['successful_count']} successful, "
        f"{aggregate['failed_count']} with errors."
    )
    print(
        "Chunk-labeled samples: "
        f"{aggregate['chunk_labeled_count']}/{aggregate['sample_count']}"
    )
    print(f"Retrieval metrics: {aggregate['retrieval_metrics'] or 'not available'}")
    if report["mode"] == "full":
        print(f"RAGAS metrics: {aggregate['ragas_metrics'] or 'not available'}")
    print(f"Latency (ms): {aggregate['latency_ms']}")
    print(f"Failures by category: {aggregate['failures_by_category'] or 'none'}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


def parse_args() -> argparse.Namespace:
    """Read the intentionally small evaluation command-line interface."""

    parser = argparse.ArgumentParser(description="Evaluate the existing RAG pipeline")
    parser.add_argument("--mode", choices=("retrieval", "full"), required=True)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--experiment", default="baseline-hybrid-rrf")
    parser.add_argument(
        "--request-delay",
        type=non_negative_float,
        default=5.0,
        help="Seconds to wait between evaluation questions (default: 5).",
    )
    parser.add_argument(
        "--max-retries",
        type=non_negative_int,
        default=4,
        help="Maximum retries for a rate-limited operation (default: 4).",
    )
    parser.add_argument(
        "--initial-retry-delay",
        type=non_negative_float,
        default=5.0,
        help="Initial 429 retry delay in seconds; later waits double (default: 5).",
    )
    parser.add_argument(
        "--metric-delay",
        type=non_negative_float,
        default=20.0,
        help="Seconds to wait between RAGAS metrics (default: 20).",
    )
    parser.add_argument(
        "--groq-tpm-limit",
        type=non_negative_int,
        default=8000,
        help="Groq token-per-minute budget used for local pacing (default: 8000).",
    )
    parser.add_argument(
        "--search-mode",
        choices=("hybrid", "hybrid_rerank", "vector", "keyword"),
        default="hybrid",
    )
    return parser.parse_args()


def main() -> None:
    """Run from ``python -m evals.runner``."""

    report, json_path, csv_path = run(parse_args())
    _print_summary(report, json_path, csv_path)


if __name__ == "__main__":
    main()
