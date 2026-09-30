"""Test rate-limit retry behavior without waiting or calling any provider."""

import pytest

from evals.rate_limit import (
    RetrySettings,
    rate_limit_details,
    run_with_metric_error_retry,
    run_with_rate_limit_retry,
)


def test_rate_limit_errors_use_exponential_backoff() -> None:
    attempts = 0
    waits: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "complete"

    result = run_with_rate_limit_retry(
        operation,
        RetrySettings(max_retries=4, initial_delay_seconds=5),
        sleep=waits.append,
    )

    assert result == "complete"
    assert attempts == 3
    assert waits == [5, 10]


def test_non_rate_limit_error_is_not_retried() -> None:
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("Database connection failed")

    with pytest.raises(RuntimeError, match="Database connection failed"):
        run_with_rate_limit_retry(
            operation,
            RetrySettings(max_retries=4, initial_delay_seconds=5),
            sleep=lambda _delay: None,
        )

    assert attempts == 1


def test_rate_limit_error_stops_after_maximum_retries() -> None:
    attempts = 0
    waits: list[float] = []

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("rate limit exceeded")

    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        run_with_rate_limit_retry(
            operation,
            RetrySettings(max_retries=2, initial_delay_seconds=2),
            sleep=waits.append,
        )

    assert attempts == 3
    assert waits == [2, 4]


def test_retry_after_header_overrides_exponential_delay() -> None:
    waits: list[float] = []
    attempts = 0

    class Response:
        headers = {"Retry-After": "12"}

    class RateLimitError(RuntimeError):
        response = Response()

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError("429 rate limit")
        return "complete"

    run_with_rate_limit_retry(
        operation,
        RetrySettings(max_retries=1, initial_delay_seconds=5),
        sleep=waits.append,
    )

    assert waits == [12]


def test_reads_safe_groq_token_limit_headers() -> None:
    """The dashboard receives quota values but never credentials or headers."""

    class Response:
        headers = {
            "retry-after": "18",
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "1200",
            "x-ratelimit-reset-tokens": "17.8s",
        }

    class RateLimitError(RuntimeError):
        response = Response()

    details = rate_limit_details(
        RateLimitError("429 tokens per minute limit reached")
    )

    assert details.limit_type == "tokens_per_minute"
    assert details.retry_after_seconds == 18
    assert details.token_limit == 8000
    assert details.remaining_tokens == 1200
    assert details.token_reset == "17.8s"


def test_incomplete_structured_output_is_retried() -> None:
    attempts = 0
    waits: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("output is incomplete due to a max_tokens length limit")
        return "valid-json"

    result = run_with_metric_error_retry(
        operation,
        max_retries=2,
        initial_delay_seconds=2,
        sleep=waits.append,
    )

    assert result == "valid-json"
    assert attempts == 3
    assert waits == [2, 4]


def test_metric_retry_does_not_duplicate_rate_limit_retries() -> None:
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("429 tokens per minute")

    with pytest.raises(RuntimeError, match="429"):
        run_with_metric_error_retry(operation, sleep=lambda _delay: None)

    assert attempts == 1
