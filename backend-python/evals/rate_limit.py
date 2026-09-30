"""Small retry helper for temporary evaluation API rate limits."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar


ResultType = TypeVar("ResultType")
RetryCallback = Callable[[int, float, Exception], None]


@dataclass(frozen=True)
class RetrySettings:
    """Control bounded exponential backoff for one external operation."""

    max_retries: int = 4
    initial_delay_seconds: float = 5.0


@dataclass(frozen=True)
class RateLimitDetails:
    """Small, safe subset of provider rate-limit information."""

    limit_type: str
    retry_after_seconds: float | None
    token_limit: int | None
    remaining_tokens: int | None
    token_reset: str | None
    message: str


def is_rate_limit_error(error: Exception) -> bool:
    """Recognize common rate-limit messages from Gemini, Groq, and OpenAI."""

    message = str(error).casefold()
    indicators = (
        "429",
        "resource_exhausted",
        "rate limit",
        "requests per minute",
        "tokens per minute",
    )
    return any(indicator in message for indicator in indicators)


def is_retryable_metric_error(error: Exception) -> bool:
    """Recognize temporary or malformed structured judge responses.

    RAGAS already validates its JSON schemas. A fresh model call can recover
    from truncated JSON, an empty generation, a transient connection failure,
    or a temporary 5xx response. Rate limits are handled separately because
    they need provider-specific reset delays.
    """

    if is_rate_limit_error(error):
        return False

    message = str(error).casefold()
    indicators = (
        "json_validate_failed",
        "failed_attempts",
        "output is incomplete",
        "max_tokens length limit",
        "timed out",
        "timeout",
        "connection error",
        "error code: 500",
        "error code: 502",
        "error code: 503",
        "error code: 504",
    )
    return any(indicator in message for indicator in indicators)


def retry_after_seconds(error: Exception) -> float | None:
    """Read a provider Retry-After header when its SDK exposes one."""

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None

    try:
        delay = float(value)
    except (TypeError, ValueError):
        return None
    return delay if delay >= 0 else None


def _integer_header(headers: object, name: str) -> int | None:
    """Read an integer response header without assuming a header class."""

    if not headers:
        return None

    value = headers.get(name)  # type: ignore[union-attr]
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def rate_limit_details(error: Exception) -> RateLimitDetails:
    """Explain whether a 429 is token-based and expose safe reset values."""

    message = str(error)
    lowered = message.casefold()
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)

    if "tokens per day" in lowered or "tpd" in lowered:
        limit_type = "tokens_per_day"
    elif "input tokens" in lowered or "itpm" in lowered:
        limit_type = "input_tokens_per_minute"
    elif "output tokens" in lowered or "otpm" in lowered:
        limit_type = "output_tokens_per_minute"
    elif "token" in lowered:
        limit_type = "tokens_per_minute"
    elif "request" in lowered:
        limit_type = "requests"
    else:
        limit_type = "unknown"

    reset = headers.get("x-ratelimit-reset-tokens") if headers else None
    return RateLimitDetails(
        limit_type=limit_type,
        retry_after_seconds=retry_after_seconds(error),
        token_limit=_integer_header(headers, "x-ratelimit-limit-tokens"),
        remaining_tokens=_integer_header(headers, "x-ratelimit-remaining-tokens"),
        token_reset=str(reset) if reset is not None else None,
        # Provider messages are useful, but API keys and headers are excluded.
        message=message[:500],
    )


def run_with_rate_limit_retry(
    operation: Callable[[], ResultType],
    settings: RetrySettings,
    on_retry: RetryCallback | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ResultType:
    """Retry only 429-style failures and immediately re-raise other errors."""

    for retry_number in range(settings.max_retries + 1):
        try:
            return operation()
        except Exception as error:
            retries_exhausted = retry_number >= settings.max_retries
            daily_token_limit = (
                is_rate_limit_error(error)
                and rate_limit_details(error).limit_type == "tokens_per_day"
            )
            if not is_rate_limit_error(error) or retries_exhausted or daily_token_limit:
                raise

            provider_delay = retry_after_seconds(error)
            delay = (
                provider_delay
                if provider_delay is not None
                else settings.initial_delay_seconds * (2**retry_number)
            )
            if on_retry:
                on_retry(retry_number + 1, delay, error)
            sleep(delay)

    raise RuntimeError("Rate-limit retry loop ended unexpectedly")


def run_with_metric_error_retry(
    operation: Callable[[], ResultType],
    max_retries: int = 2,
    initial_delay_seconds: float = 2.0,
    on_retry: RetryCallback | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ResultType:
    """Retry only temporary structured-output failures with a small bound."""

    for retry_number in range(max_retries + 1):
        try:
            return operation()
        except Exception as error:
            retries_exhausted = retry_number >= max_retries
            if retries_exhausted or not is_retryable_metric_error(error):
                raise

            delay = initial_delay_seconds * (2**retry_number)
            if on_retry:
                on_retry(retry_number + 1, delay, error)
            sleep(delay)

    raise RuntimeError("Metric retry loop ended unexpectedly")
