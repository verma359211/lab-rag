"""Test deterministic token pacing without real waits or API calls."""

from evals.token_budget import TokenBudget, estimate_tokens


def test_estimate_includes_text_and_output_reserve() -> None:
    assert estimate_tokens("a" * 200, output_reserve=50) == 150


def test_waits_until_rolling_budget_resets() -> None:
    current_time = [0.0]
    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        current_time[0] += seconds

    budget = TokenBudget(
        tokens_per_minute=1000,
        clock=lambda: current_time[0],
        sleep=sleep,
    )
    budget.reserve(600)
    budget.reserve(300)

    assert waits == [60]
