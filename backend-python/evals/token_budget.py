"""Conservative local token pacing for Groq evaluation calls."""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


WaitCallback = Callable[[float, int], None]


def estimate_tokens(*texts: str, output_reserve: int = 500) -> int:
    """Estimate tokens simply and conservatively without another API call.

    Four characters per token is a common rough estimate for English text.
    RAGAS may create more than one prompt internally, so the estimate is
    doubled and includes room for the structured judge response.
    """

    characters = sum(len(text) for text in texts if text)
    return max(1, characters // 2 + output_reserve)


@dataclass
class TokenBudget:
    """Keep estimated usage below a safe share of a per-minute limit."""

    tokens_per_minute: int
    safety_ratio: float = 0.75
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    usage: deque[tuple[float, int]] = field(default_factory=deque)

    @property
    def safe_limit(self) -> int:
        """Reserve headroom because token counts are estimates."""

        return max(1, int(self.tokens_per_minute * self.safety_ratio))

    def reserve(self, estimated_tokens: int, on_wait: WaitCallback | None = None) -> None:
        """Wait until the rolling one-minute budget can fit one operation."""

        amount = min(max(1, estimated_tokens), self.safe_limit)

        while True:
            now = self.clock()
            while self.usage and now - self.usage[0][0] >= 60:
                self.usage.popleft()

            used = sum(tokens for _timestamp, tokens in self.usage)
            if used + amount <= self.safe_limit:
                self.usage.append((now, amount))
                return

            wait_seconds = max(0.1, 60 - (now - self.usage[0][0]))
            if on_wait:
                on_wait(wait_seconds, amount)
            self.sleep(wait_seconds)
