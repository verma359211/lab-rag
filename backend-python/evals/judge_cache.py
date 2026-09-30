"""Small persistent cache for expensive evaluator LLM work.

Every cache entry is stored as readable JSON under ``evals/results/cache``.
The filename is a SHA-256 hash of all inputs that can change a judgment. This
keeps lookup fast while the JSON body keeps the cached value easy to inspect.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


CACHE_DIRECTORY = Path(__file__).resolve().parent / "results" / "cache"
CACHE_FORMAT_VERSION = "v1"


def stable_hash(value: dict[str, Any]) -> str:
    """Return the same hash for dictionaries with the same values."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class JudgeCache:
    """Persist successful judge outputs for reuse across evaluation runs."""

    def __init__(self, directory: Path = CACHE_DIRECTORY):
        self.directory = directory
        self.judgments = directory / "judgments"

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        """Return a valid cache entry or ignore a missing/corrupt file."""

        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _write(path: Path, value: dict[str, Any]) -> None:
        """Atomically replace one entry so interrupted writes stay harmless."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    def judgment_key(self, inputs: dict[str, Any]) -> str:
        """Build a versioned key for one complete metric judgment."""

        return stable_hash({"cache_format": CACHE_FORMAT_VERSION, **inputs})

    def get_judgment(self, key: str) -> dict[str, Any] | None:
        """Load one previously successful metric result."""

        entry = self._read(self.judgments / f"{key}.json")
        if not entry:
            return None
        result = entry.get("result")
        return result if isinstance(result, dict) else None

    def save_judgment(
        self,
        key: str,
        inputs: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Save only a successful metric result with its cache-key inputs."""

        self._write(
            self.judgments / f"{key}.json",
            {
                "cache_format": CACHE_FORMAT_VERSION,
                "key_inputs": inputs,
                "result": result,
            },
        )
