"""Environment settings used only by the RAGAS evaluator models."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluatorConfig:
    """Keep evaluator credentials and models separate from application Groq."""

    llm_provider: str
    llm_model: str
    llm_api_key: str
    llm_base_url: str | None
    embedding_provider: str
    embedding_model: str
    embedding_api_key: str
    embedding_base_url: str | None
    llm_max_tokens: int = 4096

    def public_values(self) -> dict[str, str | None]:
        """Return safe configuration values for result artifacts, never keys."""

        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "llm_max_tokens": self.llm_max_tokens,
        }


def _required(name: str) -> str:
    """Read one required evaluator setting with a direct error message."""

    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Full evaluation requires {name}")
    return value


def _positive_integer(name: str, default: int) -> int:
    """Read one optional positive integer evaluator setting."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def load_evaluator_config() -> EvaluatorConfig:
    """Load evaluator settings only when full evaluation is requested."""

    llm_provider = os.getenv("EVAL_LLM_PROVIDER", "openai").lower()
    embedding_provider = os.getenv("EVAL_EMBEDDING_PROVIDER", "openai").lower()

    if llm_provider not in {"openai", "groq"}:
        raise RuntimeError("EVAL_LLM_PROVIDER must be 'openai' or 'groq'")

    if embedding_provider not in {"openai", "google"}:
        raise RuntimeError(
            "EVAL_EMBEDDING_PROVIDER must be 'openai' or 'google'"
        )

    return EvaluatorConfig(
        llm_provider=llm_provider,
        llm_model=_required("EVAL_LLM_MODEL"),
        llm_api_key=_required("EVAL_LLM_API_KEY"),
        llm_base_url=os.getenv("EVAL_LLM_BASE_URL"),
        embedding_provider=embedding_provider,
        embedding_model=_required("EVAL_EMBEDDING_MODEL"),
        embedding_api_key=_required("EVAL_EMBEDDING_API_KEY"),
        embedding_base_url=os.getenv("EVAL_EMBEDDING_BASE_URL"),
        llm_max_tokens=_positive_integer("EVAL_LLM_MAX_TOKENS", 4096),
    )
