"""Validated request shapes for evaluation dashboard actions."""

from typing import Literal

from pydantic import BaseModel, Field


class EvaluationCreate(BaseModel):
    """Small set of safe options exposed to the browser."""

    experiment: str = Field(default="dashboard-baseline", min_length=1, max_length=80)
    dataset: Literal["core15", "full60"] = "core15"
    mode: Literal["retrieval", "full"] = "retrieval"
    search_mode: Literal[
        "hybrid",
        "hybrid_rerank",
        "vector",
        "keyword",
    ] = "hybrid"
    question_delay: float = Field(default=10, ge=0, le=600)
    metric_delay: float = Field(default=20, ge=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)
    initial_retry_delay: float = Field(default=10, ge=0, le=600)
    groq_tpm_limit: int = Field(default=8000, ge=1000, le=10_000_000)
