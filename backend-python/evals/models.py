"""Validated shapes for human-written evaluation examples."""

from pydantic import BaseModel, Field


class ReferenceSource(BaseModel):
    """Identify a source document and, when useful, a specific page."""

    source: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    location: str | None = None


class GoldenSample(BaseModel):
    """Represent one question and its human-reviewed expected result."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    # An intentionally unanswerable question can correctly have no source.
    reference_sources: list[ReferenceSource]
    expected_chunk_ids: list[str] | None = None
    # Some answers have more than one complete evidence path. Each inner list
    # is one valid combination; retrieval only needs to satisfy one of them.
    expected_evidence_sets: list[list[str]] | None = None
    category: str = Field(min_length=1)
    answerable: bool
    notes: str | None = None
