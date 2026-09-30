"""A small judge prompt for questions intentionally lacking a definite answer."""

import json

from pydantic import BaseModel, Field


class AbstentionInput(BaseModel):
    """Information needed to judge one deliberately unanswerable question."""

    question: str
    response: str
    reference: str
    contexts: str


class AbstentionOutput(BaseModel):
    """Binary result with a short explanation that can be audited later."""

    verdict: int = Field(ge=0, le=1)
    reason: str


def build_abstention_prompt(values: AbstentionInput) -> str:
    """Build a compact prompt without importing RAGAS before its shim runs."""

    instructions = """Judge a response to a question that the human dataset marks as unanswerable from the available information.
Return verdict 1 when the response correctly avoids inventing a definite answer and its explanation is consistent with the reference answer and retrieved context.
Return verdict 0 when it guesses an unsupported answer, contradicts the reference, or gives an unrelated refusal.
A concise statement such as 'the corpus does not provide that information' is a correct abstention when the reference confirms the information is absent.
The reason must briefly explain the verdict using the supplied reference and context."""
    data = json.dumps(values.model_dump(), ensure_ascii=False)
    return f"{instructions}\n\nEvaluation input:\n{data}"
