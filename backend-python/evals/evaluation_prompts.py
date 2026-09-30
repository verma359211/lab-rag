"""Compact, question-aware prompts used by generation-quality evaluation."""

import json

from pydantic import BaseModel, Field


class ClaimVerdict(BaseModel):
    """One substantive answer claim and whether the evidence supports it."""

    statement: str
    verdict: int = Field(ge=0, le=1)
    reason: str


class FaithfulnessJudgment(BaseModel):
    """All substantive claims extracted and judged in one model call."""

    claims: list[ClaimVerdict]


class FactualCorrectnessJudgment(BaseModel):
    """A semantic whole-answer score with a short audit explanation."""

    score: float = Field(ge=0, le=1)
    reason: str
    correct_points: list[str]
    problems: list[str]


class RelevancyQuestion(BaseModel):
    """One reverse-generated question used for embedding similarity."""

    question: str
    noncommittal: int = Field(ge=0, le=1)


def faithfulness_prompt(
    question: str,
    answer: str,
    contexts: list[str],
) -> str:
    """Ask for claim extraction and evidence checking in one call."""

    values = {
        "question": question,
        "answer": answer,
        "retrieved_contexts": contexts,
    }
    return """Evaluate whether the answer is faithful to the available evidence.
Treat facts stated in the question and retrieved contexts as evidence.
Break the answer into substantive factual claims and judge every claim once.
Do not create meta-claims from standalone words such as 'yes' or 'no'.
A claim is supported when stated by the evidence or derived through straightforward deterministic logic or arithmetic.
Use verdict 1 for supported and 0 for unsupported or contradicted.
Keep every reason under 20 words.

Input:
""" + json.dumps(values, ensure_ascii=False)


def factual_correctness_prompt(
    question: str,
    answer: str,
    reference_answer: str,
) -> str:
    """Ask for one semantic correctness score instead of claim-level F1."""

    values = {
        "question": question,
        "generated_answer": answer,
        "reference_answer": reference_answer,
    }
    return """Judge the generated answer against the human reference in light of the question.
Compare meaning, not exact wording.
Score from 0 to 1: 1 means fully correct, 0.75 mostly correct with a minor issue, 0.5 partially correct, 0.25 mostly incorrect, and 0 completely incorrect.
Do not penalize a concise answer for omitting optional explanation.
Do not penalize harmless added detail merely because the reference omits it; lower the score only when it conflicts with the question or reference.
Keep the reason under 25 words and list only meaningful correct points or problems.

Input:
""" + json.dumps(values, ensure_ascii=False)


def answer_relevancy_prompt(answer: str) -> str:
    """Generate one comparison question with minimal prompt overhead."""

    return """Generate one question that this answer directly answers.
Set noncommittal to 1 only when the answer is evasive or avoids a definite answer; otherwise use 0.

Answer:
""" + json.dumps(answer, ensure_ascii=False)
