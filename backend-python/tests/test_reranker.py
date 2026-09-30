"""Check local cross-encoder ordering without downloading a real model."""

from langchain_core.documents import Document

from app.search.models import SearchResult
from app.search.reranker import rerank_results


class FakeCrossEncoder:
    """Return predictable scores for a small reranker unit test."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Give the candidate containing 'best' the strongest score."""

        return [0.9 if "best" in passage else 0.1 for _question, passage in pairs]


def make_result(name: str, content: str, fusion_rank: int) -> SearchResult:
    """Create one candidate with its original RRF information."""

    return SearchResult(
        document=Document(id=name, page_content=content),
        fusion_score=1 / (60 + fusion_rank),
        fusion_rank=fusion_rank,
    )


def test_cross_encoder_reorders_candidates_and_preserves_rrf_data() -> None:
    """Reranking should change final order without erasing RRF evidence."""

    first = make_result("first", "less relevant text", 1)
    second = make_result("second", "best supporting text", 2)

    ranked = rerank_results(
        "question",
        [first, second],
        model=FakeCrossEncoder(),
    )

    assert [item.document.id for item in ranked] == ["second", "first"]
    assert ranked[0].reranker_score == 0.9
    assert ranked[0].reranker_rank == 1
    assert ranked[0].fusion_rank == 2


def test_reranker_limit_is_applied_after_scoring() -> None:
    """Only the best requested chunks should be sent to generation."""

    ranked = rerank_results(
        "question",
        [
            make_result("first", "less relevant text", 1),
            make_result("second", "best supporting text", 2),
        ],
        limit=1,
        model=FakeCrossEncoder(),
    )

    assert len(ranked) == 1
    assert ranked[0].document.id == "second"
