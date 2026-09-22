"""Check hybrid rank fusion without calling a database or external API."""

import unittest

from langchain_core.documents import Document

from app.search.fusion import fuse_results
from app.search.models import SearchResult


def make_document(chunk_id: str) -> Document:
    """Create a tiny document with the stable ID used during fusion."""

    return Document(
        id=chunk_id,
        page_content=f"Text for {chunk_id}",
        metadata={"chunk_id": chunk_id, "source": "test.pdf", "page": 0},
    )


class FusionTests(unittest.TestCase):
    """Verify that both rankings contribute to the final order."""

    def test_chunk_found_by_both_searches_moves_to_the_top(self):
        vector_results = [
            SearchResult(make_document("a"), vector_score=0.95),
            SearchResult(make_document("b"), vector_score=0.90),
        ]
        keyword_results = [
            SearchResult(make_document("b"), keyword_score=0.80),
            SearchResult(make_document("c"), keyword_score=0.70),
        ]

        results = fuse_results(vector_results, keyword_results)

        self.assertEqual(results[0].document.metadata["chunk_id"], "b")
        self.assertEqual(results[0].vector_rank, 2)
        self.assertEqual(results[0].keyword_rank, 1)
        self.assertGreater(results[0].fusion_score, results[1].fusion_score)

    def test_result_can_come_from_only_one_search_method(self):
        vector_results = [
            SearchResult(make_document("vector-only"), vector_score=0.85)
        ]
        keyword_results = [
            SearchResult(make_document("keyword-only"), keyword_score=0.75)
        ]

        results = fuse_results(vector_results, keyword_results)
        result_ids = {
            result.document.metadata["chunk_id"]
            for result in results
        }

        self.assertEqual(result_ids, {"vector-only", "keyword-only"})


if __name__ == "__main__":
    unittest.main()
