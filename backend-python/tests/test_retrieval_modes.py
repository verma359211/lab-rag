"""Verify that each user-selected mode runs only the expected search path."""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.retrieval import retrieve_documents
from app.search.models import SearchResult


def sample_result(name: str) -> SearchResult:
    """Create a small search result without using PostgreSQL or Gemini."""

    return SearchResult(
        document=Document(id=name, page_content=name, metadata={"source": name})
    )


class RetrievalModeTests(unittest.TestCase):
    """Keep vector, keyword, and hybrid routing easy to reason about."""

    @patch("app.retrieval.search_by_keyword")
    @patch("app.retrieval.search_by_vector")
    def test_vector_mode_does_not_run_keyword_search(self, vector, keyword):
        vector.return_value = [sample_result("vector")]

        results = retrieve_documents("question", "vector")

        self.assertEqual(results[0].document.id, "vector")
        vector.assert_called_once_with("question")
        keyword.assert_not_called()

    @patch("app.retrieval.search_by_keyword")
    @patch("app.retrieval.search_by_vector")
    def test_keyword_mode_does_not_run_vector_search(self, vector, keyword):
        keyword.return_value = [sample_result("keyword")]

        results = retrieve_documents("question", "keyword")

        self.assertEqual(results[0].document.id, "keyword")
        keyword.assert_called_once_with("question")
        vector.assert_not_called()

    @patch("app.retrieval.fuse_results")
    @patch("app.retrieval.search_by_keyword")
    @patch("app.retrieval.search_by_vector")
    def test_hybrid_mode_combines_both_searches(self, vector, keyword, fuse):
        vector.return_value = [sample_result("vector")]
        keyword.return_value = [sample_result("keyword")]
        fuse.return_value = [sample_result("fused")]

        results = retrieve_documents("question", "hybrid")

        self.assertEqual(results[0].document.id, "fused")
        fuse.assert_called_once_with(
            vector.return_value,
            keyword.return_value,
            limit=None,
        )

    @patch("app.retrieval.rerank_results")
    @patch("app.retrieval.fuse_results")
    @patch("app.retrieval.search_by_keyword")
    @patch("app.retrieval.search_by_vector")
    def test_reranked_hybrid_runs_the_cross_encoder(
        self,
        vector,
        keyword,
        fuse,
        rerank,
    ):
        """The new mode must rerank RRF candidates before returning five."""

        vector.return_value = [sample_result("vector")]
        keyword.return_value = [sample_result("keyword")]
        fuse.return_value = [sample_result("fused")]
        rerank.return_value = [sample_result("reranked")]

        results = retrieve_documents("question", "hybrid_rerank")

        self.assertEqual(results[0].document.id, "reranked")
        rerank.assert_called_once_with(
            "question",
            fuse.return_value,
            limit=None,
        )

    @patch("app.retrieval.rerank_results", side_effect=RuntimeError("offline"))
    @patch("app.retrieval.fuse_results")
    @patch("app.retrieval.search_by_keyword")
    @patch("app.retrieval.search_by_vector")
    def test_reranker_failure_falls_back_to_rrf(
        self,
        vector,
        keyword,
        fuse,
        _rerank,
    ):
        """A missing local model must not stop the normal RAG response."""

        vector.return_value = [sample_result("vector")]
        keyword.return_value = [sample_result("keyword")]
        fuse.return_value = [sample_result("fused")]

        results = retrieve_documents("question", "hybrid_rerank")

        self.assertEqual(results[0].document.id, "fused")


if __name__ == "__main__":
    unittest.main()
