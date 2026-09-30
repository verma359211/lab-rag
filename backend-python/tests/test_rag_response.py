"""Verify that answer responses expose understandable retrieval details."""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.rag import answer_question
from app.retrieval import RetrievalTrace
from app.search.models import SearchResult


class RagResponseTests(unittest.TestCase):
    @patch("app.rag.generate_answer", return_value="Test answer")
    @patch("app.rag.retrieve_with_diagnostics")
    def test_response_contains_method_and_chunk_scores(self, retrieve, _generate):
        search_results = [
            SearchResult(
                document=Document(
                    id="chunk-1",
                    page_content="Useful context",
                    metadata={"source": "guide.pdf", "page": 0},
                ),
                vector_score=0.91,
                vector_rank=1,
                fusion_score=0.02,
                fusion_rank=2,
                reranker_score=0.85,
                reranker_rank=1,
            )
        ]
        retrieve.return_value = RetrievalTrace(final_results=search_results)

        response = answer_question("Question", "vector")

        self.assertEqual(response["retrievalMethod"], "vector")
        self.assertEqual(response["retrievalDetails"][0]["source"], "guide.pdf")
        self.assertEqual(response["retrievalDetails"][0]["page"], 1)
        self.assertEqual(response["retrievalDetails"][0]["vectorScore"], 0.91)
        self.assertEqual(response["retrievalDetails"][0]["hybridRank"], 2)
        self.assertEqual(response["retrievalDetails"][0]["rerankerScore"], 0.85)
        self.assertEqual(response["retrievalDetails"][0]["rerankerRank"], 1)


if __name__ == "__main__":
    unittest.main()
