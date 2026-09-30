"""Check chat request defaults and frontend-facing field names."""

import unittest

from pydantic import ValidationError

from app.schemas import ChatRequest


class ChatRequestTests(unittest.TestCase):
    def test_search_mode_defaults_to_hybrid(self):
        request = ChatRequest.model_validate({"question": "Hello"})

        self.assertEqual(request.search_mode, "hybrid")

    def test_search_mode_accepts_frontend_camel_case_name(self):
        request = ChatRequest.model_validate(
            {"question": "Hello", "searchMode": "keyword"}
        )

        self.assertEqual(request.search_mode, "keyword")

    def test_unknown_search_mode_is_rejected(self):
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate(
                {"question": "Hello", "searchMode": "unknown"}
            )

    def test_search_mode_accepts_hybrid_reranker(self):
        """The API should expose reranking as a separate retrieval mode."""

        request = ChatRequest.model_validate(
            {"question": "Hello", "searchMode": "hybrid_rerank"}
        )

        self.assertEqual(request.search_mode, "hybrid_rerank")


if __name__ == "__main__":
    unittest.main()
