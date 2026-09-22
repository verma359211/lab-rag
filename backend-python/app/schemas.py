"""Describe the JSON data accepted by the API.

Pydantic models act like a clear contract between the React frontend and the
FastAPI backend. FastAPI validates incoming JSON against these models before
our route function runs.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """The request body expected by ``POST /chat``.

    Example JSON sent by the frontend:

    ``{"question": "What does the document say about refunds?"}``
    """

    question: str
