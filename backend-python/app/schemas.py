"""Describe the JSON data accepted by the API.

Pydantic models act like a clear contract between the React frontend and the
FastAPI backend. FastAPI validates incoming JSON against these models before
our route function runs.
"""

from pydantic import BaseModel, Field

from app.search.models import SearchMode


class ChatRequest(BaseModel):
    """The request body expected by ``POST /chat``.

    Example JSON sent by the frontend:

    ``{"question": "What does the document say?", "searchMode": "hybrid"}``

    The alias keeps JavaScript's camelCase request style while Python code uses
    the conventional snake_case name ``search_mode``.
    """

    question: str
    search_mode: SearchMode = Field(default="hybrid", alias="searchMode")
