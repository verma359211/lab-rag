"""Generate a grounded answer from the chunks found during retrieval.

The chat model does not search the database itself. This module gives it the
retrieved PDF text as context and asks it to answer only from that context.
"""

from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config import get_chat_model_name, require_env


# The prompt sets the most important rule for this demonstration: retrieved
# documents are the source of truth. If the answer is absent, the model should
# say so instead of filling the gap with general knowledge.
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using only the supplied context. "
            "If the context does not contain the answer, say that you do not "
            "have enough information. Keep the answer clear and concise.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}",
        ),
    ]
)


@lru_cache(maxsize=1)
def get_answer_chain():
    """Build and cache the small LangChain answer-generation pipeline.

    LangChain uses the ``|`` operator to connect simple steps:

    1. The prompt inserts the context and question into messages.
    2. Groq sends those messages to the configured language model.
    3. ``StrOutputParser`` turns the model response into a normal string.

    The chain is cached because its configuration does not change per request.
    """

    chat_model = ChatGroq(
        model=get_chat_model_name(),
        api_key=require_env("GROQ_API_KEY"),
        temperature=0,
    )

    return ANSWER_PROMPT | chat_model | StrOutputParser()


def generate_answer(context: str, question: str) -> str:
    """Send retrieved context and the user's question through the chain."""

    return get_answer_chain().invoke(
        {
            "context": context,
            "question": question,
        }
    )
