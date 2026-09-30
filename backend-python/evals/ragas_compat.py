"""Temporary compatibility for RAGAS 0.4.3 and modern LangChain.

RAGAS 0.4.3 imports an old Vertex AI module during startup even when Vertex
AI is not used. Modern ``langchain-community`` removed that module. This small
shim supplies the unused class so OpenAI and Groq evaluators can start. It can
be deleted after the upstream RAGAS import is fixed.
"""

import sys
from types import ModuleType


def install_vertex_import_shim() -> None:
    """Provide the removed optional module only when it is unavailable."""

    module_name = "langchain_community.chat_models.vertexai"

    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    class ChatVertexAI:
        """Unused placeholder required only while importing RAGAS 0.4.3."""

    compatibility_module = ModuleType(module_name)
    compatibility_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = compatibility_module
