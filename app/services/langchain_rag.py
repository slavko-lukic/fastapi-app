from pathlib import Path

from botocore.client import BaseClient
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import Settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.bedrock import generate_reply


DOC_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge.txt"


def _load_doc() -> str:
    """
    Load the knowledge document into memory.
    For now this is a single text file, but you could later expand to multiple docs.
    """
    try:
        return DOC_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "No additional project documentation is available."


DOC_CONTENT = _load_doc()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use the provided context to answer the user, "
            "but do not mention that you are using a document.",
        ),
        (
            "system",
            "Context:\n{context}",
        ),
        ("user", "{question}"),
    ]
)


async def answer_with_doc(
    request: ChatRequest, client: BaseClient, settings: Settings
) -> ChatResponse:
    """
    Use LangChain's prompt template to inject a local document as context,
    then delegate the final call to our existing Bedrock service.
    """
    formatted = prompt.format(context=DOC_CONTENT, question=request.message)
    wrapped_request = ChatRequest(message=formatted)
    return await generate_reply(wrapped_request, client, settings)
