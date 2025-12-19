from pathlib import Path
from typing import Any, Dict

from botocore.client import BaseClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

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


def _build_doc_chain(client: BaseClient, settings: Settings) -> RunnableLambda:
    """
    Build a small multi-step LangChain pipeline:

      1) Start from the raw question.
      2) Add the loaded document as a 'context' field.
      3) Apply the ChatPromptTemplate using {context, question}.
      4) Call Bedrock via our existing generate_reply service.
    """

    def _add_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Inputs should contain {"question": "..."}; we enrich with the doc content.
        return {**inputs, "context": DOC_CONTENT}

    add_context = RunnableLambda(_add_context)

    async def _call_bedrock(prompt_value: Any) -> str:
        # ChatPromptTemplate returns a PromptValue; turn it into text
        if hasattr(prompt_value, "to_string"):
            message_text = prompt_value.to_string()
        else:
            message_text = str(prompt_value)

        request = ChatRequest(message=message_text)
        response = await generate_reply(request, client, settings)
        return response.reply

    bedrock_step = RunnableLambda(_call_bedrock)

    # A true "chain": input dict -> add_context -> prompt -> bedrock_step
    chain = RunnablePassthrough() | add_context | prompt | bedrock_step
    return chain


async def answer_with_doc(
    request: ChatRequest, client: BaseClient, settings: Settings
) -> ChatResponse:
    """
    Use a LangChain Runnable pipeline that:
      - attaches the project document as context
      - formats a chat prompt
      - calls Bedrock and returns the reply
    """
    chain = _build_doc_chain(client, settings)
    reply_text = await chain.ainvoke({"question": request.message})
    return ChatResponse(reply=reply_text)
