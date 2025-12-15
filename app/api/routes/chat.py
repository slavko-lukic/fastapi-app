from botocore.client import BaseClient
from fastapi import APIRouter, Depends

from app.core.config import Settings
from app.dependencies.bedrock import get_bedrock_runtime
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.bedrock import generate_reply
from app.services.langchain_rag import answer_with_doc


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    bedrock_runtime: tuple[BaseClient, Settings] = Depends(get_bedrock_runtime),
) -> ChatResponse:
    """
    Basic chat endpoint that delegates directly to the Bedrock service layer via DI.
    """
    client, settings = bedrock_runtime
    return await generate_reply(payload, client, settings)


@router.post("/with-doc", response_model=ChatResponse)
async def chat_with_doc_endpoint(
    payload: ChatRequest,
    bedrock_runtime: tuple[BaseClient, Settings] = Depends(get_bedrock_runtime),
) -> ChatResponse:
    """
    Chat endpoint that uses LangChain to inject a local document as context
    before calling the Bedrock model.
    """
    client, settings = bedrock_runtime
    return await answer_with_doc(payload, client, settings)
