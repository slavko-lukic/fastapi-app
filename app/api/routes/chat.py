from botocore.client import BaseClient
from fastapi import APIRouter, Depends

from app.core.config import Settings
from app.dependencies.bedrock import get_bedrock_runtime
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.bedrock import generate_reply


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    bedrock_runtime: tuple[BaseClient, Settings] = Depends(get_bedrock_runtime),
) -> ChatResponse:
    """
    Chat endpoint that delegates to the Bedrock service layer via DI.
    """
    client, settings = bedrock_runtime
    return await generate_reply(payload, client, settings)
