from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.bedrock import generate_reply


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Chat endpoint that delegates to the (simulated) Bedrock service layer.
    """
    return await generate_reply(payload)
