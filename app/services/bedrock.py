from app.schemas.chat import ChatRequest, ChatResponse


async def generate_reply(request: ChatRequest) -> ChatResponse:
    """
    Simulated AI model call.

    In a real implementation, this function would:
      - Build an AWS Bedrock client
      - Send the user's message to a chosen model
      - Parse and return the model's response
    """
    reply_text = f'You said: "{request.message}"'
    return ChatResponse(reply=reply_text)
