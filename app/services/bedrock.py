import json
import re
from typing import Any

from botocore.client import BaseClient

from app.core.config import Settings
from app.schemas.chat import ChatRequest, ChatResponse


async def generate_reply(
    request: ChatRequest, client: BaseClient, settings: Settings
) -> ChatResponse:
    """
    Call the GPT-OSS 20B model on Amazon Bedrock and return its reply.

    GPT-OSS models imported as OpenAI-compatible expect an OpenAI-style
    chat completions payload (messages, max_completion_tokens, etc.).
    """
    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": request.message,
            }
        ],
        "max_completion_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.9,
    }

    response = client.invoke_model(
        modelId=settings.bedrock_model_id,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )

    raw_body = response["body"].read()
    data = json.loads(raw_body)

    # OpenAI-style response: choices[0].message.content
    reply_text = "Sorry, I could not generate a response."
    try:
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                text = content

                # Strip out any <reasoning>...</reasoning> blocks the model adds
                text = re.sub(
                    r"<reasoning>.*?</reasoning>",
                    "",
                    text,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()

                reply_text = text or reply_text
    except Exception:
        # Fall back to generic error text if parsing changes
        pass

    return ChatResponse(reply=reply_text)
