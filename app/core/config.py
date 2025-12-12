from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

# Load variables from .env into process environment once at startup
load_dotenv()


@dataclass
class Settings:
    aws_region: str
    bedrock_model_id: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Application settings loaded from environment variables.

    Required:
      - BEDROCK_MODEL_ID

    Optional:
      - AWS_REGION (default: "us-east-1")
    """
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    bedrock_model_id = os.environ.get("BEDROCK_MODEL_ID")

    if not bedrock_model_id:
        raise RuntimeError(
            "BEDROCK_MODEL_ID environment variable must be set "
            "to use the Bedrock client."
        )

    return Settings(aws_region=aws_region, bedrock_model_id=bedrock_model_id)
