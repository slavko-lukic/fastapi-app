import boto3
from botocore.client import BaseClient
from fastapi import Depends

from app.core.config import Settings, get_settings


def get_bedrock_client(settings: Settings = Depends(get_settings)) -> BaseClient:
    """
    Dependency that provides a configured AWS Bedrock Runtime client.
    """

    return boto3.client("bedrock-runtime", region_name=settings.aws_region)


def get_bedrock_runtime(
    client: BaseClient = Depends(get_bedrock_client),
    settings: Settings = Depends(get_settings),
) -> tuple[BaseClient, Settings]:
    """
    Convenience dependency that returns both the client and settings.
    """

    runtime: tuple[BaseClient, Settings] = (client, settings)
    return runtime
