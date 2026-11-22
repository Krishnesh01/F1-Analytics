
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)


class AzureBlobStorage:
    """Upload ApexF1 datasets to Azure Blob Storage using Entra ID."""

    def __init__(self):
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT")

        if not account_name:
            raise ValueError("AZURE_STORAGE_ACCOUNT is missing from .env")

        account_url = f"https://{account_name}.blob.core.windows.net"

        self.credential = DefaultAzureCredential()

        self.service_client = BlobServiceClient(
            account_url=account_url,
            credential=self.credential
        )

    def upload_json(
        self,
        data: list[dict],
        container_name: str,
        blob_name: str
    ) -> str:
        """Upload JSON records to an Azure container."""

        container_client = self.service_client.get_container_client(
            container_name
        )

        blob_client = container_client.get_blob_client(blob_name)

        payload = json.dumps(
            data,
            indent=2,
            default=str
        ).encode("utf-8")

        blob_client.upload_blob(
            payload,
            overwrite=True,
            content_settings=ContentSettings(
                content_type="application/json"
            )
        )

        logger.info(
            "Uploaded %s records to %s/%s",
            len(data),
            container_name,
            blob_name
        )

        return blob_client.url

    def close(self):
        self.service_client.close()
        self.credential.close()
