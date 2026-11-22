
import os
import uuid

import pytest

from dotenv import load_dotenv

load_dotenv()
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


pytestmark = pytest.mark.integration


@pytest.fixture
def azure_blob_client():
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_BRONZE")

    if not account_name or not container_name:
        pytest.skip("Azure storage configuration is missing.")

    credential = DefaultAzureCredential()

    service_client = BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=credential,
    )

    try:
        yield service_client, container_name
    finally:
        service_client.close()
        credential.close()


def test_azure_storage_can_upload_download_and_delete(azure_blob_client):
    service_client, container_name = azure_blob_client

    container_client = service_client.get_container_client(container_name)

    blob_name = f"integration-tests/connectivity-{uuid.uuid4().hex}.txt"
    expected_content = b"ApexF1 Azure integration test"

    blob_client = container_client.get_blob_client(blob_name)

    try:
        blob_client.upload_blob(expected_content, overwrite=False)

        downloaded_content = blob_client.download_blob().readall()

        assert downloaded_content == expected_content

    finally:
        if blob_client.exists():
            blob_client.delete_blob()

    assert not blob_client.exists()
