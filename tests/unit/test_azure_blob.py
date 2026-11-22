
import json
from unittest.mock import MagicMock, patch

import pytest

from src.storage.azure_blob import AzureBlobStorage


@pytest.fixture
def storage():
    with patch("src.storage.azure_blob.DefaultAzureCredential") as mock_credential, \
         patch("src.storage.azure_blob.BlobServiceClient") as mock_service:

        credential_instance = mock_credential.return_value
        service_instance = mock_service.return_value

        instance = AzureBlobStorage()

        yield instance, mock_credential, mock_service, credential_instance, service_instance


def test_storage_initializes_with_account_url(storage):
    instance, mock_credential, mock_service, _, _ = storage

    mock_credential.assert_called_once_with()

    mock_service.assert_called_once_with(
        account_url="https://apexf1648219.blob.core.windows.net",
        credential=mock_credential.return_value
    )


def test_storage_raises_error_when_account_name_missing():
    with patch("src.storage.azure_blob.os.getenv", return_value=None):
        with pytest.raises(
            ValueError,
            match="AZURE_STORAGE_ACCOUNT is missing"
        ):
            AzureBlobStorage()


def test_upload_json_success(storage):
    instance, _, _, _, service_instance = storage

    container_client = service_instance.get_container_client.return_value
    blob_client = container_client.get_blob_client.return_value

    records = [
        {"driver_number": 1, "name": "Driver A"},
        {"driver_number": 2, "name": "Driver B"}
    ]

    result = instance.upload_json(
        data=records,
        container_name="bronze",
        blob_name="test/drivers.json"
    )

    service_instance.get_container_client.assert_called_once_with("bronze")

    container_client.get_blob_client.assert_called_once_with(
        "test/drivers.json"
    )

    uploaded_payload = blob_client.upload_blob.call_args.args[0]
    uploaded_data = json.loads(uploaded_payload.decode("utf-8"))

    assert uploaded_data == records
    assert result == blob_client.url

    upload_kwargs = blob_client.upload_blob.call_args.kwargs

    assert upload_kwargs["overwrite"] is True
    assert (
        upload_kwargs["content_settings"].content_type
        == "application/json"
    )


def test_upload_json_handles_datetime_values(storage):
    instance, _, _, _, service_instance = storage

    blob_client = (
        service_instance
        .get_container_client.return_value
        .get_blob_client.return_value
    )

    records = [{"timestamp": "2025-03-16T14:30:00"}]

    instance.upload_json(
        data=records,
        container_name="bronze",
        blob_name="test/session.json"
    )

    uploaded_payload = blob_client.upload_blob.call_args.args[0]
    uploaded_data = json.loads(uploaded_payload.decode("utf-8"))

    assert uploaded_data == records


def test_close_releases_clients(storage):
    instance, _, _, credential_instance, service_instance = storage

    instance.close()

    service_instance.close.assert_called_once()
    credential_instance.close.assert_called_once()
