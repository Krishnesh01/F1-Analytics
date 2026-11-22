
import io
import json
import uuid

import pandas as pd
import pytest
from dotenv import load_dotenv

from src.processing.bronze_to_silver import BronzeToSilver


load_dotenv()

pytestmark = pytest.mark.integration


def test_bronze_to_silver_processes_real_azure_session():
    pipeline = BronzeToSilver()

    # Generate unique numeric identifiers for an isolated test session.
    unique_id = int(uuid.uuid4().hex[:10], 16)
    meeting_key = str(900000000 + unique_id)
    session_key = str(910000000 + unique_id)

    base_path = (
        f"openf1/year={pipeline.__class__.__module__ and 2025}/"
        f"meeting_key={meeting_key}/session_key={session_key}"
    )

    bronze_blobs = [
        f"{base_path}/session.json",
        f"{base_path}/drivers.json",
        f"{base_path}/laps.json",
    ]

    silver_blobs = [
        f"{base_path}/session.parquet",
        f"{base_path}/drivers.parquet",
        f"{base_path}/laps_enriched.parquet",
    ]

    session_records = [
        {
            "meeting_key": meeting_key,
            "session_key": session_key,
            "session_name": "Integration Test Race",
        }
    ]

    driver_records = [
        {
            "meeting_key": meeting_key,
            "session_key": session_key,
            "driver_number": "1",
            "full_name": "Test Driver One",
            "name_acronym": "TST",
            "team_name": "ApexF1 Test Team",
            "team_colour": "FFFFFF",
            "country_code": "IND",
        },
        {
            "meeting_key": meeting_key,
            "session_key": session_key,
            "driver_number": "2",
            "full_name": "Test Driver Two",
            "name_acronym": "TWO",
            "team_name": "ApexF1 Test Team",
            "team_colour": "FFFFFF",
            "country_code": "IND",
        },
    ]

    lap_records = [
        {
            "meeting_key": meeting_key,
            "session_key": session_key,
            "driver_number": "1",
            "lap_number": 1,
            "lap_duration": 90.5,
        },
        {
            "meeting_key": meeting_key,
            "session_key": session_key,
            "driver_number": "1",
            "lap_number": 2,
            "lap_duration": 91.2,
        },
        {
            "meeting_key": meeting_key,
            "session_key": session_key,
            "driver_number": "2",
            "lap_number": 1,
            "lap_duration": 92.1,
        },
    ]

    datasets = [
        session_records,
        driver_records,
        lap_records,
    ]

    try:
        # 1. Upload isolated Bronze JSON test datasets.
        for blob_name, records in zip(bronze_blobs, datasets):
            pipeline.bronze_client.upload_blob(
                name=blob_name,
                data=json.dumps(records).encode("utf-8"),
                overwrite=False,
            )

        # 2. Run the actual production transformation.
        result = pipeline.process_session(
            meeting_key,
            session_key,
        )

        assert result["status"] == "SUCCESS"
        assert result["meeting_key"] == meeting_key
        assert result["session_key"] == session_key
        assert result["drivers"] == 2
        assert result["laps"] == 3

        # 3. Download and inspect the generated Silver Parquet files.
        expected_rows = {
            silver_blobs[0]: 1,
            silver_blobs[1]: 2,
            silver_blobs[2]: 3,
        }

        for blob_name, row_count in expected_rows.items():
            blob = pipeline.silver_client.get_blob_client(blob_name)

            assert blob.exists(), f"Missing Silver output: {blob_name}"

            content = blob.download_blob().readall()
            dataframe = pd.read_parquet(io.BytesIO(content))

            assert len(dataframe) == row_count
            assert "processed_at_utc" in dataframe.columns
            assert dataframe["session_key"].astype(str).eq(
                session_key
            ).all()

        # 4. Verify driver enrichment in the lap dataset.
        lap_blob = pipeline.silver_client.get_blob_client(
            silver_blobs[2]
        )

        lap_data = pd.read_parquet(
            io.BytesIO(lap_blob.download_blob().readall())
        )

        assert "full_name" in lap_data.columns
        assert "team_name" in lap_data.columns
        assert lap_data["full_name"].notna().all()
        assert set(lap_data["full_name"]) == {
            "Test Driver One",
            "Test Driver Two",
        }

    finally:
        # 5. Delete only the uniquely named integration test blobs.
        for blob_name in bronze_blobs:
            blob = pipeline.bronze_client.get_blob_client(blob_name)
            if blob.exists():
                blob.delete_blob()

        for blob_name in silver_blobs:
            blob = pipeline.silver_client.get_blob_client(blob_name)
            if blob.exists():
                blob.delete_blob()

        pipeline.close()
