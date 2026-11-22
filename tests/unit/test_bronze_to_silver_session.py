
import pandas as pd
import pytest

from src.processing.bronze_to_silver import BronzeToSilver


@pytest.fixture
def pipeline(monkeypatch):
    """Create a pipeline without initializing Azure clients."""

    instance = BronzeToSilver.__new__(BronzeToSilver)

    datasets = {
        "session.json": [
            {
                "session_key": 9839,
                "meeting_key": 1276,
                "session_name": "Race"
            }
        ],
        "drivers.json": [
            {
                "session_key": 9839,
                "driver_number": 1,
                "full_name": "Driver One",
                "name_acronym": "ONE",
                "team_name": "Team A",
                "team_colour": "FF0000"
            },
            {
                "session_key": 9839,
                "driver_number": 2,
                "full_name": "Driver Two",
                "name_acronym": "TWO",
                "team_name": "Team B",
                "team_colour": "0000FF"
            }
        ],
        "laps.json": [
            {
                "session_key": 9839,
                "driver_number": 1,
                "lap_number": "1",
                "lap_duration": "92.5",
                "duration_sector_1": "30.1"
            },
            {
                "session_key": 9839,
                "driver_number": 1,
                "lap_number": "2",
                "lap_duration": "91.8",
                "duration_sector_1": "29.9"
            },
            {
                "session_key": 9839,
                "driver_number": 2,
                "lap_number": "1",
                "lap_duration": "93.2",
                "duration_sector_1": "30.4"
            }
        ]
    }

    uploaded = []

    def mock_download_json(blob_name):
        filename = blob_name.split("/")[-1]
        return datasets[filename]

    def mock_upload_parquet(df, blob_name):
        uploaded.append((df.copy(), blob_name))

    monkeypatch.setattr(
        instance,
        "download_json",
        mock_download_json
    )

    monkeypatch.setattr(
        instance,
        "upload_parquet",
        mock_upload_parquet
    )

    instance._test_datasets = datasets
    instance._test_uploaded = uploaded

    return instance


def test_process_session_success(pipeline):
    result = pipeline.process_session("1276", "9839")

    assert result == {
        "session_key": "9839",
        "meeting_key": "1276",
        "drivers": 2,
        "laps": 3,
        "status": "SUCCESS"
    }

    assert len(pipeline._test_uploaded) == 3


def test_process_session_uploads_correct_silver_paths(pipeline):
    pipeline.process_session("1276", "9839")

    uploaded_paths = [
        path for _, path in pipeline._test_uploaded
    ]

    assert uploaded_paths == [
        "openf1/year=2025/meeting_key=1276/session_key=9839/session.parquet",
        "openf1/year=2025/meeting_key=1276/session_key=9839/drivers.parquet",
        "openf1/year=2025/meeting_key=1276/session_key=9839/laps_enriched.parquet"
    ]


def test_process_session_enriches_laps_with_driver_details(pipeline):
    pipeline.process_session("1276", "9839")

    enriched_laps = pipeline._test_uploaded[2][0]

    driver_one = enriched_laps[
        enriched_laps["driver_number"] == "1"
    ]

    assert len(driver_one) == 2
    assert set(driver_one["full_name"]) == {"Driver One"}
    assert set(driver_one["team_name"]) == {"Team A"}


def test_process_session_converts_lap_fields_to_numeric(pipeline):
    pipeline.process_session("1276", "9839")

    enriched_laps = pipeline._test_uploaded[2][0]

    assert pd.api.types.is_numeric_dtype(
        enriched_laps["lap_number"]
    )

    assert pd.api.types.is_numeric_dtype(
        enriched_laps["lap_duration"]
    )

    assert enriched_laps["lap_duration"].tolist() == [
        92.5,
        91.8,
        93.2
    ]


def test_process_session_adds_processing_timestamp(pipeline):
    pipeline.process_session("1276", "9839")

    for df, _ in pipeline._test_uploaded:
        assert "processed_at_utc" in df.columns
        assert df["processed_at_utc"].notna().all()


def test_process_session_adds_partition_identifiers(pipeline):
    pipeline.process_session("1276", "9839")

    for df, _ in pipeline._test_uploaded:
        assert set(df["meeting_key"]) == {"1276"}
        assert set(df["session_key"]) == {"9839"}


@pytest.mark.parametrize(
    ("dataset_name", "expected_error"),
    [
        ("session.json", "Session dataset is empty"),
        ("drivers.json", "Driver dataset is empty"),
        ("laps.json", "Lap dataset is empty")
    ]
)
def test_process_session_rejects_empty_datasets(
    pipeline,
    dataset_name,
    expected_error
):
    pipeline._test_datasets[dataset_name] = []

    with pytest.raises(ValueError, match=expected_error):
        pipeline.process_session("1276", "9839")

    assert pipeline._test_uploaded == []


def test_process_session_rejects_missing_lap_columns(pipeline):
    pipeline._test_datasets["laps.json"] = [
        {
            "session_key": 9839,
            "driver_number": 1
        }
    ]

    with pytest.raises(
        ValueError,
        match="Lap dataset missing columns"
    ):
        pipeline.process_session("1276", "9839")

    assert pipeline._test_uploaded == []


def test_process_session_rejects_missing_driver_columns(pipeline):
    pipeline._test_datasets["drivers.json"] = [
        {
            "session_key": 9839
        }
    ]

    with pytest.raises(
        ValueError,
        match="Driver dataset missing columns"
    ):
        pipeline.process_session("1276", "9839")

    assert pipeline._test_uploaded == []
