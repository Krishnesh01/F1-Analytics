
import pandas as pd
import pytest

from src.processing.bronze_to_silver import BronzeToSilver


def test_standardise_keys_converts_column_names():
    df = pd.DataFrame({
        " Session_Key ": [9839],
        " DRIVER_NUMBER ": [1],
        "Lap_Number": [10]
    })

    result = BronzeToSilver.standardise_keys(df)

    assert list(result.columns) == [
        "session_key",
        "driver_number",
        "lap_number"
    ]


def test_standardise_keys_converts_identifiers_to_string():
    df = pd.DataFrame({
        "session_key": [9839],
        "meeting_key": [1276],
        "driver_number": [1]
    })

    result = BronzeToSilver.standardise_keys(df)

    assert str(result["session_key"].dtype) == "string"
    assert str(result["meeting_key"].dtype) == "string"
    assert str(result["driver_number"].dtype) == "string"


def test_standardise_keys_does_not_modify_original_dataframe():
    df = pd.DataFrame({
        " DRIVER_NUMBER ": [1]
    })

    BronzeToSilver.standardise_keys(df)

    assert " DRIVER_NUMBER " in df.columns


def test_standardise_keys_rejects_duplicate_columns():
    df = pd.DataFrame(
        [[9839, 9840]],
        columns=["Session_Key", " session_key "]
    )

    with pytest.raises(
        ValueError,
        match="Duplicate column names"
    ):
        BronzeToSilver.standardise_keys(df)


def test_remove_duplicates_keeps_last_record():
    df = pd.DataFrame({
        "session_key": ["9839", "9839", "9839"],
        "driver_number": ["1", "1", "2"],
        "lap_number": [1, 1, 1],
        "lap_duration": [92.5, 91.8, 93.1]
    })

    result = BronzeToSilver.remove_duplicates(
        df,
        ["session_key", "driver_number", "lap_number"],
        "Laps"
    )

    assert len(result) == 2

    driver_one = result[
        result["driver_number"] == "1"
    ]

    assert driver_one.iloc[0]["lap_duration"] == 91.8


def test_remove_duplicates_rejects_missing_keys():
    df = pd.DataFrame({
        "lap_duration": [92.5, 93.1]
    })

    with pytest.raises(
        ValueError,
        match="No deduplication keys available"
    ):
        BronzeToSilver.remove_duplicates(
            df,
            ["session_key", "driver_number"],
            "Laps"
        )


def test_remove_duplicates_preserves_original_dataframe():
    df = pd.DataFrame({
        "session_key": ["9839", "9839"],
        "driver_number": ["1", "1"],
        "lap_number": [1, 1]
    })

    BronzeToSilver.remove_duplicates(
        df,
        ["session_key", "driver_number", "lap_number"],
        "Laps"
    )

    assert len(df) == 2


def test_remove_duplicates_uses_available_keys():
    df = pd.DataFrame({
        "session_key": ["9839", "9839"],
        "lap_duration": [92.5, 93.1]
    })

    result = BronzeToSilver.remove_duplicates(
        df,
        ["session_key", "driver_number"],
        "Laps"
    )

    assert len(result) == 1
