
import pandas as pd
import pytest

from src.processing.silver_to_gold import SilverToGold


@pytest.fixture
def sample_laps():
    return pd.DataFrame({
        "meeting_key": ["100", "100", "100", "100", "200", "200"],
        "session_key": ["1001", "1001", "1001", "1001", "2001", "2001"],
        "driver_number": ["1", "1", "2", "2", "1", "2"],
        "lap_number": [1, 2, 1, 2, 1, 1],
        "lap_duration": [90.0, 91.0, 92.0, 93.0, 100.0, 102.0],
        "team_name": ["Team A", "Team A", "Team B", "Team B", "Team A", "Team B"],
        "full_name": ["Driver One", "Driver One", "Driver Two", "Driver Two", "Driver One", "Driver Two"],
        "name_acronym": ["ONE", "ONE", "TWO", "TWO", "ONE", "TWO"],
        "country_code": ["AAA", "AAA", "BBB", "BBB", "AAA", "BBB"],
        "is_pit_out_lap": [False, False, False, False, False, False],
    })


# ---------------------------------------------------------
# LAP PREPARATION
# ---------------------------------------------------------

def test_prepare_laps_removes_invalid_records(sample_laps):
    data = pd.concat([
        sample_laps,
        pd.DataFrame({
            "meeting_key": ["100", "100", "100"],
            "session_key": ["1001", "1001", "1001"],
            "driver_number": ["3", "3", "3"],
            "lap_number": [0, 3, 4],
            "lap_duration": [95, -1, None],
            "team_name": ["Team C"] * 3,
            "is_pit_out_lap": [False] * 3,
        }),
    ], ignore_index=True)

    result = SilverToGold.prepare_laps(data)

    assert len(result) == len(sample_laps)
    assert (result["lap_number"] > 0).all()
    assert (result["lap_duration"] > 0).all()


def test_prepare_laps_removes_pit_out_laps(sample_laps):
    sample_laps.loc[0, "is_pit_out_lap"] = True

    result = SilverToGold.prepare_laps(sample_laps)

    assert len(result) == len(sample_laps) - 1


def test_prepare_laps_requires_required_columns():
    data = pd.DataFrame({
        "driver_number": ["1"],
        "lap_number": [1],
    })

    with pytest.raises(ValueError, match="Missing lap columns"):
        SilverToGold.prepare_laps(data)


def test_prepare_laps_converts_numeric_fields():
    data = pd.DataFrame({
        "driver_number": [1, 2],
        "lap_number": ["1", "2"],
        "lap_duration": ["90.5", "91.5"],
    })

    result = SilverToGold.prepare_laps(data)

    assert str(result["driver_number"].dtype) == "string"
    assert pd.api.types.is_numeric_dtype(result["lap_number"])
    assert pd.api.types.is_numeric_dtype(result["lap_duration"])


# ---------------------------------------------------------
# RACE-LEVEL ANALYTICS
# ---------------------------------------------------------

def test_build_driver_performance(sample_laps):
    result = SilverToGold.build_driver_performance(sample_laps)

    assert len(result) == 2
    assert "pace_rank" in result.columns
    assert result.iloc[0]["driver_number"] == "1"
    assert result.iloc[0]["fastest_lap_seconds"] == 90.0
    assert result.iloc[0]["valid_lap_count"] == 3


def test_build_driver_performance_rejects_empty_valid_laps():
    data = pd.DataFrame({
        "driver_number": ["1"],
        "lap_number": [0],
        "lap_duration": [90.0],
    })

    with pytest.raises(ValueError, match="No valid laps"):
        SilverToGold.build_driver_performance(data)


def test_build_team_performance(sample_laps):
    result = SilverToGold.build_team_performance(sample_laps)

    assert len(result) == 2
    assert "team_name" in result.columns
    assert result.iloc[0]["team_name"] == "Team A"
    assert result.iloc[0]["valid_lap_count"] == 3


def test_build_team_performance_requires_team_name(sample_laps):
    data = sample_laps.drop(columns=["team_name"])

    with pytest.raises(ValueError, match="missing team_name"):
        SilverToGold.build_team_performance(data)


def test_build_lap_analysis(sample_laps):
    result = SilverToGold.build_lap_analysis(sample_laps)

    assert len(result) == 2
    assert result["lap_number"].tolist() == [1, 2]
    assert result.iloc[0]["recorded_laps"] == 4
    assert result.iloc[0]["fastest_lap_seconds"] == 90.0


def test_build_lap_analysis_rejects_empty_valid_laps():
    data = pd.DataFrame({
        "driver_number": ["1"],
        "lap_number": [0],
        "lap_duration": [90.0],
    })

    with pytest.raises(ValueError, match="No valid lap data"):
        SilverToGold.build_lap_analysis(data)


# ---------------------------------------------------------
# RACE AND SEASON SUMMARIES
# ---------------------------------------------------------

def test_build_race_summary(sample_laps):
    session = pd.DataFrame([{
        "session_name": "Race",
        "date_start": "2025-03-01",
        "date_end": "2025-03-01",
    }])

    drivers = pd.DataFrame({
        "driver_number": ["1", "2"],
    })

    result = SilverToGold.build_race_summary(
        session, drivers, sample_laps, "100", "1001"
    )

    assert len(result) == 1
    assert result.iloc[0]["meeting_key"] == "100"
    assert result.iloc[0]["session_key"] == "1001"
    assert result.iloc[0]["driver_count"] == 2
    assert result.iloc[0]["valid_lap_time_records"] == 6
    assert result.iloc[0]["fastest_lap_seconds"] == 90.0


def test_build_season_driver_performance(sample_laps):
    result = SilverToGold.build_season_driver_performance(sample_laps)

    assert len(result) == 2
    assert "normalized_pace_rank" in result.columns
    assert "average_race_relative_pace_pct" in result.columns
    assert result["normalized_pace_rank"].tolist() == [1, 2]
    assert result.iloc[0]["driver_number"] == "1"
    assert result["races_with_pace"].tolist() == [2, 2]


def test_build_season_team_performance(sample_laps):
    result = SilverToGold.build_season_team_performance(sample_laps)

    assert len(result) == 2
    assert "normalized_pace_rank" in result.columns
    assert "average_race_relative_pace_pct" in result.columns
    assert result["normalized_pace_rank"].tolist() == [1, 2]
    assert result.iloc[0]["team_name"] == "Team A"
    assert result["races_with_pace"].tolist() == [2, 2]


def test_build_season_summary(sample_laps):
    drivers = pd.DataFrame({"driver_number": ["1", "2"]})
    teams = pd.DataFrame({"team_name": ["Team A", "Team B"]})
    sessions = [{"meeting_key": "100", "session_key": "1001"}]

    result = SilverToGold.build_season_summary(
        sessions, sample_laps, drivers, teams
    )

    assert len(result) == 1
    assert result.iloc[0]["race_count"] == 2
    assert result.iloc[0]["meeting_count"] == 2
    assert result.iloc[0]["driver_count"] == 2
    assert result.iloc[0]["team_count"] == 2
    assert result.iloc[0]["valid_lap_count"] == 6


# ---------------------------------------------------------
# OUTPUT VALIDATION
# ---------------------------------------------------------

def test_validate_outputs_accepts_valid_dataframes():
    outputs = {
        "drivers": pd.DataFrame({"driver_number": ["1"]}),
        "teams": pd.DataFrame({"team_name": ["Team A"]}),
    }

    SilverToGold.validate_outputs(outputs)


def test_validate_outputs_rejects_empty_dataframe():
    outputs = {
        "drivers": pd.DataFrame(),
    }

    with pytest.raises(ValueError, match="Gold output is empty"):
        SilverToGold.validate_outputs(outputs)


def test_validate_outputs_rejects_duplicate_columns():
    data = pd.DataFrame([[1, 2]], columns=["driver_number", "driver_number"])

    with pytest.raises(ValueError, match="Duplicate columns"):
        SilverToGold.validate_outputs({"drivers": data})


def test_round_metrics_rounds_to_three_decimal_places():
    data = pd.DataFrame({
        "lap_seconds": [90.123456],
    })

    result = SilverToGold.round_metrics(data, ["lap_seconds"])

    assert result.iloc[0]["lap_seconds"] == 90.123
