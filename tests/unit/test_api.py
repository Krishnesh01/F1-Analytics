
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main


client = TestClient(main.app)


@pytest.fixture
def race_drivers():
    return pd.DataFrame([
        {
            "driver_number": 16,
            "name_acronym": "LEC",
            "team_name": "Ferrari",
            "fastest_lap_seconds": 86.7,
        },
        {
            "driver_number": 81,
            "name_acronym": "PIA",
            "team_name": "McLaren",
            "fastest_lap_seconds": 86.9,
        },
    ])


@pytest.fixture
def season_teams():
    return pd.DataFrame([
        {
            "team_name": "McLaren",
            "normalized_pace_rank": 1,
            "average_race_relative_pace_pct": -1.398,
            "median_race_relative_pace_pct": -1.374,
            "races_with_pace": 24,
        },
        {
            "team_name": "Ferrari",
            "normalized_pace_rank": 2,
            "average_race_relative_pace_pct": -0.630,
            "median_race_relative_pace_pct": -0.604,
            "races_with_pace": 20,
        },
    ])


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["application"] == "ApexF1"


def test_race_summary_success(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_dataset",
        lambda name: pd.DataFrame([
            {"driver_count": 20, "valid_lap_time_records": 1129}
        ]),
    )

    response = client.get("/api/v1/race/summary")

    assert response.status_code == 200
    assert response.json()["driver_count"] == 20


def test_race_summary_not_found(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_dataset",
        lambda name: pd.DataFrame(),
    )

    response = client.get("/api/v1/race/summary")

    assert response.status_code == 404


def test_get_drivers_sorted_by_fastest_lap(
    monkeypatch, race_drivers
):
    monkeypatch.setattr(
        main,
        "get_dataset",
        lambda name: race_drivers,
    )

    response = client.get("/api/v1/drivers?limit=2")

    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert response.json()["data"][0]["name_acronym"] == "LEC"


def test_get_drivers_respects_limit(
    monkeypatch, race_drivers
):
    monkeypatch.setattr(
        main,
        "get_dataset",
        lambda name: race_drivers,
    )

    response = client.get("/api/v1/drivers?limit=1")

    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_get_drivers_rejects_invalid_limit():
    response = client.get("/api/v1/drivers?limit=101")

    assert response.status_code == 422


def test_get_driver_success(monkeypatch, race_drivers):
    monkeypatch.setattr(
        main,
        "get_dataset",
        lambda name: race_drivers,
    )

    response = client.get("/api/v1/drivers/16")

    assert response.status_code == 200
    assert response.json()["name_acronym"] == "LEC"


def test_get_driver_not_found(monkeypatch, race_drivers):
    monkeypatch.setattr(
        main,
        "get_dataset",
        lambda name: race_drivers,
    )

    response = client.get("/api/v1/drivers/99")

    assert response.status_code == 404


def test_get_teams(monkeypatch):
    teams = pd.DataFrame([
        {"team_name": "Ferrari"},
        {"team_name": "McLaren"},
    ])

    monkeypatch.setattr(
        main,
        "get_dataset",
        lambda name: teams,
    )

    response = client.get("/api/v1/teams")

    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_get_laps_sorted(monkeypatch):
    laps = pd.DataFrame([
        {"lap_number": 2, "average_lap_seconds": 90.0},
        {"lap_number": 1, "average_lap_seconds": 89.0},
    ])

    monkeypatch.setattr(
        main,
        "get_dataset",
        lambda name: laps,
    )

    response = client.get("/api/v1/laps")

    assert response.status_code == 200
    assert response.json()["data"][0]["lap_number"] == 1


def test_season_teams_normalized_ranking(
    monkeypatch, season_teams
):
    monkeypatch.setattr(
        main,
        "get_season_dataset",
        lambda name: season_teams,
    )

    response = client.get("/api/v1/season/teams")

    assert response.status_code == 200

    result = response.json()

    assert result["count"] == 2
    assert result["metric"] == "average_race_relative_pace_pct"
    assert result["data"][0]["team_name"] == "McLaren"


def test_season_teams_minimum_race_filter(
    monkeypatch, season_teams
):
    monkeypatch.setattr(
        main,
        "get_season_dataset",
        lambda name: season_teams,
    )

    response = client.get(
        "/api/v1/season/teams?min_races=21"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["count"] == 1
    assert result["data"][0]["team_name"] == "McLaren"


def test_season_teams_rejects_invalid_min_races():
    response = client.get(
        "/api/v1/season/teams?min_races=0"
    )

    assert response.status_code == 422


def test_season_teams_missing_normalized_columns(
    monkeypatch
):
    invalid_data = pd.DataFrame([
        {"team_name": "McLaren"}
    ])

    monkeypatch.setattr(
        main,
        "get_season_dataset",
        lambda name: invalid_data,
    )

    response = client.get("/api/v1/season/teams")

    assert response.status_code == 500


def test_season_dataset_failure(monkeypatch):
    def failing_dataset(name):
        raise RuntimeError("Storage unavailable")

    monkeypatch.setattr(
        main,
        "load_season_dataset",
        failing_dataset,
    )

    response = client.get("/api/v1/season/summary")

    assert response.status_code == 503


def test_session_metadata():
    response = client.get("/api/v1/session")

    assert response.status_code == 200

    result = response.json()

    assert result["data_source"] == "OpenF1"
    assert result["storage_layer"] == "Azure Blob Storage - Gold"
