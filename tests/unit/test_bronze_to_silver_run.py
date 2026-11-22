
import pytest

from src.processing.bronze_to_silver import BronzeToSilver


@pytest.fixture
def pipeline():
    instance = BronzeToSilver.__new__(BronzeToSilver)
    return instance


def test_run_processes_all_sessions(pipeline, monkeypatch):
    sessions = [
        {"meeting_key": "1276", "session_key": "9839"},
        {"meeting_key": "1277", "session_key": "9840"}
    ]

    monkeypatch.setattr(
        pipeline,
        "discover_sessions",
        lambda: sessions
    )

    processed = []

    def mock_process(meeting_key, session_key):
        processed.append((meeting_key, session_key))

        return {
            "meeting_key": meeting_key,
            "session_key": session_key,
            "drivers": 20,
            "laps": 1000,
            "status": "SUCCESS"
        }

    monkeypatch.setattr(
        pipeline,
        "process_session",
        mock_process
    )

    pipeline.run()

    assert processed == [
        ("1276", "9839"),
        ("1277", "9840")
    ]


def test_run_rejects_empty_season(pipeline, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "discover_sessions",
        lambda: []
    )

    with pytest.raises(
        ValueError,
        match="No Bronze race sessions found"
    ):
        pipeline.run()


def test_run_raises_error_when_session_fails(pipeline, monkeypatch):
    sessions = [
        {"meeting_key": "1276", "session_key": "9839"},
        {"meeting_key": "1277", "session_key": "9840"}
    ]

    monkeypatch.setattr(
        pipeline,
        "discover_sessions",
        lambda: sessions
    )

    def mock_process(meeting_key, session_key):
        if session_key == "9840":
            raise ValueError("Simulated transformation failure")

        return {
            "meeting_key": meeting_key,
            "session_key": session_key,
            "drivers": 20,
            "laps": 1000,
            "status": "SUCCESS"
        }

    monkeypatch.setattr(
        pipeline,
        "process_session",
        mock_process
    )

    with pytest.raises(
        RuntimeError,
        match="1 failed sessions"
    ):
        pipeline.run()


def test_run_continues_after_session_failure(pipeline, monkeypatch):
    sessions = [
        {"meeting_key": "1276", "session_key": "9839"},
        {"meeting_key": "1277", "session_key": "9840"},
        {"meeting_key": "1278", "session_key": "9841"}
    ]

    monkeypatch.setattr(
        pipeline,
        "discover_sessions",
        lambda: sessions
    )

    processed = []

    def mock_process(meeting_key, session_key):
        processed.append(session_key)

        if session_key == "9840":
            raise ValueError("Simulated failure")

        return {
            "meeting_key": meeting_key,
            "session_key": session_key,
            "drivers": 20,
            "laps": 1000,
            "status": "SUCCESS"
        }

    monkeypatch.setattr(
        pipeline,
        "process_session",
        mock_process
    )

    with pytest.raises(RuntimeError):
        pipeline.run()

    assert processed == ["9839", "9840", "9841"]
