
import io
import uuid

import pandas as pd
import pytest
from azure.storage.blob import ContentSettings
from dotenv import load_dotenv

load_dotenv()

from src.processing.silver_to_gold import SilverToGold


def upload_test_parquet(container_client, blob_name, dataframe):
    """Upload a test DataFrame as a Parquet blob."""

    buffer = io.BytesIO()
    dataframe.to_parquet(
        buffer,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )
    buffer.seek(0)

    container_client.get_blob_client(blob_name).upload_blob(
        buffer.getvalue(),
        overwrite=True,
        content_settings=ContentSettings(
            content_type="application/vnd.apache.parquet"
        ),
    )


@pytest.mark.integration
def test_silver_to_gold_transforms_azure_data():
    pipeline = SilverToGold()

    run_id = uuid.uuid4().hex
    created_silver_blobs = []
    created_gold_blobs = []

    try:
        all_laps = []
        race_summaries = []
        session_frames = []

        # Create two isolated synthetic race sessions.
        for race_number in range(2):
            meeting_key = str(900000 + race_number)
            session_key = str(910000 + race_number)

            base_path = (
                f"integration-tests/{run_id}/silver/"
                f"meeting_key={meeting_key}/"
                f"session_key={session_key}"
            )

            session_df = pd.DataFrame([{
                "session_key": session_key,
                "meeting_key": meeting_key,
                "session_name": "Race",
                "date_start": "2025-01-01T14:00:00",
                "date_end": "2025-01-01T16:00:00",
            }])

            drivers_df = pd.DataFrame([
                {
                    "driver_number": str(number),
                    "full_name": f"Test Driver {number}",
                }
                for number in range(1, 5)
            ])

            lap_rows = []

            # Two teams, four drivers, three laps per driver.
            for driver_number in range(1, 5):
                team_name = (
                    "Test Racing A"
                    if driver_number <= 2
                    else "Test Racing B"
                )

                for lap_number in range(1, 4):
                    lap_rows.append({
                        "driver_number": str(driver_number),
                        "lap_number": lap_number,
                        "lap_duration": (
                            90.0
                            + driver_number * 0.4
                            + race_number * 2
                            + lap_number * 0.1
                        ),
                        "team_name": team_name,
                        "full_name": f"Test Driver {driver_number}",
                        "name_acronym": f"T{driver_number}",
                        "country_code": "TST",
                        "is_pit_out_lap": False,
                    })

            laps_df = pd.DataFrame(lap_rows)

            # Upload synthetic Silver datasets to Azure.
            silver_datasets = {
                "session.parquet": session_df,
                "drivers.parquet": drivers_df,
                "laps_enriched.parquet": laps_df,
            }

            for filename, dataframe in silver_datasets.items():
                blob_name = f"{base_path}/{filename}"

                upload_test_parquet(
                    pipeline.silver_client,
                    blob_name,
                    dataframe,
                )

                created_silver_blobs.append(blob_name)

            # Read the datasets back from Azure.
            session = pipeline.read_parquet(
                pipeline.silver_client,
                f"{base_path}/session.parquet",
            )

            drivers = pipeline.read_parquet(
                pipeline.silver_client,
                f"{base_path}/drivers.parquet",
            )

            laps = pipeline.read_parquet(
                pipeline.silver_client,
                f"{base_path}/laps_enriched.parquet",
            )

            # Attach race partition identifiers.
            for dataframe in (session, drivers, laps):
                dataframe["meeting_key"] = meeting_key
                dataframe["session_key"] = session_key

            # Build race-level Gold analytics.
            driver_performance = (
                pipeline.build_driver_performance(laps)
            )

            team_performance = (
                pipeline.build_team_performance(laps)
            )

            lap_analysis = (
                pipeline.build_lap_analysis(laps)
            )

            race_summary = pipeline.build_race_summary(
                session,
                driver_performance,
                laps,
                meeting_key,
                session_key,
            )

            for dataframe in (
                driver_performance,
                team_performance,
                lap_analysis,
            ):
                dataframe.insert(0, "meeting_key", meeting_key)
                dataframe.insert(1, "session_key", session_key)

            race_outputs = {
                "driver_performance": driver_performance,
                "team_performance": team_performance,
                "lap_analysis": lap_analysis,
                "race_summary": race_summary,
            }

            pipeline.validate_outputs(race_outputs)

            # Upload race-level outputs to an isolated Gold folder.
            gold_race_path = (
                f"integration-tests/{run_id}/gold/"
                f"race_{race_number}"
            )

            for name, dataframe in race_outputs.items():
                blob_name = f"{gold_race_path}/{name}.parquet"

                pipeline.upload_parquet(dataframe, blob_name)
                created_gold_blobs.append(blob_name)

            all_laps.append(laps)
            race_summaries.append(race_summary)
            session_frames.append(session)

        # Build season-level analytics from both test sessions.
        season_laps = pd.concat(all_laps, ignore_index=True)

        season_driver_performance = (
            pipeline.build_season_driver_performance(season_laps)
        )

        season_team_performance = (
            pipeline.build_season_team_performance(season_laps)
        )

        race_summary_df = pd.concat(
            race_summaries,
            ignore_index=True,
        )

        season_summary = pipeline.build_season_summary(
            session_frames,
            season_laps,
            season_driver_performance,
            season_team_performance,
        )

        season_outputs = {
            "season_driver_performance": season_driver_performance,
            "season_team_performance": season_team_performance,
            "race_summary": race_summary_df,
            "season_summary": season_summary,
        }

        pipeline.validate_outputs(season_outputs)

        # Verify normalized driver and team pace calculations.
        assert len(season_driver_performance) == 4
        assert len(season_team_performance) == 2

        assert (
            season_driver_performance["races_with_pace"] == 2
        ).all()

        assert (
            season_team_performance["races_with_pace"] == 2
        ).all()

        assert (
            season_driver_performance[
                "average_race_relative_pace_pct"
            ].notna().all()
        )

        assert (
            season_team_performance[
                "average_race_relative_pace_pct"
            ].notna().all()
        )

        assert (
            season_driver_performance["normalized_pace_rank"]
            .is_unique
        )

        assert (
            season_team_performance["normalized_pace_rank"]
            .is_unique
        )

        # Upload season-level outputs to the isolated Gold folder.
        gold_season_path = (
            f"integration-tests/{run_id}/gold/season"
        )

        for name, dataframe in season_outputs.items():
            blob_name = f"{gold_season_path}/{name}.parquet"

            pipeline.upload_parquet(dataframe, blob_name)
            created_gold_blobs.append(blob_name)

        # Read Gold outputs back from Azure and verify them.
        for blob_name in created_gold_blobs:
            if "/gold/" not in blob_name:
                continue

            result = pipeline.read_parquet(
                pipeline.gold_client,
                blob_name,
            )

            assert not result.empty

        assert len(race_summary_df) == 2
        assert int(season_summary.iloc[0]["race_count"]) == 2
        assert int(season_summary.iloc[0]["driver_count"]) == 4
        assert int(season_summary.iloc[0]["team_count"]) == 2

    finally:
        # Delete only blobs created by this test run.
        for blob_name in created_gold_blobs:
            try:
                pipeline.gold_client.get_blob_client(
                    blob_name
                ).delete_blob()
            except Exception:
                pass

        for blob_name in created_silver_blobs:
            try:
                pipeline.silver_client.get_blob_client(
                    blob_name
                ).delete_blob()
            except Exception:
                pass

        pipeline.close()
