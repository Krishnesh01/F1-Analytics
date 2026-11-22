
import io
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ApexF1.Gold")

ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
SILVER = os.getenv("AZURE_STORAGE_CONTAINER_SILVER", "silver")
GOLD = os.getenv("AZURE_STORAGE_CONTAINER_GOLD", "gold")
YEAR = int(os.getenv("OPENF1_YEAR", "2025"))

if not ACCOUNT:
    raise ValueError("AZURE_STORAGE_ACCOUNT is missing from .env")


# =========================================================
# SILVER TO GOLD PIPELINE
# =========================================================

class SilverToGold:
    """Build race-level and season-level Formula 1 analytics."""

    def __init__(self):

        self.credential = DefaultAzureCredential()

        self.service = BlobServiceClient(
            account_url=f"https://{ACCOUNT}.blob.core.windows.net",
            credential=self.credential,
        )

        self.silver_client = self.service.get_container_client(SILVER)
        self.gold_client = self.service.get_container_client(GOLD)

    # =====================================================
    # STORAGE OPERATIONS
    # =====================================================

    def discover_sessions(self):
        """Discover Silver sessions containing session.parquet."""

        prefix = f"openf1/year={YEAR}/"
        sessions = []

        for blob in self.silver_client.list_blobs(
            name_starts_with=prefix
        ):

            if not blob.name.endswith("/session.parquet"):
                continue

            parts = blob.name.split("/")

            meeting_key = next(
                (
                    p.split("=", 1)[1]
                    for p in parts
                    if p.startswith("meeting_key=")
                ),
                None,
            )

            session_key = next(
                (
                    p.split("=", 1)[1]
                    for p in parts
                    if p.startswith("session_key=")
                ),
                None,
            )

            if meeting_key and session_key:
                sessions.append({
                    "meeting_key": meeting_key,
                    "session_key": session_key,
                })

        sessions.sort(
            key=lambda item: (
                int(item["meeting_key"]),
                int(item["session_key"]),
            )
        )

        logger.info("Discovered %s Silver sessions", len(sessions))

        return sessions

    def read_parquet(self, container_client, blob_name):

        blob = container_client.get_blob_client(blob_name)
        content = blob.download_blob().readall()

        dataframe = pd.read_parquet(io.BytesIO(content))

        if dataframe.empty:
            raise ValueError(f"Dataset is empty: {blob_name}")

        logger.info(
            "Read %s rows from %s",
            len(dataframe),
            blob_name,
        )

        return dataframe

    def upload_parquet(self, dataframe, blob_name):

        if dataframe.empty:
            raise ValueError(
                f"Refusing to upload empty Gold dataset: {blob_name}"
            )

        buffer = io.BytesIO()

        dataframe.to_parquet(
            buffer,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )

        buffer.seek(0)

        blob = self.gold_client.get_blob_client(blob_name)

        blob.upload_blob(
            buffer.getvalue(),
            overwrite=True,
            content_settings=ContentSettings(
                content_type="application/vnd.apache.parquet"
            ),
        )

        logger.info(
            "Uploaded %s rows to gold/%s",
            len(dataframe),
            blob_name,
        )

    # =====================================================
    # DATA PREPARATION
    # =====================================================

    @staticmethod
    def prepare_laps(laps):
        """Convert lap fields and retain valid non-pit-out laps."""

        data = laps.copy()

        required = {
            "driver_number",
            "lap_number",
            "lap_duration",
        }

        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing lap columns: {sorted(missing)}"
            )

        data["driver_number"] = (
            data["driver_number"].astype("string")
        )

        data["lap_number"] = pd.to_numeric(
            data["lap_number"],
            errors="coerce",
        )

        data["lap_duration"] = pd.to_numeric(
            data["lap_duration"],
            errors="coerce",
        )

        valid = data[
            data["lap_number"].notna()
            & data["lap_duration"].notna()
            & (data["lap_number"] > 0)
            & (data["lap_duration"] > 0)
        ].copy()

        if "is_pit_out_lap" in valid.columns:

            pit_out = (
                valid["is_pit_out_lap"]
                .astype("string")
                .str.strip()
                .str.lower()
            )

            valid = valid[
                ~pit_out.isin(["true", "1", "yes"])
            ].copy()

        return valid

    @staticmethod
    def round_metrics(dataframe, columns):

        for column in columns:

            if column in dataframe.columns:
                dataframe[column] = dataframe[column].round(3)

        return dataframe

    # =====================================================
    # RACE DRIVER ANALYTICS
    # =====================================================

    @staticmethod
    def build_driver_performance(laps):

        valid = SilverToGold.prepare_laps(laps)

        if valid.empty:
            raise ValueError("No valid laps for driver analytics")

        result = (
            valid.groupby("driver_number")
            .agg(
                valid_lap_count=("lap_duration", "count"),
                average_lap_seconds=("lap_duration", "mean"),
                median_lap_seconds=("lap_duration", "median"),
                fastest_lap_seconds=("lap_duration", "min"),
                lap_time_std_seconds=("lap_duration", "std"),
            )
            .reset_index()
        )

        attributes = [
            column for column in [
                "full_name",
                "name_acronym",
                "team_name",
                "team_colour",
                "country_code",
            ]
            if column in valid.columns
        ]

        if attributes:

            lookup = (
                valid.sort_values("lap_number")
                .groupby("driver_number")[attributes]
                .agg(
                    lambda values: (
                        values.dropna().iloc[-1]
                        if not values.dropna().empty
                        else None
                    )
                )
                .reset_index()
            )

            result = result.merge(
                lookup,
                on="driver_number",
                how="left",
                validate="one_to_one",
            )

        result = SilverToGold.round_metrics(
            result,
            [
                "average_lap_seconds",
                "median_lap_seconds",
                "fastest_lap_seconds",
                "lap_time_std_seconds",
            ],
        )

        result = result.sort_values(
            ["fastest_lap_seconds", "average_lap_seconds"]
        ).reset_index(drop=True)

        result.insert(
            0,
            "pace_rank",
            range(1, len(result) + 1),
        )

        return result

    # =====================================================
    # RACE TEAM ANALYTICS
    # =====================================================

    @staticmethod
    def build_team_performance(laps):

        valid = SilverToGold.prepare_laps(laps)

        if "team_name" not in valid.columns:
            raise ValueError("Lap dataset is missing team_name")

        valid = valid.dropna(subset=["team_name"]).copy()

        if valid.empty:
            raise ValueError("No team data available")

        result = (
            valid.groupby("team_name")
            .agg(
                driver_count=("driver_number", "nunique"),
                valid_lap_count=("lap_duration", "count"),
                average_driver_lap_seconds=("lap_duration", "mean"),
                fastest_lap_seconds=("lap_duration", "min"),
                median_lap_seconds=("lap_duration", "median"),
            )
            .reset_index()
        )

        result = SilverToGold.round_metrics(
            result,
            [
                "average_driver_lap_seconds",
                "fastest_lap_seconds",
                "median_lap_seconds",
            ],
        )

        return result.sort_values(
            "average_driver_lap_seconds"
        ).reset_index(drop=True)

    # =====================================================
    # RACE LAP ANALYTICS
    # =====================================================

    @staticmethod
    def build_lap_analysis(laps):

        valid = SilverToGold.prepare_laps(laps)

        if valid.empty:
            raise ValueError("No valid lap data for lap analysis")

        result = (
            valid.groupby("lap_number")
            .agg(
                recorded_laps=("lap_duration", "count"),
                average_lap_seconds=("lap_duration", "mean"),
                fastest_lap_seconds=("lap_duration", "min"),
                slowest_lap_seconds=("lap_duration", "max"),
            )
            .reset_index()
        )

        result = SilverToGold.round_metrics(
            result,
            [
                "average_lap_seconds",
                "fastest_lap_seconds",
                "slowest_lap_seconds",
            ],
        )

        return result.sort_values(
            "lap_number"
        ).reset_index(drop=True)

    # =====================================================
    # RACE SUMMARY
    # =====================================================

    @staticmethod
    def build_race_summary(
        session,
        drivers,
        laps,
        meeting_key,
        session_key,
    ):

        valid = SilverToGold.prepare_laps(laps)

        if valid.empty:
            raise ValueError("No valid lap times for race summary")

        first_session = session.iloc[0]

        summary = {
            "season": YEAR,
            "meeting_key": str(meeting_key),
            "session_key": str(session_key),
            "session_name": first_session.get("session_name"),
            "date_start": first_session.get("date_start"),
            "date_end": first_session.get("date_end"),
            "driver_count": int(
                drivers["driver_number"].nunique()
            ),
            "lap_records_received": int(len(laps)),
            "valid_lap_time_records": int(len(valid)),
            "fastest_lap_seconds": round(
                float(valid["lap_duration"].min()), 3
            ),
            "average_lap_seconds": round(
                float(valid["lap_duration"].mean()), 3
            ),
            "processed_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        return pd.DataFrame([summary])

    # =====================================================
    # SEASON DRIVER ANALYTICS - NORMALIZED PACE
    # =====================================================

    @staticmethod
    def build_season_driver_performance(laps):
        """
        Calculate season driver performance using race-normalized pace.

        Each driver's median lap time is compared against the median
        of all participating drivers' median lap times in that race.

        Negative relative pace = faster than race reference.
        Positive relative pace = slower than race reference.

        Each race contributes equally to the season pace metric.
        """

        valid = SilverToGold.prepare_laps(laps)

        if valid.empty:
            raise ValueError("No valid season laps")

        required = {
            "meeting_key",
            "session_key",
            "driver_number",
        }

        missing = required - set(valid.columns)

        if missing:
            raise ValueError(
                f"Missing columns for normalized pace: {sorted(missing)}"
            )

        # 1. Calculate each driver's median lap time per race.

        race_driver_pace = (
            valid.groupby(
                ["meeting_key", "session_key", "driver_number"],
                as_index=False,
            )
            .agg(
                race_median_lap_seconds=("lap_duration", "median"),
                race_valid_lap_count=("lap_duration", "count"),
            )
        )

        # 2. Calculate the median reference pace for each race.

        race_reference = (
            race_driver_pace.groupby(
                ["meeting_key", "session_key"]
            )["race_median_lap_seconds"]
            .median()
            .reset_index(
                name="race_reference_median_seconds"
            )
        )

        race_driver_pace = race_driver_pace.merge(
            race_reference,
            on=["meeting_key", "session_key"],
            how="left",
            validate="many_to_one",
        )

        # 3. Calculate relative pace percentage.

        race_driver_pace["race_relative_pace_pct"] = (
            (
                race_driver_pace["race_median_lap_seconds"]
                / race_driver_pace["race_reference_median_seconds"]
            ) - 1
        ) * 100

        # 4. Aggregate normalized performance across races.

        normalized_performance = (
            race_driver_pace.groupby("driver_number")
            .agg(
                races_with_pace=("session_key", "nunique"),
                average_race_relative_pace_pct=(
                    "race_relative_pace_pct", "mean"
                ),
                median_race_relative_pace_pct=(
                    "race_relative_pace_pct", "median"
                ),
                average_race_median_lap_seconds=(
                    "race_median_lap_seconds", "mean"
                ),
                total_pace_sample_laps=(
                    "race_valid_lap_count", "sum"
                ),
            )
            .reset_index()
        )

        # 5. Preserve existing raw season metrics.

        raw_performance = (
            valid.groupby("driver_number")
            .agg(
                race_count=("session_key", "nunique"),
                valid_lap_count=("lap_duration", "count"),
                average_lap_seconds=("lap_duration", "mean"),
                median_lap_seconds=("lap_duration", "median"),
                fastest_lap_seconds=("lap_duration", "min"),
                lap_time_std_seconds=("lap_duration", "std"),
            )
            .reset_index()
        )

        result = raw_performance.merge(
            normalized_performance,
            on="driver_number",
            how="left",
            validate="one_to_one",
        )

        # 6. Preserve driver identity information.

        attributes = [
            column for column in [
                "full_name",
                "name_acronym",
                "country_code",
            ]
            if column in valid.columns
        ]

        if attributes:

            lookup = (
                valid.groupby("driver_number")[attributes]
                .agg(
                    lambda values: (
                        values.dropna().mode().iloc[0]
                        if not values.dropna().mode().empty
                        else None
                    )
                )
                .reset_index()
            )

            result = result.merge(
                lookup,
                on="driver_number",
                how="left",
                validate="one_to_one",
            )

        # 7. Preserve all teams represented during the season.

        if "team_name" in valid.columns:

            teams = (
                valid.dropna(subset=["team_name"])
                .groupby("driver_number")["team_name"]
                .agg(
                    lambda values: ", ".join(
                        sorted(values.astype(str).unique())
                    )
                )
                .reset_index(name="teams_represented")
            )

            result = result.merge(
                teams,
                on="driver_number",
                how="left",
                validate="one_to_one",
            )

        # 8. Round numerical metrics.

        result = SilverToGold.round_metrics(
            result,
            [
                "average_lap_seconds",
                "median_lap_seconds",
                "fastest_lap_seconds",
                "lap_time_std_seconds",
                "average_race_relative_pace_pct",
                "median_race_relative_pace_pct",
                "average_race_median_lap_seconds",
            ],
        )

        # 9. Rank drivers by normalized pace.
        # A lower relative pace percentage is faster.

        result = result.sort_values(
            [
                "average_race_relative_pace_pct",
                "races_with_pace",
            ],
            ascending=[True, False],
        ).reset_index(drop=True)

        result.insert(
            0,
            "normalized_pace_rank",
            range(1, len(result) + 1),
        )

        return result

    # =====================================================
    # SEASON TEAM ANALYTICS
    # =====================================================

    @staticmethod
    def build_season_team_performance(laps):
        """Build raw and race-normalized season team pace metrics.

        For each race, calculate each team's median valid lap time. Compare
        that median with the median of the participating teams' medians in
        the same race. Aggregate those race-relative percentages equally
        across races so circuit lap length does not dominate the season metric.

        Negative relative pace = faster than that race's team reference.
        Positive relative pace = slower than that race's team reference.
        This is descriptive lap pace, not an official championship ranking.
        """

        valid = SilverToGold.prepare_laps(laps)

        required = {
            "meeting_key",
            "session_key",
            "team_name",
            "driver_number",
            "lap_duration",
        }
        missing = required - set(valid.columns)
        if missing:
            raise ValueError(
                f"Missing columns for normalized team pace: {sorted(missing)}"
            )

        valid = valid.dropna(subset=["team_name"]).copy()
        if valid.empty:
            raise ValueError("No team data for season analytics")

        # Raw season metrics are retained for transparency/context only.
        raw = (
            valid.groupby("team_name")
            .agg(
                race_count=("session_key", "nunique"),
                driver_count=("driver_number", "nunique"),
                valid_lap_count=("lap_duration", "count"),
                average_driver_lap_seconds=("lap_duration", "mean"),
                fastest_lap_seconds=("lap_duration", "min"),
                median_lap_seconds=("lap_duration", "median"),
            )
            .reset_index()
        )

        # One observation per team per race. Using the median reduces the
        # influence of unusually slow laps and unequal lap sample counts.
        race_team_pace = (
            valid.groupby(
                ["meeting_key", "session_key", "team_name"],
                as_index=False,
            )
            .agg(
                race_median_lap_seconds=("lap_duration", "median"),
                race_valid_lap_count=("lap_duration", "count"),
            )
        )

        # Reference is the median of team medians within the same race.
        race_reference = (
            race_team_pace.groupby(["meeting_key", "session_key"])
            ["race_median_lap_seconds"]
            .median()
            .reset_index(name="race_reference_median_seconds")
        )

        race_team_pace = race_team_pace.merge(
            race_reference,
            on=["meeting_key", "session_key"],
            how="left",
            validate="many_to_one",
        )

        race_team_pace["race_relative_pace_pct"] = (
            (
                race_team_pace["race_median_lap_seconds"]
                / race_team_pace["race_reference_median_seconds"]
            ) - 1
        ) * 100

        # Each race contributes one percentage to the season average.
        normalized = (
            race_team_pace.groupby("team_name")
            .agg(
                races_with_pace=("session_key", "count"),
                average_race_relative_pace_pct=(
                    "race_relative_pace_pct", "mean"
                ),
                median_race_relative_pace_pct=(
                    "race_relative_pace_pct", "median"
                ),
                average_race_median_lap_seconds=(
                    "race_median_lap_seconds", "mean"
                ),
                total_pace_sample_laps=("race_valid_lap_count", "sum"),
            )
            .reset_index()
        )

        result = raw.merge(
            normalized,
            on="team_name",
            how="left",
            validate="one_to_one",
        )

        result = SilverToGold.round_metrics(
            result,
            [
                "average_driver_lap_seconds",
                "fastest_lap_seconds",
                "median_lap_seconds",
                "average_race_relative_pace_pct",
                "median_race_relative_pace_pct",
                "average_race_median_lap_seconds",
            ],
        )

        result = result.sort_values(
            ["average_race_relative_pace_pct", "races_with_pace"],
            ascending=[True, False],
        ).reset_index(drop=True)

        result.insert(0, "normalized_pace_rank", range(1, len(result) + 1))
        return result

    # =====================================================
    # SEASON SUMMARY
    # =====================================================

    @staticmethod
    def build_season_summary(
        sessions,
        laps,
        drivers,
        teams,
    ):

        valid = SilverToGold.prepare_laps(laps)

        if valid.empty:
            raise ValueError("No valid laps for season summary")

        summary = {
            "season": YEAR,
            "race_count": int(
                valid["session_key"].nunique()
            ),
            "meeting_count": int(
                valid["meeting_key"].nunique()
            ),
            "driver_count": int(
                valid["driver_number"].nunique()
            ),
            "team_count": int(
                valid["team_name"].nunique()
            ) if "team_name" in valid.columns else 0,
            "valid_lap_count": int(len(valid)),
            "fastest_lap_seconds": round(
                float(valid["lap_duration"].min()), 3
            ),
            "average_lap_seconds": round(
                float(valid["lap_duration"].mean()), 3
            ),
            "processed_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        return pd.DataFrame([summary])

    # =====================================================
    # OUTPUT VALIDATION
    # =====================================================

    @staticmethod
    def validate_outputs(outputs):

        for name, dataframe in outputs.items():

            if dataframe.empty:
                raise ValueError(
                    f"Gold output is empty: {name}"
                )

            if dataframe.columns.duplicated().any():
                raise ValueError(
                    f"Duplicate columns in Gold output: {name}"
                )

            logger.info(
                "%s validation passed: %s rows, %s columns",
                name,
                len(dataframe),
                len(dataframe.columns),
            )

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def run(self):

        logger.info("=" * 65)
        logger.info("APEXF1 SEASON-WIDE SILVER-TO-GOLD PIPELINE")
        logger.info("Season: %s", YEAR)

        sessions = self.discover_sessions()

        if not sessions:
            raise ValueError(
                f"No Silver sessions found for season {YEAR}"
            )

        all_laps = []
        race_summaries = []
        successful = []
        failed = []

        # -------------------------------------------------
        # Process each race independently.
        # -------------------------------------------------

        for item in sessions:

            meeting_key = item["meeting_key"]
            session_key = item["session_key"]

            base = (
                f"openf1/year={YEAR}/"
                f"meeting_key={meeting_key}/"
                f"session_key={session_key}"
            )

            logger.info("-" * 65)
            logger.info(
                "Processing meeting %s | session %s",
                meeting_key,
                session_key,
            )

            try:

                session = self.read_parquet(
                    self.silver_client,
                    f"{base}/session.parquet",
                )

                drivers_silver = self.read_parquet(
                    self.silver_client,
                    f"{base}/drivers.parquet",
                )

                laps = self.read_parquet(
                    self.silver_client,
                    f"{base}/laps_enriched.parquet",
                )

                # Attach partition identifiers explicitly.

                for dataframe in (
                    session,
                    drivers_silver,
                    laps,
                ):
                    dataframe["meeting_key"] = str(meeting_key)
                    dataframe["session_key"] = str(session_key)

                # Build race-level Gold datasets.

                driver_performance = (
                    self.build_driver_performance(laps)
                )

                team_performance = (
                    self.build_team_performance(laps)
                )

                lap_analysis = (
                    self.build_lap_analysis(laps)
                )

                race_summary = self.build_race_summary(
                    session,
                    driver_performance,
                    laps,
                    meeting_key,
                    session_key,
                )

                # Add race identifiers to analytical outputs.

                for dataframe in (
                    driver_performance,
                    team_performance,
                    lap_analysis,
                ):
                    dataframe.insert(
                        0, "meeting_key", str(meeting_key)
                    )

                    dataframe.insert(
                        1, "session_key", str(session_key)
                    )

                race_outputs = {
                    "driver_performance": driver_performance,
                    "team_performance": team_performance,
                    "lap_analysis": lap_analysis,
                    "race_summary": race_summary,
                }

                self.validate_outputs(race_outputs)

                # Preserve existing race partition paths.

                for name, dataframe in race_outputs.items():

                    self.upload_parquet(
                        dataframe,
                        f"{base}/{name}.parquet",
                    )

                # Collect records for season aggregation.

                all_laps.append(laps)
                race_summaries.append(race_summary)

                successful.append(session_key)

                logger.info(
                    "Race session %s completed successfully",
                    session_key,
                )

            except Exception as exc:

                logger.exception(
                    "Failed processing session %s: %s",
                    session_key,
                    exc,
                )

                failed.append({
                    "meeting_key": meeting_key,
                    "session_key": session_key,
                    "error": str(exc),
                })

        # -------------------------------------------------
        # Ensure all race sessions succeeded.
        # -------------------------------------------------

        if failed:

            logger.error("Some race sessions failed:")

            for item in failed:

                logger.error(
                    "Meeting %s | Session %s | %s",
                    item["meeting_key"],
                    item["session_key"],
                    item["error"],
                )

            raise RuntimeError(
                f"{len(failed)} of {len(sessions)} race sessions failed. "
                "Season-level datasets were not generated."
            )

        if len(successful) != len(sessions):
            raise RuntimeError(
                "Not all discovered sessions were processed."
            )

        # =================================================
        # BUILD SEASON-LEVEL DATASETS
        # =================================================

        logger.info("=" * 65)
        logger.info("Building season-level analytics")

        season_laps = pd.concat(
            all_laps,
            ignore_index=True,
        )

        season_driver_performance = (
            self.build_season_driver_performance(season_laps)
        )

        # -------------------------------------------------
        # NORMALIZED PACE DIAGNOSTICS
        # -------------------------------------------------

        logger.info("=" * 65)
        logger.info("NORMALIZED PACE VALIDATION")

        logger.info(
            "Season driver columns: %s",
            season_driver_performance.columns.tolist(),
        )

        diagnostic_columns = [
            "driver_number",
            "normalized_pace_rank",
            "average_race_relative_pace_pct",
            "median_race_relative_pace_pct",
            "races_with_pace",
            "valid_lap_count",
        ]

        logger.info(
            "\n%s",
            season_driver_performance[
                diagnostic_columns
            ].head(10).to_string(index=False),
        )

        # Validate normalized pace values.

        if season_driver_performance[
            "average_race_relative_pace_pct"
        ].isna().any():

            raise ValueError(
                "Normalized pace contains missing values."
            )

        if season_driver_performance[
            "normalized_pace_rank"
        ].duplicated().any():

            raise ValueError(
                "Normalized pace ranks are not unique."
            )

        if not (
            season_driver_performance["races_with_pace"]
            <= season_driver_performance["race_count"]
        ).all():

            raise ValueError(
                "Race coverage exceeds driver's race count."
            )

        logger.info(
            "Normalized pace validation passed for %s drivers",
            len(season_driver_performance),
        )

        # -------------------------------------------------
        # Build remaining season datasets.
        # -------------------------------------------------

        season_team_performance = (
            self.build_season_team_performance(season_laps)
        )

        logger.info("=" * 65)
        logger.info("NORMALIZED TEAM PACE VALIDATION")

        team_diagnostic_columns = [
            "team_name",
            "normalized_pace_rank",
            "average_race_relative_pace_pct",
            "median_race_relative_pace_pct",
            "races_with_pace",
            "race_count",
            "valid_lap_count",
        ]
        logger.info(
            "\n%s",
            season_team_performance[
                team_diagnostic_columns
            ].to_string(index=False),
        )

        if season_team_performance[
            "average_race_relative_pace_pct"
        ].isna().any():
            raise ValueError("Normalized team pace contains missing values.")

        if season_team_performance["normalized_pace_rank"].duplicated().any():
            raise ValueError("Normalized team pace ranks are not unique.")

        if not (
            season_team_performance["races_with_pace"]
            <= season_team_performance["race_count"]
        ).all():
            raise ValueError("Team race coverage exceeds team race count.")

        logger.info(
            "Normalized team pace validation passed for %s teams",
            len(season_team_performance),
        )

        race_summary_df = pd.concat(
            race_summaries,
            ignore_index=True,
        )

        season_summary = self.build_season_summary(
            sessions,
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

        self.validate_outputs(season_outputs)

        # -------------------------------------------------
        # Upload season-level Gold files.
        # -------------------------------------------------

        season_path = f"openf1/year={YEAR}/season"

        for name, dataframe in season_outputs.items():

            self.upload_parquet(
                dataframe,
                f"{season_path}/{name}.parquet",
            )

        # =================================================
        # FINAL PIPELINE REPORT
        # =================================================

        logger.info("=" * 65)
        logger.info("APEXF1 GOLD PIPELINE REPORT")
        logger.info("Season: %s", YEAR)
        logger.info("Sessions discovered: %s", len(sessions))
        logger.info("Successful sessions: %s", len(successful))
        logger.info("Failed sessions: %s", len(failed))
        logger.info("Season lap records loaded: %s", len(season_laps))

        logger.info(
            "Season drivers: %s",
            len(season_driver_performance),
        )

        logger.info(
            "Season teams: %s",
            len(season_team_performance),
        )

        logger.info(
            "Normalized pace metric: average_race_relative_pace_pct"
        )
        logger.info(
            "Normalized team pace metric: average_race_relative_pace_pct"
        )

        logger.info("Gold pipeline completed successfully")

    # =====================================================
    # CLEANUP
    # =====================================================

    def close(self):

        self.service.close()
        self.credential.close()


# =========================================================
# ENTRY POINT
# =========================================================

def main():

    pipeline = SilverToGold()

    try:
        pipeline.run()

    except Exception:

        logger.exception(
            "Silver-to-Gold transformation failed"
        )

        raise

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
