
import io
import json
import logging
import os
import re
from datetime import datetime, timezone

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ApexF1.Silver")

ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT")
BRONZE = os.getenv("AZURE_STORAGE_CONTAINER_BRONZE", "bronze")
SILVER = os.getenv("AZURE_STORAGE_CONTAINER_SILVER", "silver")
YEAR = int(os.getenv("OPENF1_YEAR", "2025"))

if not ACCOUNT_NAME:
    raise ValueError("AZURE_STORAGE_ACCOUNT is missing from .env")


class BronzeToSilver:
    """Season-wide Bronze-to-Silver transformation pipeline."""

    SESSION_PATTERN = re.compile(
        r"meeting_key=([^/]+)/session_key=([^/]+)/session\.json$"
    )

    def __init__(self):
        self.credential = DefaultAzureCredential()

        account_url = (
            f"https://{ACCOUNT_NAME}.blob.core.windows.net"
        )

        self.service = BlobServiceClient(
            account_url=account_url,
            credential=self.credential
        )

        self.bronze_client = self.service.get_container_client(BRONZE)
        self.silver_client = self.service.get_container_client(SILVER)

    def discover_sessions(self):
        """Discover race sessions from Bronze session.json files."""

        prefix = f"openf1/year={YEAR}/"
        sessions = []

        logger.info("Discovering Bronze sessions under %s", prefix)

        for blob in self.bronze_client.list_blobs(
            name_starts_with=prefix
        ):
            match = self.SESSION_PATTERN.search(blob.name)

            if match:
                meeting_key, session_key = match.groups()

                sessions.append({
                    "meeting_key": meeting_key,
                    "session_key": session_key
                })

        sessions.sort(
            key=lambda item: (
                int(item["meeting_key"]),
                int(item["session_key"])
            )
        )

        logger.info("Discovered %s sessions", len(sessions))
        return sessions

    def download_json(self, blob_name):
        """Download and validate a Bronze JSON dataset."""

        blob = self.bronze_client.get_blob_client(blob_name)
        content = blob.download_blob().readall()

        records = json.loads(content.decode("utf-8"))

        if not isinstance(records, list):
            raise ValueError(
                f"Expected a JSON list: {blob_name}"
            )

        if any(not isinstance(row, dict) for row in records):
            raise ValueError(
                f"Invalid record structure: {blob_name}"
            )

        logger.info(
            "Downloaded %s records from %s",
            len(records),
            blob_name
        )

        return records

    @staticmethod
    def standardise_keys(df):
        """Standardise column names and identifier types."""

        df = df.copy()

        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

        if df.columns.duplicated().any():
            raise ValueError("Duplicate column names after standardisation")

        for column in (
            "session_key",
            "meeting_key",
            "driver_number"
        ):
            if column in df.columns:
                df[column] = df[column].astype("string")

        return df

    @staticmethod
    def remove_duplicates(df, keys, dataset_name):
        """Remove duplicate records using business keys."""

        available_keys = [
            key for key in keys if key in df.columns
        ]

        if not available_keys:
            raise ValueError(
                f"No deduplication keys available for {dataset_name}"
            )

        before = len(df)

        df = df.drop_duplicates(
            subset=available_keys,
            keep="last"
        ).copy()

        logger.info(
            "%s: removed %s duplicate records",
            dataset_name,
            before - len(df)
        )

        return df

    def upload_parquet(self, df, blob_name):
        """Upload a DataFrame as compressed Parquet."""

        buffer = io.BytesIO()

        df.to_parquet(
            buffer,
            index=False,
            engine="pyarrow",
            compression="snappy"
        )

        buffer.seek(0)

        blob = self.silver_client.get_blob_client(blob_name)

        blob.upload_blob(
            buffer.getvalue(),
            overwrite=True,
            content_settings=ContentSettings(
                content_type="application/vnd.apache.parquet"
            )
        )

        logger.info(
            "Uploaded %s rows to %s/%s",
            len(df),
            SILVER,
            blob_name
        )

    def process_session(self, meeting_key, session_key):
        """Transform one race session independently."""

        base_path = (
            f"openf1/year={YEAR}/"
            f"meeting_key={meeting_key}/"
            f"session_key={session_key}"
        )

        logger.info("=" * 65)
        logger.info(
            "Processing meeting %s | session %s",
            meeting_key,
            session_key
        )

        # 1. Download Bronze datasets
        session_records = self.download_json(
            f"{base_path}/session.json"
        )

        driver_records = self.download_json(
            f"{base_path}/drivers.json"
        )

        lap_records = self.download_json(
            f"{base_path}/laps.json"
        )

        # 2. Convert to DataFrames
        session_df = pd.DataFrame(session_records)
        drivers_df = pd.DataFrame(driver_records)
        laps_df = pd.DataFrame(lap_records)

        # 3. Standardise column names and identifiers
        session_df = self.standardise_keys(session_df)
        drivers_df = self.standardise_keys(drivers_df)
        laps_df = self.standardise_keys(laps_df)

        # 4. Validate source datasets
        if session_df.empty:
            raise ValueError("Session dataset is empty")

        if drivers_df.empty:
            raise ValueError("Driver dataset is empty")

        if laps_df.empty:
            raise ValueError("Lap dataset is empty")

        required_lap_columns = {
            "session_key",
            "driver_number",
            "lap_number"
        }

        missing = required_lap_columns - set(laps_df.columns)

        if missing:
            raise ValueError(
                f"Lap dataset missing columns: {sorted(missing)}"
            )

        required_driver_columns = {
            "session_key",
            "driver_number"
        }

        missing = required_driver_columns - set(drivers_df.columns)

        if missing:
            raise ValueError(
                f"Driver dataset missing columns: {sorted(missing)}"
            )

        # 5. Ensure partition identifiers are present
        for df in (session_df, drivers_df, laps_df):
            if "meeting_key" not in df.columns:
                df["meeting_key"] = str(meeting_key)

            if "session_key" not in df.columns:
                df["session_key"] = str(session_key)

        # 6. Remove duplicates
        session_df = self.remove_duplicates(
            session_df,
            ["session_key"],
            "Sessions"
        )

        drivers_df = self.remove_duplicates(
            drivers_df,
            ["session_key", "driver_number"],
            "Drivers"
        )

        laps_df = self.remove_duplicates(
            laps_df,
            ["session_key", "driver_number", "lap_number"],
            "Laps"
        )

        # 7. Convert numeric lap fields
        numeric_columns = [
            "lap_number",
            "lap_duration",
            "duration_sector_1",
            "duration_sector_2",
            "duration_sector_3",
            "i1_speed",
            "i2_speed",
            "st_speed"
        ]

        for column in numeric_columns:
            if column in laps_df.columns:
                laps_df[column] = pd.to_numeric(
                    laps_df[column],
                    errors="coerce"
                )

        # 8. Validate driver lookup uniqueness
        driver_keys = ["session_key", "driver_number"]

        if drivers_df.duplicated(subset=driver_keys).any():
            raise ValueError(
                "Driver lookup contains duplicate session/driver keys"
            )

        # 9. Enrich laps with driver and team information
        driver_columns = [
            "session_key",
            "driver_number",
            "full_name",
            "name_acronym",
            "team_name",
            "team_colour",
            "country_code"
        ]

        available_columns = [
            column for column in driver_columns
            if column in drivers_df.columns
        ]

        driver_lookup = drivers_df[available_columns].copy()

        laps_enriched = laps_df.merge(
            driver_lookup,
            on=driver_keys,
            how="left",
            validate="many_to_one"
        )

        # 10. Log unmatched driver records
        if "full_name" in laps_enriched.columns:
            missing_drivers = (
                laps_enriched["full_name"].isna().sum()
            )

            logger.info(
                "Lap records without matched driver names: %s",
                missing_drivers
            )

        # 11. Add processing metadata
        processed_at = datetime.now(timezone.utc).isoformat()

        for df in (session_df, drivers_df, laps_enriched):
            df["processed_at_utc"] = processed_at

        # 12. Validate Silver datasets
        output_datasets = {
            "session": session_df,
            "drivers": drivers_df,
            "laps_enriched": laps_enriched
        }

        for name, df in output_datasets.items():
            if df.empty:
                raise ValueError(
                    f"Silver dataset is empty: {name}"
                )

            if df.columns.duplicated().any():
                raise ValueError(
                    f"Duplicate columns in Silver dataset: {name}"
                )

            if df["session_key"].isna().any():
                raise ValueError(
                    f"Null session_key found in {name}"
                )

            logger.info(
                "%s validation passed: %s rows, %s columns",
                name,
                len(df),
                len(df.columns)
            )

        # 13. Upload Silver Parquet files
        output_path = base_path

        self.upload_parquet(
            session_df,
            f"{output_path}/session.parquet"
        )

        self.upload_parquet(
            drivers_df,
            f"{output_path}/drivers.parquet"
        )

        self.upload_parquet(
            laps_enriched,
            f"{output_path}/laps_enriched.parquet"
        )

        logger.info(
            "Session %s processed successfully",
            session_key
        )

        return {
            "session_key": session_key,
            "meeting_key": meeting_key,
            "drivers": len(drivers_df),
            "laps": len(laps_enriched),
            "status": "SUCCESS"
        }

    def run(self):
        """Run the transformation across the entire season."""

        logger.info("Starting ApexF1 season-wide Bronze-to-Silver pipeline")
        logger.info("Season: %s", YEAR)

        sessions = self.discover_sessions()

        if not sessions:
            raise ValueError(
                f"No Bronze race sessions found for year {YEAR}"
            )

        successful = []
        failed = []

        total_drivers = 0
        total_laps = 0

        for session in sessions:
            meeting_key = session["meeting_key"]
            session_key = session["session_key"]

            try:
                result = self.process_session(
                    meeting_key,
                    session_key
                )

                successful.append(result)
                total_drivers += result["drivers"]
                total_laps += result["laps"]

            except Exception as exc:
                logger.exception(
                    "Failed processing session %s: %s",
                    session_key,
                    exc
                )

                failed.append({
                    "meeting_key": meeting_key,
                    "session_key": session_key,
                    "error": str(exc)
                })

        logger.info("=" * 65)
        logger.info("SEASON-WIDE SILVER PIPELINE REPORT")
        logger.info("Year: %s", YEAR)
        logger.info("Sessions discovered: %s", len(sessions))
        logger.info("Successful sessions: %s", len(successful))
        logger.info("Failed sessions: %s", len(failed))
        logger.info("Driver records processed: %s", total_drivers)
        logger.info("Lap records processed: %s", total_laps)

        if failed:
            logger.error("FAILED SESSION DETAILS:")

            for item in failed:
                logger.error(
                    "Meeting %s | Session %s | %s",
                    item["meeting_key"],
                    item["session_key"],
                    item["error"]
                )

        if failed:
            raise RuntimeError(
                f"Silver pipeline completed with {len(failed)} "
                f"failed sessions. Check the logs above."
            )

        logger.info(
            "All %s sessions processed successfully.",
            len(successful)
        )

    def close(self):
        self.service.close()
        self.credential.close()


def main():
    pipeline = BronzeToSilver()

    try:
        pipeline.run()
    except Exception:
        logger.exception(
            "Season-wide Bronze-to-Silver pipeline failed"
        )
        raise
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
