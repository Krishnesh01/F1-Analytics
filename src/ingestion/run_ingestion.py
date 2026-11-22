
import logging
import os

from dotenv import load_dotenv

from src.ingestion.providers.openf1 import OpenF1Client
from src.storage.azure_blob import AzureBlobStorage


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

YEAR = int(os.getenv("OPENF1_YEAR", "2025"))
BRONZE_CONTAINER = os.getenv(
    "AZURE_STORAGE_CONTAINER_BRONZE",
    "bronze",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("apexf1.season_ingestion")


# --------------------------------------------------
# INGESTION PIPELINE
# --------------------------------------------------

def run_season_ingestion():

    api = OpenF1Client()
    storage = None

    successful_sessions = 0
    failed_sessions = []
    total_drivers = 0
    total_laps = 0

    try:
        logger.info("=" * 60)
        logger.info("APEXF1 SEASON INGESTION STARTED")
        logger.info("Target season: %s", YEAR)
        logger.info("=" * 60)

        storage = AzureBlobStorage()

        # Step 1: Retrieve all race sessions
        sessions = api.get_race_sessions(YEAR)

        if not sessions:
            raise RuntimeError(
                f"No race sessions found for {YEAR}"
            )

        sessions = sorted(
            sessions,
            key=lambda item: item.get("date_start") or "",
        )

        logger.info(
            "Found %s race sessions for %s",
            len(sessions),
            YEAR,
        )

        # Step 2: Process each race session
        for index, session in enumerate(sessions, start=1):

            session_key = session.get("session_key")
            meeting_key = session.get("meeting_key")

            if not session_key or not meeting_key:
                logger.warning(
                    "Skipping session with missing keys: %s",
                    session,
                )
                failed_sessions.append({
                    "session_key": session_key,
                    "reason": "Missing meeting_key or session_key",
                })
                continue

            prefix = (
                f"openf1/year={YEAR}/"
                f"meeting_key={meeting_key}/"
                f"session_key={session_key}/"
            )

            logger.info("-" * 60)
            logger.info(
                "Processing race %s of %s | Meeting %s | Session %s",
                index,
                len(sessions),
                meeting_key,
                session_key,
            )

            try:
                # Retrieve data
                drivers = api.get_drivers(int(session_key))
                laps = api.get_laps(int(session_key))

                logger.info(
                    "Retrieved %s drivers and %s lap records",
                    len(drivers),
                    len(laps),
                )

                # Upload raw session metadata
                storage.upload_json(
                    [session],
                    BRONZE_CONTAINER,
                    prefix + "session.json",
                )

                # Upload raw driver records
                storage.upload_json(
                    drivers,
                    BRONZE_CONTAINER,
                    prefix + "drivers.json",
                )

                # Upload raw lap records
                storage.upload_json(
                    laps,
                    BRONZE_CONTAINER,
                    prefix + "laps.json",
                )

                successful_sessions += 1
                total_drivers += len(drivers)
                total_laps += len(laps)

                logger.info(
                    "SUCCESS: Meeting %s | Session %s",
                    meeting_key,
                    session_key,
                )

            except Exception as exc:
                logger.exception(
                    "FAILED: Meeting %s | Session %s",
                    meeting_key,
                    session_key,
                )

                failed_sessions.append({
                    "meeting_key": meeting_key,
                    "session_key": session_key,
                    "reason": str(exc),
                })

                continue

        # Step 3: Final report
        logger.info("=" * 60)
        logger.info("APEXF1 SEASON INGESTION REPORT")
        logger.info("Season: %s", YEAR)
        logger.info("Sessions discovered: %s", len(sessions))
        logger.info("Successful sessions: %s", successful_sessions)
        logger.info("Failed sessions: %s", len(failed_sessions))
        logger.info("Total driver records: %s", total_drivers)
        logger.info("Total lap records: %s", total_laps)

        if failed_sessions:
            logger.warning("Sessions requiring retry:")

            for failure in failed_sessions:
                logger.warning("%s", failure)

        logger.info("=" * 60)

        if successful_sessions == 0:
            raise RuntimeError(
                "No sessions were ingested successfully."
            )

    finally:
        api.session.close()

        if storage is not None:
            storage.close()


if __name__ == "__main__":
    run_season_ingestion()
