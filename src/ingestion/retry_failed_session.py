
import logging
import os

from dotenv import load_dotenv

from src.ingestion.providers.openf1 import OpenF1Client
from src.storage.azure_blob import AzureBlobStorage

load_dotenv()

YEAR = 2025
SESSION_KEY = 9888

BRONZE_CONTAINER = os.getenv(
    "AZURE_STORAGE_CONTAINER_BRONZE",
    "bronze",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("apexf1.retry")


def retry_session():

    api = OpenF1Client()
    storage = None

    try:
        storage = AzureBlobStorage()

        # Find the original session metadata
        sessions = api.get_race_sessions(YEAR)

        session = next(
            (
                item for item in sessions
                if int(item.get("session_key", -1)) == SESSION_KEY
            ),
            None,
        )

        if session is None:
            raise RuntimeError(
                f"Session {SESSION_KEY} was not found."
            )

        meeting_key = session["meeting_key"]

        logger.info(
            "Retrying meeting %s, session %s",
            meeting_key,
            SESSION_KEY,
        )

        # Retrieve the missing datasets
        drivers = api.get_drivers(SESSION_KEY)
        laps = api.get_laps(SESSION_KEY)

        if not drivers:
            raise RuntimeError("Driver dataset is empty.")

        if not laps:
            raise RuntimeError("Lap dataset is empty.")

        prefix = (
            f"openf1/year={YEAR}/"
            f"meeting_key={meeting_key}/"
            f"session_key={SESSION_KEY}/"
        )

        # Re-upload all three files to ensure consistency
        storage.upload_json(
            [session],
            BRONZE_CONTAINER,
            prefix + "session.json",
        )

        storage.upload_json(
            drivers,
            BRONZE_CONTAINER,
            prefix + "drivers.json",
        )

        storage.upload_json(
            laps,
            BRONZE_CONTAINER,
            prefix + "laps.json",
        )

        logger.info("RETRY SUCCESSFUL")
        logger.info("Session: %s", SESSION_KEY)
        logger.info("Drivers: %s", len(drivers))
        logger.info("Laps: %s", len(laps))

    finally:
        api.session.close()

        if storage is not None:
            storage.close()


if __name__ == "__main__":
    retry_session()
