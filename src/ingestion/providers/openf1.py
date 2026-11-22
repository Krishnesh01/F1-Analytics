
import time
import logging
from typing import Any

import requests

BASE_URL = "https://api.openf1.org/v1"

logger = logging.getLogger(__name__)


class OpenF1Client:
    """Client for retrieving historical Formula 1 data."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ApexF1-Analytics-Platform/1.0"
        })
        self.last_request_time = 0.0

    def get(self, endpoint: str, params: dict | None = None) -> list[dict[str, Any]]:
        """Fetch JSON data with pacing, timeout and retry handling."""

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(4):
            # Stay below OpenF1's free-tier request rate.
            elapsed = time.monotonic() - self.last_request_time
            if elapsed < 0.4:
                time.sleep(0.4 - elapsed)

            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=45
                )
                self.last_request_time = time.monotonic()

                if response.status_code == 429:
                    wait = min(2 ** (attempt + 1), 30)
                    logger.warning("Rate limited. Waiting %s seconds.", wait)
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                if not isinstance(data, list):
                    raise ValueError(
                        f"Expected a JSON list from {endpoint}"
                    )

                logger.info(
                    "Retrieved %s records from %s",
                    len(data),
                    endpoint
                )
                return data

            except requests.RequestException:
                if attempt == 3:
                    raise

                wait = min(2 ** attempt, 15)
                logger.exception(
                    "Request failed for %s. Retrying in %s seconds.",
                    endpoint,
                    wait
                )
                time.sleep(wait)

        raise RuntimeError(f"Unable to retrieve endpoint: {endpoint}")

    def get_race_sessions(self, year: int) -> list[dict]:
        return self.get(
            "sessions",
            {"year": year, "session_name": "Race"}
        )

    def get_drivers(self, session_key: int) -> list[dict]:
        return self.get(
            "drivers",
            {"session_key": session_key}
        )

    def get_laps(self, session_key: int) -> list[dict]:
        return self.get(
            "laps",
            {"session_key": session_key}
        )
