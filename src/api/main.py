
import io
import logging
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from threading import RLock
import time

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("apexf1.api")

ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT")
GOLD_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER_GOLD", "gold")

YEAR = os.getenv("OPENF1_YEAR", "2025")
MEETING_KEY = os.getenv("OPENF1_MEETING_KEY", "1276")
SESSION_KEY = os.getenv("OPENF1_SESSION_KEY", "9839")

# Gold datasets are cached in memory to reduce repeated Azure downloads.
# Set APEXF1_CACHE_TTL_SECONDS in .env; default is 300 seconds (5 minutes).
CACHE_TTL_SECONDS = int(
    os.getenv("APEXF1_CACHE_TTL_SECONDS", "300")
)
if CACHE_TTL_SECONDS < 1:
    raise ValueError("APEXF1_CACHE_TTL_SECONDS must be at least 1")

DATASET_CACHE_MAXSIZE = 16
_DATASET_CACHE = OrderedDict()
_DATASET_CACHE_LOCK = RLock()

RACE_BASE_PATH = (
    f"openf1/year={YEAR}/"
    f"meeting_key={MEETING_KEY}/"
    f"session_key={SESSION_KEY}/"
)

SEASON_BASE_PATH = (
    f"openf1/year={YEAR}/season/"
)

RACE_DATASETS = {
    "drivers": "driver_performance.parquet",
    "teams": "team_performance.parquet",
    "laps": "lap_analysis.parquet",
    "summary": "race_summary.parquet",
}

SEASON_DATASETS = {
    "drivers": "season_driver_performance.parquet",
    "teams": "season_team_performance.parquet",
    "races": "race_summary.parquet",
    "summary": "season_summary.parquet",
}


# ==================================================
# FASTAPI APPLICATION
# ==================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application resources without connecting to Azure at startup."""
    logger.info("ApexF1 API starting")
    try:
        yield
    finally:
        close_azure_clients()
        logger.info("ApexF1 API shutdown complete")


app = FastAPI(
    title="ApexF1 Race Intelligence API",
    description=(
        "REST API for historical Formula 1 race and "
        "season analytics powered by Azure Blob Storage."
    ),
    version="2.6.0",
    lifespan=lifespan,
)


# ==================================================
# AZURE STORAGE CONNECTION
# ==================================================

_AZURE_CLIENT_LOCK = RLock()
_azure_credential = None
_blob_service_client = None


def get_blob_service():
    """Create the Azure credential and Blob client once, on first use."""
    global _azure_credential, _blob_service_client

    with _AZURE_CLIENT_LOCK:
        if _blob_service_client is not None:
            return _blob_service_client

        if not ACCOUNT_NAME:
            raise RuntimeError(
                "AZURE_STORAGE_ACCOUNT is missing from .env"
            )

        credential = DefaultAzureCredential()

        try:
            blob_service = BlobServiceClient(
                account_url=(
                    f"https://{ACCOUNT_NAME}.blob.core.windows.net"
                ),
                credential=credential,
            )
        except Exception:
            credential.close()
            raise

        _azure_credential = credential
        _blob_service_client = blob_service
        return _blob_service_client


def close_azure_clients() -> None:
    """Close Azure SDK resources created by this API process."""
    global _azure_credential, _blob_service_client

    with _AZURE_CLIENT_LOCK:
        blob_service = _blob_service_client
        credential = _azure_credential

        # Reset references first so a later app lifespan can recreate them.
        _blob_service_client = None
        _azure_credential = None

        if blob_service is not None:
            try:
                blob_service.close()
                logger.info("Azure BlobServiceClient closed")
            except Exception:
                logger.exception("Error while closing Azure BlobServiceClient")

        if credential is not None:
            try:
                credential.close()
                logger.info("Azure credential closed")
            except Exception:
                logger.exception("Error while closing Azure credential")


# ==================================================
# DATA ACCESS
# ==================================================

def _get_cached_dataset(cache_key: tuple, loader) -> pd.DataFrame:
    """Return a cached DataFrame, refreshing it after the configured TTL.

    The lock prevents concurrent requests from downloading the same dataset
    multiple times when its cache entry is missing or expired.
    """
    now = time.monotonic()

    with _DATASET_CACHE_LOCK:
        cached_entry = _DATASET_CACHE.get(cache_key)

        if cached_entry is not None:
            expires_at, dataframe = cached_entry

            if now < expires_at:
                _DATASET_CACHE.move_to_end(cache_key)
                return dataframe

            del _DATASET_CACHE[cache_key]
            logger.info("Dataset cache expired: %s", cache_key)

        dataframe = loader()

        _DATASET_CACHE[cache_key] = (
            time.monotonic() + CACHE_TTL_SECONDS,
            dataframe,
        )
        _DATASET_CACHE.move_to_end(cache_key)

        while len(_DATASET_CACHE) > DATASET_CACHE_MAXSIZE:
            evicted_key, _ = _DATASET_CACHE.popitem(last=False)
            logger.info("Evicted dataset cache entry: %s", evicted_key)

        return dataframe


def load_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in RACE_DATASETS:
        raise ValueError("Unknown race dataset")

    def load_from_azure():
        blob_name = RACE_BASE_PATH + RACE_DATASETS[dataset_name]
        return download_parquet(blob_name, dataset_name)

    return _get_cached_dataset(("race", dataset_name), load_from_azure)


def load_season_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in SEASON_DATASETS:
        raise ValueError("Unknown season dataset")

    def load_from_azure():
        blob_name = SEASON_BASE_PATH + SEASON_DATASETS[dataset_name]
        return download_parquet(
            blob_name,
            f"season_{dataset_name}",
        )

    return _get_cached_dataset(
        ("season", dataset_name),
        load_from_azure,
    )


def clear_dataset_caches() -> None:
    """Clear all in-memory race and season dataset cache entries."""
    with _DATASET_CACHE_LOCK:
        _DATASET_CACHE.clear()
    logger.info("ApexF1 dataset caches cleared")


def download_parquet(blob_name: str, dataset_name: str):

    blob_client = get_blob_service().get_blob_client(
        container=GOLD_CONTAINER,
        blob=blob_name,
    )

    try:

        content = blob_client.download_blob().readall()

        dataframe = pd.read_parquet(
            io.BytesIO(content)
        )

        logger.info(
            "Loaded %s records from %s",
            len(dataframe),
            blob_name,
        )

        return dataframe

    except Exception as exc:

        logger.exception(
            "Failed to load dataset: %s",
            dataset_name,
        )

        raise RuntimeError(
            f"Unable to load {dataset_name} dataset"
        ) from exc


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Fail clearly when a Gold dataset does not match its API contract."""
    missing = required_columns - set(dataframe.columns)

    if missing:
        logger.error(
            "Gold dataset %s is missing required columns: %s",
            dataset_name,
            sorted(missing),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Gold dataset '{dataset_name}' is missing required columns.",
        )


def require_single_summary_record(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> dict:
    """Return a summary only when the Gold dataset has exactly one row."""
    if dataframe.empty:
        raise HTTPException(
            status_code=404,
            detail=f"{dataset_name} not found",
        )

    if len(dataframe) != 1:
        logger.error(
            "Expected one row in %s, found %s",
            dataset_name,
            len(dataframe),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Gold dataset '{dataset_name}' must contain exactly one summary record.",
        )

    return dataframe_to_records(dataframe)[0]


def dataframe_to_records(dataframe: pd.DataFrame):
    """Convert a DataFrame to JSON-safe records, replacing null and infinite values."""
    safe_df = dataframe.astype(object).where(
        pd.notna(dataframe),
        None,
    )
    safe_df = safe_df.replace([float("inf"), float("-inf")], None)
    return safe_df.to_dict(orient="records")


def get_dataset(dataset_name: str):

    try:
        return load_dataset(dataset_name)

    except Exception as exc:
        logger.exception(
            "API failed to load race dataset: %s",
            dataset_name,
        )

        raise HTTPException(
            status_code=503,
            detail="Race dataset is temporarily unavailable.",
        ) from exc


def get_season_dataset(dataset_name: str):

    try:
        return load_season_dataset(dataset_name)

    except Exception as exc:
        logger.exception(
            "API failed to load season dataset: %s",
            dataset_name,
        )

        raise HTTPException(
            status_code=503,
            detail="Season dataset is temporarily unavailable.",
        ) from exc


# ==================================================
# HEALTH CHECKS
# ==================================================

@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe: confirms that the API process is responding.

    This endpoint deliberately does not call Azure. A temporary Azure
    outage should not make the application appear to have crashed.
    """
    return {
        "status": "healthy",
        "application": "ApexF1",
        "version": "2.6.0",
    }


@app.get("/health/ready", tags=["System"])
def readiness_check():
    """Readiness probe: verifies access to the configured Gold data.

    Checks that the Gold container can be reached and that the season
    summary Parquet file exists. Returns HTTP 503 if the dependency check
    fails, without exposing Azure exception details to API consumers.
    """
    try:
        blob_service = get_blob_service()
        summary_blob = (
            SEASON_BASE_PATH + SEASON_DATASETS["summary"]
        )
        blob_client = blob_service.get_blob_client(
            container=GOLD_CONTAINER,
            blob=summary_blob,
        )
        blob_client.get_blob_properties()

        return {
            "status": "ready",
            "application": "ApexF1",
            "version": "2.6.0",
            "dependencies": {
                "azure_blob_storage": "available",
                "gold_season_summary": "available",
            },
        }

    except Exception as exc:
        logger.exception("Readiness check failed: Azure Gold data unavailable")
        raise HTTPException(
            status_code=503,
            detail="Service is not ready: required data storage is unavailable.",
        ) from exc


# ==================================================
# RACE SUMMARY
# ==================================================

@app.get("/api/v1/race/summary", tags=["Race"])
def race_summary():

    dataframe = get_dataset("summary")

    return require_single_summary_record(dataframe, "Race summary")


# ==================================================
# RACE DRIVER PERFORMANCE
# ==================================================

@app.get("/api/v1/drivers", tags=["Drivers"])
def get_drivers(
    limit: int = Query(default=20, ge=1, le=100),
):

    dataframe = get_dataset("drivers")
    require_columns(dataframe, {"fastest_lap_seconds"}, "race drivers")

    dataframe = dataframe.copy()
    dataframe["fastest_lap_seconds"] = pd.to_numeric(
        dataframe["fastest_lap_seconds"],
        errors="coerce",
    )
    dataframe = dataframe.sort_values(
        "fastest_lap_seconds",
        ascending=True,
        na_position="last",
    )

    return {
        "count": min(len(dataframe), limit),
        "data": dataframe_to_records(
            dataframe.head(limit)
        ),
    }


@app.get(
    "/api/v1/drivers/{driver_number}",
    tags=["Drivers"],
)
def get_driver(driver_number: int):

    dataframe = get_dataset("drivers")
    require_columns(dataframe, {"driver_number"}, "race drivers")

    result = dataframe[
        pd.to_numeric(
            dataframe["driver_number"],
            errors="coerce",
        ) == driver_number
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="Driver not found",
        )

    return dataframe_to_records(result)[0]


# ==================================================
# RACE TEAM PERFORMANCE
# ==================================================

@app.get("/api/v1/teams", tags=["Teams"])
def get_teams():

    dataframe = get_dataset("teams")

    return {
        "count": len(dataframe),
        "data": dataframe_to_records(dataframe),
    }


# ==================================================
# RACE LAP ANALYSIS
# ==================================================

@app.get("/api/v1/laps", tags=["Laps"])
def get_laps():

    dataframe = get_dataset("laps")
    require_columns(dataframe, {"lap_number"}, "race laps")

    dataframe = dataframe.copy()
    dataframe["lap_number"] = pd.to_numeric(
        dataframe["lap_number"],
        errors="coerce",
    )
    dataframe = dataframe.sort_values(
        "lap_number",
        na_position="last",
    )

    return {
        "count": len(dataframe),
        "data": dataframe_to_records(dataframe),
    }


# ==================================================
# SESSION METADATA
# ==================================================

@app.get("/api/v1/session", tags=["Race"])
def get_session():

    return {
        "season": YEAR,
        "meeting_key": MEETING_KEY,
        "session_key": SESSION_KEY,
        "data_source": "OpenF1",
        "storage_layer": "Azure Blob Storage - Gold",
    }


# ==================================================
# SEASON SUMMARY
# ==================================================

@app.get(
    "/api/v1/season/summary",
    tags=["Season Analytics"],
)
def season_summary():

    dataframe = get_season_dataset("summary")

    return require_single_summary_record(dataframe, "Season summary")


# ==================================================
# SEASON DRIVER PERFORMANCE
# ==================================================

@app.get(
    "/api/v1/season/drivers",
    tags=["Season Analytics"],
)
def season_drivers(
    limit: int = Query(default=21, ge=1, le=100),
    min_races: int = Query(default=1, ge=1),
):

    dataframe = get_season_dataset("drivers")

    required_columns = {
        "normalized_pace_rank",
        "average_race_relative_pace_pct",
        "races_with_pace",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Normalized pace fields missing from Gold dataset: "
                f"{sorted(missing)}"
            ),
        )

    dataframe = dataframe.copy()
    dataframe["normalized_pace_rank"] = pd.to_numeric(
        dataframe["normalized_pace_rank"],
        errors="coerce",
    )
    dataframe["average_race_relative_pace_pct"] = pd.to_numeric(
        dataframe["average_race_relative_pace_pct"],
        errors="coerce",
    )
    dataframe["races_with_pace"] = pd.to_numeric(
        dataframe["races_with_pace"],
        errors="coerce",
    )

    # Keep missing country codes as JSON null, not the strings "nan" or "<NA>".
    if "country_code" in dataframe.columns:
        dataframe["country_code"] = dataframe["country_code"].astype("string")
        dataframe["country_code"] = dataframe["country_code"].str.strip()
        dataframe["country_code"] = dataframe["country_code"].replace("", pd.NA)

    dataframe = dataframe[
        dataframe["races_with_pace"].fillna(0) >= min_races
    ].copy()
    dataframe = dataframe.sort_values(
        ["normalized_pace_rank", "average_race_relative_pace_pct"],
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
    )

    return {
        "season": YEAR,
        "metric": "average_race_relative_pace_pct",
        "metric_description": (
            "Average driver median lap pace relative to "
            "the median driver pace in each race. "
            "Negative values indicate faster relative pace."
        ),
        "minimum_races_required": min_races,
        "count": min(len(dataframe), limit),
        "data": dataframe_to_records(
            dataframe.head(limit)
        ),
    }


# ==================================================
# SEASON TEAM PERFORMANCE
# ==================================================

@app.get(
    "/api/v1/season/teams",
    tags=["Season Analytics"],
)
def season_teams(
    limit: int = Query(default=10, ge=1, le=100),
    min_races: int = Query(default=1, ge=1),
):
    """
    Return season teams ordered by race-normalized relative pace.

    Negative average_race_relative_pace_pct values indicate faster
    than the per-race team reference. This is a descriptive pace
    metric, not an official constructors' championship ranking.
    """

    dataframe = get_season_dataset("teams")

    required_columns = {
        "normalized_pace_rank",
        "average_race_relative_pace_pct",
        "median_race_relative_pace_pct",
        "races_with_pace",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Normalized team pace fields missing from Gold dataset: "
                f"{sorted(missing)}. Rerun the Silver-to-Gold pipeline."
            ),
        )

    dataframe = dataframe.copy()
    for column in (
        "normalized_pace_rank",
        "average_race_relative_pace_pct",
        "median_race_relative_pace_pct",
        "races_with_pace",
    ):
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )
    dataframe = dataframe[
        dataframe["races_with_pace"].fillna(0) >= min_races
    ].copy()

    dataframe = dataframe.sort_values(
        ["normalized_pace_rank", "average_race_relative_pace_pct"],
        ascending=[True, True],
        na_position="last",
    )

    return {
        "season": YEAR,
        "metric": "average_race_relative_pace_pct",
        "metric_description": (
            "Average team median lap pace relative to the median team "
            "pace in each race. Negative values indicate faster relative "
            "pace. Race-level percentages are averaged across races."
        ),
        "minimum_races_required": min_races,
        "count": min(len(dataframe), limit),
        "data": dataframe_to_records(dataframe.head(limit)),
    }


# ==================================================
# SEASON RACE SUMMARIES
# ==================================================

@app.get(
    "/api/v1/season/races",
    tags=["Season Analytics"],
)
def season_races(
    limit: int = Query(default=100, ge=1, le=200),
):

    dataframe = get_season_dataset("races")
    require_columns(dataframe, {"date_start"}, "season races")

    dataframe = dataframe.copy()
    dataframe["_sort_date"] = pd.to_datetime(
        dataframe["date_start"],
        errors="coerce",
        utc=True,
    )
    dataframe = dataframe.sort_values(
        "_sort_date",
        ascending=True,
        na_position="last",
        kind="mergesort",
    ).drop(columns=["_sort_date"])

    total_count = len(dataframe)
    result = dataframe.head(limit)

    return {
        "season": YEAR,
        "count": len(result),
        "total_count": total_count,
        "limit": limit,
        "data": dataframe_to_records(result),
    }


# ==================================================
# SEASON DRIVER DETAIL
# ==================================================

@app.get(
    "/api/v1/season/drivers/{driver_number}",
    tags=["Season Analytics"],
)
def season_driver_detail(driver_number: int):

    dataframe = get_season_dataset("drivers")
    require_columns(dataframe, {"driver_number"}, "season drivers")

    result = dataframe[
        pd.to_numeric(
            dataframe["driver_number"],
            errors="coerce",
        ) == driver_number
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="Season driver not found",
        )

    return dataframe_to_records(result)[0]
