
import io
import os
import sys

import numpy as np
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
GOLD = os.getenv("AZURE_STORAGE_CONTAINER_GOLD", "gold")
YEAR = int(os.getenv("OPENF1_YEAR", "2025"))

if not ACCOUNT:
    raise ValueError("AZURE_STORAGE_ACCOUNT is missing from .env")


def read_gold_dataset(container, blob_name):
    blob = container.get_blob_client(blob_name)
    content = blob.download_blob().readall()
    return pd.read_parquet(io.BytesIO(content))


def check(condition, message, issues):
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        issues.append(message)


def check_numeric_validity(df, columns, dataset_name, issues):
    for column in columns:
        if column not in df.columns:
            check(False, f"{dataset_name}: missing {column}", issues)
            continue

        values = pd.to_numeric(df[column], errors="coerce")

        check(
            values.notna().all(),
            f"{dataset_name}: {column} contains no null/non-numeric values",
            issues,
        )

        check(
            np.isfinite(values.dropna()).all(),
            f"{dataset_name}: {column} contains only finite values",
            issues,
        )


def main():
    credential = DefaultAzureCredential()
    service = BlobServiceClient(
        account_url=f"https://{ACCOUNT}.blob.core.windows.net",
        credential=credential,
    )

    issues = []

    try:
        container = service.get_container_client(GOLD)
        prefix = f"openf1/year={YEAR}/season"

        print("=" * 65)
        print(f"APEXF1 GOLD DATA QUALITY AUDIT | YEAR {YEAR}")
        print("=" * 65)

        datasets = {
            "drivers": f"{prefix}/season_driver_performance.parquet",
            "teams": f"{prefix}/season_team_performance.parquet",
            "races": f"{prefix}/race_summary.parquet",
            "summary": f"{prefix}/season_summary.parquet",
        }

        data = {}

        for name, blob_name in datasets.items():
            try:
                data[name] = read_gold_dataset(container, blob_name)
                print(
                    f"[READ] {name}: "
                    f"{len(data[name])} rows, "
                    f"{len(data[name].columns)} columns"
                )
            except Exception as exc:
                issues.append(f"Could not read {name}: {exc}")
                print(f"[FAIL] Could not read {name}: {exc}")

        if issues:
            return issues

        drivers = data["drivers"]
        teams = data["teams"]
        races = data["races"]
        summary = data["summary"]

        print("\n--- 1. Dataset structure ---")

        for name, df in data.items():
            check(not df.empty, f"{name} dataset is not empty", issues)
            check(
                not df.columns.duplicated().any(),
                f"{name} has no duplicate column names",
                issues,
            )

        check(
            len(summary) == 1,
            "Season summary contains exactly one row",
            issues,
        )

        print("\n--- 2. Driver data quality ---")

        driver_required = [
            "driver_number",
            "normalized_pace_rank",
            "race_count",
            "races_with_pace",
            "valid_lap_count",
            "average_race_relative_pace_pct",
            "fastest_lap_seconds",
        ]

        for column in driver_required:
            check(
                column in drivers.columns,
                f"Driver dataset contains {column}",
                issues,
            )

        if all(column in drivers.columns for column in driver_required):
            check(
                drivers["driver_number"].notna().all(),
                "Driver numbers are not null",
                issues,
            )

            check(
                drivers["driver_number"].is_unique,
                "Driver numbers are unique",
                issues,
            )

            check(
                drivers["normalized_pace_rank"].is_unique,
                "Driver normalized pace ranks are unique",
                issues,
            )

            check(
                sorted(drivers["normalized_pace_rank"].tolist())
                == list(range(1, len(drivers) + 1)),
                "Driver ranks are continuous from 1 to N",
                issues,
            )

            check(
                (
                    drivers["races_with_pace"]
                    <= drivers["race_count"]
                ).all(),
                "Driver pace coverage does not exceed race count",
                issues,
            )

            check(
                (drivers["valid_lap_count"] > 0).all(),
                "Every driver has valid lap records",
                issues,
            )

            check(
                (drivers["fastest_lap_seconds"] > 0).all(),
                "Driver fastest lap times are positive",
                issues,
            )

            check_numeric_validity(
                drivers,
                [
                    "average_race_relative_pace_pct",
                    "median_race_relative_pace_pct",
                    "average_race_median_lap_seconds",
                ],
                "Driver dataset",
                issues,
            )

        print("\n--- 3. Team data quality ---")

        team_required = [
            "team_name",
            "normalized_pace_rank",
            "race_count",
            "races_with_pace",
            "valid_lap_count",
            "average_race_relative_pace_pct",
            "fastest_lap_seconds",
        ]

        for column in team_required:
            check(
                column in teams.columns,
                f"Team dataset contains {column}",
                issues,
            )

        if all(column in teams.columns for column in team_required):
            check(
                teams["team_name"].notna().all(),
                "Team names are not null",
                issues,
            )

            check(
                teams["team_name"].is_unique,
                "Team names are unique",
                issues,
            )

            check(
                teams["normalized_pace_rank"].is_unique,
                "Team normalized pace ranks are unique",
                issues,
            )

            check(
                sorted(teams["normalized_pace_rank"].tolist())
                == list(range(1, len(teams) + 1)),
                "Team ranks are continuous from 1 to N",
                issues,
            )

            check(
                (
                    teams["races_with_pace"]
                    <= teams["race_count"]
                ).all(),
                "Team pace coverage does not exceed race count",
                issues,
            )

            check(
                (teams["valid_lap_count"] > 0).all(),
                "Every team has valid lap records",
                issues,
            )

            check(
                (teams["fastest_lap_seconds"] > 0).all(),
                "Team fastest lap times are positive",
                issues,
            )

            check_numeric_validity(
                teams,
                [
                    "average_race_relative_pace_pct",
                    "median_race_relative_pace_pct",
                    "average_race_median_lap_seconds",
                ],
                "Team dataset",
                issues,
            )

        print("\n--- 4. Race data quality ---")

        race_required = [
            "meeting_key",
            "session_key",
            "driver_count",
            "lap_records_received",
            "valid_lap_time_records",
            "fastest_lap_seconds",
        ]

        for column in race_required:
            check(
                column in races.columns,
                f"Race dataset contains {column}",
                issues,
            )

        if all(column in races.columns for column in race_required):
            check(
                races["session_key"].is_unique,
                "Race session keys are unique",
                issues,
            )

            check(
                (races["valid_lap_time_records"] > 0).all(),
                "Every race has valid lap records",
                issues,
            )

            check(
                (
                    races["valid_lap_time_records"]
                    <= races["lap_records_received"]
                ).all(),
                "Valid lap records do not exceed received lap records",
                issues,
            )

            check(
                (races["fastest_lap_seconds"] > 0).all(),
                "Race fastest lap times are positive",
                issues,
            )

        print("\n--- 5. Season consistency ---")

        if len(summary) == 1:
            season = summary.iloc[0]

            consistency_checks = {
                "race_count": races["session_key"].nunique(),
                "driver_count": drivers["driver_number"].nunique(),
                "team_count": teams["team_name"].nunique(),
            }

            for column, expected in consistency_checks.items():
                if column in summary.columns:
                    actual = int(season[column])
                    check(
                        actual == expected,
                        f"Season {column}: summary={actual}, datasets={expected}",
                        issues,
                    )
                else:
                    check(
                        False,
                        f"Season summary is missing {column}",
                        issues,
                    )

        print("\n--- AUDIT RESULT ---")

        if issues:
            print(f"[FAIL] {len(issues)} issue(s) found.")
            for issue in issues:
                print(f" - {issue}")
        else:
            print("[PASS] All Gold data quality checks passed.")

        print("\n--- Dataset overview ---")
        print(f"Drivers: {len(drivers)}")
        print(f"Teams: {len(teams)}")
        print(f"Races: {len(races)}")

        return issues

    finally:
        service.close()
        credential.close()


if __name__ == "__main__":
    problems = main()
    sys.exit(1 if problems else 0)
