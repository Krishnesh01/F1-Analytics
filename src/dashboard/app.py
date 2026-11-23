
import os
import logging

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("apexf1.dashboard")

API_BASE_URL = os.getenv(
    "APEXF1_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

st.set_page_config(
    page_title="ApexF1 | Race Intelligence",
    page_icon="🏎️",
    layout="wide",
)


# --------------------------------------------------
# API CLIENT
# --------------------------------------------------

def api_get(endpoint):
    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        logger.exception("API request failed: %s", endpoint)
        raise RuntimeError(
            f"Could not retrieve data from {endpoint}. "
            "Check whether FastAPI is running."
        ) from exc


# --------------------------------------------------
# LOAD DASHBOARD DATA
# --------------------------------------------------

@st.cache_data(ttl=60)
def load_dashboard_data():

    # Race-level datasets
    race_summary = api_get("/api/v1/race/summary")
    race_drivers_response = api_get("/api/v1/drivers?limit=100")
    race_teams_response = api_get("/api/v1/teams")
    race_laps_response = api_get("/api/v1/laps")
    session = api_get("/api/v1/session")

    # Season-level datasets
    season_summary = api_get("/api/v1/season/summary")
    season_drivers_response = api_get(
        "/api/v1/season/drivers?limit=100"
    )
    season_teams_response = api_get("/api/v1/season/teams")
    season_races_response = api_get(
        "/api/v1/season/races?limit=100"
    )

    return {
        "race_summary": race_summary,
        "race_drivers": pd.DataFrame(
            race_drivers_response.get("data", [])
        ),
        "race_teams": pd.DataFrame(
            race_teams_response.get("data", [])
        ),
        "race_laps": pd.DataFrame(
            race_laps_response.get("data", [])
        ),
        "session": session,
        "season_summary": season_summary,
        "season_drivers": pd.DataFrame(
            season_drivers_response.get("data", [])
        ),
        "season_teams": pd.DataFrame(
            season_teams_response.get("data", [])
        ),
        "season_races": pd.DataFrame(
            season_races_response.get("data", [])
        ),
    }


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def format_seconds(value):

    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"

        value = float(value)

        if value <= 0:
            return "N/A"

        minutes = int(value // 60)
        seconds = value % 60

        return f"{minutes}:{seconds:06.3f}"

    except (TypeError, ValueError):
        return "N/A"


def get_value(data, possible_keys, default=0):
    """Return the first available value from a dictionary."""

    for key in possible_keys:
        value = data.get(key)

        if value is not None:
            return value

    return default


def first_existing_column(dataframe, candidates):
    """Find the first available column from a candidate list."""

    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def get_driver_label(dataframe):
    if "name_acronym" in dataframe.columns:
        return dataframe["name_acronym"].fillna(
            dataframe["driver_number"].astype(str)
        )

    if "full_name" in dataframe.columns:
        return dataframe["full_name"].fillna(
            dataframe["driver_number"].astype(str)
        )

    return dataframe["driver_number"].astype(str)




def clean_display_dataframe(dataframe):
    """Replace missing values with readable labels in dashboard tables."""
    display_data = dataframe.copy()
    return display_data.astype(object).where(
        pd.notna(display_data), "N/A"
    )


def race_display_label(dataframe):
    """Create readable race labels using only fields returned by the API."""
    for column in ["meeting_name", "location", "country_name"]:
        if column in dataframe.columns:
            values = dataframe[column].fillna("").astype(str).str.strip()
            if values.ne("").any():
                return values.where(values.ne(""), "Unknown race")

    if "meeting_key" in dataframe.columns:
        return "Meeting " + dataframe["meeting_key"].astype(str)

    return pd.Series(
        [f"Race {index + 1}" for index in range(len(dataframe))],
        index=dataframe.index,
    )


def team_color_map(drivers_dataframe):
    """Build team colors from driver metadata when available."""
    if not {"team_name", "team_colour"}.issubset(drivers_dataframe.columns):
        return {}

    colors = {}
    for _, row in drivers_dataframe[["team_name", "team_colour"]].dropna().iterrows():
        team = str(row["team_name"]).strip()
        color = str(row["team_colour"]).strip().lstrip("#")
        if team and color and color.lower() != "none":
            colors[team] = f"#{color}"
    return colors


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏎️ ApexF1")
st.subheader("Formula 1 Race Intelligence & Predictive Analytics")

st.markdown(
    """
    Explore historical Formula 1 race and season analytics,
    driver performance, team comparisons and lap-level trends.
    """
)

st.caption(
    "Data platform: Azure Blob Storage | "
    "Backend: FastAPI | Dashboard: Streamlit"
)

if st.button("🔄 Refresh dashboard"):
    st.cache_data.clear()
    st.rerun()

st.divider()


# --------------------------------------------------
# FETCH DATA
# --------------------------------------------------

try:
    with st.spinner("Fetching race and season analytics..."):
        data = load_dashboard_data()

except Exception:
    logger.exception("Dashboard startup data load failed")
    st.error(
        "Unable to load dashboard data. "
        "Check that the FastAPI service is running and reachable."
    )
    st.caption(f"Configured API endpoint: {API_BASE_URL}")
    st.stop()


race_summary = data["race_summary"]
race_drivers = data["race_drivers"]
race_teams = data["race_teams"]
race_laps = data["race_laps"]
session = data["session"]

season_summary = data["season_summary"]
season_drivers = data["season_drivers"]
season_teams = data["season_teams"]
season_races = data["season_races"]


# ==================================================
# SEASON ANALYTICS
# ==================================================

st.header("🌍 Season Analytics")

st.caption(
    f"Historical Formula 1 season: {session.get('season', 'N/A')}"
)

if season_summary:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Race Sessions",
        get_value(
            season_summary,
            ["race_count", "session_count"],
            len(season_races),
        ),
    )

    col2.metric(
        "Drivers",
        get_value(
            season_summary,
            ["driver_count", "total_drivers"],
            len(season_drivers),
        ),
    )

    col3.metric(
        "Teams",
        get_value(
            season_summary,
            ["team_count", "total_teams"],
            len(season_teams),
        ),
    )

    col4.metric(
    "Valid Lap Records",
    f"{int(season_summary.get('valid_lap_count', 0)):,}",
)

else:
    st.warning("Season summary is unavailable.")


# --------------------------------------------------
# SEASON DRIVER ANALYSIS — NORMALIZED PACE
# --------------------------------------------------

st.subheader("🏁 Season Driver Performance")

if not season_drivers.empty:
    pace_column = "average_race_relative_pace_pct"
    rank_column = "normalized_pace_rank"
    coverage_column = "races_with_pace"

    required_pace_columns = {pace_column, rank_column, coverage_column}
    if required_pace_columns.issubset(season_drivers.columns):
        pace_data = season_drivers.copy()

        for column in [pace_column, rank_column, coverage_column, "valid_lap_count"]:
            if column in pace_data.columns:
                pace_data[column] = pd.to_numeric(
                    pace_data[column], errors="coerce"
                )

        pace_data["driver_label"] = get_driver_label(pace_data)
        pace_data = pace_data.dropna(
            subset=[pace_column, rank_column, coverage_column]
        )

        max_races = int(pace_data[coverage_column].max()) if not pace_data.empty else 1
        min_races = min(10, max_races)
        selected_min_races = st.slider(
            "Minimum races required for the pace leaderboard",
            min_value=1,
            max_value=max(1, max_races),
            value=min_races,
            help=(
                "Drivers with fewer races than this threshold are hidden from "
                "the ranking chart and leaderboard. Lower the threshold to "
                "include drivers with partial-season data."
            ),
        )

        ranked_drivers = pace_data[
            pace_data[coverage_column] >= selected_min_races
        ].sort_values(rank_column, ascending=True)

        if ranked_drivers.empty:
            st.warning("No drivers meet the selected race-coverage threshold.")
        else:
            st.caption(
                "Normalized pace compares each driver's median lap time with "
                "the median driver pace in each race, then averages the "
                "race-relative percentages across races."
            )

            metric_cols = st.columns(3)
            metric_cols[0].metric("Ranked drivers", len(ranked_drivers))
            metric_cols[1].metric(
                "Minimum race coverage", f"{selected_min_races} races"
            )
            metric_cols[2].metric(
                "Season races available", f"{max_races} races"
            )

            display_columns = [
                column for column in [
                    rank_column,
                    "full_name",
                    "name_acronym",
                    "country_code",
                    pace_column,
                    "median_race_relative_pace_pct",
                    coverage_column,
                    "race_count",
                    "valid_lap_count",
                    "teams_represented",
                ] if column in ranked_drivers.columns
            ]

            leaderboard = ranked_drivers[display_columns].copy()
            if "country_code" in leaderboard.columns:
                leaderboard["country_code"] = (
                    leaderboard["country_code"]
                    .fillna("Not available")
                    .replace(r"^\s*$", "Not available", regex=True)
                )
            leaderboard = leaderboard.rename(columns={
                rank_column: "Pace rank",
                "full_name": "Driver",
                "name_acronym": "Code",
                "country_code": "Country code",
                pace_column: "Average relative pace (%)",
                "median_race_relative_pace_pct": "Median relative pace (%)",
                coverage_column: "Races with pace data",
                "race_count": "Race sessions",
                "valid_lap_count": "Valid laps",
                "teams_represented": "Teams represented",
            })

            st.dataframe(
                leaderboard,
                width="stretch",
                hide_index=True,
                column_config={
                    "Pace rank": st.column_config.NumberColumn(format="%d"),
                    "Average relative pace (%)": st.column_config.NumberColumn(
                        format="%.3f%%"
                    ),
                    "Median relative pace (%)": st.column_config.NumberColumn(
                        format="%.3f%%"
                    ),
                    "Races with pace data": st.column_config.NumberColumn(
                        format="%d"
                    ),
                    "Valid laps": st.column_config.NumberColumn(format="%d"),
                },
            )

            pace_chart = ranked_drivers.sort_values(
                pace_column, ascending=True
            ).copy()
            pace_chart["pace_category"] = pace_chart[pace_column].apply(
                lambda value: (
                    "Faster than reference" if value < 0
                    else "Slower than reference" if value > 0
                    else "At reference"
                )
            )

            hover_fields = {
                "full_name": True,
                coverage_column: True,
                "valid_lap_count": True,
                "median_race_relative_pace_pct": True,
                pace_column: True,
                "pace_category": True,
            }
            hover_fields = {
                column: include
                for column, include in hover_fields.items()
                if column in pace_chart.columns
            }

            fig = px.bar(
                pace_chart,
                x="driver_label",
                y=pace_column,
                color="pace_category",
                color_discrete_map={
                    "Faster than reference": "#16A34A",
                    "Slower than reference": "#DC2626",
                    "At reference": "#64748B",
                },
                category_orders={
                    "pace_category": [
                        "Faster than reference",
                        "At reference",
                        "Slower than reference",
                    ]
                },
                hover_data=hover_fields,
                title="Normalized Season Driver Pace",
                labels={
                    "driver_label": "Driver",
                    pace_column: "Average race-relative pace (%)",
                    "pace_category": "Classification",
                    "full_name": "Driver name",
                    coverage_column: "Races with pace data",
                    "valid_lap_count": "Valid laps",
                    "median_race_relative_pace_pct": "Median relative pace (%)",
                },
            )
            fig.update_layout(
                xaxis_title="Driver",
                yaxis_title="Average race-relative pace (%)",
                yaxis_ticksuffix="%",
                hovermode="closest",
                legend_title_text="Compared with race reference",
                bargap=0.22,
                height=430,
                margin=dict(b=90),
            )
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="#475569",
                line_width=1.5,
                annotation_text="Race reference (0%)",
                annotation_position="top left",
            )
            # Keep Plotly Express' generated hover labels so the
            # pace_category column is shown reliably for every bar.
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, width="stretch")

            st.info(
                "Interpretation: negative values indicate faster-than-reference "
                "pace; positive values indicate slower-than-reference pace. "
                "This is a descriptive lap-pace metric, not official championship "
                "standings. It does not fully adjust for tyre compounds, fuel, "
                "traffic, weather, or race strategy."
            )

        with st.expander("View all raw season driver metrics"):
            st.dataframe(
                clean_display_dataframe(season_drivers),
                width="stretch",
                hide_index=True,
            )
    else:
        st.warning(
            "The season driver API response does not contain all normalized "
            "pace fields. Expected: normalized_pace_rank, "
            "average_race_relative_pace_pct, and races_with_pace."
        )
        st.dataframe(
            season_drivers,
            width="stretch",
            hide_index=True,
        )
else:
    st.warning("Season driver data is unavailable.")

st.divider()


# --------------------------------------------------
# SEASON TEAM ANALYSIS — NORMALIZED PACE
# --------------------------------------------------

st.subheader("🏆 Season Team Performance")

if not season_teams.empty:
    team_pace_column = "average_race_relative_pace_pct"
    team_rank_column = "normalized_pace_rank"
    team_coverage_column = "races_with_pace"

    required_team_columns = {
        "team_name",
        team_pace_column,
        team_rank_column,
        team_coverage_column,
    }

    if required_team_columns.issubset(season_teams.columns):
        team_pace_data = season_teams.copy()

        numeric_columns = [
            team_pace_column,
            team_rank_column,
            team_coverage_column,
            "median_race_relative_pace_pct",
            "valid_lap_count",
            "driver_count",
            "race_count",
            "total_pace_sample_laps",
            "average_race_median_lap_seconds",
        ]

        for column in numeric_columns:
            if column in team_pace_data.columns:
                team_pace_data[column] = pd.to_numeric(
                    team_pace_data[column], errors="coerce"
                )

        team_pace_data = team_pace_data.dropna(
            subset=[
                "team_name",
                team_pace_column,
                team_rank_column,
                team_coverage_column,
            ]
        )

        max_team_races = (
            int(team_pace_data[team_coverage_column].max())
            if not team_pace_data.empty
            else 1
        )

        selected_team_min_races = st.slider(
            "Minimum races required for the team pace ranking",
            min_value=1,
            max_value=max(1, max_team_races),
            value=min(20, max(1, max_team_races)),
            key="team_min_race_coverage",
            help=(
                "Teams with fewer races contributing to normalized pace "
                "than this threshold are excluded from the ranking."
            ),
        )

        ranked_teams = team_pace_data[
            team_pace_data[team_coverage_column] >= selected_team_min_races
        ].sort_values(team_rank_column, ascending=True)

        if ranked_teams.empty:
            st.warning(
                "No teams meet the selected race-coverage threshold."
            )
        else:
            st.caption(
                "Team pace is calculated by comparing each team's median "
                "lap time with the median team pace within each race. "
                "The race-relative percentages are then averaged across "
                "races, giving each race equal weight."
            )

            team_metric_cols = st.columns(3)
            team_metric_cols[0].metric(
                "Ranked teams", len(ranked_teams)
            )
            team_metric_cols[1].metric(
                "Minimum race coverage",
                f"{selected_team_min_races} races",
            )
            team_metric_cols[2].metric(
                "Season races available",
                f"{max_team_races} races",
            )

            team_display_columns = [
                column for column in [
                    team_rank_column,
                    "team_name",
                    team_pace_column,
                    "median_race_relative_pace_pct",
                    team_coverage_column,
                    "race_count",
                    "driver_count",
                    "valid_lap_count",
                    "total_pace_sample_laps",
                ]
                if column in ranked_teams.columns
            ]

            team_leaderboard = ranked_teams[
                team_display_columns
            ].copy()

            team_leaderboard = team_leaderboard.rename(columns={
                team_rank_column: "Pace rank",
                "team_name": "Team",
                team_pace_column: "Average relative pace (%)",
                "median_race_relative_pace_pct": "Median relative pace (%)",
                team_coverage_column: "Races with pace data",
                "race_count": "Race sessions",
                "driver_count": "Drivers represented",
                "valid_lap_count": "Valid laps",
                "total_pace_sample_laps": "Pace sample laps",
            })

            st.dataframe(
                team_leaderboard,
                width="stretch",
                hide_index=True,
                column_config={
                    "Pace rank": st.column_config.NumberColumn(format="%d"),
                    "Average relative pace (%)": st.column_config.NumberColumn(
                        format="%.3f%%"
                    ),
                    "Median relative pace (%)": st.column_config.NumberColumn(
                        format="%.3f%%"
                    ),
                    "Races with pace data": st.column_config.NumberColumn(
                        format="%d"
                    ),
                    "Valid laps": st.column_config.NumberColumn(format="%d"),
                    "Pace sample laps": st.column_config.NumberColumn(
                        format="%d"
                    ),
                },
            )

            team_chart = ranked_teams.sort_values(
                team_pace_column, ascending=True
            ).copy()

            team_chart["pace_category"] = team_chart[
                team_pace_column
            ].apply(
                lambda value: (
                    "Faster than reference" if value < 0
                    else "Slower than reference" if value > 0
                    else "At reference"
                )
            )

            team_hover_data = {
                "team_name": True,
                team_pace_column: True,
                "median_race_relative_pace_pct": True,
                team_coverage_column: True,
                "valid_lap_count": True,
                "driver_count": True,
                "pace_category": True,
            }
            team_hover_data = {
                column: include
                for column, include in team_hover_data.items()
                if column in team_chart.columns
            }

            team_fig = px.bar(
                team_chart,
                x="team_name",
                y=team_pace_column,
                color="pace_category",
                color_discrete_map={
                    "Faster than reference": "#16A34A",
                    "Slower than reference": "#DC2626",
                    "At reference": "#64748B",
                },
                category_orders={
                    "pace_category": [
                        "Faster than reference",
                        "At reference",
                        "Slower than reference",
                    ]
                },
                hover_data=team_hover_data,
                title="Normalized Season Team Pace",
                labels={
                    "team_name": "Team",
                    team_pace_column: "Average race-relative pace (%)",
                    "pace_category": "Classification",
                    "median_race_relative_pace_pct": "Median relative pace (%)",
                    team_coverage_column: "Races with pace data",
                    "valid_lap_count": "Valid laps",
                    "driver_count": "Drivers represented",
                },
            )

            team_fig.update_layout(
                xaxis_title="Team",
                yaxis_title="Average race-relative pace (%)",
                yaxis_ticksuffix="%",
                hovermode="closest",
                legend_title_text="Compared with race reference",
                bargap=0.22,
                height=430,
                margin=dict(b=90),
            )

            team_fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="#475569",
                line_width=1.5,
                annotation_text="Race reference (0%)",
                annotation_position="top left",
            )

            team_fig.update_traces(marker_line_width=0)
            st.plotly_chart(team_fig, width="stretch")

            st.info(
                "Interpretation: negative values indicate faster-than-reference "
                "team pace; positive values indicate slower-than-reference pace. "
                "This is a descriptive lap-pace metric, not an official "
                "Constructors' Championship ranking. It does not fully adjust "
                "for tyre compounds, fuel loads, traffic, weather, or strategy."
            )

        with st.expander("View raw season team metrics"):
            st.dataframe(
                clean_display_dataframe(season_teams),
                width="stretch",
                hide_index=True,
            )

    else:
        st.warning(
            "The season team API response does not contain all normalized "
            "pace fields. Expected: normalized_pace_rank, "
            "average_race_relative_pace_pct, races_with_pace, and team_name."
        )
        st.dataframe(
            season_teams,
            width="stretch",
            hide_index=True,
        )
else:
    st.warning("Season team data is unavailable.")

st.divider()


# --------------------------------------------------
# SEASON RACE ANALYSIS
# --------------------------------------------------

st.subheader("🏎️ Race-by-Race Season Analysis")

if not season_races.empty:
    race_analysis = season_races.copy()

    # Convert numeric fields defensively; missing optional columns are allowed.
    numeric_race_columns = [
        "driver_count",
        "lap_records_received",
        "valid_lap_time_records",
        "fastest_lap_seconds",
        "average_lap_seconds",
    ]
    for column in numeric_race_columns:
        if column in race_analysis.columns:
            race_analysis[column] = pd.to_numeric(
                race_analysis[column], errors="coerce"
            )

    if {"lap_records_received", "valid_lap_time_records"}.issubset(
        race_analysis.columns
    ):
        denominator = race_analysis["lap_records_received"].where(
            race_analysis["lap_records_received"] > 0
        )
        race_analysis["valid_lap_percentage"] = (
            race_analysis["valid_lap_time_records"] / denominator * 100
        )

    race_analysis["race_label"] = race_display_label(race_analysis)

    if "date_start" in race_analysis.columns:
        race_analysis["race_date"] = pd.to_datetime(
            race_analysis["date_start"], errors="coerce", utc=True
        ).dt.strftime("%d %b %Y")
        race_analysis["race_date"] = race_analysis["race_date"].fillna("N/A")
    else:
        race_analysis["race_date"] = "N/A"

    metric_columns = st.columns(4)
    metric_columns[0].metric("Race sessions", len(race_analysis))
    if "lap_records_received" in race_analysis.columns:
        metric_columns[1].metric(
            "Lap records received",
            f"{int(race_analysis['lap_records_received'].fillna(0).sum()):,}",
        )
    if "valid_lap_time_records" in race_analysis.columns:
        metric_columns[2].metric(
            "Valid lap-time records",
            f"{int(race_analysis['valid_lap_time_records'].fillna(0).sum()):,}",
        )
    if "valid_lap_percentage" in race_analysis.columns:
        total_received = race_analysis["lap_records_received"].fillna(0).sum()
        total_valid = race_analysis["valid_lap_time_records"].fillna(0).sum()
        overall_valid_pct = (
            total_valid / total_received * 100 if total_received else 0
        )
        metric_columns[3].metric("Overall valid-lap rate", f"{overall_valid_pct:.1f}%")

    st.caption(
        "Race-level quality metrics show how many received lap records contain "
        "a usable lap time. Lap-time values are circuit-dependent and should not "
        "be interpreted as direct performance rankings across different tracks."
    )

    # Curated table: retain technical identifiers in an optional raw-data view.
    preferred_columns = [
        "race_label",
        "race_date",
        "driver_count",
        "lap_records_received",
        "valid_lap_time_records",
        "valid_lap_percentage",
        "fastest_lap_seconds",
        "average_lap_seconds",
    ]
    visible_columns = [
        column for column in preferred_columns if column in race_analysis.columns
    ]
    display_races = race_analysis[visible_columns].rename(
        columns={
            "race_label": "Race",
            "race_date": "Date",
            "driver_count": "Drivers",
            "lap_records_received": "Lap records received",
            "valid_lap_time_records": "Valid lap-time records",
            "valid_lap_percentage": "Valid lap records (%)",
            "fastest_lap_seconds": "Fastest lap (seconds)",
            "average_lap_seconds": "Raw average lap (seconds)",
        }
    )
    st.dataframe(
        display_races,
        width="stretch",
        hide_index=True,
        column_config={
            "Valid lap records (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Fastest lap (seconds)": st.column_config.NumberColumn(format="%.3f"),
            "Raw average lap (seconds)": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    chart_col1, chart_col2 = st.columns(2)

    if "valid_lap_percentage" in race_analysis.columns:
        quality_data = race_analysis.dropna(subset=["valid_lap_percentage"])
        with chart_col1:
            if not quality_data.empty:
                quality_fig = px.bar(
                    quality_data.sort_values("valid_lap_percentage"),
                    x="race_label",
                    y="valid_lap_percentage",
                    hover_data=[
                        column for column in [
                            "race_date",
                            "lap_records_received",
                            "valid_lap_time_records",
                        ] if column in quality_data.columns
                    ],
                    title="Valid Lap-Time Records by Race",
                    labels={
                        "race_label": "Race",
                        "valid_lap_percentage": "Valid lap records (%)",
                        "race_date": "Race date",
                        "lap_records_received": "Lap records received",
                        "valid_lap_time_records": "Valid lap-time records",
                    },
                )
                quality_fig.update_layout(
                    yaxis_title="Valid lap records (%)",
                    yaxis_ticksuffix="%",
                    xaxis_title="Race",
                    xaxis_tickangle=-45,
                    height=420,
                    margin=dict(b=100),
                )
                st.plotly_chart(quality_fig, width="stretch")

    if "driver_count" in race_analysis.columns:
        with chart_col2:
            driver_data = race_analysis.dropna(subset=["driver_count"])
            if not driver_data.empty:
                driver_fig = px.bar(
                    driver_data.sort_values("driver_count"),
                    x="race_label",
                    y="driver_count",
                    hover_data=[
                        column for column in ["race_date", "meeting_key", "session_key"]
                        if column in driver_data.columns
                    ],
                    title="Driver Participation by Race",
                    labels={
                        "race_label": "Race",
                        "driver_count": "Drivers represented",
                        "race_date": "Race date",
                    },
                )
                driver_fig.update_layout(
                    xaxis_title="Race",
                    yaxis_title="Drivers represented",
                    xaxis_tickangle=-45,
                    height=420,
                    margin=dict(b=100),
                )
                st.plotly_chart(driver_fig, width="stretch")

    with st.expander("View raw season race data"):
        st.dataframe(
            clean_display_dataframe(season_races),
            width="stretch",
            hide_index=True,
        )

else:
    st.warning("Season race summaries are unavailable.")


# ==================================================
# INDIVIDUAL RACE ANALYTICS
# ==================================================

st.header("🏁 Individual Race Analytics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Drivers",
    race_summary.get("driver_count", len(race_drivers)),
)

col2.metric(
    "Valid Lap Records",
    race_summary.get("valid_lap_time_records", 0),
)

col3.metric(
    "Fastest Lap",
    format_seconds(
        race_summary.get("fastest_lap_seconds")
    ),
)

col4.metric(
    "Average Lap Time",
    format_seconds(
        race_summary.get("average_lap_seconds")
    ),
)

st.caption(
    f"Season: {session.get('season', 'N/A')} | "
    f"Meeting: {session.get('meeting_key', 'N/A')} | "
    f"Session: {session.get('session_key', 'N/A')}"
)

st.divider()


# --------------------------------------------------
# RACE DRIVER PERFORMANCE
# --------------------------------------------------

st.subheader("🏁 Race Driver Performance")

if not race_drivers.empty:
    leaderboard = race_drivers.copy()

    if "fastest_lap_seconds" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(
            "fastest_lap_seconds", ascending=True
        )

    # Make missing metadata explicit rather than displaying Python None values.
    if "country_code" in leaderboard.columns:
        leaderboard["country_code"] = leaderboard["country_code"].fillna("N/A")

    st.dataframe(
        clean_display_dataframe(leaderboard),
        width="stretch",
        hide_index=True,
    )

    if {"driver_number", "average_lap_seconds"}.issubset(leaderboard.columns):
        leaderboard["driver_label"] = get_driver_label(leaderboard)
        hover_columns = [
            column for column in [
                "full_name", "team_name", "valid_lap_count",
                "median_lap_seconds", "fastest_lap_seconds",
            ] if column in leaderboard.columns
        ]
        fig = px.bar(
            leaderboard.sort_values("average_lap_seconds", ascending=True),
            x="driver_label",
            y="average_lap_seconds",
            color="team_name" if "team_name" in leaderboard.columns else None,
            hover_data=hover_columns,
            title="Average Lap Time by Driver",
            labels={
                "driver_label": "Driver",
                "average_lap_seconds": "Average lap time (seconds)",
                "team_name": "Team",
            },
        )
        fig.update_layout(
            xaxis_title="Driver",
            yaxis_title="Average lap time (seconds)",
            hovermode="closest",
        )
        st.plotly_chart(fig, width="stretch")

else:
    st.warning("Race driver data is unavailable.")


# --------------------------------------------------
# RACE TEAM PERFORMANCE
# --------------------------------------------------

st.subheader("🏆 Race Team Performance")

if not race_teams.empty:
    team_avg_column = first_existing_column(
        race_teams,
        ["average_driver_lap_seconds", "average_lap_seconds"],
    )

    if team_avg_column and "team_name" in race_teams.columns:
        team_chart_data = race_teams.copy()
        team_chart_data[team_avg_column] = pd.to_numeric(
            team_chart_data[team_avg_column], errors="coerce"
        )
        team_chart_data = team_chart_data.dropna(subset=[team_avg_column])

        colors = team_color_map(race_drivers)
        fig = px.bar(
            team_chart_data.sort_values(team_avg_column, ascending=True),
            x="team_name",
            y=team_avg_column,
            color="team_name",
            color_discrete_map=colors if colors else None,
            hover_data=[
                column for column in [
                    "driver_count", "valid_lap_count",
                    "fastest_lap_seconds", "median_lap_seconds",
                ] if column in team_chart_data.columns
            ],
            title="Team Average Lap Time",
            labels={
                "team_name": "Team",
                team_avg_column: "Average lap time (seconds)",
                "driver_count": "Drivers",
                "valid_lap_count": "Valid laps",
                "fastest_lap_seconds": "Fastest lap (seconds)",
                "median_lap_seconds": "Median lap (seconds)",
            },
        )
        fig.update_layout(
            xaxis_title="Team",
            yaxis_title="Average lap time (seconds)",
            showlegend=False,
            hovermode="closest",
        )
        st.plotly_chart(fig, width="stretch")

    st.dataframe(
        clean_display_dataframe(race_teams),
        width="stretch",
        hide_index=True,
    )
else:
    st.warning("Race team data is unavailable.")


# --------------------------------------------------
# RACE LAP ANALYSIS
# --------------------------------------------------

st.subheader("📈 Lap-by-Lap Analysis")

if not race_laps.empty:
    lap_chart_data = race_laps.copy()
    for column in [
        "lap_number", "average_lap_seconds",
        "fastest_lap_seconds", "slowest_lap_seconds",
    ]:
        if column in lap_chart_data.columns:
            lap_chart_data[column] = pd.to_numeric(
                lap_chart_data[column], errors="coerce"
            )

    if {"lap_number", "average_lap_seconds"}.issubset(lap_chart_data.columns):
        lap_chart_data = lap_chart_data.sort_values("lap_number")
        trend_columns = [
            column for column in [
                "average_lap_seconds",
                "fastest_lap_seconds",
                "slowest_lap_seconds",
            ] if column in lap_chart_data.columns
        ]
        trend_data = lap_chart_data.melt(
            id_vars=["lap_number"],
            value_vars=trend_columns,
            var_name="lap_metric",
            value_name="lap_time_seconds",
        ).dropna(subset=["lap_time_seconds"])
        metric_names = {
            "average_lap_seconds": "Average",
            "fastest_lap_seconds": "Fastest",
            "slowest_lap_seconds": "Slowest",
        }
        trend_data["lap_metric"] = trend_data["lap_metric"].map(metric_names)

        fig = px.line(
            trend_data,
            x="lap_number",
            y="lap_time_seconds",
            color="lap_metric",
            markers=True,
            title="Lap-Time Trend Across the Race",
            labels={
                "lap_number": "Lap number",
                "lap_time_seconds": "Lap time (seconds)",
                "lap_metric": "Lap metric",
            },
        )
        fig.update_layout(
            xaxis_title="Lap number",
            yaxis_title="Lap time (seconds)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "The first lap can be slower because of race-start conditions. "
            "It is retained in the data; interpret it in its race context."
        )

    st.dataframe(
        clean_display_dataframe(lap_chart_data),
        width="stretch",
        hide_index=True,
    )
else:
    st.warning("Race lap data is unavailable.")


# --------------------------------------------------
# RACE DRIVER COMPARISON
# --------------------------------------------------

st.subheader("🔍 Compare Race Drivers")

if not race_drivers.empty and "driver_number" in race_drivers.columns:
    driver_options = race_drivers["driver_number"].dropna().tolist()

    if len(driver_options) >= 2:
        driver_labels = {}
        for _, row in race_drivers.iterrows():
            number = row["driver_number"]
            name = row.get("full_name")
            if pd.isna(name) or not str(name).strip():
                name = row.get("name_acronym")
            if pd.isna(name) or not str(name).strip():
                name = str(number)
            team = row.get("team_name")
            driver_labels[number] = (
                f"{name} ({team})"
                if pd.notna(team) and str(team).strip()
                else str(name)
            )

        selected_drivers = st.multiselect(
            "Select drivers to compare",
            options=driver_options,
            default=driver_options[:2],
            format_func=lambda value: driver_labels.get(value, str(value)),
            key="race_driver_comparison",
        )

        comparison = race_drivers[
            race_drivers["driver_number"].isin(selected_drivers)
        ].copy()

        comparison_columns = [
            column for column in [
                "driver_number", "full_name", "name_acronym", "team_name",
                "valid_lap_count", "average_lap_seconds",
                "median_lap_seconds", "fastest_lap_seconds",
            ] if column in comparison.columns
        ]

        st.dataframe(
            clean_display_dataframe(comparison[comparison_columns]),
            width="stretch",
            hide_index=True,
        )

        comparison_metrics = [
            column for column in [
                "average_lap_seconds", "median_lap_seconds",
                "fastest_lap_seconds",
            ] if column in comparison.columns
        ]
        if not comparison.empty and comparison_metrics:
            comparison["driver_label"] = get_driver_label(comparison)
            comparison_long = comparison.melt(
                id_vars=["driver_label"],
                value_vars=comparison_metrics,
                var_name="metric",
                value_name="lap_time_seconds",
            ).dropna(subset=["lap_time_seconds"])
            comparison_long["metric"] = comparison_long["metric"].map({
                "average_lap_seconds": "Average lap",
                "median_lap_seconds": "Median lap",
                "fastest_lap_seconds": "Fastest lap",
            })
            comparison_fig = px.bar(
                comparison_long,
                x="driver_label",
                y="lap_time_seconds",
                color="metric",
                barmode="group",
                title="Selected Driver Lap-Time Comparison",
                labels={
                    "driver_label": "Driver",
                    "lap_time_seconds": "Lap time (seconds)",
                    "metric": "Metric",
                },
            )
            comparison_fig.update_layout(
                xaxis_title="Driver",
                yaxis_title="Lap time (seconds)",
                hovermode="closest",
            )
            st.plotly_chart(comparison_fig, width="stretch")
else:
    st.info("Driver comparison is unavailable because driver data is missing.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "ApexF1 | Historical Formula 1 Analytics | "
    "OpenF1 | Azure Blob Storage | FastAPI | Streamlit"
)

st.caption(
    "Season analytics aggregate historical race sessions. "
    "Individual race analytics refer to the configured session."
)
