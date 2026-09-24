"""Interactive timeline dashboard for one Tesserae participant."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from charts import (
    TimelineSeries,
    build_circadian_heatmap,
    build_circadian_profile,
    build_explorer_plot,
    build_sleep_timing,
    build_timeline,
)
from data import (
    DEFAULT_PARQUET_ROOT,
    DataPathError,
    aggregate_metric,
    build_index,
    circadian_values,
    coverage_table,
    extract_sleep,
    index_date_bounds,
    list_participant_paths,
    load_subtype,
    missing_dates,
    resolve_resolution,
    subtype_index,
)


st.set_page_config(
    page_title="Tesserae Timeline",
    page_icon="⌁",
    layout="wide",
)

## one chosen participant 
DEFAULT_PARTICIPANT_PATH = Path(
    DEFAULT_PARQUET_ROOT / "36fd7269ce06b988c4f4347550a4a939"
)
# DEFAULT_PARTICIPANT_PATH = Path(
#     os.environ.get("TESSERAE_PARTICIPANT", str(DEFAULT_PARTICIPANT))
# ).expanduser()
# DEFAULT_PARQUET_ROOT = os.environ.get(
#     "TESSERAE_PARQUET_ROOT", str(DEFAULT_PARTICIPANT_PATH.parent)
# )

SURVEY_METRICS = {
    "Stress score": "stress",
    "Anxiety score": "anxiety",
    "Perceived sleep": "sleep",
    "Positive affect": "pos_affect",
    "Negative affect": "neg_affect",
    "Extraversion": "extraversion",
    "Agreeableness": "agreeableness",
    "Conscientiousness": "conscientiousness",
    "Neuroticism": "neuroticism",
    "Openness": "openness",
    "Daily physical activity score": "total_pa_d",
}

ACTIVITY_METRICS = {
    "Steps": "steps",
    "Metabolic Equivalent of Task": "met",
    "Active time": "active_time_seconds",
    "Active calories": "active_kilocalories",
    "Distance": "distance_meters",
    "Maximum motion intensity": "max_motion_intensity",
    "Mean motion intensity": "mean_motion_intensity",
}

PHYSIO_METRICS = {
    "Heart rate": ("GarminPA", "HR", "hr", "bpm"),
    "RMSSD": ("GarminPA", "HRVSeries", "rmssd", "ms"),
    "SDNN": ("GarminPA", "HRVSeries", "sdnn", "ms"),
    "Daily average stress": ("GarminSummary", "StressSummary", "ave_stress", "score"),
}

CIRCADIAN_METRICS = {
    "Heart rate": ("GarminPA", "HR", "hr"),
    "Sleep stages": None,
    "Steps": ("GarminHealthAPI", "Epochs", "steps"),
    "Metabolic Equivalent of Task": ("GarminHealthAPI", "Epochs", "met"),
    "Mean motion intensity": (
        "GarminHealthAPI",
        "Epochs",
        "mean_motion_intensity",
    ),
    "Beat-to-beat interval": ("GarminPA", "BBI", "bbi"),
}

STATISTIC_LABELS = {
    "Default for metric": "default",
    "Native values": "native",
    "Mean": "mean",
    "Median": "median",
    "Standard deviation": "sd",
    # "Minimum": "min",
    # "Maximum": "max",
    "Sum": "sum",
}


@st.cache_data(show_spinner=False, persist="disk", max_entries=8)
def cached_index(path: str) -> pd.DataFrame:
    return build_index(path, inspect_metadata=False)


@st.cache_data(
    show_spinner="Scanning all Parquet metadata for Coverage…",
    persist="disk",
    max_entries=4,
)
def cached_metadata_index(path: str) -> pd.DataFrame:
    return build_index(path, inspect_metadata=True)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_participant_paths(path: str) -> tuple[str, ...]:
    return tuple(str(item) for item in list_participant_paths(path))


@st.cache_data(show_spinner="Loading selected Parquet rows…", max_entries=64)
def cached_load(
    path: str,
    category: str,
    subtype: str,
    start_iso: str,
    end_iso: str,
    columns: tuple[str, ...] | None,
    row_limit: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    index = cached_index(path)
    return load_subtype(
        index,
        category,
        subtype,
        pd.Timestamp(start_iso),
        pd.Timestamp(end_iso),
        columns=columns,
        row_limit=row_limit,
    )


def load_view(
    participant_path: str,
    category: str,
    subtype: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    columns: list[str] | tuple[str, ...] | None,
    row_limit: int | None = None,
) -> pd.DataFrame:
    frame, warnings = cached_load(
        participant_path,
        category,
        subtype,
        start.isoformat(),
        end.isoformat(),
        tuple(columns) if columns is not None else None,
        row_limit,
    )
    for warning in warnings[:3]:
        st.warning(warning)
    if len(warnings) > 3:
        st.warning(f"{len(warnings) - 3} additional files could not be read.")
    return frame


def date_controls(
    first_date: pd.Timestamp, last_date: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    preset = st.sidebar.radio(
        "Time window",
        ("Full range", "Custom"),
        index=0,
    )
    if preset == "Full range":
        start_date, end_date = first_date.date(), last_date.date()
    else:
        selected = st.sidebar.date_input(
            "Date range",
            value=(max(first_date.date(), (last_date - pd.Timedelta(days=30)).date()), last_date.date()),
            min_value=first_date.date(),
            max_value=last_date.date(),
        )
        if isinstance(selected, tuple) and len(selected) == 2:
            start_date, end_date = selected
        else:
            start_date = end_date = selected[0] if isinstance(selected, tuple) else selected
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return start, end


def app_header(participant_path: str, index: pd.DataFrame) -> None:
    st.title("Tesserae participant timeline")
    st.caption(
        f"Participant `{Path(participant_path).name}` · "
        f"{len(index):,} Parquet files · metadata-first, on-demand loading"
    )


def timeline_tab(
    participant_path: str,
    index: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    st.subheader("Aligned physiological and behavioural timeline")
    controls = st.columns((1.1, 1.2, 0.8))
    resolution_label = controls[0].selectbox(
        "Resolution",
        ("Auto", "Raw", "1 minute", "5 minutes", "15 minutes", "Hourly", "Daily"),
    )
    statistic_label = controls[1].selectbox("Statistic", tuple(STATISTIC_LABELS))
    show_sd = controls[2].checkbox("Mean ± SD ribbon", value=True)
    resolution = resolve_resolution(resolution_label, start, end)
    statistic = STATISTIC_LABELS[statistic_label]

    with st.expander("Signals", expanded=True):
        columns = st.columns(3)
        survey_selected = columns[0].multiselect(
            "Survey scores",
            tuple(SURVEY_METRICS),
            default=("Stress score", "Positive affect","Negative affect"),
        )
        activity_selected = columns[1].multiselect(
            "Activity",
            tuple(ACTIVITY_METRICS),
            default=("Steps", "Metabolic Equivalent of Task"),
        )
        physiology_selected = columns[2].multiselect(
            "Physiology",
            tuple(PHYSIO_METRICS),
            default=("Heart rate", "RMSSD"),
        )
        # Sleep stages are always included as the final timeline track.

    plotted: list[TimelineSeries] = []
    if survey_selected:
        survey_columns = [SURVEY_METRICS[label] for label in survey_selected]
        survey = load_view(
            participant_path,
            "GroundTruthSurvey",
            "DailiesScored",
            start,
            end,
            survey_columns,
        )
        for label in survey_selected:
            metric = SURVEY_METRICS[label]
            values = aggregate_metric(
                survey, metric, "raw", "native", include_sd=False
            )
            plotted.append(TimelineSeries(label, values, mode="markers"))

    if activity_selected:
        activity_columns = [ACTIVITY_METRICS[label] for label in activity_selected]
        activity = load_view(
            participant_path,
            "GarminHealthAPI",
            "Epochs",
            start,
            end,
            activity_columns,
        )
        for label in activity_selected:
            metric = ACTIVITY_METRICS[label]
            values = aggregate_metric(
                activity,
                metric,
                resolution,
                statistic,
                include_sd=show_sd and statistic in {"default", "mean"},
            )
            unit = "steps" if metric == "steps" else ""
            plotted.append(TimelineSeries(label, values, unit=unit))

    for label in physiology_selected:
        category, subtype, metric, unit = PHYSIO_METRICS[label]
        frame = load_view(
            participant_path, category, subtype, start, end, [metric]
        )
        values = aggregate_metric(
            frame,
            metric,
            resolution,
            statistic,
            include_sd=show_sd and statistic in {"default", "mean"},
        )
        plotted.append(TimelineSeries(label, values, unit=unit))

    sleep = pd.DataFrame()
    sleep_records = load_view(
        participant_path,
        "GarminHealthAPI",
        "Sleep",
        start,
        end,
        ["sleep_levels_map", "validation"],
    )
    sleep = extract_sleep(sleep_records)
    if not sleep.empty:
        sleep = sleep.loc[
            (sleep["sleep_offset"] >= start) & (sleep["sleep_onset"] < end)
        ]

    st.caption(
        f"Effective resolution: **{resolution}**. Native values bypass binning; "
        "survey observations and nightly sleep-stage summaries are always discrete."
    )
    figure = build_timeline(plotted, sleep, start, end)
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
    )

def circadian_tab(
    participant_path: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    st.subheader("Circadian rhythm")
    controls = st.columns(4)
    metric_label = controls[0].selectbox("Metric", tuple(CIRCADIAN_METRICS))
    day_filter = controls[2].selectbox("Days", ("All days", "Weekdays", "Weekends"))

    if metric_label == "Sleep stages":
        sleep_bin_minutes = controls[1].selectbox(
            "Clock-time bin", (5, 15, 30, 60), index=1, key="sleep_clock_bin"
        )
        sleep_frame = load_view(
            participant_path,
            "GarminHealthAPI",
            "Sleep",
            start,
            end,
            ["sleep_levels_map", "validation"],
        )
        sleep = extract_sleep(sleep_frame)
        if not sleep.empty:
            sleep = sleep.loc[
                (sleep["sleep_offset"] >= start) & (sleep["sleep_onset"] < end)
            ]
            if day_filter == "Weekdays":
                sleep = sleep.loc[sleep["night"].dt.dayofweek < 5]
            elif day_filter == "Weekends":
                sleep = sleep.loc[sleep["night"].dt.dayofweek >= 5]
        st.caption(
            "Categorical sleep stages from Garmin `sleep_levels_map`; all "
            "unassigned clock-time bins are shown as Awake."
        )
        st.plotly_chart(
            build_sleep_timing(sleep, sleep_bin_minutes),
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
        )
        return

    bin_minutes = controls[1].selectbox("Clock-time bin", (5, 15, 30, 60), index=1)
    split_profile = controls[3].checkbox("Compare weekday/weekend", value=False)
    metric_source = CIRCADIAN_METRICS[metric_label]
    assert metric_source is not None
    category, subtype, metric = metric_source
    frame = load_view(participant_path, category, subtype, start, end, [metric])
    matrix, profile = circadian_values(frame, metric, bin_minutes, day_filter)

    st.plotly_chart(
        build_circadian_heatmap(matrix, metric_label),
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    if split_profile:
        profiles = []
        for group in ("Weekdays", "Weekends"):
            _, grouped_profile = circadian_values(frame, metric, bin_minutes, group)
            profiles.append(grouped_profile)
        profile = pd.concat(profiles, ignore_index=True)
    st.plotly_chart(
        build_circadian_profile(profile, metric_label),
        use_container_width=True,
        config={"displaylogo": False},
    )


def explorer_tab(
    participant_path: str,
    index: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    st.subheader("All-data explorer")
    categories = sorted(index["category"].dropna().unique())
    selectors = st.columns((1, 1.4, 1))
    category = selectors[0].selectbox("Category", categories)
    subtypes = sorted(index.loc[index["category"] == category, "subtype"].unique())
    subtype = selectors[1].selectbox("Subtype", subtypes)
    row_limit = selectors[2].select_slider(
        "Maximum rows",
        options=(1_000, 5_000, 10_000, 25_000, 50_000, 100_000),
        value=25_000,
    )
    records = subtype_index(index, category, subtype)

    frame = load_view(
        participant_path,
        category,
        subtype,
        start,
        end,
        None,
        row_limit=row_limit,
    )
    if frame.empty:
        st.info("No rows from this subtype fall within the selected date range.")
        return
    visible_columns = [column for column in frame.columns if column != "_time"]
    metrics = st.columns(4)
    metrics[0].metric("Files", f"{len(records):,}")
    metrics[1].metric("Loaded rows", f"{len(frame):,}")
    metrics[2].metric("Loaded columns", len(visible_columns))
    metrics[3].metric("Load limit", f"{row_limit:,}")
    schema = pd.DataFrame(
        {"column": visible_columns, "loaded dtype": [str(frame[name].dtype) for name in visible_columns]}
    )
    with st.expander("Loaded schema", expanded=False):
        st.dataframe(schema, width="stretch", hide_index=True)
    numeric_columns = [
        column
        for column in frame.columns
        if column != "_time"
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    selected = st.multiselect(
        "Numeric columns to plot",
        numeric_columns,
        default=numeric_columns[:1],
        max_selections=6,
    )
    st.plotly_chart(
        build_explorer_plot(frame, selected),
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    preview_tab, missing_tab = st.tabs(("Preview", "Missingness"))
    with preview_tab:
        st.dataframe(frame.head(1_000), width="stretch", hide_index=True)
        if len(frame) >= row_limit:
            st.caption(f"Display/load limit reached: {row_limit:,} rows.")
    with missing_tab:
        missing = pd.DataFrame(
            {
                "column": frame.columns,
                "missing": [int(frame[column].isna().sum()) for column in frame.columns],
                "missing_percent": [
                    round(float(frame[column].isna().mean() * 100), 2)
                    for column in frame.columns
                ],
            }
        ).sort_values("missing_percent", ascending=False)
        st.dataframe(missing, width="stretch", hide_index=True)


def coverage_tab(participant_path: str) -> None:
    st.subheader("Data coverage and quality")
    st.caption(
        "Coverage opens every Parquet footer once. The result is cached on disk "
        "for later visits and app restarts."
    )
    index = cached_metadata_index(participant_path)
    coverage = coverage_table(index)
    st.dataframe(coverage, width="stretch", hide_index=True)

    st.markdown("#### Daily chunk gaps")
    options = [f"{row.category} / {row.subtype}" for row in coverage.itertuples() if row.files >= 5]
    if options:
        selected = st.selectbox("Subtype", options)
        category, subtype = selected.split(" / ", 1)
        gaps = missing_dates(subtype_index(index, category, subtype))
        if gaps:
            st.write(", ".join(day.strftime("%Y-%m-%d") for day in gaps[:100]))
            if len(gaps) > 100:
                st.caption(f"Showing 100 of {len(gaps):,} missing calendar dates.")
        else:
            st.success("No missing calendar dates between the first and last chunk.")

    problems = index.loc[index["status"] != "ok", ["category", "subtype", "path", "status", "error"]]
    st.markdown("#### Empty or unreadable files")
    if problems.empty:
        st.success("No empty or unreadable Parquet files were detected in metadata.")
    else:
        st.dataframe(problems, width="stretch", hide_index=True)


def main() -> None:
    st.sidebar.header("Data source")
    if st.sidebar.button("Refresh metadata cache"):
        st.cache_data.clear()
        st.rerun()
    parquet_root = st.sidebar.text_input(
        "Parquet root directory", value=DEFAULT_PARQUET_ROOT
    )
    try:
        participant_paths = cached_participant_paths(parquet_root)
    except DataPathError as error:
        st.error(str(error))
        st.stop()
    preferred_index = next(
        (
            index
            for index, path in enumerate(participant_paths)
            if Path(path).name == DEFAULT_PARTICIPANT_PATH.name
        ),
        0,
    )
    participant_path = st.sidebar.selectbox(
        "Participant",
        participant_paths,
        index=preferred_index,
        format_func=lambda path: Path(path).name,
    )
    st.sidebar.caption(f"{len(participant_paths):,} participant folders available")
    try:
        with st.spinner("Indexing Parquet metadata…"):
            index = cached_index(participant_path)
    except DataPathError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.error(f"Could not index the participant data: {type(error).__name__}: {error}")
        st.stop()

    first_date, last_date = index_date_bounds(index)
    start, end = date_controls(first_date, last_date)
    st.sidebar.caption(
        f"Available chunk dates: {first_date:%Y-%m-%d} to {last_date:%Y-%m-%d}"
    )
    app_header(participant_path, index)
    view = st.radio(
        "Dashboard view",
        ("Timeline", "Circadian", "Data Explorer", "Coverage"),
        horizontal=True,
        label_visibility="collapsed",
    )
    if view == "Timeline":
        timeline_tab(participant_path, index, start, end)
    elif view == "Circadian":
        circadian_tab(participant_path, start, end)
    elif view == "Data Explorer":
        explorer_tab(participant_path, index, start, end)
    else:
        coverage_tab(participant_path)


if __name__ == "__main__":
    main()
