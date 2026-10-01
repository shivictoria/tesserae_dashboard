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
    build_group_coverage_heatmap,
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
from group_coverage import (
    build_and_cache_coverages,
    coverage_cache_directory,
    discover_variables,
    load_cached_coverage,
    load_variable_catalogue,
    prepare_coverage_view,
    save_variable_catalogue,
    variable_from_record,
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
            default=("Stress score", "Positive affect","Negative affect","Anxiety score"),
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
        "survey observations and nightly sleep-stage summaries are always discrete. "
        "Light grey shading marks weekends (Saturday–Sunday)."
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


def group_coverage_view(
    parquet_root: str,
    participant_paths: tuple[str, ...],
) -> None:
    """Render population-level coverage without loading participant views."""

    st.title("Tesserae group coverage")
    st.caption(
        "Daily coverage means at least one non-null value for the selected "
        "category, subtype, and column. Raw physiological values are not "
        "stored in the coverage cache."
    )
    cache_directory = coverage_cache_directory(parquet_root)
    catalogue = load_variable_catalogue(parquet_root)
    with st.expander(
        "Variable catalogue and cache",
        expanded=catalogue.empty,
    ):
        st.caption(f"Persistent cache: `{cache_directory}`")
        catalogue_controls = st.columns((1.2, 3), vertical_alignment="center")
        catalogue_action = catalogue_controls[0].button(
            "Discover variables" if catalogue.empty else "Refresh catalogue",
            type="primary" if catalogue.empty else "secondary",
            use_container_width=True,
        )
        catalogue_controls[1].caption(
            "The saved catalogue is reused after dashboard restarts. Refresh only "
            "when source schemas change. Discovery samples one Parquet schema per "
            "participant/source and unions the columns found across participants."
        )

    if catalogue_action:
        progress_bar = st.progress(0.0, text="Starting variable discovery…")

        def discovery_progress(done: int, total: int, label: str) -> None:
            progress_bar.progress(
                done / total if total else 1.0,
                text=f"Discovering variables · participant {done:,} of {total:,}",
            )

        catalogue, discovery_errors = discover_variables(
            parquet_root,
            participant_paths,
            progress=discovery_progress,
        )
        save_variable_catalogue(catalogue, parquet_root)
        progress_bar.empty()
        if discovery_errors.empty:
            st.success(f"Discovered {len(catalogue):,} physical columns.")
        else:
            st.warning(
                f"Discovered {len(catalogue):,} columns with "
                f"{len(discovery_errors):,} unreadable representative files."
            )
            with st.expander("Variable-discovery errors"):
                st.dataframe(discovery_errors, width="stretch", hide_index=True)

    if catalogue.empty:
        st.info("Discover variables before building group coverage.")
        return

    supported = catalogue.loc[catalogue["timestamp_column"].notna()].copy()
    unsupported = catalogue.loc[catalogue["timestamp_column"].isna()].copy()
    if not unsupported.empty:
        with st.expander(
            f"Columns without a usable date ({len(unsupported):,})",
            expanded=False,
        ):
            st.caption(
                "These columns are discoverable but cannot be plotted as daily "
                "coverage because their source has no recognised timestamp."
            )
            st.dataframe(
                unsupported[["category", "subtype", "column", "arrow_type"]],
                width="stretch",
                hide_index=True,
            )
    if supported.empty:
        st.warning(
            "No discovered columns belong to a source with a recognised timestamp."
        )
        return

    labels = {
        row.variable_id: f"{row.category} / {row.subtype} / {row.column}"
        for row in supported.itertuples()
    }
    available_ids = set(supported["variable_id"])
    selection_key = "group_coverage_selected_variables"
    selection_root_key = "group_coverage_selection_root"
    resolved_root = str(Path(parquet_root).expanduser().resolve())
    if st.session_state.get(selection_root_key) != resolved_root:
        st.session_state[selection_root_key] = resolved_root
        st.session_state[selection_key] = []
    selected_ids = [
        identifier
        for identifier in st.session_state.get(selection_key, [])
        if identifier in available_ids
    ]
    st.session_state[selection_key] = selected_ids

    with st.container(border=True):
        st.markdown("#### Select variables")
        st.caption(
            "Choose a physical column and add it to the comparison. Change the "
            "category or subtype to combine variables from different sources."
        )
        selector_columns = st.columns(
            (1, 1, 1.5, 0.55),
            vertical_alignment="bottom",
        )
        categories = sorted(supported["category"].dropna().unique())
        selected_category = selector_columns[0].selectbox(
            "Category",
            categories,
            key="group_coverage_category",
        )
        category_rows = supported.loc[supported["category"] == selected_category]
        subtypes = sorted(category_rows["subtype"].dropna().unique())
        selected_subtype = selector_columns[1].selectbox(
            "Subtype",
            subtypes,
            key="group_coverage_subtype",
        )
        schema_rows = category_rows.loc[category_rows["subtype"] == selected_subtype]
        schema_lookup = schema_rows.set_index("variable_id")
        candidate_id = selector_columns[2].selectbox(
            "Schema column",
            schema_rows["variable_id"].tolist(),
            format_func=lambda identifier: (
                f"{schema_lookup.loc[identifier, 'column']} "
                f"({schema_lookup.loc[identifier, 'arrow_type']})"
            ),
            key="group_coverage_schema_column",
        )
        already_selected = candidate_id in selected_ids
        add_requested = selector_columns[3].button(
            "Added" if already_selected else "Add",
            type="primary" if not already_selected else "secondary",
            disabled=already_selected,
            use_container_width=True,
        )
        if add_requested:
            st.session_state[selection_key] = [*selected_ids, candidate_id]
            st.rerun()

        if selected_ids:
            st.markdown(f"**Selected variables ({len(selected_ids)})**")
            selected_lookup = supported.set_index("variable_id")
            for index, identifier in enumerate(selected_ids):
                record = selected_lookup.loc[identifier]
                variable_columns = st.columns(
                    (5, 0.8),
                    vertical_alignment="center",
                )
                variable_columns[0].markdown(
                    f"**{record['column']}** &nbsp; "
                    f"{record['category']} / {record['subtype']} &nbsp; "
                    f"`{record['arrow_type']}`"
                )
                if variable_columns[1].button(
                    "Remove",
                    key=f"group_coverage_remove_{index}",
                    use_container_width=True,
                ):
                    st.session_state[selection_key] = [
                        item for item in selected_ids if item != identifier
                    ]
                    st.rerun()

    if not selected_ids:
        st.info(
            "Choose a category, subtype, and schema column, then click Add. "
            "Repeat to compare variables from different sources."
        )
        return

    selected_records = supported.loc[
        supported["variable_id"].isin(selected_ids)
    ].set_index("variable_id").loc[selected_ids].reset_index()
    variables = [variable_from_record(row) for _, row in selected_records.iterrows()]

    cached: dict[str, tuple[pd.DataFrame, dict | None, pd.DataFrame]] = {
        variable.variable_id: load_cached_coverage(parquet_root, variable)
        for variable in variables
    }
    missing_variables = [
        variable
        for variable in variables
        if cached[variable.variable_id][1] is None
    ]

    building_state_key = "group_coverage_building"
    was_building = bool(st.session_state.get(building_state_key, False))
    with st.container(border=True):
        st.markdown("#### Daily coverage cache")
        build_controls = st.columns(
            (1.3, 1.4, 1.0, 2.5),
            vertical_alignment="center",
        )
        build_requested = build_controls[0].button(
            "Build missing coverage",
            type="primary",
            use_container_width=True,
        )
        refresh_requested = build_controls[1].button(
            "Rebuild selected coverage",
            use_container_width=True,
        )
        requested_targets = (
            variables
            if refresh_requested
            else missing_variables
            if build_requested
            else []
        )
        stop_requested = build_controls[2].button(
            "Stop building",
            disabled=not (was_building or requested_targets),
            use_container_width=True,
        )
        if missing_variables:
            build_controls[3].caption(
                f"{len(missing_variables):,} of {len(variables):,} selected "
                "variable(s) still need a daily summary."
            )
        else:
            build_controls[3].caption(
                "All selected daily summaries are ready. Rebuild only when the "
                "source data have changed."
            )

    targets = requested_targets
    if stop_requested:
        st.session_state[building_state_key] = False
        targets = []
        st.warning(
            "Coverage building stopped. Previously completed cache files were kept; "
            "the unfinished summary was not saved."
        )
    elif was_building and not targets:
        st.session_state[building_state_key] = False
        st.warning(
            "The previous coverage build was interrupted by a dashboard update. "
            "Previously completed cache files were kept."
        )
    if build_requested and not targets:
        st.success("All selected coverage summaries are already cached.")
    if targets:
        st.session_state[building_state_key] = True
        progress_bar = st.progress(0.0, text="Preparing daily coverage…")

        def build_progress(done: int, total: int, label: str) -> None:
            progress_bar.progress(
                done / total if total else 1.0,
                text=f"Building daily coverage · participant {done:,} of {total:,}",
            )

        try:
            build_and_cache_coverages(
                parquet_root,
                targets,
                participant_paths,
                progress=build_progress,
            )
        except Exception:
            st.session_state[building_state_key] = False
            raise
        st.session_state[building_state_key] = False
        progress_bar.empty()
        cached = {
            variable.variable_id: load_cached_coverage(parquet_root, variable)
            for variable in variables
        }
        st.success(f"Built {len(targets):,} coverage summary file(s).")

    status_records = []
    coverage_frames = []
    error_frames = []
    unavailable = []
    for variable in variables:
        coverage, manifest, errors = cached[variable.variable_id]
        if manifest is None:
            unavailable.append(variable.label)
            status_records.append(
                {
                    "variable": variable.label,
                    "status": "Not built",
                    "built_at": None,
                    "files_processed": 0,
                    "participants_with_data": 0,
                    "participant_days": 0,
                    "errors": 0,
                }
            )
            continue
        coverage_frames.append(coverage)
        if not errors.empty:
            error_frames.append(errors.assign(variable=variable.label))
        status_records.append(
            {
                "variable": variable.label,
                "status": "Cached",
                "built_at": manifest.get("built_at"),
                "files_processed": manifest.get("files_processed", 0),
                "participants_with_data": manifest.get("participants_with_data", 0),
                "participant_days": manifest.get("participant_days", 0),
                "errors": manifest.get("errors", 0),
            }
        )

    with st.expander("Coverage cache status", expanded=bool(unavailable)):
        st.dataframe(
            pd.DataFrame.from_records(status_records),
            width="stretch",
            hide_index=True,
        )
    if unavailable:
        st.info("Build every selected variable before plotting their overlap.")
        return

    coverage = (
        pd.concat(coverage_frames, ignore_index=True)
        if coverage_frames
        else pd.DataFrame(columns=["participant_id", "date", "variable_id"])
    )
    participant_ids = tuple(Path(path).name for path in participant_paths)

    controls = st.columns((1, 1.3, 1.2))
    alignment_label = controls[0].radio(
        "Alignment",
        ("Participant Day 0", "Calendar date"),
        horizontal=True,
    )
    display_options = ["Overlap count", *selected_ids]
    display_choice = controls[1].selectbox(
        "Colour by",
        display_options,
        format_func=lambda value: (
            value if value == "Overlap count" else labels.get(value, value)
        ),
    )
    participant_sort = controls[2].selectbox(
        "Participant order",
        ("Study span", "Study start", "Completeness", "Participant ID"),
    )
    alignment = "relative" if alignment_label == "Participant Day 0" else "calendar"
    display_variable_id = None if display_choice == "Overlap count" else display_choice

    start = end = None
    valid_dates = pd.to_datetime(coverage["date"], errors="coerce").dropna()
    if alignment == "calendar" and not valid_dates.empty:
        minimum_date = valid_dates.min().date()
        maximum_date = valid_dates.max().date()
        selected_range = st.date_input(
            "Calendar window",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start = pd.Timestamp(selected_range[0])
            end = pd.Timestamp(selected_range[1])

    view = prepare_coverage_view(
        coverage,
        participant_ids,
        variables,
        alignment=alignment,
        display_variable_id=display_variable_id,
        participant_sort=participant_sort,
        start=start,
        end=end,
    )
    selected_variable_label = (
        labels.get(display_variable_id) if display_variable_id is not None else None
    )
    st.plotly_chart(
        build_group_coverage_heatmap(
            view.matrix,
            view.hover_text,
            maximum_value=view.maximum_value,
            alignment=view.alignment,
            selected_variable_label=selected_variable_label,
        ),
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
    )

    anchored = int((view.participant_summary["span_days"] > 0).sum())
    complete_days = int(view.participant_summary["all_variables_days"].sum())
    missing_days = int(view.participant_summary["internal_missing_days"].sum())
    metrics = st.columns(4)
    metrics[0].metric("Participants", f"{len(participant_ids):,}")
    metrics[1].metric("With selected data", f"{anchored:,}")
    metrics[2].metric("Complete-overlap days", f"{complete_days:,}")
    metrics[3].metric("Internally missing days", f"{missing_days:,}")

    participant_tab, variable_tab, combination_tab = st.tabs(
        ("Participants", "Variables", "Coverage combinations")
    )
    with participant_tab:
        st.dataframe(view.participant_summary, width="stretch", hide_index=True)
        st.download_button(
            "Download participant coverage CSV",
            view.participant_summary.to_csv(index=False),
            file_name="tesserae_participant_coverage.csv",
            mime="text/csv",
        )
    with variable_tab:
        st.dataframe(view.variable_summary, width="stretch", hide_index=True)
        st.download_button(
            "Download variable coverage CSV",
            view.variable_summary.to_csv(index=False),
            file_name="tesserae_variable_coverage.csv",
            mime="text/csv",
        )
    with combination_tab:
        st.dataframe(view.combination_summary, width="stretch", hide_index=True)

    if error_frames:
        errors = pd.concat(error_frames, ignore_index=True)
        with st.expander(f"Coverage build errors ({len(errors):,})"):
            st.dataframe(errors, width="stretch", hide_index=True)


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
    analysis_level = st.sidebar.radio(
        "Analysis level",
        ("Participant", "Group"),
        horizontal=True,
    )
    st.sidebar.caption(f"{len(participant_paths):,} participant folders available")
    if analysis_level == "Group":
        group_coverage_view(parquet_root, participant_paths)
        return

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
        ("Timeline", "Circadian", "Data Explorer", "Coverage", "EDM"),
        horizontal=True,
        label_visibility="collapsed",
    )
    if view == "Timeline":
        timeline_tab(participant_path, index, start, end)
    elif view == "Circadian":
        circadian_tab(participant_path, start, end)
    elif view == "Data Explorer":
        explorer_tab(participant_path, index, start, end)
    elif view == "EDM":
        from edm_ui import edm_tab
        edm_tab(participant_path, index, start, end, load_view)
    else:
        coverage_tab(participant_path)


if __name__ == "__main__":
    main()
