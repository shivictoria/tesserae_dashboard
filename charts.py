"""Plotly figure builders for the Tesserae dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTE = (
    "#45B8AC",
    "#F2A65A",
    "#7E6BC4",
    "#E85D75",
    "#4C9BE8",
    "#9AB973",
    "#D983B7",
    "#E4C34A",
)

SLEEP_STAGE_COLORS = {
    "awake_seconds": "#FFE0C1",
    "light_sleep_seconds": "#8EA6E8",
    "deep_sleep_seconds": "#4556A6",
    "rem_sleep_seconds": "#FE5B5B",
    "unmeasurable_sleep_seconds": "#A9AFBC",
}


@dataclass
class TimelineSeries:
    label: str
    data: pd.DataFrame
    unit: str = ""
    color: str | None = None
    mode: str = "lines"


def empty_figure(message: str, height: int = 360) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": "#667085"},
    )
    figure.update_layout(height=height, template="plotly_white")
    return figure


def build_timeline(
    series: list[TimelineSeries],
    sleep: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> go.Figure:
    usable = [item for item in series if not item.data.empty]
    has_sleep = sleep is not None and not sleep.empty
    row_count = len(usable) + int(has_sleep)
    if row_count == 0:
        return empty_figure("Select at least one signal with data in this period.")

    titles = [item.label for item in usable]
    if has_sleep:
        titles.append("Sleep stages")
    figure = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.04, 0.18 / row_count),
        subplot_titles=titles,
    )

    for row, item in enumerate(usable, start=1):
        color = item.color or PALETTE[(row - 1) % len(PALETTE)]
        data = item.data
        if "sd" in data and data["sd"].notna().any():
            upper = data["value"] + data["sd"]
            lower = data["value"] - data["sd"]
            figure.add_trace(
                go.Scatter(
                    x=data["_time"],
                    y=upper,
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=data["_time"],
                    y=lower,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=_rgba(color, 0.18),
                    hoverinfo="skip",
                    name=f"{item.label} ± SD",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
        figure.add_trace(
            go.Scattergl(
                x=data["_time"],
                y=data["value"],
                mode=item.mode,
                name=item.label,
                line={"color": color, "width": 1.6},
                marker={"color": color, "size": 7},
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M}<br>%{y:.3g}"
                    + (f" {item.unit}" if item.unit else "")
                    + "<extra>" + item.label + "</extra>"
                ),
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text=item.unit or None, row=row, col=1)

    if has_sleep:
        assert sleep is not None
        sleep_row = row_count
        for column, color in SLEEP_STAGE_COLORS.items():
            if column not in sleep.columns:
                continue
            label = column.replace("_seconds", "").replace("_", " ").title()
            figure.add_trace(
                go.Bar(
                    x=sleep["night"],
                    y=pd.to_numeric(sleep[column], errors="coerce") / 3600,
                    name=label,
                    marker={"color": color},
                    width=18 * 60 * 60 * 1000,
                    customdata=sleep[["sleep_onset", "sleep_offset", "validation"]],
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>%{y:.2f} h<br>"
                        "Onset: %{customdata[0]|%H:%M}<br>"
                        "Offset: %{customdata[1]|%H:%M}<br>"
                        "Validation: %{customdata[2]}<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=sleep_row,
                col=1,
            )
        figure.update_yaxes(title_text="Hours", rangemode="tozero", row=sleep_row, col=1)

    # Shade Saturday midnight through Monday midnight on every timeline track.
    # Batch shapes to avoid repeatedly validating all previously added rectangles.
    weekend_shapes = []
    weekend_start = start.normalize() - pd.DateOffset(days=(start.weekday() - 5) % 7)
    while weekend_start < end:
        weekend_end = weekend_start + pd.DateOffset(days=2)
        if weekend_end > start:
            for row in range(1, row_count + 1):
                suffix = str(row) if row > 1 else ""
                weekend_shapes.append(
                    dict(
                        type="rect",
                        x0=max(start, weekend_start),
                        x1=min(end, weekend_end),
                        xref=f"x{suffix}",
                        yref=f"y{suffix} domain",
                        y0=0,
                        y1=1,
                        fillcolor="rgba(128,128,128,0.12)",
                        line_width=0,
                        layer="below",
                    )
                )
        weekend_start += pd.DateOffset(days=7)

    figure.update_xaxes(range=[start, end], showspikes=True, spikemode="across")
    figure.update_xaxes(rangeslider={"visible": True}, row=row_count, col=1) ## later should use cycle for this

    figure.update_layout(
        shapes=weekend_shapes,
        template="plotly_white",
        barmode="stack",
        height=max(460, 185 * row_count),
        hovermode="x unified",
        margin={"l": 70, "r": 25, "t": 45, "b": 35},
        legend={"orientation": "h", "yanchor": "top", "y": -0.01},
        uirevision="timeline",
    )
    return figure


def build_circadian_heatmap(
    matrix: pd.DataFrame,
    metric_label: str,
) -> go.Figure:
    if matrix.empty:
        return empty_figure("No circadian data in the selected period.")
    # Plotly infers each heatmap row's height from the distance to adjacent y
    # values. With a sparse DatetimeIndex, one observed row can consequently
    # stretch across weeks or months. Insert explicit empty daily rows so every
    # coloured cell represents exactly one calendar day.
    matrix = matrix.copy()
    matrix.index = pd.DatetimeIndex(pd.to_datetime(matrix.index)).normalize()
    if matrix.index.has_duplicates:
        matrix = matrix.groupby(level=0).mean()
    matrix = matrix.sort_index().reindex(
        pd.date_range(matrix.index.min(), matrix.index.max(), freq="D")
    )
    clock_minutes = matrix.columns.to_numpy(dtype=float)
    clock_labels = [f"{int(value // 60):02d}:{int(value % 60):02d}" for value in clock_minutes]
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=clock_minutes / 60,
            y=matrix.index,
            colorscale="Viridis",
            colorbar={"title": metric_label},
            customdata=np.tile(clock_labels, (len(matrix), 1)),
            hovertemplate="%{y|%Y-%m-%d}<br>%{customdata}<br>%{z:.3g}<extra></extra>",
        )
    )
    figure.update_xaxes(title="Local clock time", range=[0, 24], dtick=2)
    figure.update_yaxes(title="Date", autorange="reversed")
    figure.update_layout(
        template="plotly_white",
        height=max(440, min(900, 160 + 10 * len(matrix))),
        margin={"l": 70, "r": 25, "t": 35, "b": 50},
    )
    return figure


def build_circadian_profile(profile: pd.DataFrame, metric_label: str) -> go.Figure:
    if profile.empty:
        return empty_figure("No profile data in the selected period.")
    figure = go.Figure()
    for index, (group, data) in enumerate(profile.groupby("group", sort=False)):
        color = PALETTE[index % len(PALETTE)]
        if data["sd"].notna().any():
            figure.add_trace(
                go.Scatter(
                    x=data["clock_hour"],
                    y=data["mean"] + data["sd"],
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=data["clock_hour"],
                    y=data["mean"] - data["sd"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=_rgba(color, 0.16),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        figure.add_trace(
            go.Scatter(
                x=data["clock_hour"],
                y=data["mean"],
                mode="lines",
                line={"color": color, "width": 2},
                name=str(group),
                hovertemplate="%{x:.2f} h<br>%{y:.3g}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_xaxes(title="Local clock time", range=[0, 24], dtick=2)
    figure.update_yaxes(title=metric_label)
    figure.update_layout(
        template="plotly_white",
        height=380,
        hovermode="x unified",
        margin={"l": 70, "r": 25, "t": 25, "b": 50},
    )
    return figure


def build_sleep_timing(sleep: pd.DataFrame, bin_minutes: int = 15) -> go.Figure:
    """Build a categorical date-by-clock-time sleep-stage heatmap."""
    if sleep.empty:
        return empty_figure("No sleep-stage records in the selected period.")
    if bin_minutes <= 0 or 1440 % bin_minutes:
        raise ValueError("bin_minutes must be a positive divisor of 1440")

    ordered = sleep.copy()
    ordered["night"] = pd.to_datetime(ordered["night"], errors="coerce").dt.normalize()
    ordered = (
        ordered.dropna(subset=["night"])
        .sort_values("night")
        .drop_duplicates("night", keep="last")
    )
    if ordered.empty:
        return empty_figure("No valid sleep-stage dates in the selected period.")
    nights = pd.date_range(ordered["night"].min(), ordered["night"].max(), freq="D")
    night_rows = {night: index for index, night in enumerate(nights)}
    stage_codes = {
        "awake": 0,
        "unmeasurable": 1,
        "light": 2,
        "deep": 3,
        "rem": 4,
    }
    stage_labels = np.array(
        ["Awake", "Unmeasurable", "Light", "Deep", "REM"], dtype=object
    )
    stage_colors = [
        SLEEP_STAGE_COLORS["awake_seconds"],
        SLEEP_STAGE_COLORS["unmeasurable_sleep_seconds"],
        SLEEP_STAGE_COLORS["light_sleep_seconds"],
        SLEEP_STAGE_COLORS["deep_sleep_seconds"],
        SLEEP_STAGE_COLORS["rem_sleep_seconds"],
    ]
    clock_minutes = np.arange(0, 1440, bin_minutes)
    # Missing nights remain blank; recorded nights default to Awake and mapped
    # sleep stages overwrite bins within that night's onset/offset.
    values = np.full((len(nights), len(clock_minutes)), np.nan)

    for record in ordered.itertuples(index=False):
        row_index = night_rows[record.night]
        values[row_index, :] = stage_codes["awake"]
        onset = record.sleep_onset
        offset = record.sleep_offset
        for stage, interval_start, interval_end in record.stage_intervals:
            # Awake is the default for the entire day. Only mapped sleep
            # stages overwrite it, and only inside calculated sleep bounds.
            if stage == "awake" or stage not in stage_codes:
                continue
            start = max(interval_start, onset)
            end = min(interval_end, offset)
            if end <= start:
                continue
            cursor = start.floor(f"{bin_minutes}min")
            while cursor < end:
                minute = cursor.hour * 60 + cursor.minute
                column = int(minute // bin_minutes)
                values[row_index, column] = max(
                    values[row_index, column], stage_codes[stage]
                )
                cursor += pd.Timedelta(minutes=bin_minutes)

    clock_labels = np.array(
        [f"{minute // 60:02d}:{minute % 60:02d}" for minute in clock_minutes],
        dtype=object,
    )
    hover_data = np.empty((*values.shape, 2), dtype=object)
    hover_data[:, :, 0] = np.tile(clock_labels, (len(nights), 1))
    hover_data[:, :, 1] = "Missing"
    present = np.isfinite(values)
    hover_data[:, :, 1][present] = stage_labels[values[present].astype(int)]
    colorscale: list[list[float | str]] = []
    for index, color in enumerate(stage_colors):
        colorscale.extend(
            [
                [index / len(stage_colors), color],
                [(index + 1) / len(stage_colors), color],
            ]
        )

    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=clock_minutes / 60,
            y=nights,
            customdata=hover_data,
            zmin=-0.5,
            zmax=4.5,
            colorscale=colorscale,
            showscale=True,
            colorbar={
                "title": "Stage",
                "tickmode": "array",
                "tickvals": list(stage_codes.values()),
                "ticktext": list(stage_labels),
            },
            hovertemplate=(
                "%{y|%Y-%m-%d}<br>%{customdata[0]}<br>"
                "%{customdata[1]}<extra></extra>"
            ),
        )
    )
    tick_values = list(range(0, 25, 2))
    figure.update_xaxes(
        title="Local clock time",
        range=[0, 24],
        tickmode="array",
        tickvals=tick_values,
        ticktext=[f"{hour:02d}:00" for hour in tick_values],
    )
    figure.update_yaxes(title="Date", autorange="reversed")
    figure.update_layout(
        template="plotly_white",
        height=max(440, min(900, 160 + 10 * len(nights))),
        margin={"l": 75, "r": 25, "t": 25, "b": 50},
    )
    return figure


def build_explorer_plot(frame: pd.DataFrame, metrics: list[str]) -> go.Figure:
    if frame.empty or not metrics:
        return empty_figure("Choose one or more numeric columns to plot.")
    figure = go.Figure()
    for index, metric in enumerate(metrics):
        values = pd.to_numeric(frame[metric], errors="coerce")
        figure.add_trace(
            go.Scattergl(
                x=frame["_time"],
                y=values,
                mode="lines+markers" if len(frame) < 500 else "lines",
                name=metric,
                line={"color": PALETTE[index % len(PALETTE)], "width": 1.4},
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=460,
        hovermode="x unified",
        xaxis={"title": "Recorded local time", "rangeslider": {"visible": True}},
        yaxis={"title": "Value"},
        margin={"l": 70, "r": 25, "t": 25, "b": 50},
    )
    return figure


def _group_coverage_colors(maximum_value: int) -> tuple[list[list[float | str]], list[str]]:
    """Return a discrete outside/missing/partial/complete colour scale."""

    maximum_value = max(int(maximum_value), 1)
    colours = ["#D9DEE7", "#F6C7C7"]
    for value in range(1, maximum_value + 1):
        if value == maximum_value:
            colours.append("#55A868")
        else:
            fraction = value / maximum_value
            lightness = 88 - int(35 * fraction)
            colours.append(f"hsl(211, 65%, {lightness}%)")

    scale: list[list[float | str]] = []
    count = len(colours)
    for index, colour in enumerate(colours):
        scale.extend(
            [
                [index / count, colour],
                [(index + 1) / count, colour],
            ]
        )
    return scale, colours


def build_group_coverage_heatmap(
    matrix: pd.DataFrame,
    hover_text: np.ndarray,
    *,
    maximum_value: int,
    alignment: str,
    selected_variable_label: str | None = None,
) -> go.Figure:
    """Build a participant-by-day population coverage heatmap."""

    if matrix.empty or matrix.shape[1] == 0:
        return empty_figure("No selected-variable coverage is available to plot.")
    if hover_text.shape != matrix.shape:
        raise ValueError("hover_text must have the same shape as matrix")

    maximum_value = max(int(maximum_value), 1)
    colorscale, _ = _group_coverage_colors(maximum_value)
    tick_values = [-1, *range(maximum_value + 1)]
    if selected_variable_label is None:
        tick_labels = [
            "Outside span",
            "Missing all",
            *[
                "Complete overlap" if value == maximum_value else f"{value} present"
                for value in range(1, maximum_value + 1)
            ],
        ]
        colourbar_title = "Signals present"
    else:
        tick_labels = ["Outside span", "Missing", "Present"]
        colourbar_title = selected_variable_label

    participant_positions = np.arange(len(matrix.index))
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=matrix.columns,
            y=participant_positions,
            customdata=hover_text,
            zmin=-1.5,
            zmax=maximum_value + 0.5,
            colorscale=colorscale,
            showscale=True,
            colorbar={
                "title": colourbar_title,
                "tickmode": "array",
                "tickvals": tick_values,
                "ticktext": tick_labels,
            },
            hovertemplate="%{customdata}<extra></extra>",
            hoverongaps=False,
            xgap=0,
            ygap=0,
            zsmooth=False,
        )
    )
    figure.update_xaxes(
        title=("Days since first selected-variable observation" if alignment == "relative" else "Calendar date"),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor="#667085",
    )
    figure.update_yaxes(
        title="Participant",
        autorange="reversed",
        showticklabels=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#667085",
    )
    figure.update_layout(
        title="Population data coverage",
        template="plotly_white",
        hovermode="closest",
        height=max(500, min(1600, 220 + int(len(matrix) * 1.8))),
        margin={"l": 70, "r": 40, "t": 55, "b": 55},
        uirevision="group-coverage",
    )
    return figure


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[offset : offset + 2], 16) for offset in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"
