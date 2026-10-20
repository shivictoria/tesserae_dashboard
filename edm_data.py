"""Regular-grid preparation for EDM; never compress or fill missing time steps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .data import extract_sleep, select_files_for_range, subtype_index, TIME_COLUMNS
except ImportError:
    from data import extract_sleep, select_files_for_range, subtype_index, TIME_COLUMNS

AGGREGATIONS = ("mean", "median", "sd", "min", "max", "sum", "lowest-30% mean")
SLEEP_VARIABLES = ("sleep_onset", "sleep_offset", "duration_hours")


def numeric_schema(index, category, subtype, start, end):
    """Inspect only selected-source footers, including schema changes across chunks."""
    records = select_files_for_range(subtype_index(index, category, subtype), start, end)
    names, warnings = set(), []
    for path in records.path:
        try:
            for field in pq.ParquetFile(path).schema_arrow:
                if (pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
                        or pa.types.is_decimal(field.type) or pa.types.is_boolean(field.type)):
                    if field.name not in TIME_COLUMNS:
                        names.add(field.name)
        except Exception as exc:
            warnings.append(f"Could not inspect {path}: {exc}")
    return sorted(names), warnings


def sleep_values(frame, start, end):
    sleep = extract_sleep(frame)
    if sleep.empty:
        return pd.DataFrame(columns=["_time", *SLEEP_VARIABLES]), sleep, 0
    sleep = sleep.copy()
    sleep["wake_date"] = sleep.sleep_offset.dt.normalize()
    sleep = sleep.loc[(sleep.wake_date >= start) & (sleep.wake_date < end)]
    before = len(sleep)
    sleep = sleep.sort_values(["wake_date", "duration_hours", "sleep_offset"])
    sleep = sleep.drop_duplicates("wake_date", keep="last").reset_index(drop=True)
    base = sleep.wake_date - pd.Timedelta(hours=12)
    values = pd.DataFrame({"_time": sleep.wake_date})
    for column in SLEEP_VARIABLES[:2]:
        values[column] = (sleep[column] - base).dt.total_seconds() / 3600
    values["duration_hours"] = sleep.duration_hours
    return values, sleep, before - len(sleep)


def regular_series(frame, column, start, end, interval, statistic):
    step = pd.Timedelta(interval)
    if step <= pd.Timedelta(0):
        raise ValueError("Sampling interval must be positive.")
    if end <= start:
        raise ValueError("The selected window is empty.")
    grid = pd.date_range(start, end, freq=step, inclusive="left")
    if column not in frame or "_time" not in frame:
        return pd.Series(np.nan, index=grid, name=column)
    data = pd.DataFrame({"time": pd.to_datetime(frame._time),
                         "value": pd.to_numeric(frame[column], errors="coerce")})
    data.value = data.value.replace([np.inf, -np.inf], np.nan)
    data = data.loc[data.time.ge(start) & data.time.lt(end)].set_index("time")
    grouped = data.value.resample(step, origin=start)
    if statistic == "sum":
        result = grouped.sum(min_count=1)
    elif statistic == "lowest-30% mean":
        def low_mean(values):
            values = values.dropna()
            return values.nsmallest(max(1, int(np.ceil(len(values) * .30)))).mean()
        result = grouped.apply(low_mean)
    elif statistic in AGGREGATIONS:
        result = grouped.agg("std" if statistic == "sd" else statistic)
    else:
        raise ValueError(f"Unknown aggregation: {statistic}")
    return result.reindex(grid).rename(column)


def prepare_grid(selections, start, end, interval):
    """selections: (alias, loaded frame, column, aggregation) tuples."""
    output = pd.DataFrame(index=pd.date_range(start, end, freq=pd.Timedelta(interval), inclusive="left"))
    for alias, frame, column, aggregation in selections:
        output[alias] = regular_series(frame, column, start, end, interval, aggregation)
    return output.rename_axis("time").reset_index()


def crop_overlap(frame):
    """Trim only the edges; preserve every row between jointly observed endpoints."""
    values = frame.drop(columns="time")
    empty = [column for column in values if not values[column].notna().any()]
    if empty:
        raise ValueError(f"No observations for {', '.join(empty)} in this window. Choose another variable or date range.")
    complete = np.flatnonzero(values.notna().all(axis=1).to_numpy())
    if not len(complete):
        raise ValueError("No overlapping rows with observations for all selected variables. "
                         "Choose another date range or a coarser sampling interval.")
    first, last = int(complete[0]), int(complete[-1])
    return frame.iloc[first:last + 1].reset_index(drop=True).copy(), first, len(frame) - last - 1


def missingness(frame, interval, max_missing_days=7, max_missing_percentage=20):
    if max_missing_days < 0 or not 0 <= max_missing_percentage <= 100:
        raise ValueError("Missingness limits must be nonnegative; percentage must be 0–100.")
    values = frame.drop(columns="time")
    masks = {column: values[column].isna().to_numpy() for column in values}
    masks["Combined (any selected variable)"] = values.isna().any(axis=1).to_numpy()
    rows = []
    for name, mask in masks.items():
        boundaries = np.diff(np.r_[False, mask, False].astype(int))
        lengths = np.flatnonzero(boundaries == -1) - np.flatnonzero(boundaries == 1)
        longest = int(lengths.max()) if len(lengths) else 0
        days = longest * pd.Timedelta(interval) / pd.Timedelta(days=1)
        percentage = int(mask.sum()) * 100 / len(mask) if len(mask) else 100.0
        rows.append({"variable": name, "missing_rows": int(mask.sum()),
                     "longest_gap_days": float(days), "missing_percentage": float(percentage),
                     "passes": bool(len(mask) and days <= max_missing_days
                                    and percentage <= max_missing_percentage)})
    return pd.DataFrame(rows)
