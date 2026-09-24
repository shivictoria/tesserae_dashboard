"""Metadata indexing and selective Parquet loading for the Tesserae dashboard."""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_PARTICIPANT = Path(
    "/Users/yvshi/Library/CloudStorage/OneDrive-SharedLibraries-TheUniversityofMelbourne/RiPL - Rhythms in Physiology Lab-TEMPO Study - Documents/Tesserae Data/full_parquets/parquets/36fd7269ce06b988c4f4347550a4a939"
    # "/Volumes/One Touch/parquets/36fd7269ce06b988c4f4347550a4a939"
)

DEFAULT_PARQUET_ROOT = Path(
    "/Users/yvshi/Library/CloudStorage/OneDrive-SharedLibraries-TheUniversityofMelbourne/RiPL - Rhythms in Physiology Lab-TEMPO Study - Documents/Tesserae Data/full_parquets/parquets"
)

TIME_COLUMNS = (
    "local_time",
    "date",
    "Timestamp",
    "timestamp",
    "start_time",
    "survey_start_datetime",
)

FILE_DATE_RE = re.compile(
    r"_(?P<date>\d{4}-\d{2}-\d{2})(?:-\d{2}-\d{2}-\d{2}-[^.]+)?$"
)

RESOLUTION_RULES = {
    "1 minute": "1min",
    "5 minutes": "5min",
    "15 minutes": "15min",
    "Hourly": "1h",
    "Daily": "1D",
}

ADDITIVE_TOKENS = (
    "steps",
    "step_count",
    "seconds",
    "secs",
    "kilocalories",
    "calories",
    "distance",
    "duration",
    "floors",
    "num_activities",
)


class DataPathError(ValueError):
    """Raised when a participant directory is missing or invalid."""


def _walk_parquets(root: Path) -> Iterable[Path]:
    """Walk Parquet files, following symlinked category directories safely."""
    seen_directories: set[str] = set()
    for current, directories, filenames in os.walk(root, followlinks=True):
        real_current = os.path.realpath(current)
        if real_current in seen_directories:
            directories[:] = []
            continue
        seen_directories.add(real_current)
        directories[:] = [
            name
            for name in directories
            if os.path.realpath(os.path.join(current, name)) not in seen_directories
        ]
        for filename in filenames:
            if filename.startswith("._") or not filename.lower().endswith(".parquet"):
                continue
            yield Path(current, filename)


def parse_file_date(path: str | Path) -> pd.Timestamp | pd.NaT:
    """Extract the chunk date encoded at the end of a Tesserae filename."""
    match = FILE_DATE_RE.search(Path(path).stem)
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group("date"), errors="coerce")


def parse_subtype(path: str | Path, participant_id: str, category: str) -> str:
    """Return the exact subtype between participant/category and timestamp."""
    stem = Path(path).stem
    prefix = f"{participant_id}_{category}_"
    remainder = stem[len(prefix) :] if stem.startswith(prefix) else stem
    match = FILE_DATE_RE.search(remainder)
    subtype = remainder[: match.start()] if match else remainder
    return subtype.rstrip("_") or "Unknown"


def validate_participant_path(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_dir():
        raise DataPathError(
            f"Participant directory not found: {root}. "
            "Connect the external drive or choose another participant directory."
        )
    try:
        first = next(_walk_parquets(root), None)
    except OSError as error:
        raise DataPathError(f"Could not read participant directory: {error}") from error
    if first is None:
        raise DataPathError(f"No Parquet files were found under: {root}")
    return root.resolve()


def list_participant_paths(path: str | Path) -> list[Path]:
    """List immediate participant directories beneath a Parquet root."""
    root = Path(path).expanduser()
    if not root.is_dir():
        raise DataPathError(
            f"Parquet root directory not found: {root}. "
            "Connect the external drive or choose another Parquet root."
        )
    try:
        participants = sorted(
            (
                entry
                for entry in root.iterdir()
                if not entry.name.startswith(".") and entry.is_dir()
            ),
            key=lambda entry: entry.name.casefold(),
        )
    except OSError as error:
        raise DataPathError(f"Could not list Parquet root directory: {error}") from error
    if not participants:
        raise DataPathError(f"No participant subfolders were found under: {root}")
    return participants


def build_index(path: str | Path, *, inspect_metadata: bool = True) -> pd.DataFrame:
    """Build a file catalogue, optionally opening every Parquet footer."""
    root = validate_participant_path(path)
    participant_id = root.name
    records: list[dict] = []

    for parquet_path in sorted(_walk_parquets(root), key=lambda item: str(item)):
        try:
            relative = parquet_path.relative_to(root)
            category = relative.parts[0] if len(relative.parts) > 1 else "Uncategorised"
        except ValueError:
            category = parquet_path.parent.name
        subtype = parse_subtype(parquet_path, participant_id, category)
        record = {
            "path": str(parquet_path),
            "category": category,
            "subtype": subtype,
            "file_date": parse_file_date(parquet_path),
            "rows": 0,
            "size_bytes": 0,
            "columns": tuple(),
            "schema_signature": tuple(),
            "status": "indexed" if not inspect_metadata else "ok",
            "error": "",
        }
        if not inspect_metadata:
            records.append(record)
            continue
        try:
            record["size_bytes"] = parquet_path.stat().st_size
            parquet = pq.ParquetFile(parquet_path)
            schema = parquet.schema_arrow
            record["rows"] = parquet.metadata.num_rows
            record["columns"] = tuple(schema.names)
            record["schema_signature"] = tuple(
                (field.name, str(field.type)) for field in schema
            )
            if parquet.metadata.num_rows == 0:
                record["status"] = "empty"
        except Exception as error:  # metadata errors must not hide the rest of the data
            record["status"] = "unreadable"
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame["file_date"] = pd.to_datetime(frame["file_date"], errors="coerce")
    return frame


def index_date_bounds(index: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.to_datetime(index.get("file_date"), errors="coerce").dropna()
    if dates.empty:
        now = pd.Timestamp.now().normalize()
        return now, now
    return dates.min().normalize(), dates.max().normalize()


def subtype_index(index: pd.DataFrame, category: str, subtype: str) -> pd.DataFrame:
    return index.loc[
        (index["category"] == category)
        & (index["subtype"] == subtype)
        & (index["status"].isin(["ok", "empty", "indexed"]))
    ].copy()


def select_files_for_range(
    records: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    """Prune daily chunks while retaining consolidated and undated files."""
    if records.empty or start is None or end is None or len(records) <= 2:
        return records
    dates = pd.to_datetime(records["file_date"], errors="coerce")
    lower = pd.Timestamp(start).normalize() - pd.Timedelta(days=1)
    upper = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    return records.loc[dates.isna() | dates.between(lower, upper)]


def _requested_columns(
    available: Sequence[str], requested: Sequence[str] | None
) -> list[str]:
    if requested is None:
        return list(available)
    wanted = list(dict.fromkeys([*TIME_COLUMNS, *requested]))
    return [name for name in wanted if name in available]


def _parse_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize(None)
    return parsed


def add_display_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `_time`, preferring recorded local clock time over UTC."""
    output = frame.copy()
    result = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns]")
    for name in TIME_COLUMNS:
        if name not in output.columns:
            continue
        parsed = _parse_datetime(output[name])
        # Parquet chunks occasionally disagree between date32 and timestamp for
        # local_time. Pandas otherwise retains a mixed object column that
        # Streamlit cannot serialize back to Arrow. Normalise local calendar
        # fields, but leave the provenance-bearing UTC Timestamp untouched.
        if name in {"local_time", "date"}:
            output[name] = parsed
        result = result.where(result.notna(), parsed)
    output["_time"] = result
    return output


def load_subtype(
    index: pd.DataFrame,
    category: str,
    subtype: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    *,
    columns: Sequence[str] | None = None,
    row_limit: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load only relevant chunks and return data plus non-fatal read warnings."""
    records = select_files_for_range(subtype_index(index, category, subtype), start, end)
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    rows_read = 0

    for record in records.itertuples(index=False):
        if row_limit is not None and rows_read >= row_limit:
            break
        try:
            source_file = pq.ParquetFile(record.path)
            available = (
                list(record.columns)
                if record.columns
                else list(source_file.schema_arrow.names)
            )
            selected_columns = _requested_columns(available, columns)
            if not selected_columns:
                warnings.append(f"No requested columns in {record.path}")
                continue
            table = source_file.read(columns=selected_columns)
            frame = table.to_pandas()
            if row_limit is not None:
                remaining = row_limit - rows_read
                frame = frame.iloc[:remaining]
            frames.append(frame)
            rows_read += len(frame)
        except Exception as error:
            warnings.append(f"{Path(record.path).name}: {type(error).__name__}: {error}")

    if not frames:
        return pd.DataFrame(columns=["_time"]), warnings

    output = add_display_time(pd.concat(frames, ignore_index=True, sort=False))
    if start is not None:
        output = output.loc[output["_time"].isna() | (output["_time"] >= start)]
    if end is not None:
        output = output.loc[output["_time"].isna() | (output["_time"] < end)]
    return output.sort_values("_time", na_position="last").reset_index(drop=True), warnings


def resolve_resolution(label: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    if label == "Raw":
        return "raw"
    if label != "Auto":
        return RESOLUTION_RULES[label]
    span = end - start
    if span <= pd.Timedelta(days=7):
        return "1min"
    if span <= pd.Timedelta(days=31):
        return "15min"
    if span <= pd.Timedelta(days=183):
        return "1h"
    return "1D"


def default_statistic(metric: str) -> str:
    lowered = metric.lower()
    return "sum" if any(token in lowered for token in ADDITIVE_TOKENS) else "mean"


def aggregate_metric(
    frame: pd.DataFrame,
    metric: str,
    resolution: str,
    statistic: str,
    *,
    include_sd: bool = False,
) -> pd.DataFrame:
    """Return `_time`, `value`, and optional `sd` for one numeric metric."""
    if frame.empty or metric not in frame.columns:
        return pd.DataFrame(columns=["_time", "value", "sd"])
    working = frame.loc[:, ["_time", metric]].copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working = working.dropna(subset=["_time", metric]).sort_values("_time")
    if working.empty:
        return pd.DataFrame(columns=["_time", "value", "sd"])

    if resolution == "raw" or statistic == "native":
        return working.rename(columns={metric: "value"}).assign(sd=np.nan)

    effective_stat = default_statistic(metric) if statistic == "default" else statistic
    grouped = working.set_index("_time")[metric].resample(resolution)
    aggregators = {
        "mean": grouped.mean,
        "median": grouped.median,
        "sd": grouped.std,
        # "min": grouped.min,
        # "max": grouped.max,
        "sum": grouped.sum,
    }
    if effective_stat not in aggregators:
        raise ValueError(f"Unsupported statistic: {effective_stat}")
    values = aggregators[effective_stat]().rename("value")
    result = values.reset_index()
    result["sd"] = grouped.std().to_numpy() if include_sd else np.nan
    return result.dropna(subset=["value"]).reset_index(drop=True)


SLEEP_STAGES = ("light", "deep", "rem", "awake", "unmeasurable")
SLEEP_VALIDATION_PRIORITY = {
    "AUTO_TENTATIVE": 1,
    "ENHANCED_TENTATIVE": 2,
    "AUTO_FINAL": 3,
    "ENHANCED_FINAL": 4,
    "AUTO_MANUAL": 5,
    "MANUAL": 6,
}


def _parse_sleep_levels_map(
    raw_map: object,
) -> tuple[
    dict[str, float],
    pd.Timestamp | pd.NaT,
    pd.Timestamp | pd.NaT,
    list[tuple[str, pd.Timestamp, pd.Timestamp]],
]:
    """Extract stage durations and sleep boundaries from Garmin stage JSON."""
    durations = {stage: 0.0 for stage in SLEEP_STAGES}
    sleep_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    unmeasurable_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    stage_intervals: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    if raw_map is None or (not isinstance(raw_map, str) and pd.isna(raw_map)):
        return durations, pd.NaT, pd.NaT, stage_intervals
    try:
        parsed = json.loads(str(raw_map))
    except (TypeError, ValueError, json.JSONDecodeError):
        return durations, pd.NaT, pd.NaT, stage_intervals
    if not isinstance(parsed, dict):
        return durations, pd.NaT, pd.NaT, stage_intervals

    for stage in SLEEP_STAGES:
        segments = parsed.get(stage, [])
        if not isinstance(segments, list):
            continue
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            start = pd.to_datetime(segment.get("startTime"), errors="coerce")
            end = pd.to_datetime(segment.get("endTime"), errors="coerce")
            if pd.isna(start) or pd.isna(end):
                continue
            if end <= start:
                end += pd.Timedelta(days=1)
            seconds = (end - start).total_seconds()
            if 0 < seconds <= 24 * 3600:
                durations[stage] += seconds
                stage_intervals.append((stage, start, end))
                if stage in {"light", "deep", "rem"}:
                    sleep_intervals.append((start, end))
                elif stage == "unmeasurable":
                    unmeasurable_intervals.append((start, end))

    # If a device provides only an unmeasurable stage, its interval is the best
    # available sleep boundary. Awake segments alone never define sleep.
    if not sleep_intervals and durations["unmeasurable"] > 0:
        sleep_intervals = unmeasurable_intervals
    if not sleep_intervals:
        return durations, pd.NaT, pd.NaT, stage_intervals
    onset = min(interval[0] for interval in sleep_intervals)
    offset = max(interval[1] for interval in sleep_intervals)
    return durations, onset, offset, stage_intervals


def extract_sleep(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract nightly stages, onset, and offset solely from sleep_levels_map.

    The input is the concatenated GarminHealthAPI/Sleep data returned by
    :func:`load_subtype`. The result has one row per onset date. Repeated
    exports are resolved by validation status and mapped stage coverage.
    """
    output_columns = [
        "night",
        "sleep_onset",
        "sleep_offset",
        "duration_hours",
        "stage_intervals",
        *[f"{stage}_sleep_seconds" for stage in ("light", "deep", "rem")],
        "awake_seconds",
        "unmeasurable_sleep_seconds",
        "duration_seconds",
        "validation",
    ]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    records: list[dict] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        durations, onset, offset, stage_intervals = _parse_sleep_levels_map(
            row.get("sleep_levels_map")
        )
        if pd.isna(onset) or pd.isna(offset) or offset <= onset:
            continue
        values = {
            "light_sleep_seconds": durations["light"],
            "deep_sleep_seconds": durations["deep"],
            "rem_sleep_seconds": durations["rem"],
            "awake_seconds": durations["awake"],
            "unmeasurable_sleep_seconds": durations["unmeasurable"],
        }
        stage_total = float(sum(values.values()))
        records.append(
            {
                "night": onset.normalize(),
                "sleep_onset": onset,
                "sleep_offset": offset,
                "duration_hours": (offset - onset).total_seconds() / 3600,
                "stage_intervals": stage_intervals,
                **values,
                "duration_seconds": stage_total,
                "validation": row.get("validation", ""),
                "_validation_rank": SLEEP_VALIDATION_PRIORITY.get(
                    str(row.get("validation", "")), 0
                ),
                "_stage_total": stage_total,
                "_position": position,
            }
        )

    if not records:
        return pd.DataFrame(columns=output_columns)
    result = pd.DataFrame.from_records(records)
    result = result.sort_values(
        ["night", "_validation_rank", "_stage_total", "_position"]
    ).drop_duplicates("night", keep="last")
    return result.loc[:, output_columns].sort_values("night").reset_index(drop=True)


def circadian_values(
    frame: pd.DataFrame,
    metric: str,
    minutes: int,
    day_filter: str = "All days",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build date×clock-time values and an across-day mean/SD profile."""
    if frame.empty or metric not in frame.columns:
        return pd.DataFrame(), pd.DataFrame(columns=["clock_hour", "mean", "sd", "group"])
    data = frame.loc[:, ["_time", metric]].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna()
    if day_filter == "Weekdays":
        data = data.loc[data["_time"].dt.dayofweek < 5]
    elif day_filter == "Weekends":
        data = data.loc[data["_time"].dt.dayofweek >= 5]
    data["date"] = data["_time"].dt.normalize()
    minute_of_day = data["_time"].dt.hour * 60 + data["_time"].dt.minute
    data["clock_minute"] = (minute_of_day // minutes) * minutes
    binned = data.groupby(["date", "clock_minute"], observed=True)[metric].mean()
    matrix = binned.unstack("clock_minute").sort_index()
    by_clock = binned.groupby("clock_minute")
    profile = pd.DataFrame({"mean": by_clock.mean(), "sd": by_clock.std()}).reset_index()
    profile["clock_hour"] = profile["clock_minute"] / 60
    profile["group"] = day_filter
    return matrix, profile


def coverage_table(index: pd.DataFrame) -> pd.DataFrame:
    if index.empty:
        return pd.DataFrame()
    grouped = index.groupby(["category", "subtype"], dropna=False)
    coverage = grouped.agg(
        files=("path", "size"),
        rows=("rows", "sum"),
        size_bytes=("size_bytes", "sum"),
        first_file=("file_date", "min"),
        last_file=("file_date", "max"),
        schema_versions=("schema_signature", "nunique"),
        problem_files=("status", lambda values: int((values != "ok").sum())),
    ).reset_index()
    coverage["size_mb"] = (coverage.pop("size_bytes") / 1024**2).round(1)
    return coverage.sort_values(["category", "subtype"]).reset_index(drop=True)


def missing_dates(records: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.to_datetime(records["file_date"], errors="coerce").dropna().dt.normalize()
    if dates.nunique() < 5:
        return []
    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    return list(expected.difference(pd.DatetimeIndex(dates.unique())))
