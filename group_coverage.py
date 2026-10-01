"""Population-level daily data coverage for the standalone dashboard.

Coverage records contain dates and presence flags only. Physiological values
are never persisted in the dashboard cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:  # Package imports used by tests and ``python -m``.
    from .data import (
        TIME_COLUMNS,
        _walk_parquets,
        list_participant_paths,
        parse_subtype,
    )
except ImportError:  # Direct imports used by ``streamlit run app.py``.
    from data import (  # type: ignore
        TIME_COLUMNS,
        _walk_parquets,
        list_participant_paths,
        parse_subtype,
    )


CATALOGUE_COLUMNS = (
    "variable_id",
    "category",
    "subtype",
    "column",
    "arrow_type",
    "timestamp_column",
    "participants_sampled",
)

COVERAGE_COLUMNS = (
    "participant_id",
    "date",
    "variable_id",
)


@dataclass(frozen=True)
class CoverageVariable:
    """A column in one physical Tesserae category/subtype source."""

    category: str
    subtype: str
    column: str
    arrow_type: str = ""
    timestamp_column: str | None = None

    @property
    def variable_id(self) -> str:
        return variable_id(self.category, self.subtype, self.column)

    @property
    def label(self) -> str:
        return f"{self.category} / {self.subtype} / {self.column}"


@dataclass
class CoverageView:
    """Prepared participant-by-day coverage data for plotting."""

    matrix: pd.DataFrame
    hover_text: np.ndarray
    participant_summary: pd.DataFrame
    variable_summary: pd.DataFrame
    combination_summary: pd.DataFrame
    alignment: str
    maximum_value: int


def variable_id(category: str, subtype: str, column: str) -> str:
    """Return an unambiguous, stable identifier for a physical column."""

    return json.dumps(
        [str(category), str(subtype), str(column)],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def variable_from_record(record: Mapping[str, object] | pd.Series) -> CoverageVariable:
    timestamp = record.get("timestamp_column")
    if timestamp is None or pd.isna(timestamp) or str(timestamp) == "":
        timestamp = None
    return CoverageVariable(
        category=str(record["category"]),
        subtype=str(record["subtype"]),
        column=str(record["column"]),
        arrow_type=str(record.get("arrow_type", "")),
        timestamp_column=str(timestamp) if timestamp is not None else None,
    )


def coverage_cache_directory(
    data_root: str | Path,
    cache_base: str | Path | None = None,
) -> Path:
    """Return the root-specific persistent dashboard cache directory."""

    root = str(Path(data_root).expanduser().resolve())
    root_hash = hashlib.sha256(root.encode("utf-8")).hexdigest()[:16]
    configured = os.environ.get("TESSERAE_DASHBOARD_CACHE_DIR")
    base = Path(
        cache_base
        if cache_base is not None
        else configured
        if configured
        else Path.home() / ".cache" / "tesserae_dashboard"
    ).expanduser()
    return base / root_hash


def _catalogue_path(data_root: str | Path, cache_base: str | Path | None = None) -> Path:
    return coverage_cache_directory(data_root, cache_base) / "variable_catalogue.parquet"


def _variable_cache_stem(variable: CoverageVariable) -> str:
    digest = hashlib.sha256(variable.variable_id.encode("utf-8")).hexdigest()[:20]
    return digest


def _coverage_path(
    data_root: str | Path,
    variable: CoverageVariable,
    cache_base: str | Path | None = None,
) -> Path:
    return coverage_cache_directory(data_root, cache_base) / "coverage" / (
        _variable_cache_stem(variable) + ".parquet"
    )


def _manifest_path(
    data_root: str | Path,
    variable: CoverageVariable,
    cache_base: str | Path | None = None,
) -> Path:
    return coverage_cache_directory(data_root, cache_base) / "coverage" / (
        _variable_cache_stem(variable) + ".json"
    )


def _error_path(
    data_root: str | Path,
    variable: CoverageVariable,
    cache_base: str | Path | None = None,
) -> Path:
    return coverage_cache_directory(data_root, cache_base) / "coverage" / (
        _variable_cache_stem(variable) + ".errors.parquet"
    )


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.stem + "-",
        suffix=".parquet",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(data: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.stem + "-",
        suffix=".json",
        dir=destination.parent,
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(data, handle, indent=2)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _source_for_path(path: Path, participant_path: Path) -> tuple[str, str]:
    participant_id = participant_path.name
    try:
        relative = path.relative_to(participant_path)
        category = relative.parts[0] if len(relative.parts) > 1 else "Uncategorised"
    except ValueError:
        category = path.parent.name
    return category, parse_subtype(path, participant_id, category)


def _preferred_time_column(columns: Iterable[str]) -> str | None:
    available = set(columns)
    return next((column for column in TIME_COLUMNS if column in available), None)


def discover_variables(
    data_root: str | Path,
    participant_paths: Sequence[str | Path] | None = None,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover source columns from one representative file per participant/source.

    Sampling every participant captures cross-participant schema variants without
    opening the footer of every daily chunk. Coverage builds still inspect every
    selected file and safely tolerate later schema drift.
    """

    root = Path(data_root).expanduser().resolve()
    participants = (
        [Path(path).expanduser().resolve() for path in participant_paths]
        if participant_paths is not None
        else list_participant_paths(root)
    )
    observed: dict[tuple[str, str, str], dict[str, object]] = {}
    errors: list[dict[str, str]] = []
    total = len(participants)

    for position, participant_path in enumerate(participants, start=1):
        if progress is not None:
            progress(position - 1, total, participant_path.name)
        samples: dict[tuple[str, str], Path] = {}
        try:
            for path in _walk_parquets(participant_path):
                source = _source_for_path(path, participant_path)
                samples.setdefault(source, path)
        except Exception as error:
            errors.append(
                {
                    "participant_id": participant_path.name,
                    "path": str(participant_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        for (category, subtype), path in samples.items():
            try:
                schema = pq.ParquetFile(path).schema_arrow
            except Exception as error:
                errors.append(
                    {
                        "participant_id": participant_path.name,
                        "path": str(path),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            time_column = _preferred_time_column(schema.names)
            for field in schema:
                key = (category, subtype, field.name)
                record = observed.setdefault(
                    key,
                    {
                        "types": set(),
                        "time_columns": set(),
                        "participants": set(),
                    },
                )
                record["types"].add(str(field.type))  # type: ignore[union-attr]
                if time_column is not None:
                    record["time_columns"].add(time_column)  # type: ignore[union-attr]
                record["participants"].add(participant_path.name)  # type: ignore[union-attr]

    records: list[dict[str, object]] = []
    for (category, subtype, column), details in observed.items():
        time_columns = details["time_columns"]
        timestamp_column = next(
            (name for name in TIME_COLUMNS if name in time_columns),
            None,
        )
        records.append(
            {
                "variable_id": variable_id(category, subtype, column),
                "category": category,
                "subtype": subtype,
                "column": column,
                "arrow_type": " | ".join(sorted(details["types"])),
                "timestamp_column": timestamp_column,
                "participants_sampled": len(details["participants"]),
            }
        )

    catalogue = pd.DataFrame.from_records(records, columns=CATALOGUE_COLUMNS)
    if not catalogue.empty:
        catalogue = catalogue.sort_values(
            ["category", "subtype", "column"]
        ).reset_index(drop=True)
    error_frame = pd.DataFrame.from_records(
        errors,
        columns=["participant_id", "path", "error"],
    )
    if progress is not None:
        progress(total, total, "Complete")
    return catalogue, error_frame


def save_variable_catalogue(
    catalogue: pd.DataFrame,
    data_root: str | Path,
    cache_base: str | Path | None = None,
) -> Path:
    path = _catalogue_path(data_root, cache_base)
    _atomic_parquet(catalogue.loc[:, list(CATALOGUE_COLUMNS)], path)
    return path


def load_variable_catalogue(
    data_root: str | Path,
    cache_base: str | Path | None = None,
) -> pd.DataFrame:
    path = _catalogue_path(data_root, cache_base)
    if not path.is_file():
        return pd.DataFrame(columns=CATALOGUE_COLUMNS)
    frame = pd.read_parquet(path)
    for column in CATALOGUE_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame.loc[:, list(CATALOGUE_COLUMNS)]


def _normalise_datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _stat_date(value: object) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _single_day_from_statistics(
    parquet: pq.ParquetFile,
    row_group_index: int,
    time_column: str,
    value_column: str,
) -> pd.Timestamp | None:
    """Return an exact covered date when row-group statistics prove one."""

    names = parquet.schema_arrow.names
    if time_column not in names or value_column not in names:
        return None
    row_group = parquet.metadata.row_group(row_group_index)
    time_statistics = row_group.column(names.index(time_column)).statistics
    value_statistics = row_group.column(names.index(value_column)).statistics
    if time_statistics is None or value_statistics is None:
        return None
    if time_statistics.has_null_count and time_statistics.null_count:
        return None
    if not value_statistics.has_null_count:
        return None
    if value_statistics.null_count >= row_group.num_rows:
        return None
    minimum = _stat_date(time_statistics.min)
    maximum = _stat_date(time_statistics.max)
    if minimum is None or maximum is None or minimum != maximum:
        return None
    return minimum


def _scan_row_group_dates(
    parquet: pq.ParquetFile,
    row_group_index: int,
    time_column: str,
    value_column: str,
) -> set[pd.Timestamp]:
    columns = list(dict.fromkeys([time_column, value_column]))
    dates: set[pd.Timestamp] = set()
    for batch in parquet.iter_batches(
        batch_size=65_536,
        row_groups=[row_group_index],
        columns=columns,
    ):
        frame = batch.to_pandas()
        times = _normalise_datetime(frame[time_column])
        values = frame[value_column]
        valid = times.notna() & values.notna()
        if valid.any():
            dates.update(times.loc[valid].dt.normalize().unique())
    return {pd.Timestamp(date) for date in dates if not pd.isna(date)}


def file_coverage_dates(
    parquet: pq.ParquetFile,
    time_column: str,
    value_column: str,
) -> set[pd.Timestamp]:
    """Return dates with at least one non-null value in a Parquet file."""

    if time_column not in parquet.schema_arrow.names:
        return set()
    if value_column not in parquet.schema_arrow.names:
        return set()
    dates: set[pd.Timestamp] = set()
    for row_group_index in range(parquet.metadata.num_row_groups):
        fast_date = _single_day_from_statistics(
            parquet,
            row_group_index,
            time_column,
            value_column,
        )
        if fast_date is not None:
            dates.add(fast_date)
        else:
            dates.update(
                _scan_row_group_dates(
                    parquet,
                    row_group_index,
                    time_column,
                    value_column,
                )
            )
    return dates


def build_and_cache_coverages(
    data_root: str | Path,
    variables: Sequence[CoverageVariable],
    participant_paths: Sequence[str | Path] | None = None,
    *,
    cache_base: str | Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, dict[str, object]]:
    """Build and persist exact daily coverage for selected variables."""

    root = Path(data_root).expanduser().resolve()
    participants = (
        [Path(path).expanduser().resolve() for path in participant_paths]
        if participant_paths is not None
        else list_participant_paths(root)
    )
    unique_variables = {item.variable_id: item for item in variables}
    by_source: dict[tuple[str, str], list[CoverageVariable]] = {}
    for item in unique_variables.values():
        if item.timestamp_column is None:
            raise ValueError(f"Variable has no usable timestamp: {item.label}")
        by_source.setdefault((item.category, item.subtype), []).append(item)

    observed: dict[str, set[tuple[str, pd.Timestamp]]] = {
        item.variable_id: set() for item in unique_variables.values()
    }
    files_processed = {item.variable_id: 0 for item in unique_variables.values()}
    errors: dict[str, list[dict[str, str]]] = {
        item.variable_id: [] for item in unique_variables.values()
    }
    total = len(participants)

    for position, participant_path in enumerate(participants, start=1):
        participant_id = participant_path.name
        if progress is not None:
            progress(position - 1, total, participant_id)
        try:
            paths = _walk_parquets(participant_path)
            matched_files = 0
            for path in paths:
                source = _source_for_path(path, participant_path)
                source_variables = by_source.get(source)
                if not source_variables:
                    continue
                matched_files += 1
                # Besides updating the UI, this gives Streamlit a regular
                # checkpoint at which a Stop/rerun request can interrupt a
                # participant with many source files.
                if progress is not None and matched_files % 100 == 0:
                    progress(position - 1, total, participant_id)
                try:
                    parquet = pq.ParquetFile(path)
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    for item in source_variables:
                        errors[item.variable_id].append(
                            {
                                "participant_id": participant_id,
                                "path": str(path),
                                "error": message,
                            }
                        )
                    continue

                names = set(parquet.schema_arrow.names)
                actual_time = _preferred_time_column(names)
                for item in source_variables:
                    if item.column not in names or actual_time is None:
                        continue
                    files_processed[item.variable_id] += 1
                    try:
                        dates = file_coverage_dates(
                            parquet,
                            actual_time,
                            item.column,
                        )
                    except Exception as error:
                        errors[item.variable_id].append(
                            {
                                "participant_id": participant_id,
                                "path": str(path),
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                        continue
                    observed[item.variable_id].update(
                        (participant_id, date) for date in dates
                    )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            for item in unique_variables.values():
                errors[item.variable_id].append(
                    {
                        "participant_id": participant_id,
                        "path": str(participant_path),
                        "error": message,
                    }
                )

    built_at = datetime.now(timezone.utc).isoformat()
    manifests: dict[str, dict[str, object]] = {}
    for item in unique_variables.values():
        rows = sorted(observed[item.variable_id], key=lambda value: (value[0], value[1]))
        coverage = pd.DataFrame.from_records(
            (
                {
                    "participant_id": participant_id,
                    "date": date,
                    "variable_id": item.variable_id,
                }
                for participant_id, date in rows
            ),
            columns=COVERAGE_COLUMNS,
        )
        if not coverage.empty:
            coverage["date"] = pd.to_datetime(coverage["date"]).dt.normalize()
        error_frame = pd.DataFrame.from_records(
            errors[item.variable_id],
            columns=["participant_id", "path", "error"],
        )
        _atomic_parquet(coverage, _coverage_path(root, item, cache_base))
        error_path = _error_path(root, item, cache_base)
        if error_frame.empty:
            error_path.unlink(missing_ok=True)
        else:
            _atomic_parquet(error_frame, error_path)
        manifest: dict[str, object] = {
            "variable_id": item.variable_id,
            "label": item.label,
            "category": item.category,
            "subtype": item.subtype,
            "column": item.column,
            "timestamp_column": item.timestamp_column,
            "data_root": str(root),
            "built_at": built_at,
            "participants_scanned": total,
            "participants_with_data": (
                int(coverage["participant_id"].nunique()) if not coverage.empty else 0
            ),
            "participant_days": len(coverage),
            "files_processed": files_processed[item.variable_id],
            "errors": len(error_frame),
        }
        _atomic_json(manifest, _manifest_path(root, item, cache_base))
        manifests[item.variable_id] = manifest

    if progress is not None:
        progress(total, total, "Complete")
    return manifests


def load_cached_coverage(
    data_root: str | Path,
    variable: CoverageVariable,
    cache_base: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object] | None, pd.DataFrame]:
    coverage_path = _coverage_path(data_root, variable, cache_base)
    manifest_path = _manifest_path(data_root, variable, cache_base)
    error_path = _error_path(data_root, variable, cache_base)
    if not coverage_path.is_file() or not manifest_path.is_file():
        return (
            pd.DataFrame(columns=COVERAGE_COLUMNS),
            None,
            pd.DataFrame(columns=["participant_id", "path", "error"]),
        )
    coverage = pd.read_parquet(coverage_path)
    coverage["date"] = pd.to_datetime(coverage["date"], errors="coerce").dt.normalize()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    errors = (
        pd.read_parquet(error_path)
        if error_path.is_file()
        else pd.DataFrame(columns=["participant_id", "path", "error"])
    )
    return coverage, manifest, errors


def _variable_labels(
    variables: Sequence[CoverageVariable],
) -> dict[str, str]:
    return {item.variable_id: item.label for item in variables}


def prepare_coverage_view(
    coverage: pd.DataFrame,
    participants: Sequence[str],
    variables: Sequence[CoverageVariable],
    *,
    alignment: str = "relative",
    display_variable_id: str | None = None,
    participant_sort: str = "Study span",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> CoverageView:
    """Create matrices and summaries for the group coverage view."""

    if alignment not in {"relative", "calendar"}:
        raise ValueError("alignment must be 'relative' or 'calendar'")
    if not variables:
        raise ValueError("At least one variable is required")
    selected_ids = [item.variable_id for item in variables]
    labels = _variable_labels(variables)
    if display_variable_id is not None and display_variable_id not in selected_ids:
        raise ValueError("display_variable_id must identify a selected variable")

    all_participants = list(dict.fromkeys(str(item) for item in participants))
    selected = coverage.loc[
        coverage.get("variable_id", pd.Series(dtype=str)).isin(selected_ids),
        list(COVERAGE_COLUMNS),
    ].copy()
    selected["participant_id"] = selected["participant_id"].astype(str)
    selected = selected.loc[selected["participant_id"].isin(all_participants)]
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce").dt.normalize()
    selected = selected.dropna(subset=["date"]).drop_duplicates()

    days_by_participant: dict[str, dict[pd.Timestamp, set[str]]] = {}
    for row in selected.itertuples(index=False):
        values = days_by_participant.setdefault(row.participant_id, {}).setdefault(
            row.date, set()
        )
        values.add(row.variable_id)

    participant_dates = selected.groupby("participant_id")["date"] if not selected.empty else None
    first_dates = participant_dates.min().to_dict() if participant_dates is not None else {}
    last_dates = participant_dates.max().to_dict() if participant_dates is not None else {}

    participant_records: list[dict[str, object]] = []
    participant_variable_records: list[dict[str, object]] = []
    for participant_id in all_participants:
        first = first_dates.get(participant_id, pd.NaT)
        last = last_dates.get(participant_id, pd.NaT)
        span_days = int((last - first).days + 1) if pd.notna(first) and pd.notna(last) else 0
        participant_days = days_by_participant.get(participant_id, {})
        days_any = sum(bool(values) for values in participant_days.values())
        days_all = sum(len(values) == len(selected_ids) for values in participant_days.values())
        record: dict[str, object] = {
            "participant_id": participant_id,
            "first_date": first,
            "last_date": last,
            "span_days": span_days,
            "days_any": days_any,
            "internal_missing_days": max(span_days - days_any, 0),
            "all_variables_days": days_all,
            "any_coverage_percent": round(100 * days_any / span_days, 2) if span_days else np.nan,
            "complete_overlap_percent": round(100 * days_all / span_days, 2) if span_days else np.nan,
        }
        for variable_identifier in selected_ids:
            covered = sum(
                variable_identifier in values for values in participant_days.values()
            )
            participant_variable_records.append(
                {
                    "participant_id": participant_id,
                    "variable_id": variable_identifier,
                    "variable": labels[variable_identifier],
                    "covered_days": covered,
                    "span_days": span_days,
                    "coverage_percent": (
                        round(100 * covered / span_days, 2) if span_days else np.nan
                    ),
                }
            )
        participant_records.append(record)

    participant_summary = pd.DataFrame.from_records(participant_records)
    participant_variable = pd.DataFrame.from_records(participant_variable_records)
    if not participant_variable.empty:
        wide = participant_variable.pivot(
            index="participant_id",
            columns="variable",
            values="coverage_percent",
        ).add_prefix("coverage_percent: ")
        participant_summary = participant_summary.merge(
            wide.reset_index(),
            on="participant_id",
            how="left",
        )

    if participant_sort == "Participant ID":
        participant_summary = participant_summary.sort_values("participant_id")
    elif participant_sort == "Study start":
        participant_summary = participant_summary.sort_values(
            ["first_date", "participant_id"], na_position="last"
        )
    elif participant_sort == "Completeness":
        participant_summary = participant_summary.sort_values(
            ["complete_overlap_percent", "participant_id"],
            ascending=[False, True],
            na_position="last",
        )
    else:
        participant_summary = participant_summary.sort_values(
            ["span_days", "participant_id"], ascending=[False, True]
        )
    ordered_participants = participant_summary["participant_id"].tolist()

    anchored = participant_summary.loc[participant_summary["span_days"] > 0]
    if alignment == "relative":
        maximum_day = int(anchored["span_days"].max() - 1) if not anchored.empty else 0
        columns: pd.Index = pd.Index(range(maximum_day + 1), name="day")
    else:
        available_start = selected["date"].min() if not selected.empty else pd.NaT
        available_end = selected["date"].max() if not selected.empty else pd.NaT
        lower = pd.Timestamp(start).normalize() if start is not None else available_start
        upper = pd.Timestamp(end).normalize() if end is not None else available_end
        if pd.isna(lower) or pd.isna(upper) or upper < lower:
            columns = pd.DatetimeIndex([], name="date")
        else:
            columns = pd.date_range(lower, upper, freq="D", name="date")

    matrix = pd.DataFrame(
        -1.0,
        index=pd.Index(ordered_participants, name="participant_id"),
        columns=columns,
    )
    hover_text = np.empty(matrix.shape, dtype=object)
    hover_text[:] = "No selected-variable data for this participant"

    combinations: dict[tuple[str, ...], int] = {}
    for row_index, participant_id in enumerate(ordered_participants):
        first = first_dates.get(participant_id, pd.NaT)
        last = last_dates.get(participant_id, pd.NaT)
        if pd.isna(first) or pd.isna(last):
            continue
        participant_days = days_by_participant.get(participant_id, {})
        span = pd.date_range(first, last, freq="D")
        for date in span:
            x_value: object = int((date - first).days) if alignment == "relative" else date
            if x_value not in matrix.columns:
                continue
            column_index = matrix.columns.get_loc(x_value)
            present_ids = participant_days.get(date, set())
            present_selected = [item for item in selected_ids if item in present_ids]
            missing_selected = [item for item in selected_ids if item not in present_ids]
            value = (
                int(display_variable_id in present_ids)
                if display_variable_id is not None
                else len(present_selected)
            )
            matrix.iat[row_index, column_index] = value
            present_labels = [labels[item] for item in present_selected]
            missing_labels = [labels[item] for item in missing_selected]
            relative_day = int((date - first).days)
            hover_text[row_index, column_index] = (
                f"Participant: {participant_id}<br>"
                f"Date: {date:%Y-%m-%d}<br>"
                f"Study day: {relative_day}<br>"
                f"Present: {', '.join(present_labels) if present_labels else 'None'}<br>"
                f"Missing: {', '.join(missing_labels) if missing_labels else 'None'}"
            )
            combinations[tuple(present_selected)] = combinations.get(
                tuple(present_selected), 0
            ) + 1

        for column_index, x_value in enumerate(matrix.columns):
            if matrix.iat[row_index, column_index] != -1:
                continue
            date = (
                first + pd.Timedelta(days=int(x_value))
                if alignment == "relative"
                else pd.Timestamp(x_value)
            )
            hover_text[row_index, column_index] = (
                f"Participant: {participant_id}<br>"
                f"Date: {date:%Y-%m-%d}<br>Outside selected-variable study span"
            )

    variable_records: list[dict[str, object]] = []
    total_span_days = int(participant_summary["span_days"].sum())
    for item in variables:
        subset = selected.loc[selected["variable_id"] == item.variable_id]
        variable_records.append(
            {
                "variable_id": item.variable_id,
                "variable": item.label,
                "participants_with_data": int(subset["participant_id"].nunique()),
                "participant_days": len(subset),
                "coverage_percent_of_selected_spans": (
                    round(100 * len(subset) / total_span_days, 2)
                    if total_span_days
                    else np.nan
                ),
            }
        )
    variable_summary = pd.DataFrame.from_records(variable_records)

    combination_records = [
        {
            "present_variables": (
                " + ".join(labels[item] for item in combination)
                if combination
                else "None"
            ),
            "participant_days": count,
        }
        for combination, count in combinations.items()
    ]
    combination_summary = pd.DataFrame.from_records(
        combination_records,
        columns=["present_variables", "participant_days"],
    )
    if not combination_summary.empty:
        denominator = int(combination_summary["participant_days"].sum())
        combination_summary["percent_of_in_span_days"] = (
            100 * combination_summary["participant_days"] / denominator
        ).round(2)
        combination_summary = combination_summary.sort_values(
            "participant_days", ascending=False
        ).reset_index(drop=True)

    return CoverageView(
        matrix=matrix,
        hover_text=hover_text,
        participant_summary=participant_summary.reset_index(drop=True),
        variable_summary=variable_summary,
        combination_summary=combination_summary,
        alignment=alignment,
        maximum_value=1 if display_variable_id is not None else len(selected_ids),
    )
