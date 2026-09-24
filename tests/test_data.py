from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tesserae_dashboard.charts import (
    SLEEP_STAGE_COLORS,
    build_circadian_heatmap,
    build_sleep_timing,
    build_timeline,
)

from tesserae_dashboard.data import (
    DataPathError,
    add_display_time,
    aggregate_metric,
    build_index,
    circadian_values,
    coverage_table,
    extract_sleep,
    list_participant_paths,
    load_subtype,
    missing_dates,
    parse_file_date,
    parse_subtype,
    resolve_resolution,
    select_files_for_range,
    validate_participant_path,
)


PARTICIPANT = "36fd7269ce06b988c4f4347550a4a939"


def write_parquet(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(data), path)


def tesserae_path(root: Path, category: str, subtype: str, date: str) -> Path:
    return root / category / f"{PARTICIPANT}_{category}_{subtype}_{date}-00-00-00-CST.parquet"


def test_parse_precise_subtypes_and_dates(tmp_path: Path) -> None:
    sleep = tesserae_path(tmp_path, "GarminHealthAPI", "Sleep", "2018-01-10")
    epochs = tesserae_path(tmp_path, "GarminHealthAPI", "Epochs", "2018-01-11")
    assert parse_subtype(sleep, PARTICIPANT, "GarminHealthAPI") == "Sleep"
    assert parse_subtype(epochs, PARTICIPANT, "GarminHealthAPI") == "Epochs"
    assert parse_file_date(sleep) == pd.Timestamp("2018-01-10")


def test_index_excludes_sidecars_and_reports_corrupt_files(tmp_path: Path) -> None:
    root = tmp_path / PARTICIPANT
    good = tesserae_path(root, "GarminHealthAPI", "Sleep", "2018-01-10")
    write_parquet(good, {"Timestamp": [pd.Timestamp("2018-01-10", tz="UTC")], "value": [1]})
    sidecar = good.with_name("._" + good.name)
    sidecar.write_bytes(b"not parquet")
    corrupt = tesserae_path(root, "GarminHealthAPI", "Epochs", "2018-01-11")
    corrupt.write_bytes(b"not parquet")

    index = build_index(root)
    assert len(index) == 2
    assert set(index["subtype"]) == {"Sleep", "Epochs"}
    assert index.set_index("subtype").loc["Epochs", "status"] == "unreadable"


def test_fast_index_defers_metadata_but_still_loads_selected_data(tmp_path: Path) -> None:
    root = tmp_path / PARTICIPANT
    path = tesserae_path(root, "GarminPA", "HR", "2018-01-10")
    write_parquet(
        path,
        {
            "Timestamp": [pd.Timestamp("2018-01-10 01:00", tz="UTC")],
            "local_time": [pd.Timestamp("2018-01-10 02:00")],
            "hr": [60],
        },
    )
    index = build_index(root, inspect_metadata=False)
    assert index.loc[0, "status"] == "indexed"
    assert index.loc[0, "columns"] == tuple()
    frame, warnings = load_subtype(
        index,
        "GarminPA",
        "HR",
        pd.Timestamp("2018-01-10"),
        pd.Timestamp("2018-01-11"),
        columns=["hr"],
    )
    assert not warnings
    assert frame.loc[0, "hr"] == 60


def test_invalid_participant_directory(tmp_path: Path) -> None:
    with pytest.raises(DataPathError, match="not found"):
        validate_participant_path(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DataPathError, match="No Parquet"):
        validate_participant_path(empty)


def test_list_participant_paths_uses_immediate_visible_subfolders(
    tmp_path: Path,
) -> None:
    (tmp_path / "participant-b").mkdir()
    (tmp_path / "participant-a").mkdir()
    (tmp_path / ".metadata").mkdir()
    (tmp_path / "README.txt").write_text("not a participant", encoding="utf-8")

    participants = list_participant_paths(tmp_path)

    assert [path.name for path in participants] == ["participant-a", "participant-b"]


def test_list_participant_paths_reports_missing_or_empty_root(tmp_path: Path) -> None:
    with pytest.raises(DataPathError, match="root directory not found"):
        list_participant_paths(tmp_path / "missing")
    with pytest.raises(DataPathError, match="No participant subfolders"):
        list_participant_paths(tmp_path)


def test_selective_load_handles_schema_drift_and_date_filter(tmp_path: Path) -> None:
    root = tmp_path / PARTICIPANT
    first = tesserae_path(root, "GarminHealthAPI", "Epochs", "2018-01-10")
    second = tesserae_path(root, "GarminHealthAPI", "Epochs", "2018-01-11")
    write_parquet(
        first,
        {
            "Timestamp": [pd.Timestamp("2018-01-10 01:00", tz="UTC")],
            "local_time": [pd.Timestamp("2018-01-10 02:00")],
            "steps": pa.array([5], type=pa.int64()),
        },
    )
    write_parquet(
        second,
        {
            "Timestamp": [pd.Timestamp("2018-01-11 01:00", tz="UTC")],
            "local_time": [pd.Timestamp("2018-01-11 02:00")],
            "steps": pa.array([None], type=pa.null()),
            "met": [1.2],
        },
    )
    index = build_index(root)
    frame, warnings = load_subtype(
        index,
        "GarminHealthAPI",
        "Epochs",
        pd.Timestamp("2018-01-10"),
        pd.Timestamp("2018-01-12"),
        columns=["steps", "met"],
    )
    assert not warnings
    assert len(frame) == 2
    assert {"steps", "met", "_time"}.issubset(frame.columns)
    assert frame.loc[0, "_time"] == pd.Timestamp("2018-01-10 02:00")


def test_file_range_pruning_keeps_buffer_and_consolidated_files() -> None:
    records = pd.DataFrame(
        {
            "file_date": pd.date_range("2018-01-01", periods=10),
            "path": [f"file-{index}" for index in range(10)],
        }
    )
    selected = select_files_for_range(
        records, pd.Timestamp("2018-01-05"), pd.Timestamp("2018-01-06")
    )
    assert selected["file_date"].min() == pd.Timestamp("2018-01-04")
    assert selected["file_date"].max() == pd.Timestamp("2018-01-07")
    assert len(select_files_for_range(records.iloc[:1], pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))) == 1
    assert select_files_for_range(
        records, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")
    ).empty


def test_local_time_precedes_utc_and_falls_back() -> None:
    frame = pd.DataFrame(
        {
            "local_time": [pd.Timestamp("2018-01-10 09:00"), pd.NaT],
            "Timestamp": [
                pd.Timestamp("2018-01-10 15:00", tz="UTC"),
                pd.Timestamp("2018-01-11 15:00", tz="UTC"),
            ],
        }
    )
    output = add_display_time(frame)
    assert output.loc[0, "_time"] == pd.Timestamp("2018-01-10 09:00")
    assert output.loc[1, "_time"] == pd.Timestamp("2018-01-11 15:00")


def test_mixed_local_date_and_timestamp_is_arrow_serializable() -> None:
    frame = pd.DataFrame(
        {
            "local_time": [
                pd.Timestamp("2018-01-10 09:15"),
                pd.Timestamp("2018-01-11").date(),
            ],
            "Timestamp": [
                pd.Timestamp("2018-01-10 15:15", tz="UTC"),
                pd.Timestamp("2018-01-11 06:00", tz="UTC"),
            ],
        }
    )
    output = add_display_time(frame)
    assert pd.api.types.is_datetime64_ns_dtype(output["local_time"])
    table = pa.Table.from_pandas(output, preserve_index=False)
    assert str(table.schema.field("local_time").type) == "timestamp[ns]"


@pytest.mark.parametrize(
    ("statistic", "expected"),
    [
        ("mean", 2.0),
        ("median", 2.0),
        ("sd", 1.0),
        ("min", 1.0),
        ("max", 3.0),
        ("sum", 6.0),
    ],
)
def test_aggregation_statistics(statistic: str, expected: float) -> None:
    frame = pd.DataFrame(
        {
            "_time": pd.to_datetime(
                ["2018-01-01 00:01", "2018-01-01 00:02", "2018-01-01 00:03"]
            ),
            "hr": [1.0, 2.0, 3.0],
        }
    )
    result = aggregate_metric(frame, "hr", "1h", statistic, include_sd=True)
    assert result.loc[0, "value"] == pytest.approx(expected)
    assert result.loc[0, "sd"] == pytest.approx(1.0)


def test_sd_with_one_observation_is_empty() -> None:
    frame = pd.DataFrame({"_time": [pd.Timestamp("2018-01-01")], "hr": [60]})
    result = aggregate_metric(frame, "hr", "1h", "sd")
    assert result.empty


def test_auto_resolution_boundaries() -> None:
    start = pd.Timestamp("2018-01-01")
    assert resolve_resolution("Auto", start, start + pd.Timedelta(days=7)) == "1min"
    assert resolve_resolution("Auto", start, start + pd.Timedelta(days=30)) == "15min"
    assert resolve_resolution("Auto", start, start + pd.Timedelta(days=90)) == "1h"
    assert resolve_resolution("Auto", start, start + pd.Timedelta(days=365)) == "1D"


def test_extract_sleep_derives_boundaries_stages_and_deduplicates_night() -> None:
    sleep_map = (
        '{"deep":[{"startTime":"2018-01-01 23:00:00",'
        '"endTime":"2018-01-02 00:00:00"}],'
        '"light":[{"startTime":"2018-01-02 00:00:00",'
        '"endTime":"2018-01-02 02:00:00"}],'
        '"awake":[{"startTime":"2018-01-02 02:00:00",'
        '"endTime":"2018-01-02 02:10:00"}]}'
    )
    frame = pd.DataFrame(
        {
            "_time": [pd.Timestamp("2018-01-02 01:00"), pd.Timestamp("2018-01-02 01:05")],
            "sleep_levels_map": [sleep_map, sleep_map],
            "validation": ["AUTO_TENTATIVE", "AUTO_FINAL"],
        }
    )
    sleep = extract_sleep(frame)
    assert len(sleep) == 1
    assert sleep.loc[0, "sleep_onset"] == pd.Timestamp("2018-01-01 23:00")
    assert sleep.loc[0, "sleep_offset"] == pd.Timestamp("2018-01-02 02:00")
    assert sleep.loc[0, "duration_hours"] == pytest.approx(3)
    assert sleep.loc[0, "deep_sleep_seconds"] == 3600
    assert sleep.loc[0, "light_sleep_seconds"] == 7200
    assert sleep.loc[0, "awake_seconds"] == 600
    assert sleep.loc[0, "validation"] == "AUTO_FINAL"
    assert [stage for stage, _, _ in sleep.loc[0, "stage_intervals"]] == [
        "light",
        "deep",
        "awake",
    ]


def test_sleep_timing_chart_is_date_by_clock_stage_heatmap() -> None:
    sleep = pd.DataFrame(
        {
            "night": [pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-02")],
            "sleep_onset": [
                pd.Timestamp("2018-01-01 23:00"),
                pd.Timestamp("2018-01-02 22:30"),
            ],
            "sleep_offset": [
                pd.Timestamp("2018-01-02 07:00"),
                pd.Timestamp("2018-01-03 06:30"),
            ],
            "stage_intervals": [
                [
                    (
                        "deep",
                        pd.Timestamp("2018-01-01 23:00"),
                        pd.Timestamp("2018-01-02 00:00"),
                    ),
                    (
                        "light",
                        pd.Timestamp("2018-01-02 00:00"),
                        pd.Timestamp("2018-01-02 02:00"),
                    ),
                ],
                [
                    (
                        "rem",
                        pd.Timestamp("2018-01-03 05:30"),
                        pd.Timestamp("2018-01-03 06:30"),
                    )
                ],
            ],
        }
    )
    figure = build_sleep_timing(sleep, 15)
    heatmap = figure.data[0]
    assert heatmap.type == "heatmap"
    assert np.asarray(heatmap.z).shape == (2, 96)
    assert heatmap.z[0][22 * 4] == 0  # outside onset/offset is awake
    assert heatmap.z[0][23 * 4] == 3  # deep
    assert heatmap.z[0][0] == 2  # light after midnight
    assert heatmap.z[0][2 * 4] == 0  # unassigned bins are awake
    assert heatmap.z[1][5 * 4 + 2] == 4  # REM at 05:30
    assert [entry[1] for entry in heatmap.colorscale[::2]] == [
        SLEEP_STAGE_COLORS["awake_seconds"],
        SLEEP_STAGE_COLORS["unmeasurable_sleep_seconds"],
        SLEEP_STAGE_COLORS["light_sleep_seconds"],
        SLEEP_STAGE_COLORS["deep_sleep_seconds"],
        SLEEP_STAGE_COLORS["rem_sleep_seconds"],
    ]


def test_circadian_heatmap_inserts_empty_daily_rows_across_data_gaps() -> None:
    matrix = pd.DataFrame(
        [[60.0, 65.0], [70.0, 75.0]],
        index=pd.to_datetime(["2018-01-01", "2018-01-04"]),
        columns=[0, 15],
    )

    figure = build_circadian_heatmap(matrix, "Heart rate")
    heatmap = figure.data[0]

    assert list(pd.to_datetime(heatmap.y)) == list(
        pd.date_range("2018-01-01", "2018-01-04", freq="D")
    )
    assert np.asarray(heatmap.z).shape == (4, 2)
    assert np.isnan(heatmap.z[1]).all()
    assert np.isnan(heatmap.z[2]).all()


def test_sleep_heatmap_inserts_blank_daily_rows_across_missing_nights() -> None:
    sleep = pd.DataFrame(
        {
            "night": pd.to_datetime(["2018-01-01", "2018-01-04"]),
            "sleep_onset": pd.to_datetime(
                ["2018-01-01 23:00", "2018-01-04 23:00"]
            ),
            "sleep_offset": pd.to_datetime(
                ["2018-01-02 07:00", "2018-01-05 07:00"]
            ),
            "stage_intervals": [[], []],
        }
    )

    heatmap = build_sleep_timing(sleep, 60).data[0]

    assert np.asarray(heatmap.z).shape == (4, 24)
    assert np.isnan(heatmap.z[1]).all()
    assert np.isnan(heatmap.z[2]).all()
    assert (heatmap.z[0] == 0).all()


def test_main_timeline_uses_stacked_sleep_stage_bars() -> None:
    sleep = pd.DataFrame(
        {
            "night": [pd.Timestamp("2018-01-01")],
            "sleep_onset": [pd.Timestamp("2018-01-01 23:00")],
            "sleep_offset": [pd.Timestamp("2018-01-02 07:00")],
            "validation": ["AUTO_FINAL"],
            "light_sleep_seconds": [4 * 3600],
            "deep_sleep_seconds": [2 * 3600],
            "rem_sleep_seconds": [3600],
            "awake_seconds": [1800],
            "unmeasurable_sleep_seconds": [0],
        }
    )
    figure = build_timeline(
        [], sleep, pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-03")
    )
    assert all(trace.type == "bar" for trace in figure.data)
    assert [trace.name for trace in figure.data] == [
        "Light Sleep",
        "Deep Sleep",
        "Rem Sleep",
        "Awake",
        "Unmeasurable Sleep",
    ]
    assert figure.layout.barmode == "stack"


def test_circadian_matrix_and_profile() -> None:
    frame = pd.DataFrame(
        {
            "_time": pd.to_datetime(
                ["2018-01-01 08:01", "2018-01-02 08:05", "2018-01-06 08:02"]
            ),
            "hr": [60, 70, 80],
        }
    )
    matrix, profile = circadian_values(frame, "hr", 15, "Weekdays")
    assert matrix.shape == (2, 1)
    assert profile.loc[0, "mean"] == pytest.approx(65)
    assert profile.loc[0, "sd"] == pytest.approx(np.sqrt(50))


def test_coverage_and_missing_dates(tmp_path: Path) -> None:
    root = tmp_path / PARTICIPANT
    for date in ("2018-01-01", "2018-01-02", "2018-01-04", "2018-01-05", "2018-01-06"):
        write_parquet(
            tesserae_path(root, "GarminPA", "HR", date),
            {"Timestamp": [pd.Timestamp(date, tz="UTC")], "hr": [60]},
        )
    index = build_index(root)
    coverage = coverage_table(index)
    assert coverage.loc[0, "files"] == 5
    gaps = missing_dates(index)
    assert gaps == [pd.Timestamp("2018-01-03")]
