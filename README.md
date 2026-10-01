# Tesserae participant and group coverage dashboard

A local Streamlit dashboard for exploring one participant's Tesserae Parquet
directory and population-level data coverage without consolidating or modifying
the source data.


## Install and run

The existing `.venv` uses Python 3.9.6. The EDM workbench targets that environment
with pyEDM 2.3.0, Streamlit 1.50.0, and pinned compatible scientific dependencies.
Do not upgrade pyEDM independently: newer versions change the API and Python
requirements. The legacy environment can emit an event-loop traceback if
Streamlit is interrupted while external-drive I/O is still active.

From the workspace root:

```bash
tesserae_dashboard/.venv/bin/python -m pip install -r tesserae_dashboard/requirements.txt
tesserae_dashboard/.venv/bin/python -m streamlit run tesserae_dashboard/app.py
```

Streamlit prints a localhost URL. In the sidebar, set the directory containing
all participant folders, then choose a participant from the generated dropdown.
The `TESSERAE_PARQUET_ROOT` environment variable can set a different initial
root, while `TESSERAE_PARTICIPANT` can preselect a participant:

```bash
TESSERAE_PARQUET_ROOT="/path/to/parquets" \
TESSERAE_PARTICIPANT="/path/to/participant" \
  python3 -m streamlit run tesserae_dashboard/app.py
```

## Views

- **Timeline** aligns selected surveys, activity, physiology, and sleep on one
  zoomable time axis. Choose raw through daily resolution and native, mean,
  median, SD, min, max, or sum aggregation.
- **Circadian** shows date-by-clock-time heatmaps, 24-hour mean/SD profiles,
  weekday/weekend comparisons, and bedtime/wakeup markers.
- **Data Explorer** exposes every indexed category and subtype with schema,
  previews, missingness, and generic numeric plotting.
- **Coverage** reports metadata row counts, file coverage, schema versions,
  calendar gaps, and unreadable or empty files.
- **Group coverage** discovers physical Parquet columns and builds persistent
  participant-by-day non-null coverage summaries. It can align participants by
  their first selected-variable observation or by calendar date, and displays
  missing, partial, and complete overlap across any number of variables.

Use the sidebar **Analysis level** control to switch between Participant and
Group modes. Group coverage is deliberately built only after clicking
**Build selected coverage**, because the first scan of a high-frequency variable
can take time on cloud storage. Only participant IDs, dates, and variable IDs are
cached; raw physiological values are not persisted.

By default the group cache is stored under
`~/.cache/tesserae_dashboard/<data-root-hash>`. Set
`TESSERAE_DASHBOARD_CACHE_DIR` to place it elsewhere.
- **EDM** prepares regular time series and runs pyEDM after explicit submission,
  displaying its original Matplotlib figures and returned numerical tables.

Startup indexes filenames and dates without opening thousands of Parquet
footers. Row data is loaded on demand for the chosen view, subtype, and date
range; hidden views do not execute in the background. Opening **Coverage** runs
the complete footer audit once and persists it across restarts. Finder sidecar
files beginning with `._` are ignored. Press **Refresh metadata cache** after
files on disk change.

## Tests

```bash
tesserae_dashboard/.venv/bin/python -m pytest tesserae_dashboard/tests
```

## EDM workflow

1. Choose the participant and window in the sidebar, then open **EDM**.
2. Select two or more category/subtype/numeric-variable combinations. Each input
   gets an alias (`x1`, `x2`, …). Only selected-source schemas and selected columns
   are read; analysis inputs are never truncated to a preview row limit.
3. Choose a fixed sampling interval (daily by default) and aggregation per input.
   The lowest-30% mean averages the lowest `ceil(0.30 * valid_count)` observations
   per bin, matching the notebook's resting-heart-rate calculation. Empty bins,
   including sums, stay NaN. **Crop to common observed overlap** is on by default:
   trim to the first and last bins with observations for all selected variables.
   Every internal calendar row and NaN is preserved; no interpolation occurs.
   Turn cropping off to use the entire sidebar window. If no jointly observed
   bin exists, select another window, variable, or coarser interval.
4. Choose a pyEDM function, edit its arguments, and press **Submit**. Editing an
   argument alone does not run the analysis. Missingness limits update immediately
   outside the analysis form. Previous submitted results remain
   visible until another run is submitted, including when a new run fails.

`lib` and `pred` are native **1-based inclusive row ranges**, for example
`1 29` and `30 58`. Multiple pairs such as `1 20 30 40` are supported. Defaults
split the analysis grid in half. Cropping renumbers rows from 1; changing the grid
resets `lib`, `pred`, `validLib`, and `libSizes` to their defaults. The effective
window and trimming counts are displayed and recorded in each download.
`Tp`, `tau`, and `exclusionRadius` are measured in rows,
so changing the sampling interval changes their physical duration. Timestamps
are serialized as ISO strings for pyEDM 2.3.0's time-extension compatibility;
their order and all input rows are retained.

The form exposes arguments from the pinned function signature, including E,
theta, neighbors, generation, masks, sampling, and multiprocessing settings.
Nullable arguments use JSON (`null`, a number, a string, or an array as applicable).
`validLib` accepts `[]` or a boolean array with one entry per original grid row.
SMap solvers use a supported scikit-learn class and JSON constructor parameters.
No Python expressions are evaluated. `dataFrame` comes from the preview;
file-input and automatic file-output parameters are managed by the dashboard.
`returnObject=True` exposes the object's result tables, not an opaque pickle.

Supported functions: `EmbedDimension`, `Simplex`, `SMap`, `PredictNonlinear`,
`PredictInterval`, `CCM`, `Multiview`, `Embed`, and `SurrogateData`. Multiview
requires the target among `columns`. Its native 2.3.0 implementation does not
forward `knn` to internal Simplex calls and does not reliably honor `excludeTarget`;
the form calls out these upstream limitations. Multivariate SMap requires `embedded=True`; `Embed` outputs
can be reused through **Analysis data**. A selected best E is copied only when
you click **Use E**. CCM directions denote reconstruction, not automatic causality.

### Sleep and missingness

For `GarminHealthAPI / Sleep`, derived `sleep_onset`, `sleep_offset`, and
`duration_hours` use the existing `extract_sleep()` logic. Episodes are assigned
to wake date. Onset and offset are numerical hours since noon preceding wake
date (23:00 → 11; 07:00 → 19). Original timestamps are exported. If multiple
extracted episodes share a wake date, the longest wins, then the latest offset;
the selection count is displayed. Adjacent sleep chunks are loaded for boundary
episodes. On subdaily grids, nightly values occupy only their wake-date bin;
they are not repeated across the day.

Before running, every selected input and the combined incomplete-row mask must
pass both editable limits, evaluated after overlap cropping when enabled:

- `max_missing_days=7`: longest consecutive missing duration, including leading
  and trailing gaps. Subdaily missing duration is converted to days.
- `max_missing_percentage=20`: all missing rows divided by the full grid length,
  including scattered gaps. Exact equality with either threshold passes.

Passing these checks sends the full NaN-containing grid to pyEDM. Embedding
feasibility, usable neighbors, masks, and library sizes are checked separately.
`ignoreNan=True` permits pyEDM to exclude invalid **embedding vectors**, after
lags have been built on the unchanged grid. Version-specific restrictions are
reported rather than worked around by altering data:

- SurrogateData and Multiview currently require complete selected inputs.
- With gaps, CCM requires `Tp=0` because pyEDM 2.3.0's nonzero-lag CCM alignment
  operates on filtered rows.
- The 2.3.0 Ebisuzaki surrogate implementation requires an even row count.

### Native plots, history, and downloads

`showPlot=True` uses pyEDM's actual plotting routines. Figures appear as static
Matplotlib images after submission, including both SMap figures. Figure creation
is serialized and new figures are closed after capture, including on failure.
Functions without native figures show tables. Native plots are not converted to
Plotly. pyEDM defaults to one process here and CCM to sequential execution.

Named runs stay in session memory with immutable parameters, prepared data,
diagnostics, native figure snapshots, and metrics. Compare up to three runs
side by side. Each run downloads as a ZIP containing CSVs, PNGs, parameter JSON,
and `reproduce.py`; run the script beside the extracted files in the pinned
environment. Surrogate runs have a separate random-seed control, and saved
surrogate realizations can be explicitly selected as analysis inputs. They do
not automatically create a significance claim. No experiment files are written
automatically; restarting the session clears its history.
