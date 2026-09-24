# Tesserae participant timeline dashboard

A local Streamlit dashboard for exploring one participant's Tesserae Parquet
directory without consolidating or modifying the source data.


## Install and run

Python 3.11 or newer is recommended. The workspace's legacy Python 3.9 virtual
environment can emit an event-loop traceback if Streamlit is interrupted while
external-drive I/O is still active.

From the workspace root:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
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

Startup indexes filenames and dates without opening thousands of Parquet
footers. Row data is loaded on demand for the chosen view, subtype, and date
range; hidden views do not execute in the background. Opening **Coverage** runs
the complete footer audit once and persists it across restarts. Finder sidecar
files beginning with `._` are ignored. Press **Refresh metadata cache** after
files on disk change.

## Tests

```bash
python3 -m pytest tesserae_dashboard/tests
```
