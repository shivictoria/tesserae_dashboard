"""Streamlit EDM workbench. Scientific calls happen only on form submission."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    from .edm_data import AGGREGATIONS, SLEEP_VARIABLES, numeric_schema, sleep_values, prepare_grid, crop_overlap, missingness
    from .edm_engine import FUNCTIONS, SOLVERS, api, defaults, execute, export_bundle
except ImportError:
    from edm_data import AGGREGATIONS, SLEEP_VARIABLES, numeric_schema, sleep_values, prepare_grid, crop_overlap, missingness
    from edm_engine import FUNCTIONS, SOLVERS, api, defaults, execute, export_bundle


@st.cache_data(show_spinner=False, max_entries=64)
def cached_schema(index, category, subtype, start, end):
    return numeric_schema(index, category, subtype, start, end)


def fingerprint(frame, context):
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(json.dumps(context, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def source_controls(index, participant_path, start, end, load_view):
    count = int(st.number_input("Number of input variables", min_value=2, max_value=12, value=2))
    interval = st.selectbox("Sampling interval", ("1D", "12h", "6h", "1h", "15min", "5min", "1min"))
    selections, specs, original_sleep = [], [], []
    for i in range(count):
        alias = f"x{i + 1}"
        with st.expander(f"{alias}: variable selection", expanded=True):
            boxes = st.columns(4)
            category = boxes[0].selectbox("Category", sorted(index.category.unique()), key=f"edm_cat_{i}")
            subtypes = sorted(index.loc[index.category.eq(category), "subtype"].unique())
            subtype = boxes[1].selectbox("Subtype", subtypes, key=f"edm_sub_{i}_{category}")
            is_sleep = (category, subtype) == ("GarminHealthAPI", "Sleep")
            names, schema_warnings = cached_schema(index, category, subtype, start, end)
            if is_sleep:
                names = list(SLEEP_VARIABLES) + [name for name in names if name not in SLEEP_VARIABLES]
            for warning in schema_warnings[:3]:
                st.warning(warning)
            if not names:
                st.info("This source has no numeric columns in the selected window.")
                return None
            column = boxes[2].selectbox("Variable", names, index=min(i, len(names) - 1), key=f"edm_col_{i}_{category}_{subtype}")
            statistic = boxes[3].selectbox("Aggregation", AGGREGATIONS, key=f"edm_agg_{i}")
            if is_sleep and column in SLEEP_VARIABLES:
                records = load_view(participant_path, category, subtype,
                                    start - pd.Timedelta(days=2), end + pd.Timedelta(days=2),
                                    ["sleep_levels_map", "validation"])
                frame, originals, removed = sleep_values(records, start, end)
                original_sleep.append(originals.assign(variable=alias))
                st.caption("Assigned to wake date. Onset/offset: hours since noon preceding wake date. "
                           "Original timestamps are included in the run download.")
                if removed:
                    st.info(f"Selected the longest episode on {removed} duplicate wake-date entries.")
            else:
                frame = load_view(participant_path, category, subtype, start, end, [column])
            selections.append((alias, frame, column, statistic))
            specs.append({"alias": alias, "category": category, "subtype": subtype,
                          "variable": column, "aggregation": statistic})
    data = prepare_grid(selections, start, end, interval)
    context = {"participant": participant_path, "start": str(start), "end": str(end),
               "interval": interval, "variables": specs, "sleep_alignment": "wake date; previous noon"}
    return data, context, pd.concat(original_sleep, ignore_index=True) if original_sleep else pd.DataFrame()


def parameter_controls(function, frame):
    """All function arguments use their API names, defaults and types."""
    values = defaults(function, len(frame))
    numeric = [name for name in frame if name != "time"]
    raw, json_fields = {}, set()
    cells = st.columns(3)
    key_prefix = f"edm_arg_{function}_"
    for i, (name, value) in enumerate(values.items()):
        box, key = cells[i % 3], key_prefix + name
        if name == "columns":
            raw[name] = box.multiselect(name, numeric, default=numeric if function == "Multiview" else numeric[:1], key=key)
        elif name in ("target", "column"):
            raw[name] = box.selectbox(name, numeric, index=min(1, len(numeric) - 1) if name == "target" else 0, key=key)
        elif name == "solver":
            solver = box.selectbox(name, ("Default (numpy)", *SOLVERS), key=key)
            text = box.text_area("solver parameters (JSON)", "{}", key=key + "_params")
            raw[name] = (solver, text)
        elif name == "method":
            raw[name] = box.selectbox(name, ("ebisuzaki", "random_shuffle", "seasonal"), key=key)
        elif name == "mpMethod":
            raw[name] = box.selectbox(name, ("spawn", "forkserver"), key=key)
        elif isinstance(value, bool):
            raw[name] = box.checkbox(name, value=value, key=key)
        elif name == "validLib" or value is None or isinstance(value, (list, dict)):
            help_text = "JSON value; null uses the pyEDM default."
            if name == "validLib":
                help_text = f"[] or {len(frame)} JSON booleans, one per original grid row."
            raw[name] = box.text_area(name, json.dumps(value), help=help_text, key=key)
            json_fields.add(name)
        elif isinstance(value, (int, float)):
            if name == "theta":
                value = float(value)
            if name == "E" and "edm_pending_E" in st.session_state:
                st.session_state[key] = st.session_state["edm_pending_E"]
            raw[name] = box.number_input(name, value=value, step=1 if isinstance(value, int) else .1, key=key)
        else:
            raw[name] = box.text_input(name, value=str(value), key=key,
                                       help="1-based inclusive row pairs" if name in ("lib", "pred") else None)
    if "E" in values:
        st.session_state.pop("edm_pending_E", None)
    return raw, json_fields


def parse_parameters(raw, json_fields):
    args = {name: json.loads(value) if name in json_fields else value for name, value in raw.items()}
    if "solver" in args:
        solver, text = args["solver"]
        params = json.loads(text)
        if not isinstance(params, dict):
            raise ValueError("solver parameters must be a JSON object.")
        args["solver"] = None if solver == "Default (numpy)" else {"name": solver, "parameters": params}
    return args


def render_run(run, key):
    st.caption(f"{run['function']} · pyEDM {run['version']} · submitted {run['created']}")
    st.json(run["parameters"], expanded=False)
    with st.expander("Inputs and preparation"):
        st.json(run["context"])
        st.dataframe(run["missingness"], hide_index=True)
        if run.get("surrogate_seed") is not None:
            st.caption(f"Surrogate random seed: {run['surrogate_seed']}")
    for i, figure in enumerate(run["figures"]):
        st.image(figure)
        st.download_button(f"Download figure {i + 1}", figure, f"edm_{i + 1}.png", "image/png", key=f"{key}_fig_{i}")
    if not run["metrics"].empty:
        st.dataframe(run["metrics"], hide_index=True)
    for name, table in run["tables"].items():
        with st.expander(name, expanded=not run["figures"]):
            st.dataframe(table, hide_index=True)
    if run["function"] == "EmbedDimension":
        table = run["tables"].get("result")
        if table is not None and table.rho.notna().any():
            best = int(table.loc[table.rho.idxmax(), "E"])
            st.info(f"Best E in this submitted scan: {best}. This describes the chosen prediction window.")
            if st.button(f"Use E={best} in the next analysis", key=f"{key}_apply_E"):
                st.session_state["edm_pending_E"] = best
                st.rerun()
    if run["function"] == "CCM":
        st.caption("CCM labels indicate reconstruction directions, not automatic causal conclusions.")
    with st.expander("Run diagnostics and warnings"):
        st.dataframe(run["diagnostics"], hide_index=True)
        for warning in run["warnings"]:
            st.warning(warning)
        if run["log"]:
            st.text(run["log"])
    st.download_button("Download complete run", export_bundle(run), "edm_run.zip", "application/zip", key=f"{key}_bundle")


def history():
    runs = st.session_state.get("edm_runs", [])
    if not runs:
        return
    st.subheader("Submitted runs")
    chosen = st.multiselect("Compare runs", range(len(runs)), default=[len(runs) - 1],
                            format_func=lambda i: f"{i + 1}: {runs[i]['name']}", max_selections=3,
                            key=f"edm_history_{len(runs)}")
    if chosen:
        for box, i in zip(st.columns(len(chosen)), chosen):
            with box:
                st.markdown(f"**{runs[i]['name']}**")
                render_run(runs[i], f"run_{i}")


def workbench(participant_path, index, start, end, load_view):
    st.subheader("Empirical dynamic modelling — pyEDM 2.3.0")
    st.caption("Select inputs, edit API arguments, then Submit to run. Library and prediction ranges use rows.")
    api()
    prepared = source_controls(index, participant_path, start, end, load_view)
    if prepared is None:
        return
    frame, context, sleep = prepared
    # An explicit route to analyse prior embeddings or surrogate realizations.
    runs = st.session_state.get("edm_runs", [])
    reusable = [i for i, run in enumerate(runs) if run["function"] in ("Embed", "SurrogateData")]
    use_run = st.selectbox("Analysis data", [None, *reusable],
                           format_func=lambda i: "Current selected variables" if i is None else f"Saved output: {runs[i]['name']}")
    if use_run is not None:
        saved = runs[use_run]
        table = saved["tables"]["result"].copy()
        # Embed may omit time; insert the unchanged parent grid, never align by dropping NaNs.
        table = table.drop(columns="time", errors="ignore")
        table.insert(0, "time", saved["data"].time.to_numpy())
        frame, sleep = table, saved.get("sleep_timestamps", pd.DataFrame())
        context = dict(saved["context"], source_run=use_run + 1)
        if saved["function"] == "SurrogateData":
            selected = st.multiselect("Surrogate realizations to analyse", list(frame.columns[1:]), default=list(frame.columns[1:3]))
            frame = frame[["time", *selected]]
        st.info("Using the saved output's original participant, window and sampling interval.")
    if len(frame.columns) < 2:
        st.info("Select at least one numeric variable.")
        return
    crop = st.checkbox("Crop to common observed overlap", value=True, key="edm_crop_overlap",
                       help="Use the first and last rows where all selected variables have values; retain every missing row between them.")
    before_rows = len(frame)
    if crop:
        frame, leading, trailing = crop_overlap(frame)
        st.caption(f"Common overlap: {frame.time.iloc[0]} to {frame.time.iloc[-1]} · "
                   f"{len(frame):,} rows · trimmed {leading:,} leading and {trailing:,} trailing rows. "
                   "Internal gaps are preserved.")
    context = dict(context, requested_start=context.get("requested_start", context["start"]),
                   requested_end=context.get("requested_end", context["end"]),
                   start=str(frame.time.iloc[0]), end=str(pd.Timestamp(frame.time.iloc[-1]) + pd.Timedelta(context["interval"])),
                   crop_to_overlap=crop, rows_before_crop=before_rows, rows_after_crop=len(frame))
    if not sleep.empty and "wake_date" in sleep:
        sleep = sleep.loc[(sleep.wake_date >= pd.Timestamp(context["start"]))
                          & (sleep.wake_date < pd.Timestamp(context["end"]))].copy()
    grid_identity = (context["participant"], context["start"], context["end"], context["interval"], len(frame))
    if st.session_state.get("edm_grid_identity") != grid_identity:
        # These arguments refer to row positions on this particular grid.
        for method in FUNCTIONS:
            for argument in ("lib", "pred", "validLib", "libSizes"):
                st.session_state.pop(f"edm_arg_{method}_{argument}", None)
        st.session_state["edm_grid_identity"] = grid_identity
    limits = st.columns(2)
    max_days = limits[0].number_input("max_missing_days", min_value=0.0, value=7.0,
                                     key="edm_max_missing_days", help="Longest consecutive missing duration within the analysis window, in days.")
    max_percent = limits[1].number_input("max_missing_percentage", min_value=0.0, max_value=100.0, value=20.0,
                                        key="edm_max_missing_percentage", help="All missing rows within the analysis window, including scattered gaps.")
    diagnostics = missingness(frame, context["interval"], max_days, max_percent)
    st.dataframe(diagnostics, hide_index=True)
    if not diagnostics.passes.all():
        reasons = []
        for row in diagnostics.loc[~diagnostics.passes].itertuples():
            failed = []
            if row.longest_gap_days > max_days:
                failed.append(f"longest gap {row.longest_gap_days:g} days > {max_days:g}")
            if row.missing_percentage > max_percent:
                failed.append(f"missing {row.missing_percentage:.2f}% > {max_percent:g}%")
            reasons.append(f"{row.variable}: {', '.join(failed)}")
        st.warning("Missingness limits exceeded. Submission is blocked. " + "; ".join(reasons))
    st.dataframe(frame.rename_axis("row").set_axis(range(1, len(frame) + 1)), height=250)
    function = st.selectbox("pyEDM function", FUNCTIONS, key="edm_function")
    with st.form(f"edm_form_{function}"):
        raw, json_fields = parameter_controls(function, frame)
        surrogate_seed = int(st.number_input("Surrogate random seed", min_value=0, max_value=2**32 - 1, value=0,
                                             help="Seeds Python and NumPy for reproducible native SurrogateData output.")) if function == "SurrogateData" else 0
        name = st.text_input("Run name", value=function)
        st.caption("dataFrame is the preview above. File input/output arguments are managed by the dashboard.")
        submitted = st.form_submit_button("Submit")
    parameters = None
    try:
        parameters = parse_parameters(raw, json_fields)
        arguments = []
        for key, value in parameters.items():
            if key == "solver" and isinstance(value, dict):
                expression = f"linear_model.{value['name']}(**{value['parameters']!r})"
            elif key == "validLib":
                expression = f"np.asarray({value!r}, dtype=bool)"
            else:
                expression = repr(value)
            arguments.append(f"{key}={expression}")
        call = ",\n    ".join(arguments)
        st.code(f"pyEDM.{function}(\n    dataFrame=data,\n    {call}\n)", language="python")
        if function == "Multiview":
            st.caption("pyEDM 2.3.0 limitations: knn is not forwarded to its internal Simplex calls; "
                       "excludeTarget does not reliably remove the target coordinates. Returned plots use the native implementation.")
    except (ValueError, TypeError) as exc:
        st.error(f"Invalid arguments: {exc}")
    current = fingerprint(frame, {"context": context, "parameters": parameters, "function": function,
                                   "max_missing_days": max_days, "max_missing_percentage": max_percent,
                                   "surrogate_seed": surrogate_seed})
    if runs and runs[-1].get("fingerprint") != current:
        st.caption("Current settings differ from the last submitted run. Existing results remain unchanged.")
    if submitted and parameters is not None and diagnostics.passes.all():
        with st.spinner(f"Running {function}…"):
            result = execute(function, frame, parameters, surrogate_seed=surrogate_seed)
        result.update(name=name or function, created=datetime.now().isoformat(timespec="seconds"),
                      context=dict(context, max_missing_days=max_days, max_missing_percentage=max_percent),
                      missingness=diagnostics.copy(), sleep_timestamps=sleep.copy(), fingerprint=current)
        st.session_state.setdefault("edm_runs", []).append(result)
        st.success("Run complete. Native figures and numerical results are below.")


def edm_tab(participant_path, index, start, end, load_view):
    try:
        workbench(participant_path, index, start, end, load_view)
    except Exception as exc:
        st.error(f"EDM: {type(exc).__name__}: {exc}")
        st.caption("Previous submitted results are retained. Internal calendar gaps are never removed or interpolated.")
    history()
