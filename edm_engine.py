"""Versioned pyEDM adapter, native figure capture, and portable run exports."""
from __future__ import annotations

import copy
import inspect
import io
import json
import random
import threading
import warnings
import zipfile
from contextlib import redirect_stdout
from importlib.metadata import version

import numpy as np
import pandas as pd

FUNCTIONS = ("EmbedDimension", "Simplex", "SMap", "PredictNonlinear",
             "PredictInterval", "CCM", "Multiview", "Embed", "SurrogateData")
MANAGED = {"dataFrame", "pathIn", "dataFile", "outputFile"}
SOLVERS = ("LinearRegression", "SGDRegressor", "Ridge", "RidgeCV", "Lasso", "LassoCV",
           "Lars", "LarsCV", "LassoLars", "LassoLarsCV", "LassoLarsIC", "ElasticNet",
           "ElasticNetCV", "OrthogonalMatchingPursuit", "OrthogonalMatchingPursuitCV")
PLOT_LOCK = threading.RLock()


def api():
    import matplotlib
    if "agg" not in matplotlib.get_backend().lower():
        matplotlib.use("Agg")
    import pyEDM
    if version("pyEDM") != "2.3.0":
        raise RuntimeError("This workbench requires pyEDM==2.3.0. Install the dashboard requirements.")
    return pyEDM


def defaults(function, nrows):
    signature = inspect.signature(getattr(api(), function))
    values = {name: copy.deepcopy(p.default) for name, p in signature.parameters.items()
              if name not in MANAGED}
    overrides = {"E": 2, "maxE": 7, "lib": f"1 {max(2, nrows // 2)}",
                 "pred": f"{max(2, nrows // 2) + 1} {nrows}", "numProcess": 1,
                 "sequential": True, "showPlot": True, "sample": 50,
                 "libSizes": f"5 {max(5, nrows - 3)} 3", "mpMethod": "spawn"}
    values.update({k: v for k, v in overrides.items() if k in values})
    return values


def row_ranges(value, nrows, name):
    try:
        parts = value.split() if isinstance(value, str) else value
        if any(isinstance(x, bool) or int(x) != float(x) for x in parts):
            raise ValueError()
        numbers = [int(x) for x in parts]
    except (ValueError, TypeError):
        raise ValueError(f"{name} must contain integer start/stop pairs.")
    if not numbers or len(numbers) % 2:
        raise ValueError(f"{name} needs 1-based inclusive start/stop pairs, e.g. '1 29'.")
    for start, stop in zip(numbers[::2], numbers[1::2]):
        if not 1 <= start <= stop <= nrows:
            raise ValueError(f"{name} range {start} {stop} is outside 1…{nrows} or reversed.")
    return numbers


def library_sizes(value):
    parts = value.split() if isinstance(value, str) else value
    if not parts or any(isinstance(v, bool) or int(v) != float(v) for v in parts):
        raise ValueError("libSizes must contain positive integers.")
    sizes = [int(v) for v in parts]
    if len(sizes) == 3 and sizes[2] < sizes[1]:
        if sizes[2] <= 0 or sizes[0] > sizes[1]:
            raise ValueError("libSizes needs a positive increment and ordered bounds.")
        sizes = list(range(sizes[0], sizes[1] + 1, sizes[2]))
    if min(sizes) < 1:
        raise ValueError("libSizes must be positive.")
    return sizes


def call_arguments(parameters):
    args = copy.deepcopy(parameters)
    solver = args.get("solver")
    if isinstance(solver, dict):
        from sklearn import linear_model
        if solver.get("name") not in SOLVERS:
            raise ValueError("Unsupported SMap solver.")
        args["solver"] = getattr(linear_model, solver["name"])(**solver.get("parameters", {}))
    if "validLib" in args:
        args["validLib"] = np.asarray(args["validLib"], dtype=bool)
    return args


def validate(function, frame, parameters):
    """Validate without running predictions or modifying the input grid."""
    edm = api()
    if function not in FUNCTIONS:
        raise ValueError("Unknown EDM function.")
    allowed = set(inspect.signature(getattr(edm, function)).parameters) - MANAGED
    if set(parameters) - allowed:
        raise ValueError(f"Unsupported arguments: {sorted(set(parameters) - allowed)}")
    n = len(frame)
    if n < 3:
        raise ValueError("At least three regular-grid rows are required.")
    columns = parameters.get("columns", parameters.get("column", []))
    columns = columns.split() if isinstance(columns, str) else list(columns)
    target = parameters.get("target")
    selected = list(dict.fromkeys(columns + ([target] if target else [])))
    if not columns or any(c not in frame or c == "time" for c in selected):
        raise ValueError("Select valid numeric columns and target.")
    for column in selected:
        if frame[column].dropna().nunique() < 2:
            raise ValueError(f"{column} is empty or constant in the selected window.")
    for name in ("lib", "pred"):
        if name in parameters:
            row_ranges(parameters[name], n, name)
    mask = parameters.get("validLib", [])
    if len(mask) and (len(mask) != n or any(type(x) not in (bool, np.bool_) for x in mask)):
        raise ValueError(f"validLib must be [] or a JSON boolean array of length {n}.")
    for name in ("E", "maxE", "maxTp", "sample", "numSurrogates", "numProcess", "chunksize"):
        if name in parameters and parameters[name] < 1:
            raise ValueError(f"{name} must be positive.")
    for name in ("knn", "exclusionRadius", "generateSteps", "D", "multiview"):
        if parameters.get(name, 0) < 0:
            raise ValueError(f"{name} must be nonnegative.")
    if parameters.get("tau", -1) == 0 and not parameters.get("embedded", False):
        raise ValueError("tau must be nonzero when embedding time series.")
    has_missing = frame[selected].isna().any().any()
    if has_missing and parameters.get("ignoreNan") is False:
        raise ValueError("These inputs contain NaNs; set ignoreNan=True to retain the grid safely.")
    if function == "SurrogateData":
        if has_missing:
            raise ValueError("SurrogateData requires complete input; this workbench does not fill or drop gaps.")
        if parameters.get("method") == "ebisuzaki" and n % 2:
            raise ValueError("pyEDM 2.3.0 ebisuzaki requires an even row count. Select an even-length window.")
        return pd.DataFrame()
    if function == "Embed":
        embedded = edm.Embed(dataFrame=frame, **call_arguments(parameters))
        if not embedded.drop(columns="time", errors="ignore").notna().all(axis=1).any():
            raise ValueError("No complete embedding vectors at this E and tau.")
        return pd.DataFrame()
    if function == "Multiview":
        if target not in columns:
            raise ValueError("Multiview requires target in columns; use excludeTarget to omit its coordinates from views.")
        if has_missing:
            raise ValueError("Multiview in pyEDM 2.3.0 is limited here to complete inputs; NaN grids are retained.")
        if parameters.get("D", 0) > len(columns) * parameters.get("E", 1):
            raise ValueError("D exceeds the available embedded coordinates.")
        if parameters.get("excludeTarget") and len(columns) == 1:
            raise ValueError("excludeTarget leaves no Multiview predictor coordinates.")
    if function == "CCM" and has_missing and parameters.get("Tp", 0) != 0:
        raise ValueError("pyEDM 2.3.0 CCM with gaps is supported at Tp=0 only: its lag alignment otherwise uses filtered rows.")
    # Use the pinned engine's own index and embedding construction for feasibility.
    from pyEDM.Simplex import Simplex
    from pyEDM.SMap import SMap
    cls = SMap if function in ("SMap", "PredictNonlinear") else Simplex
    args = call_arguments(parameters)
    base = {k: v for k, v in args.items() if k in inspect.signature(cls).parameters}
    base.update(dataFrame=frame, columns=columns, target=target, verbose=False)
    base.setdefault("E", 2)
    base.setdefault("lib", f"1 {n}")
    base.setdefault("pred", f"1 {n}")
    if function == "Multiview":
        base["columns"] = columns[:1]
    if function == "PredictNonlinear":
        base["theta"] = 0
    dims = range(1, parameters["maxE"] + 1) if function == "EmbedDimension" else [base["E"]]
    horizons = range(1, parameters["maxTp"] + 1) if function == "PredictInterval" else [parameters.get("Tp", 1)]
    directions = [(columns, target)]
    if function == "CCM":
        directions.append(([target], columns[0]))
    diagnostics = []
    for dim in dims:
        for horizon in horizons:
            for source, destination in directions:
                trial = dict(base, E=dim, Tp=horizon, columns=source, target=destination)
                if function == "Multiview":
                    trial["columns"] = source[:1]
                model = cls(**trial)
                model.EmbedData()
                model.RemoveNan()
                libs = model.lib_i
                if len(mask):
                    libs = libs[np.asarray(mask)[libs]]
                target_values = frame[destination].to_numpy()
                usable = libs[(libs + horizon >= 0) & (libs + horizon < n)]
                usable = usable[np.isfinite(target_values[usable + horizon])]
                needed = model.knn if parameters.get("knn", 0) else model.E + 1
                if len(usable) < max(needed + 1, 3) or len(model.pred_i) < 2:
                    raise ValueError(f"Insufficient usable vectors for E={dim}, Tp={horizon}: "
                                     f"{len(usable)} library, {len(model.pred_i)} prediction.")
                if function != "CCM":
                    radius = parameters.get("exclusionRadius", 0)
                    sorted_lib = np.sort(libs)
                    predictions = model.pred_i
                    excluded = (np.searchsorted(sorted_lib, predictions + radius, side="right")
                                - np.searchsorted(sorted_lib, predictions - radius, side="left"))
                    if (len(libs) - excluded < model.knn).any():
                        raise ValueError("Too few neighbors for knn, validLib and exclusionRadius. "
                                         "Reduce knn or exclusionRadius, or expand lib.")
                if function == "CCM":
                    sizes = library_sizes(parameters["libSizes"])
                    if min(sizes) < needed + 1 or max(sizes) > len(libs):
                        raise ValueError(f"CCM libSizes must be between {needed + 1} and {len(libs)} usable vectors.")
                diagnostics.append({"columns": " ".join(source), "target": destination,
                                    "E": dim, "Tp": horizon, "usable_library": len(usable),
                                    "usable_prediction": len(model.pred_i)})
    return pd.DataFrame(diagnostics)


def tables_from_result(result):
    if isinstance(result, pd.DataFrame):
        return {"result": result.copy(deep=True)}
    if isinstance(result, dict):
        return {str(k): v.copy(deep=True) for k, v in result.items() if isinstance(v, pd.DataFrame)}
    names = ("Projection", "Coefficients", "SingularValues", "libMeans", "PredictStats1", "PredictStats2", "View")
    return {name: getattr(result, name).copy(deep=True) for name in names
            if isinstance(getattr(result, name, None), pd.DataFrame)}


def execute(function, frame, parameters, surrogate_seed=0):
    """Call pyEDM once, preserving native figures and releasing all new figures."""
    diagnostics = validate(function, frame, parameters)
    with PLOT_LOCK:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from unittest.mock import patch
        before = set(plt.get_fignums())
        figures, log = [], io.StringIO()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            if function == "SurrogateData":
                np.random.seed(surrogate_seed)
                random.seed(surrogate_seed)
            with warnings.catch_warnings(record=True) as caught, redirect_stdout(log):
                # pyEDM imports show into both modules; prevent GUI display without
                # changing its plotting code or closing figures before capture.
                with patch("pyEDM.API.show"), patch("pyEDM.AuxFunc.show"):
                    result = getattr(api(), function)(dataFrame=api_frame(frame), **call_arguments(parameters))
                for number in sorted(set(plt.get_fignums()) - before):
                    output = io.BytesIO()
                    plt.figure(number).savefig(output, format="png", dpi=150, bbox_inches="tight")
                    figures.append(output.getvalue())
            tables = tables_from_result(result)
            metrics = []
            for name, table in tables.items():
                if {"Observations", "Predictions"}.issubset(table):
                    finite = table[["Observations", "Predictions"]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(finite) >= 2:
                        metrics.append({"table": name, **api().ComputeError(finite.Observations, finite.Predictions)})
            return {"function": function, "version": version("pyEDM"),
                    "parameters": copy.deepcopy(parameters), "data": frame.copy(deep=True),
                    "tables": tables, "figures": figures, "metrics": pd.DataFrame(metrics),
                    "diagnostics": diagnostics, "warnings": [str(w.message) for w in caught],
                    "log": log.getvalue(), "surrogate_seed": surrogate_seed if function == "SurrogateData" else None}
        finally:
            if function == "SurrogateData":
                np.random.set_state(numpy_state)
                random.setstate(python_state)
            for number in set(plt.get_fignums()) - before:
                plt.close(number)


def api_frame(frame):
    """pyEDM 2.3.0 extends ISO times correctly, but not numpy datetime64 times."""
    output = frame.copy(deep=True)
    if pd.api.types.is_datetime64_any_dtype(output["time"]):
        output["time"] = output["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return output


def reproduction_code(run):
    return ("# Install the dashboard requirements; run beside the extracted CSV and JSON.\n"
            "import json\nimport random\nimport numpy as np\nimport pandas as pd\nimport pyEDM\n"
            "from pathlib import Path\nfrom sklearn import linear_model\n\n"
            "def main():\n"
            "    folder = Path(__file__).resolve().parent\n"
            "    data = pd.read_csv(folder / 'prepared_data.csv')\n"
            "    with open(folder / 'parameters.json') as f:\n        config = json.load(f)\n"
            "    args = config['parameters']\n"
            "    if config.get('surrogate_seed') is not None:\n"
            "        np.random.seed(config['surrogate_seed'])\n        random.seed(config['surrogate_seed'])\n"
            "    if isinstance(args.get('solver'), dict):\n"
            "        spec = args['solver']\n"
            "        args['solver'] = getattr(linear_model, spec['name'])(**spec['parameters'])\n"
            "    if 'validLib' in args:\n        args['validLib'] = np.asarray(args['validLib'], dtype=bool)\n"
            f"    result = pyEDM.{run['function']}(dataFrame=data, **args)\n"
            "    print(result)\n    return result\n\n"
            "# Required for pyEDM's multiprocessing scans on macOS.\n"
            "if __name__ == '__main__':\n    result = main()\n")


def export_bundle(run):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("prepared_data.csv", run["data"].to_csv(index=False))
        config = {k: run[k] for k in ("function", "version", "parameters", "context", "surrogate_seed") if k in run}
        archive.writestr("parameters.json", json.dumps(config, indent=2, default=str))
        archive.writestr("reproduce.py", reproduction_code(run))
        for name, table in run["tables"].items():
            archive.writestr(f"results/{name}.csv", table.to_csv(index=False))
        for i, figure in enumerate(run["figures"], 1):
            archive.writestr(f"figures/{i}.png", figure)
        for name in ("diagnostics", "metrics", "missingness", "sleep_timestamps"):
            if name in run:
                archive.writestr(f"{name}.csv", run[name].to_csv(index=False))
    return output.getvalue()
