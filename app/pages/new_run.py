"""New run: point the analysis at a data folder of your own and run every step into a workspace of its own.

Reached from Home (it is not in the nav: ``nav=False``). The page only starts and watches
``python -m uiwue.pipeline`` (src/uiwue/pipeline.py): the steps run in a separate process and write to
``runs/user/<name>/``, never to the repo's ``derived/``, so this app keeps showing the committed tables.
``View`` starts a second app on a finished run's tables, on its own port.

Start is off unless the app was started with ``--runs`` (``UIWUE_RUNS=1``), so nothing can be started by a
stray click during a demo; without it the page still checks a folder and shows the steps and the command.
The learned models (plant mask, refined depth, fusion score) are off until switched on, run or no run.
"""

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from components import chip, details, header, next_steps, panel, stat, stat_missing
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from uiwue import pipeline as pl
from uiwue.core import REPO

QUESTION = "How do I run my own data through the same analysis?"
dash.register_page(__name__, path="/new-run", name="New run", order=95, description=QUESTION, nav=False)

KICKER = "Your data"
CAMERAS = ("thermal", "depth", "rgb", "hsi")
LAYOUT_HOVER = (
    "The folder follows the ADA 2025 layout: thermal/, depth/ and rgb/ with one <bed>_<plant>/<date>/<hour>/ "
    "folder per run, plus air_readings.parquet, leaf_readings.parquet and metadata_rwc_par_en.xlsx. "
    "Beds W1 (control), W2 and W3 (deficit), on the trial's calendar."
)
OFF_HOVER = (
    "Start is off: this app was started without --runs, so nothing can be started from here. "
    "Run the command below in a terminal, or start the app with: python app/app.py --runs"
)
STATE_CHIP = {  # run state -> (chip kind, hover)
    "running": ("context", "The steps are running in their own process"),
    "done": ("best", "Every step finished"),
    "partial": ("caveat", "Some steps failed; the others finished"),
    "failed": ("caveat", "No step finished"),
    "stopped": ("rule", "Stopped before the last step"),
}
VIEW_PORTS = range(8061, 8100)


def runs_enabled() -> bool:
    return os.environ.get("UIWUE_RUNS") == "1"


# ---------------------------------------------------------------- what the folder holds
def _span(f):
    """'12 Aug – 22 Sep' from the first and last date folder."""
    if not f.get("first"):
        return "zipped" if f.get("zipped") else None
    a, b = (datetime.fromisoformat(f[k]) for k in ("first", "last"))
    return f"{a.day} {a:%b} – {b.day} {b:%b}"


def tiles(found):
    """One tile per input: plants and dates for a camera folder, found or not for a file."""
    out = []
    for key, (name, where) in pl.INPUTS.items():
        f = found[key] if found else None
        if f is None:
            out.append(stat_missing(name, f"No folder checked yet. Looked for: {where}"))
        elif not f["found"]:
            out.append(stat_missing(name, f"Not in this folder: {where}"))
        elif key in CAMERAS:
            beds = ", ".join(f"{b} {n}" for b, n in f["beds"].items())
            out.append(
                stat(
                    name,
                    f"{f['plants']}",
                    "plants",
                    "gray",
                    sub=_span(f),
                    hover=f"{where}: {f['plants']} plants ({beds}), {f['plant_days']:,} plant-days"
                    + (f", {f['zipped']} zipped" if f["zipped"] else ""),
                )
            )
        else:
            out.append(stat(name, "found", "", "gray", hover=where))
    return out


def notes(folder, found):
    """Caveat chips for a folder the analysis only half fits."""
    if not folder:
        return []
    if found is None:
        return [chip("no folder", f"{folder} is not a folder on this machine", "caveat")]
    out = []
    beds = {b for k in ("thermal", "depth", "rgb") for b in found[k].get("beds", {})}
    if beds and "W1" not in beds:
        out.append(chip("no control bed", "No W1 plants: every comparison is against the control bed W1", "caveat"))
    if other := pl.other_beds(found):
        out.append(
            chip(
                "other beds",
                f"{', '.join(other)}: only W1, W2 and W3 are compared; these plants are left out",
                "caveat",
            )
        )
    return out


def step_options(found):
    """Checklist options for the steps: off and dashed when the folder lacks the step's input."""
    out = []
    for s in pl.STEPS:
        if s.ml:
            continue
        why = pl.missing_input(s, found) if found else None
        hover = s.about + (f" Writes {s.writes}." if s.writes else "") + (f" Off: {why}." if why else "")
        out.append(dict(label=html.Span(s.label, title=hover), value=s.key, disabled=bool(why)))
    return out


def ml_options(found):
    ready = pl.ml_ready(found) if found else {k: None for k in pl.ML}
    out = []
    for key, (label, about) in pl.ML_ABOUT.items():
        why = ready[key]
        out.append(
            dict(label=html.Span(label, title=about + (f" Off: {why}." if why else "")), value=key, disabled=bool(why))
        )
    return out


def checks(id, options, value, label, hover, cls=""):
    """Multi-select drawn as segmented buttons (native checkboxes), as Export's filters."""
    return html.Fieldset(
        [
            html.Legend(label, title=hover, className="uw-kicker nr-legend"),
            dbc.Checklist(
                id=id,
                options=options,
                value=value,
                inline=True,
                className=f"uw-seg nr-seg {cls}".strip(),
                inputClassName="btn-check",
                labelClassName="uw-seg-btn",
                labelCheckedClassName="uw-seg-on",
            ),
        ],
        className="nr-field",
    )


def field(label, control, hover):
    return html.Label([html.Span(label, title=hover), control], className="uw-label nr-label")


# ---------------------------------------------------------------- the runs
def _when(iso):
    t = datetime.fromisoformat(iso).astimezone()
    return f"{t.day} {t:%b} {t:%H:%M}"


def _took(seconds):
    if seconds is None:
        return "–"
    m, s = divmod(int(seconds), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def viewer_port(ws) -> int | None:
    """The port a viewer of this run answers on, None when there is none."""
    try:
        port = int(json.loads((Path(ws) / "viewer.json").read_text())["port"])
    except (OSError, ValueError, KeyError):
        return None
    return port if _answers(port) else None


def _answers(port) -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def open_viewer(st, wait=45) -> int | None:
    """Start a second app on the run's tables (its own port) and wait until it answers."""
    ws = Path(st["workspace"])
    if port := viewer_port(ws):
        return port
    port = next((p for p in VIEW_PORTS if not _answers(p)), None)
    if port is None:
        return None
    env = os.environ | dict(
        UIWUE_DERIVED=str(ws / "derived"),
        UIWUE_CLOUDS=str(ws / "threed" / "clouds"),
        UIWUE_DATA=st["data_dir"],
        UIWUE_RUN_NAME=st["name"],
    )
    env.pop("UIWUE_RUNS", None)  # a viewer starts no runs
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "app" / "app.py"), "--port", str(port)],
        cwd=REPO,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    (ws / "viewer.json").write_text(json.dumps(dict(port=port, pid=proc.pid)))
    t0 = time.time()
    while time.time() - t0 < wait and proc.poll() is None:
        if _answers(port):
            return port
        time.sleep(0.5)
    return None


def has_tables(st) -> bool:
    return any((Path(st["workspace"]) / "derived").glob("*/*.parquet"))


def step_chips(st):
    out = []
    for s in st["steps"]:
        hover = {
            "done": f"{s['label']}: done in {_took(s.get('seconds'))}",
            "running": f"{s['label']}: running",
            "failed": f"{s['label']}: {s.get('why') or 'failed'}",
            "skipped": f"{s['label']}: skipped, {s.get('why') or 'not reached'}",
            "stopped": f"{s['label']}: stopped",
            "pending": f"{s['label']}: waiting",
        }[s["state"]]
        out.append(html.Span(s["label"], title=hover, className=f"nr-step nr-step-{s['state']}"))
    return html.Div(out, className="nr-steps", **{"aria-label": "Steps"})


def run_card(st, first=False):
    kind, hover = STATE_CHIP.get(st["state"], ("rule", ""))
    ran = [s for s in st["steps"] if s["state"] != "skipped"]
    done = sum(s["state"] == "done" for s in ran)
    took = sum(s["seconds"] or 0 for s in st["steps"] if s.get("seconds"))
    actions = []
    if st["alive"]:
        actions.append(
            html.Button(
                "Stop",
                id={"type": "nr-stop", "name": st["name"]},
                n_clicks=0,
                title="Stop this run and its steps",
                className="uw-btn uw-btn-outline",
            )
        )
    elif has_tables(st):
        if port := viewer_port(st["workspace"]):
            actions.append(
                html.A(
                    "Open",
                    href=f"http://127.0.0.1:{port}/",
                    target="_blank",
                    rel="noopener",
                    title=f"This run's pages, in a second app on port {port}",
                    className="uw-btn uw-btn-primary",
                )
            )
        else:
            actions.append(
                html.Button(
                    "View",
                    id={"type": "nr-view", "name": st["name"]},
                    n_clicks=0,
                    title="Start a second app on this run's tables; this app keeps the committed ones",
                    className="uw-btn uw-btn-outline",
                )
            )
    ml = [chip("GPU", "Used: " + ", ".join(pl.ML_ABOUT[m][0] for m in st["ml"]), "rule")] if st.get("ml") else []
    return html.Div(
        [
            html.Div(
                [
                    html.B(st["name"], className="nr-run-name", title=f"{st['data_dir']} → {st['workspace']}"),
                    html.Span(st["state"], title=hover, className=f"uw-chip uw-chip-{kind}"),
                    *ml,
                    html.Div(className="uw-spacer"),
                    *actions,
                ],
                className="nr-run-head",
            ),
            html.Div(
                [
                    html.Span(
                        [html.B(f"{done} / {len(ran)}"), " steps"], title="Steps finished of the steps with input"
                    ),
                    html.Span(_when(st["started"]), title="Started"),
                    html.Span(_took(took), title="Time in finished steps"),
                ],
                className="nr-run-meta",
            ),
            step_chips(st),
            details("Log", html.Pre(pl.log_tail(st["workspace"]) or "–", className="nr-log"), id=f"nr-log-{st['name']}")
            if first
            else None,
        ],
        className="nr-run",
    )


def run_cards(runs):
    if not runs:
        return html.Div(
            [html.Strong("Runs"), html.Span("none yet")], className="uw-empty", title=f"No run in {pl.RUNS_DIR} yet"
        )
    return [run_card(st, first=i == 0) for i, st in enumerate(runs)]


def _signature(runs):
    """What the runs panel shows, as a string: the panel is redrawn only when this changes."""
    return json.dumps(
        [
            (
                st["name"],
                st["state"],
                st["alive"],
                [(s["key"], s["state"]) for s in st["steps"]],
                viewer_port(st["workspace"]),
                has_tables(st),
                pl.log_tail(st["workspace"], 6) if i == 0 else "",
            )
            for i, st in enumerate(runs)
        ]
    )


# ---------------------------------------------------------------- layout
def layout(folder=None, **_):
    folder = (folder or "").strip()
    found = pl.scan(folder) if folder and Path(folder).is_dir() else None
    on = runs_enabled()
    runs = pl.list_runs()
    head = header(
        "New run",
        QUESTION,
        chip("ADA layout", LAYOUT_HOVER, "rule"),
        *([] if on else [chip("start off", OFF_HOVER, "caveat")]),
        kicker=KICKER,
    )
    pick = html.Div(
        [
            field(
                "Folder",
                dbc.Input(
                    id="nr-folder",
                    type="text",
                    value=folder,
                    debounce=True,
                    placeholder="D:\\trial\\data",
                    className="nr-input nr-input-wide",
                    autoComplete="off",
                ),
                "A folder on the machine this app runs on. It is read, never changed (the Environment step adds one "
                "weather file under external/).",
            ),
            html.Button(
                "Check", id="nr-check", n_clicks=0, title="Look for each input", className="uw-btn uw-btn-outline"
            ),
            html.Div(notes(folder, found), id="nr-notes", className="nr-notes"),
        ],
        className="nr-pick",
    )
    core = [s.key for s in pl.STEPS if not s.ml and not (found and pl.missing_input(s, found))]
    steps = panel(
        "Steps",
        checks(
            "nr-steps",
            step_options(found),
            core,
            "Analysis",
            "Each step is one module's own command, run in order. A step is off when the folder lacks its input.",
        ),
        checks(
            "nr-ml",
            ml_options(found),
            [],
            "Models",
            "The learned models. Off until switched on; they need a GPU and their weights on this machine.",
            "nr-seg-ml",
        ),
        details(
            "Options",
            html.Div(
                [
                    field(
                        "Name",
                        dbc.Input(
                            id="nr-name",
                            type="text",
                            value=Path(folder).name if folder else "",
                            debounce=True,
                            className="nr-input",
                            autoComplete="off",
                        ),
                        "The workspace, runs/user/<name>. A name already in use resumes that run.",
                    ),
                    field(
                        "Plants",
                        dbc.Input(
                            id="nr-plants",
                            type="text",
                            value="",
                            debounce=True,
                            placeholder="all",
                            className="nr-input",
                            autoComplete="off",
                        ),
                        "Full IDs such as W1_A2 W2_J5, to run a few plants only. Empty: every plant found.",
                    ),
                    field(
                        "Jobs",
                        dbc.Input(
                            id="nr-jobs",
                            type="number",
                            value=pl.default_jobs(),
                            min=1,
                            max=64,
                            step=1,
                            className="nr-input nr-input-num",
                        ),
                        "Processes the RGB and 3D steps use at once",
                    ),
                ],
                className="nr-options",
            ),
            id="nr-options",
        ),
        html.Div(
            [
                html.Button(
                    "Start",
                    id="nr-start",
                    n_clicks=0,
                    disabled=True,
                    className="uw-btn uw-btn-primary",
                    title="Run the steps in a process of their own, into runs/user/<name>" if on else OFF_HOVER,
                ),
                html.Div(id="nr-note", className="nr-notes", **{"aria-live": "polite"}),
            ],
            className="nr-start",
        ),
        details("Command", html.Pre(id="nr-command", className="nr-log"), id="nr-command-box"),
        hover="The steps the run takes, in order. They run outside this app and write to a workspace of their own.",
    )
    ran = panel(
        "Runs",
        html.Div(run_cards(runs), id="nr-runs", className="nr-runs"),
        hover="Every workspace under runs/user, newest first. Nothing here changes the tables this app shows.",
    )
    return html.Div(
        [
            head,
            pick,
            html.Div(tiles(found), id="nr-tiles", className="uw-tiles uw-lead", **{"aria-label": "Inputs found"}),
            html.Div([steps, ran], className="nr-cols"),
            dcc.Interval(id="nr-tick", interval=3000),
            dcc.Store(id="nr-poke", data=0),
            dcc.Store(id="nr-shown", data=_signature(runs)),
            next_steps(("Home", "/", "Back to the trial"), ("Methods & limits", "/methods", "Checks and limits")),
        ],
        className="nr-page",
    )


# ---------------------------------------------------------------- callbacks
def _plants(text):
    return [p for p in (text or "").replace(",", " ").split() if pl.PLANT_RE.match(p)]


@callback(
    Output("nr-tiles", "children"),
    Output("nr-notes", "children"),
    Output("nr-steps", "options"),
    Output("nr-steps", "value"),
    Output("nr-ml", "options"),
    Output("nr-ml", "value"),
    Output("nr-name", "value"),
    Input("nr-check", "n_clicks"),
    Input("nr-folder", "value"),
    State("nr-name", "value"),
    prevent_initial_call=True,
)
def check(_, folder, name):
    folder = (folder or "").strip().strip('"')
    found = pl.scan(folder) if folder and Path(folder).is_dir() else None
    core = [s.key for s in pl.STEPS if not s.ml and not (found and pl.missing_input(s, found))]
    name = name or (Path(folder).name if found else "")
    return tiles(found), notes(folder, found), step_options(found), core if found else [], ml_options(found), [], name


@callback(
    Output("nr-command", "children"),
    Output("nr-start", "disabled"),
    Input("nr-folder", "value"),
    Input("nr-name", "value"),
    Input("nr-steps", "value"),
    Input("nr-ml", "value"),
    Input("nr-plants", "value"),
    Input("nr-jobs", "value"),
)
def command(folder, name, steps, ml, plants, jobs):
    folder = (folder or "").strip().strip('"')
    ready = bool(folder) and Path(folder).is_dir() and bool(pl.safe_name(name)) and bool(steps or ml)
    line = pl.command_line(folder, name, steps or [], ml or [], _plants(plants), jobs)
    return line, not (ready and runs_enabled())


@callback(
    Output("nr-note", "children"),
    Output("nr-poke", "data"),
    Input("nr-start", "n_clicks"),
    Input({"type": "nr-stop", "name": ALL}, "n_clicks"),
    Input({"type": "nr-view", "name": ALL}, "n_clicks"),
    State("nr-folder", "value"),
    State("nr-name", "value"),
    State("nr-steps", "value"),
    State("nr-ml", "value"),
    State("nr-plants", "value"),
    State("nr-jobs", "value"),
    State("nr-poke", "data"),
    prevent_initial_call=True,
)
def act(_start, _stops, _views, folder, name, steps, ml, plants, jobs, poke):
    """Start, Stop and View: each changes what is on disk, then pokes the runs panel to redraw."""
    trig = ctx.triggered_id
    if not trig or not any(t["value"] for t in ctx.triggered):  # a redrawn button, not a click
        return no_update, no_update
    poke = (poke or 0) + 1
    if trig == "nr-start":
        folder = (folder or "").strip().strip('"')
        if not runs_enabled():
            return chip("start off", OFF_HOVER, "caveat"), no_update
        if not folder or not Path(folder).is_dir() or not pl.safe_name(name) or not (steps or ml):
            return chip("not ready", "Pick a folder, a name and at least one step", "caveat"), no_update
        if other := pl.running():
            return chip("one at a time", f"Run {other['name']} is still going", "caveat"), no_update
        pl.start(Path(folder).resolve(), name, steps or [], ml or [], _plants(plants), int(jobs) if jobs else None)
        t0 = time.time()
        while time.time() - t0 < 10 and not (pl.workspace(name) / "status.json").exists():
            time.sleep(0.2)
        return chip("started", f"Writing to {pl.workspace(name)}", "context"), poke
    st = pl.read_status(pl.workspace(trig["name"]))
    if st is None:
        return no_update, poke
    if trig["type"] == "nr-stop":
        pl.stop(trig["name"])
        return chip("stopped", f"Run {trig['name']} stopped", "rule"), poke
    port = open_viewer(st)
    if port is None:
        return chip("no viewer", "The second app did not start; see the terminal", "caveat"), poke
    return chip("viewer on", f"Run {trig['name']} is open on port {port}: use Open", "context"), poke


@callback(
    Output("nr-runs", "children"),
    Output("nr-shown", "data"),
    Input("nr-tick", "n_intervals"),
    Input("nr-poke", "data"),
    State("nr-shown", "data"),
    prevent_initial_call=True,
)
def refresh(_, _poke, shown):
    runs = pl.list_runs()
    sig = _signature(runs)
    if sig == shown:
        return no_update, no_update
    return run_cards(runs), sig
