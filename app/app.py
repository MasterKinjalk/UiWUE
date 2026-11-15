"""UiWUE app shell. Each page registers itself from app/pages/<page>.py.

    python app/app.py                      # then open http://127.0.0.1:8050
    python app/app.py --port 8060 --debug  # other port, hot reload
    python app/app.py --runs               # the New run page may start runs on a folder of your own data

Works from the repo root or from app/. The navbar draws the pages as a path in four groups
(``components.NAV``: See, Detect, Act, Data); a registered page missing there still shows, after
them, by its ``order``. On a phone the groups fold into one menu button that names the current page.
A nav link carries the selection (?bed=&plant=&day=) to the next page when that page's layout reads it.
A page registered with ``nav=False`` (New run) is reached from Home only. UIWUE_DERIVED points the app at
another set of tables (a run's workspace); UIWUE_RUN_NAME then names that run beside the wordmark.
"""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # pages import helpers that sit next to this file
    sys.path.insert(0, str(HERE))
try:  # a fresh clone without `pip install -e .`: use the package from src/
    import uiwue  # noqa: F401
except ImportError:
    sys.path.insert(0, str(HERE.parent / "src"))

import components  # registers the shared figure theme before pages load
import dash
import dash_bootstrap_components as dbc
from dash import ClientsideFunction, Input, Output, dcc, html

FONTS = "https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Source+Serif+4:wght@600;700&display=swap"
app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=str(HERE / "pages"),
    assets_folder=str(HERE / "assets"),
    title="UiWUE",
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.FLATLY, FONTS],
)
# design/tokens.css is the one copy of the CSS tokens (--uw-*); assets/uiwue.css uses them
TOKENS_CSS = (HERE.parent / "design" / "tokens.css").read_text(encoding="utf-8")
app.index_string = app.index_string.replace("{%css%}", "{%css%}\n<style>" + TOKENS_CSS + "</style>")
server = app.server  # for a WSGI server, e.g. waitress-serve --port 8050 app:server

pages = sorted(  # the pages of the nav; a page registered with nav=False is reached from Home only
    (p for p in dash.page_registry.values() if p.get("nav", True)),
    key=lambda p: (p.get("order") is None, p.get("order") or 0),
)
RUN_NAME = os.environ.get("UIWUE_RUN_NAME")  # set when this app shows a run's workspace, not the repo's derived/


def nav_groups(registry=None):
    """[(group, question, [page, ...])]: components.NAV filled from the registry, every other page (but Home,
    which the wordmark links) in a last group, so a new page is never left out of the nav."""
    registry = pages if registry is None else registry
    by_path = {p["path"]: p for p in registry}
    out, placed = [], {"/"}
    for group, question, items in components.NAV:
        found = [by_path[path] for path, _ in items if path in by_path]
        placed |= {p["path"] for p in found}
        if found:
            out.append((group, question, found))
    rest = [p for p in registry if p["path"] not in placed]
    if rest:
        out.append(("More", "Other pages", rest))
    return out


def nav_links(prefix):
    """The grouped links: a caps label (its question on hover) over a row of links. Links are dcc.Link, so they
    navigate without a reload; nav.js marks the current one (class active, aria-current)."""
    out = []
    for group, question, found in nav_groups():
        gid = f"{prefix}-{group.lower()}"
        links = [
            dcc.Link(p["name"], href=p["relative_path"], title=components.BLURB.get(p["path"]), className="nav-link")
            for p in found
        ]
        out.append(
            html.Div(
                [
                    html.Span(group, id=gid, title=question, className="uw-navkicker"),
                    html.Div(links, className="uw-links"),
                ],
                role="group",
                className="uw-navgroup",
                **{"aria-labelledby": gid},
            )
        )
    return out


navbar = html.Header(
    [
        dcc.Location(id="uw-url"),
        dcc.Link("UiWUE", href="/", title="Home: the experiment and its findings", className="uw-brand"),
        html.Span(
            RUN_NAME,
            title=f"This app shows the tables of run {RUN_NAME} (New run), not the committed ones",
            className="uw-chip uw-chip-context uw-runchip",
        )
        if RUN_NAME
        else None,
        html.Nav(nav_links("uw-ng"), className="uw-navrow", **{"aria-label": "Pages"}),
        html.Details(  # phone width: one button naming the page, the same groups below it
            [
                html.Summary(
                    [
                        html.Span("Pages", id="uw-menu-current"),
                        html.Span(" · all pages", className="uw-sr"),
                        html.Span("▾", className="uw-menu-caret", **{"aria-hidden": "true"}),
                    ],
                    className="uw-menu-btn",
                ),
                html.Nav(nav_links("uw-mg"), className="uw-menu-list", **{"aria-label": "Pages"}),
            ],
            id="uw-menu",
            className="uw-menu",
        ),
    ],
    className="uw-nav",
)
app.layout = html.Div([navbar, dash.page_container], className="uw-page")

CARRY = ("bed", "plant", "day")  # the selection a nav link carries to the next page, when that page reads it


def carried(registry=None):
    """{relative path: the CARRY keys its layout() takes}: read from each page's layout signature, so a page that
    starts reading ?day= gets it from the nav without an edit here. Only pages that show one plant (they take
    ``plant``): Export's ``bed`` is a row filter, and a nav click should not narrow a download unasked."""
    out = {}
    for p in pages if registry is None else registry:
        layout = p.get("layout")
        try:
            names = set(inspect.signature(layout).parameters) if callable(layout) else set()
        except (TypeError, ValueError):
            names = set()
        keys = [k for k in CARRY if k in names]
        if "plant" in keys and p["path"] != "/":
            out[p["relative_path"]] = keys
    return out


app.index_string = app.index_string.replace(
    "{%scripts%}", "<script>window.UW_CARRY = " + json.dumps(carried()) + ";</script>\n{%scripts%}"
)

# mark the current page in both navs, name it on the menu button, and fold the menu after a pick (assets/nav.js)
app.clientside_callback(
    ClientsideFunction(namespace="uiwue", function_name="markPage"),
    Output("uw-menu-current", "children"),
    Input("uw-url", "pathname"),
)


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the UiWUE Dash app.")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve the local network")
    p.add_argument("--debug", action="store_true", help="hot reload and the Dash error panel")
    p.add_argument("--runs", action="store_true", help="let the New run page start runs (off by default: demo-safe)")
    args = p.parse_args(argv)
    if args.runs:
        os.environ["UIWUE_RUNS"] = "1"  # read by pages/new_run.py when a run is started
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
