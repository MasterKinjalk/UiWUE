"""UiWUE app shell. Each page registers itself from app/pages/<page>.py.

    python app/app.py                      # then open http://127.0.0.1:8050
    python app/app.py --port 8060 --debug  # other port, hot reload

Works from the repo root or from app/. The navbar lists every registered page by its ``order``,
so adding a page needs no edit here.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # pages import helpers that sit next to this file
    sys.path.insert(0, str(HERE))

import components  # noqa: F401  (registers the shared figure theme before pages load)
import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=str(HERE / "pages"),
    assets_folder=str(HERE / "assets"),
    title="UiWUE",
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.FLATLY],
)
server = app.server  # for a WSGI server, e.g. waitress-serve --port 8050 app:server

pages = sorted(dash.page_registry.values(), key=lambda p: (p.get("order") is None, p.get("order") or 0))
navbar = dbc.NavbarSimple(
    [dbc.NavLink(p["name"], href=p["relative_path"]) for p in pages],
    brand="UiWUE · water-deficit workbench",
    color="primary",
    dark=True,
    fluid=True,
)
app.layout = html.Div([navbar, dbc.Container(dash.page_container, fluid=True, className="py-3")])


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the UiWUE Dash app.")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve the local network")
    p.add_argument("--debug", action="store_true", help="hot reload and the Dash error panel")
    args = p.parse_args(argv)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
