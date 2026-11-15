"""UiWUE Dash app: the shell (theme, navigation, home page) for every team member's pages.

Run from the repository root:

    .venv/Scripts/python.exe app/app.py            # http://127.0.0.1:8050
    .venv/Scripts/python.exe app/app.py --port 8060 --debug

Each page is one file in ``app/pages/`` that calls ``dash.register_page``; the navigation bar and
the home page list every registered page, so adding a page needs no edit here. See
docs/threed/APP.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # pages import helpers that sit next to this file
    sys.path.insert(0, str(HERE))

import dash  # noqa: E402
import dash_bootstrap_components as dbc  # noqa: E402
from dash import Dash, html  # noqa: E402

TITLE = "UiWUE"
SUBTITLE = "Water-deficit stress in sugar beet (ADA 2025)"


def nav_pages() -> list[dict]:
    """Registered pages except the home page, in ``order``."""
    pages = [p for p in dash.page_registry.values() if p["path"] != "/"]
    return sorted(pages, key=lambda p: (p.get("order") is None, p.get("order") or 0, p["name"]))


def home_layout(**_):
    cards = [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(p["name"], className="card-title"),
                        html.P(p.get("description") or "", className="card-text small"),
                        dbc.Button("Open", href=p["relative_path"], size="sm", color="primary"),
                    ]
                ),
                class_name="h-100",
            ),
            md=4,
            class_name="mb-3",
        )
        for p in nav_pages()
    ]
    return dbc.Container(
        [
            html.H3(TITLE, className="mt-4 mb-0"),
            html.P(SUBTITLE, className="text-muted"),
            dbc.Row(cards) if cards else html.P("No pages registered yet."),
        ],
        class_name="py-2",
    )


def navbar() -> dbc.Navbar:
    links = [
        dbc.NavItem(dbc.NavLink(p["name"], href=p["relative_path"], active="exact"))
        for p in nav_pages()
    ]
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(TITLE, href="/", class_name="fw-semibold"),
                dbc.Nav(links, navbar=True, class_name="me-auto"),
                html.Span(SUBTITLE, className="navbar-text small d-none d-lg-inline"),
            ],
            fluid=True,
        ),
        color="primary",
        dark=True,
        class_name="mb-0",
    )


def create_app() -> Dash:
    """Build the app. Create it once per process: Dash pages register their callbacks globally."""
    app = Dash(
        __name__,
        use_pages=True,
        pages_folder=str(HERE / "pages"),
        assets_folder=str(HERE / "assets"),
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
        title=TITLE,
    )
    dash.register_page("home", path="/", name="Home", layout=home_layout, order=0)
    app.layout = html.Div([navbar(), dash.page_container])
    return app


app = create_app()
server = app.server  # for a WSGI server, e.g. waitress-serve --port 8050 app:server


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run the UiWUE Dash app.")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve the local network")
    p.add_argument("--debug", action="store_true", help="hot reload and the Dash error panel")
    args = p.parse_args(argv)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
