"""UiWUE app shell. Each page registers itself from app/pages/<page>.py.

python app/app.py          # then open http://127.0.0.1:8050
"""

import components  # noqa: F401  (registers the shared figure theme before pages load)
import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(
    __name__, use_pages=True, title="UiWUE", suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY]
)

pages = sorted(dash.page_registry.values(), key=lambda p: p.get("order", 99))
navbar = dbc.NavbarSimple(
    [dbc.NavLink(p["name"], href=p["relative_path"]) for p in pages],
    brand="UiWUE · water-deficit workbench",
    color="primary",
    dark=True,
    fluid=True,
)
app.layout = html.Div([navbar, dbc.Container(dash.page_container, fluid=True, className="py-3")])

if __name__ == "__main__":
    app.run(debug=False, port=8050)
