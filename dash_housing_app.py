import dash
from dash import dcc
from dash import html
import pandas as pd
import numpy as np
import flask
from dash.dependencies import Output, Input
import plotly.express as px
import dash_leaflet as dl
from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import dash_bootstrap_components as dbc
from etl import *
from model import *

city_list = [
    "Birmingham",
    "Bristol",
    "Cardiff",
    "Edinburgh",
    "Glasgow",
    "Leeds",
    "Liverpool",
    "London",
    "Manchester",
    "Newcastle-upon-Tyne",
    "York",
]

server = flask.Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    [
        html.Div(
            children=[
                html.H1(children="House Price Analytics", className="header-title"),
                html.P(
                    children="Start by choosing the city you are interested in, then choose the key features you'd like the house to have.",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.P(
            html.B(
                children="City",
                className="header-description",
            )
        ),
        dcc.Dropdown(city_list, id="location", value="Manchester"),
        html.P(
            html.B(
                children="Number of Bedrooms",
                className="header-description",
            )
        ),
        dcc.Input(id="num-bedrooms", type="number", value=2),
        html.P(
            html.B(
                children="Number of Bathrooms",
                className="header-description",
            )
        ),
        dcc.Input(id="num-bathrooms", type="number", value=1),
        html.P(
            html.B(
                children="Property Type",
                className="header-description",
            )
        ),
        dcc.Dropdown(
            ["Terraced", "Semi-Detached", "Detached"],
            id="property-type",
            value="Terraced",
        ),
        html.P(
            children="""There are three models to choose from, but some are perhaps more helpful that others. 
                        Can you think why linear regression might not be ideal for predicting house prices by location?""",
            className="header-description",
        ),
        html.P(
            html.B(
                children="Model Type",
                className="header-description",
            )
        ),
        dcc.Dropdown(
            ["Linear Regression", "K-Nearest Neighbors", "XGBoost"],
            id="model-type",
            value="XGBoost",
        ),
        html.Div(dash.dash_table.DataTable(id="table"), style={"display": "None"}),
        html.P(
            children="Now click on the map, and view the model's prediction for that location.",
            className="header-description",
        ),
        dl.Map(
            [dl.TileLayer(), dl.LayerGroup(id="layer")],
            id="map",
            zoom=5,
            center=[54.5, -1.5],
            style={
                "width": "100%",
                "height": "50vh",
                "margin": "auto",
                "display": "block",
            },
        ),
    ]
)


@app.callback(
    Output("map", "center"),
    Output("map", "zoom"),
    Output("table", "data"),
    Input("location", "value"),
)
def get_map(location):
    # data = create_table(location)
    # cleaned_data = clean_data(data)
    cleaned_data = pd.read_csv("data//" + location.lower())
    print(len(cleaned_data))
    Latitude_Midpoint = (
        cleaned_data["Latitude"].min()
        + (cleaned_data["Latitude"].max() - cleaned_data["Latitude"].min()) / 2
    )
    Longitude_Midpoint = (
        cleaned_data["Longitude"].min()
        + (cleaned_data["Longitude"].max() - cleaned_data["Longitude"].min()) / 2
    )
    coord = [Latitude_Midpoint, Longitude_Midpoint]
    zoom = 12
    return (
        coord,
        zoom,
        cleaned_data.to_dict("records"),
    )


@app.callback(
    Output("layer", "children"),
    Input("map", "click_lat_lng"),
    Input("table", "data"),
    Input("num-bedrooms", "value"),
    Input("num-bathrooms", "value"),
    Input("property-type", "value"),
    Input("model-type", "value"),
)
def map_click(
    click_lat_lng, data, num_bedrooms, num_bathrooms, property_type, model_type
):
    if click_lat_lng is None:
        raise PreventUpdate
    lat, lng = click_lat_lng[0], click_lat_lng[1]
    data = pd.DataFrame(data)
    model_dict = {
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "XGBoost": xgb.XGBRegressor(),
    }
    model = model_dict[model_type]
    prediction = predict(
        data,
        num_bedrooms,
        num_bathrooms,
        lat,
        lng,
        property_type == "Detached",
        property_type == "Semi-Detached",
        property_type == "Terraced",
        model,
    )
    prediction = "£" + "{:.2f}".format(prediction)
    return [dl.Marker(position=click_lat_lng, children=dl.Tooltip(prediction))]


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8050)
