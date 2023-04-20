import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import flask
import numpy as np
import pandas as pd
import xgboost as xgb
from dash import Dash, Input, Output, dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def fit_model(data: pd.DataFrame, model) -> pd.DataFrame:
    """
    Given a pandas DataFrame of housing data and a scikit-learn model,
    this function fits a scaler and then fits the model to the DataFrame.
    It returns the model and the scaler.
    """
    scaler = StandardScaler()
    X = data.drop(columns=["Price_GBP", "Address"])
    X = scaler.fit_transform(X.values)
    y = data["Price_GBP"]
    model.fit(X, y)
    return model, scaler


def predict(
    cleaned_data: pd.DataFrame,
    num_Bedrooms: int,
    num_Bathrooms: int,
    latitude: float,
    longitude: float,
    detached: int,
    semi_detached: int,
    terraced: int,
    model: int,
) -> float:
    """
    Given a pandas DataFrame of cleaned housing data, a scikit-learn model, and an example one wishes to make a prediction on,
    this function fits the model to the data and returns the predicted price of the example.
    """
    example = np.array(
        [
            num_Bedrooms,
            num_Bathrooms,
            latitude,
            longitude,
            detached,
            semi_detached,
            terraced,
        ]
    ).reshape(1, -1)
    fitted_model, scaler = fit_model(cleaned_data, model)
    example = scaler.transform(example)
    prediction = fitted_model.predict(example)
    return prediction[0]


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
    cleaned_data = pd.read_csv("data//" + location.lower())
    print(len(cleaned_data))
    latitude_midpoint = (
        cleaned_data["Latitude"].min()
        + (cleaned_data["Latitude"].max() - cleaned_data["Latitude"].min()) / 2
    )
    longitude_midpoint = (
        cleaned_data["Longitude"].min()
        + (cleaned_data["Longitude"].max() - cleaned_data["Longitude"].min()) / 2
    )
    coord = [latitude_midpoint, longitude_midpoint]
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
