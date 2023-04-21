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
from sklearn.model_selection import cross_val_score


def fit_model(data: pd.DataFrame, model):
    """
    Given a pandas DataFrame of housing data and a scikit-learn model,
    this function fits a scaler and then fits the model to the DataFrame.
    It returns the model, scaler features and target.
    """
    scaler = StandardScaler()
    features = data.drop(columns=["Price_GBP", "Address"])
    features = scaler.fit_transform(features.values)
    target = data["Price_GBP"]
    model.fit(features, target)
    return model, scaler, features, target

def get_model_score(model, features: pd.DataFrame, target: pd.Series) -> float:
    """
    Performs k-fold cross validation on a model with k=10. Returns the mean of the scores.
    """
    scores = cross_val_score(model, features, target, cv=10, scoring ='neg_mean_squared_error')
    avg_score = int(np.sqrt(np.abs(scores.mean())))
    return avg_score

data = pd.read_csv("data//" + "manchester".lower())

model, scaler, features, target = fit_model(data, LinearRegression())

print(get_model_score(model, features, target))