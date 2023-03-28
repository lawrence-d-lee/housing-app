from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
import pandas as pd


def fit_model(data, model):
    """
    Given a pandas DataFrame of housing data and a scikit-learn model, this function fits a scaler and then fits the model to the DataFrame. 
    It returns the model and the scaler.
    """
    scaler = StandardScaler()
    X = data.drop(columns=["Price_GBP", "Address"])
    X = scaler.fit_transform(X.values)
    y = data["Price_GBP"]
    model.fit(X, y)
    return model, scaler


def predict(
    cleaned_data,
    Num_Bedrooms,
    Num_Bathrooms,
    Latitude,
    Longitude,
    Detached,
    Semi_Detached,
    Terraced,
    model,
):
    """
    Given a pandas DataFrame of cleaned housing data, a scikit-learn model, and an example one wishes to make a prediction on,
    this function fits the model to the data and returns the predicted price of the example.
    """
    example = np.array(
        [
            Num_Bedrooms,
            Num_Bathrooms,
            Latitude,
            Longitude,
            Detached,
            Semi_Detached,
            Terraced,
        ]
    ).reshape(1, -1)
    fitted_model, scaler = fit_model(cleaned_data, model)
    example = scaler.transform(example)
    prediction = fitted_model.predict(example)
    return prediction[0]
