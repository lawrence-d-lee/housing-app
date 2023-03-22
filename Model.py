from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
import pandas as pd

scaler = StandardScaler()
LR = LinearRegression()
XGB = xgb.XGBRegressor()

def fit_model(data, model):
    X = data.drop(columns=["Price_GBP", "Address"])
    y = data["Price_GBP"]
    X = scaler.fit_transform(X.values)
    model.fit(X, y)
    return model

def predict(cleaned_data, Num_Bedrooms, Num_Bathrooms, Latitude, Longitude, Detached, Semi_Detached, Terraced,
model):
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
    fitted_model = fit_model(cleaned_data, model)
    example = scaler.transform(example)
    prediction = fitted_model.predict(example)
    return prediction[0]
