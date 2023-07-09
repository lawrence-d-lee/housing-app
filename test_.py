import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

import etl as etl
import dash_housing_app as dha

def test_fit_model():
    data = pd.read_csv("data//manchester")
    xgb_model = xgb.XGBRegressor()
    knn_model = KNeighborsRegressor(n_neighbors=7)
    assert type(dha.fit_model(data, xgb_model)[0]) is xgb.sklearn.XGBRegressor
    assert type(dha.fit_model(data, knn_model)[0]) is KNeighborsRegressor
    assert type(dha.fit_model(data, xgb_model)[1]) is StandardScaler
    assert type(dha.fit_model(data, xgb_model)[2]) is np.ndarray
    assert type(dha.fit_model(data, xgb_model)[3]) is pd.Series

def test_get_model_score():
    data = pd.read_csv("data//liverpool")
    model = xgb.XGBRegressor()
    model, scaler, features, target = dha.fit_model(data, model) 
    assert type(dha.get_model_score(model, features, target)) is float
    assert dha.get_model_score(model, features, target) >= 0
       
def test_predict():
    data = pd.read_csv("data//manchester")
    xgb_model = xgb.XGBRegressor()
    knn_model = KNeighborsRegressor(n_neighbors=7)
    assert type(dha.predict(data, 2, 1, 53.4808, 2.2426, 1, 0, 0, xgb_model)) is float
    assert dha.predict(data, 2, 1, 53.4808, 2.2426, 1, 0, 0, xgb_model) >= 0
    assert type(dha.predict(data, 1, 2, 53.4808, 2.2426, 0, 1, 0, knn_model)) is float
    assert dha.predict(data, 1, 2, 53.4808, 2.2426, 0, 1, 0, knn_model) >= 0