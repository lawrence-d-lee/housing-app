import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

import dash_housing_app as dha


def test_sum():
    assert sum([1, 2]) == 3


def test_fit_model():
    data = pd.read_csv("data//manchester")
    model = xgb.XGBRegressor()
    assert type(dha.fit_model(data, model)[0]) is xgb.sklearn.XGBRegressor
    assert type(dha.fit_model(data, model)[1]) is StandardScaler
    assert type(dha.fit_model(data, model)[2]) is np.ndarray
    assert type(dha.fit_model(data, model)[3]) is pd.Series
