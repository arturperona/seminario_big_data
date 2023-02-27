import pandas as pd
from typing import Any
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from models.model_interface import ModelInterface




class AutoregModel(ModelInterface):
    regressor: Any
    random_state: int
    lags: int
    model: Any


    def __init__(self, regressor: Any, lags: int = 25, random_state: int = 0):
        self.regressor = regressor
        self.random_state = random_state
        self.lags = lags
        self.model = ForecasterAutoreg(regressor = regressor(random_state=random_state), lags=lags)

    def train(self, train_df: pd.DataFrame):
        self.forecaster.fit(y=train_df['consumption'])

    def __zero_if_negative(self, value):
        if value < 0:
            return 0
        return value

    def predict(self, n_days: int):
        predictions = self.forecaster.predict(n_days)
        predictions =  [self.__zero_if_negative(x) for x in predictions]
        return predictions