import pandas as pd
import numpy as np
import statsmodels.api as sm
from models.model_interface import ModelInterface


class SarimaModel(ModelInterface):

    order: bool
    seasonal_order: bool
    enforce_stationarity: bool
    enforce_invertibility: bool
    scaler: any
    model: any

    def __init__(self, order: int, seasonal_order: int, enforce_stationarity: bool, enforce_invertibility: bool, scaler):
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.scaler = scaler

    def train(self, train_df: pd.DataFrame):
        ts = train_df.consumption.values
        y_scaled = self.scaler.fit_transform(np.array(ts)[:, np.newaxis])

        self.model = sm.tsa.statespace.SARIMAX(y_scaled,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=self.enforce_stationarity,
                        enforce_invertibility=self.enforce_invertibility)
        self.model = self.model.fit(method='cg', disp=0)

    def __zero_if_negative(self, value):
        if value < 0:
            return 0
        return value

    def __high_if_negative_mean(self, mean, high):
        if mean >= 0:
            return mean
        return high

    def __combine_predictions(self, mean, high):
        return [self.__zero_if_negative(self.__high_if_negative_mean(x, high[ind])) for ind, x in enumerate(mean)]


    def predict(self, n_days: int) -> list:
        predicted_scaled = self.model.get_forecast(steps=n_days)
        pred_ci = predicted_scaled.conf_int(alpha=0.5)
        predicted_mean = self.scaler.inverse_transform(predicted_scaled.predicted_mean.reshape(-1, 1))
        prediced_high = self.scaler.inverse_transform(pred_ci[:, 1].reshape(-1, 1))
        return self.__combine_predictions(predicted_mean, prediced_high)
