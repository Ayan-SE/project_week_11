import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

def arima_model(data, order, forecast_steps=10, plot=True):
    """
    Fits an ARIMA model to time series data and makes forecasts.
    """

    try:
        # Ensure data is a pandas Series with a DatetimeIndex if possible.
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        model = ARIMA(data, order=order)
        model_fit = model.fit()

        # Generate forecasts
        forecasts = model_fit.forecast(steps=forecast_steps)
        fitted_values = model_fit.fittedvalues

        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(data, label='Original Data')
            plt.plot(fitted_values, color='red', label='Fitted Values')

            # Plot forecasts
            forecast_index = pd.RangeIndex(start=len(data), stop=len(data) + forecast_steps)
            plt.plot(forecast_index, forecasts, color='green', label='Forecasts')

            plt.legend()
            plt.title(f'ARIMA({order}) Model')
            plt.show()

        return forecasts, fitted_values

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def evaluate_arima(data, order, train_size=0.8):
    """
    Evaluates an ARIMA model using a train/test split.
    """
    try:
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        train_len = int(len(data) * train_size)
        train, test = data[:train_len], data[train_len:]

        model = ARIMA(train, order=order)
        model_fit = model.fit()

        predictions = model_fit.forecast(steps=len(test))
        rmse = sqrt(mean_squared_error(test, predictions))
        print(f"ARIMA({order}) RMSE: {rmse}")
        return rmse

    except Exception as e:
         print(f"An error occurred: {e}")
         return None

