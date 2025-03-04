import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

def plot_time_series(series, title="Time Series Data"):
    """Plots the given time series."""
    plt.figure(figsize=(10, 5))
    plt.plot(series, label="Time Series")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.show()

def adf_test(series):
    """Performs the Augmented Dickey-Fuller test and prints results."""
    result = adfuller(series.dropna())
    print("ADF Test Results:")
    print(f"Test Statistic: {result[0]}")
    print(f"P-Value: {result[1]}")
    print(f"Critical Values: {result[4]}")
    
    if result[1] <= 0.05:
        print("✅ The series is likely stationary (Reject H0).")
    else:
        print("❌ The series is likely non-stationary (Fail to reject H0).")

def plot_rolling_statistics(series, window=12):
    """Plots rolling mean and standard deviation for visual stationarity check."""
    plt.figure(figsize=(10, 5))
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    plt.plot(series, label="Original Data", color='blue')
    plt.plot(rolling_mean, label="Rolling Mean", color='red')
    plt.plot(rolling_std, label="Rolling Std Dev", color='black')

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Rolling Mean & Standard Deviation")
    #plt.legend()
    plt.show()