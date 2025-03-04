import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_closing_prices(data, title="Closing Prices Over Time"):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, dashes=False)
    plt.title(title, fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Closing Price (USD)", fontsize=12)
    plt.legend(labels=data.columns, loc="upper left")
    plt.grid(True)
    plt.show()


def calculate_daily_returns(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a Pandas DataFrame.")

    # Ensure data contains numeric values
    if not all(data.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
        raise TypeError("Data contains non-numeric values. Check the dataset.")

    returns = data.pct_change().dropna()  # Compute daily returns
    return returns

# Function to plot daily returns for volatility analysis
def plot_daily_returns(returns, title="Daily Percentage Change (Volatility)"):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=returns)
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)  # Zero line for reference
    plt.title(title, fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Daily Return (%)", fontsize=12)
    plt.legend(labels=returns.columns, loc="upper left")
    plt.grid(True)
    plt.show()

def calculate_daily_percentage_change(data):
    """
    Calculate daily percentage change for the 'Close' price.
    """
    data['Daily_Change'] = data['Close'].pct_change() * 100
    return data

def plot_percentage_change(data):
    """
    Plot the daily percentage change.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Daily_Change'], label='Daily Percentage Change', color='blue')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title('Daily Percentage Change in Prices')
    plt.xlabel('Date')
    plt.ylabel('Daily % Change')
    plt.legend()
    plt.grid()
    plt.show()

def calculate_rolling_stats(data, window):
    """Calculates rolling mean and standard deviation."""
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        return "Error: Input data must be a pandas Series or DataFrame."
    if not isinstance(window, int) or window <= 0:
        return "Error: Window size must be a positive integer."

    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    return pd.DataFrame({'rolling_mean': rolling_mean, 'rolling_std': rolling_std})

def plot_rolling_stats(data, rolling_stats, ticker):
    """Plots the rolling mean and standard deviation."""
    if isinstance(rolling_stats, str):
        print(f"Error for {ticker}: {rolling_stats}")
        return #stops the rest of the function from executing.
    rolling_mean = rolling_stats['rolling_mean']
    rolling_std = rolling_stats['rolling_std']
    plt.figure(figsize=(6, 3))
    plt.plot(data, label=f"{ticker} Closing Price", color="blue", alpha=0.6)
    plt.plot(rolling_mean, label=f"{ticker} {len(rolling_mean)}-day Moving Avg", color="orange", linewidth=2)
    plt.fill_between(data.index, rolling_mean - rolling_std, rolling_mean + rolling_std, color="gray", alpha=0.3, label="Volatility Range")
    plt.title(f"{ticker} Rolling Mean and Volatility")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to decompose time series into trend, seasonal, and residual components
def decompose_time_series(data, ticker, period=252):
    """
    Decomposes the time series of stock data into trend, seasonal, and residual components.
    """
    # Decompose the series using seasonal decomposition
    decomposition = seasonal_decompose(data, model='additive', period=period)
    
    # Plot the decomposition results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(411)
    plt.plot(data, label=f'{ticker} Closing Price', color='blue')
    plt.title(f'{ticker} Original Time Series')
    plt.legend(loc='best')

    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend', color='orange')
    plt.title('Trend Component')
    plt.legend(loc='best')

    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonality', color='green')
    plt.title('Seasonal Component')
    plt.legend(loc='best')

    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuals', color='red')
    plt.title('Residual Component')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.show()