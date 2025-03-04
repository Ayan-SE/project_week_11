import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_clean_understand(tickers, start_date, end_date):
    """
    """

    data = {}
    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if not stock_data.empty:
                data[ticker] = stock_data
                print(f"Data loaded successfully for {ticker}")
            else:
                print(f"No data available for {ticker} within the specified date range.")
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

    # Cleaning and Understanding
    for ticker, df in data.items():
        # Check for missing values
        print(f"\n--- {ticker} Data ---")
        print("Missing Values:\n", df.isnull().sum())

        # Handle missing values (e.g., forward fill or drop)
        df.fillna(method='ffill', inplace=True) # Forward fill missing values.
        #df.dropna(inplace=True) # alternative: drop rows with missing values

        # Basic descriptive statistics
        print("\nDescriptive Statistics:\n", df.describe())

        # Visualize adjusted closing prices
        plt.figure(figsize=(12, 6))
        plt.plot(df['Adj Close'], label=ticker)
        plt.title(f"{ticker} Adjusted Closing Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Calculate daily returns
        df['Daily Return'] = df['Adj Close'].pct_change()
        print("\nDaily Return statistics:\n", df['Daily Return'].describe())
        plt.figure(figsize=(12, 6))
        plt.plot(df['Daily Return'], label=ticker + ' Daily Return')
        plt.title(f"{ticker} Daily Returns")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True)
        plt.show()

        #calculate volatility
        df['Volatility'] = df['Daily Return'].rolling(window=20).std() * np.sqrt(252) #20 day rolling volatility, annualized
        print("\nVolatility statistics:\n", df['Volatility'].describe())
        plt.figure(figsize=(12, 6))
        plt.plot(df['Volatility'], label=ticker + ' Volatility')
        plt.title(f"{ticker} Volatility")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(True)
        plt.show()

    return data